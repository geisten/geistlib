/*
 * test_glm4moe_unit - the GLM-4.5 family on in-memory models (#242).
 *
 * mf_glm4moe_gguf writes a three-layer GLM-4.5 with F32 weights: one dense
 * lead layer, then two MoE layers of 8 experts (2 per token) plus one shared
 * expert; q/k/v biases, per-head QK norms, NEOX RoPE on the first 16 of 32
 * head dims, 4 query heads over 2 KV heads (q rows 128 against d_model 64);
 * a sigmoid router with exp_probs_b, renormalised and scaled by 2.5; one
 * NextN layer after the stack in block_count.
 *
 * The reference is this file's own forward pass in double, written from
 * llama.cpp's graph for the architecture (models/glm4-moe.cpp,
 * build_moe_ffn), not from geist's code.
 *
 *   - GLM-4.5 shape (QK norms) and Air shape (none): cpu_scalar within
 *     relative L2 1e-5 of the reference after a 21-token prompt (FP32 KV);
 *   - each mechanism matters: the reference without the attention biases,
 *     with RoPE over the whole head, without the shared expert or without
 *     the selection bias moves the logits by more than 1e-2;
 *   - every backend in the build: chunked and single-token prefill agree,
 *     a reset repeats its logits bit for bit, and the logits match
 *     cpu_scalar (cpu_neon and Metal within 1e-3, others cosine 0.999);
 *     a GPU backend's logits with its experts on the GPU match the reference
 *     within 1e-5 and (Metal) its own run with GEIST_MOE_HOT=0.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "model_fixtures.h"
#include "test_helpers.h"

#include "gguf_dequant.h"
#include "gguf_reader.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 96, N_PROMPT = 21, LAYERS = 3, LEAD = 1, D = 64, HEADS = 4, KV_HEADS = 2 };
enum { HD = 32, ROT = 16, QO = HEADS * HD, KO = KV_HEADS * HD, FF = 48 };
enum { EXPERTS = 8, USED = 2, EFF = 16, SHARED = 1, SFF = SHARED * EFF };

static const struct mf_glm4moe GLM45 = {.layers    = LAYERS,
                                        .lead      = LEAD,
                                        .d_model   = D,
                                        .heads     = HEADS,
                                        .kv_heads  = KV_HEADS,
                                        .head_dim  = HD,
                                        .rot       = ROT,
                                        .ff        = FF,
                                        .experts   = EXPERTS,
                                        .used      = USED,
                                        .expert_ff = EFF,
                                        .shared    = SHARED,
                                        .vocab     = VOCAB,
                                        .seed      = 47,
                                        .w_scale   = 2.5f};

static const struct geist_session_opts OPTS = {
        .top_p = 1.0f, .m_max = 8, .max_seq_len = 256, .kv_mode = GEIST_KV_FP32};

static geist_token_t prompt_id(size_t i) {
    return (geist_token_t) ((i * 37 + 5) % VOCAB);
}

/* ---- The reference --------------------------------------------------- */

struct knobs {
    bool no_bias, full_rope, no_shared, no_sel_bias;
};

struct ref_layer {
    float *attn_norm, *wq, *bq, *wk, *bk, *wv, *bv, *wo, *q_norm, *k_norm, *ffn_norm;
    float *wg, *wu, *wd;            /* dense */
    float *wr, *sel, *eg, *eu, *ed; /* experts */
    float *sg, *su, *sd;            /* shared */
};

struct ref_model {
    float           *embd, *out, *out_norm;
    struct ref_layer l[LAYERS];
};

[[gnu::format(printf, 2, 3)]] static float *tensor(struct gguf_ctx *g, const char *fmt, ...) {
    char    name[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    const struct gguf_tensor_t *t = gguf_get_tensor(g, name);
    return t != nullptr ? gguf_dequant_to_fp32(t) : nullptr;
}

static void ref_load(struct gguf_ctx *g, struct ref_model *m) {
    m->embd     = tensor(g, "token_embd.weight");
    m->out      = tensor(g, "output.weight");
    m->out_norm = tensor(g, "output_norm.weight");
    for (int l = 0; l < LAYERS; l++) {
        struct ref_layer *r = &m->l[l];
        r->attn_norm        = tensor(g, "blk.%d.attn_norm.weight", l);
        r->wq               = tensor(g, "blk.%d.attn_q.weight", l);
        r->bq               = tensor(g, "blk.%d.attn_q.bias", l);
        r->wk               = tensor(g, "blk.%d.attn_k.weight", l);
        r->bk               = tensor(g, "blk.%d.attn_k.bias", l);
        r->wv               = tensor(g, "blk.%d.attn_v.weight", l);
        r->bv               = tensor(g, "blk.%d.attn_v.bias", l);
        r->wo               = tensor(g, "blk.%d.attn_output.weight", l);
        r->q_norm           = tensor(g, "blk.%d.attn_q_norm.weight", l);
        r->k_norm           = tensor(g, "blk.%d.attn_k_norm.weight", l);
        r->ffn_norm         = tensor(g, "blk.%d.post_attention_norm.weight", l);
        r->wg               = tensor(g, "blk.%d.ffn_gate.weight", l);
        r->wu               = tensor(g, "blk.%d.ffn_up.weight", l);
        r->wd               = tensor(g, "blk.%d.ffn_down.weight", l);
        r->wr               = tensor(g, "blk.%d.ffn_gate_inp.weight", l);
        r->sel              = tensor(g, "blk.%d.exp_probs_b.bias", l);
        r->eg               = tensor(g, "blk.%d.ffn_gate_exps.weight", l);
        r->eu               = tensor(g, "blk.%d.ffn_up_exps.weight", l);
        r->ed               = tensor(g, "blk.%d.ffn_down_exps.weight", l);
        r->sg               = tensor(g, "blk.%d.ffn_gate_shexp.weight", l);
        r->su               = tensor(g, "blk.%d.ffn_up_shexp.weight", l);
        r->sd               = tensor(g, "blk.%d.ffn_down_shexp.weight", l);
    }
}

static void ref_free(struct ref_model *m) {
    free(m->embd);
    free(m->out);
    free(m->out_norm);
    for (int l = 0; l < LAYERS; l++) {
        float **all = &m->l[l].attn_norm;
        for (size_t i = 0; i < sizeof m->l[l] / sizeof(float *); i++)
            free(all[i]);
    }
}

static void rmsnorm(size_t n, const double *x, const float *w, double *y) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++)
        ss += x[i] * x[i];
    const double r = 1.0 / sqrt(ss / (double) n + 1e-6);
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] * r * w[i];
}

/* y = w x (+ b when b is non-null and not switched off). */
static void
matvec(size_t n_out, size_t n_in, const float *w, const float *b, const double *x, double *y) {
    for (size_t o = 0; o < n_out; o++) {
        double acc = b != nullptr ? (double) b[o] : 0.0;
        for (size_t i = 0; i < n_in; i++)
            acc += (double) w[o * n_in + i] * x[i];
        y[o] = acc;
    }
}

static void
swiglu(size_t ff, const float *wg, const float *wu, const float *wd, const double *x, double *y) {
    double g[FF], u[FF];
    matvec(ff, D, wg, nullptr, x, g);
    matvec(ff, D, wu, nullptr, x, u);
    for (size_t i = 0; i < ff; i++)
        g[i] = g[i] / (1.0 + exp(-g[i])) * u[i];
    matvec(D, ff, wd, nullptr, g, y);
}

/* ggml's NEOX rope on the first `rot` dims at x: pairs (i, i + rot/2),
 * theta = p * base^(-2i/rot). */
static void rope(double *x, size_t p, size_t rot) {
    for (size_t i = 0; i < rot / 2; i++) {
        const double th = (double) p * pow(10000.0, -2.0 * (double) i / (double) rot);
        const double x0 = x[i], x1 = x[i + rot / 2];
        x[i]           = x0 * cos(th) - x1 * sin(th);
        x[i + rot / 2] = x0 * sin(th) + x1 * cos(th);
    }
}

/* build_moe_ffn's sigmoid router: selection by sigmoid + exp_probs_b (ties:
 * lower id), weights the unbiased sigmoids, renormalised, times w_scale. */
static void route(const struct ref_layer *r,
                  const struct knobs      k,
                  double                  w_scale,
                  const double           *x,
                  size_t                  ids[static USED],
                  double                  w[static USED]) {
    double logit[EXPERTS], s[EXPERTS], sel[EXPERTS];
    bool   taken[EXPERTS] = {0};
    matvec(EXPERTS, D, r->wr, nullptr, x, logit);
    for (size_t e = 0; e < EXPERTS; e++) {
        s[e]   = 1.0 / (1.0 + exp(-logit[e]));
        sel[e] = s[e] + (k.no_sel_bias ? 0.0 : (double) r->sel[e]);
    }
    double sum = 0.0;
    for (size_t j = 0; j < USED; j++) {
        size_t best = EXPERTS;
        for (size_t e = 0; e < EXPERTS; e++)
            if (!taken[e] && (best == EXPERTS || sel[e] > sel[best]))
                best = e;
        taken[best] = true;
        ids[j]      = best;
        w[j]        = s[best];
        sum += w[j];
    }
    for (size_t j = 0; j < USED; j++)
        w[j] = w[j] / fmax(sum, 6.103515625e-5) * w_scale;
}

static void ref_forward(const struct ref_model  *m,
                        const struct mf_glm4moe *g,
                        struct knobs             k,
                        double                   logits[static VOCAB]) {
    static double kc[LAYERS][N_PROMPT][KO], vc[LAYERS][N_PROMPT][KO];
    const size_t  rot = k.full_rope ? HD : g->rot;
    double        h[D], x[D], q[QO], kk[KO], v[KO], att[QO], o[D], y[D], sc[N_PROMPT];
    for (size_t p = 0; p < N_PROMPT; p++) {
        const float *e = m->embd + (size_t) prompt_id(p) * D;
        for (size_t i = 0; i < D; i++)
            h[i] = e[i];
        for (size_t l = 0; l < LAYERS; l++) {
            const struct ref_layer *r = &m->l[l];
            rmsnorm(D, h, r->attn_norm, x);
            matvec(QO, D, r->wq, k.no_bias ? nullptr : r->bq, x, q);
            matvec(KO, D, r->wk, k.no_bias ? nullptr : r->bk, x, kk);
            matvec(KO, D, r->wv, k.no_bias ? nullptr : r->bv, x, v);
            for (size_t hh = 0; hh < HEADS; hh++) {
                if (r->q_norm != nullptr)
                    rmsnorm(HD, q + hh * HD, r->q_norm, q + hh * HD);
                rope(q + hh * HD, p, rot);
            }
            for (size_t hh = 0; hh < KV_HEADS; hh++) {
                if (r->k_norm != nullptr)
                    rmsnorm(HD, kk + hh * HD, r->k_norm, kk + hh * HD);
                rope(kk + hh * HD, p, rot);
            }
            memcpy(kc[l][p], kk, sizeof kk);
            memcpy(vc[l][p], v, sizeof v);
            for (size_t hh = 0; hh < HEADS; hh++) {
                const size_t kvh = hh / (HEADS / KV_HEADS);
                double       mx = -INFINITY, den = 0.0;
                for (size_t t = 0; t <= p; t++) {
                    double dot = 0.0;
                    for (size_t i = 0; i < HD; i++)
                        dot += q[hh * HD + i] * kc[l][t][kvh * HD + i];
                    sc[t] = dot / sqrt((double) HD);
                    mx    = fmax(mx, sc[t]);
                }
                for (size_t t = 0; t <= p; t++) {
                    sc[t] = exp(sc[t] - mx);
                    den += sc[t];
                }
                for (size_t i = 0; i < HD; i++) {
                    double acc = 0.0;
                    for (size_t t = 0; t <= p; t++)
                        acc += sc[t] * vc[l][t][kvh * HD + i];
                    att[hh * HD + i] = acc / den;
                }
            }
            matvec(D, QO, r->wo, nullptr, att, o);
            for (size_t i = 0; i < D; i++)
                h[i] += o[i];
            rmsnorm(D, h, r->ffn_norm, x);
            if (l < LEAD) {
                swiglu(FF, r->wg, r->wu, r->wd, x, y);
                for (size_t i = 0; i < D; i++)
                    h[i] += y[i];
                continue;
            }
            size_t ids[USED];
            double w[USED];
            route(r, k, g->w_scale, x, ids, w);
            for (size_t j = 0; j < USED; j++) {
                const size_t ge = ids[j] * EFF * D;
                swiglu(EFF, r->eg + ge, r->eu + ge, r->ed + ge, x, y);
                for (size_t i = 0; i < D; i++)
                    h[i] += w[j] * y[i];
            }
            if (!k.no_shared) {
                swiglu(SFF, r->sg, r->su, r->sd, x, y);
                for (size_t i = 0; i < D; i++)
                    h[i] += y[i];
            }
        }
    }
    rmsnorm(D, h, m->out_norm, x);
    matvec(VOCAB, D, m->out, nullptr, x, logits);
}

/* ---- Engine runs ------------------------------------------------------ */

static bool logits_after(struct geist_session *s, size_t step, float out[static VOCAB]) {
    geist_token_t ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = prompt_id(i);
    if (geist_session_reset(s) != GEIST_OK)
        return false;
    for (size_t i = 0; i < N_PROMPT; i += step) {
        const size_t k = N_PROMPT - i < step ? N_PROMPT - i : step;
        if (geist_session_prefill_tokens(s, k, ids + i) != GEIST_OK)
            return false;
    }
    size_t       nl = 0;
    const float *lg = geist_session_peek_logits(&nl, s);
    if (lg == nullptr || nl != VOCAB)
        return false;
    memcpy(out, lg, VOCAB * sizeof(float));
    return true;
}

static double rel_l2_d(const float a[static VOCAB], const double b[static VOCAB]) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += b[i] * b[i];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

static double rel_l2_f(const float a[static VOCAB], const float b[static VOCAB]) {
    double num = 0.0, den = 0.0;
    for (size_t j = 0; j < VOCAB; j++) {
        num += ((double) a[j] - b[j]) * ((double) a[j] - b[j]);
        den += (double) b[j] * b[j];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

static double cosine(const float a[static VOCAB], const float b[static VOCAB]) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / sqrt(aa * bb > 0.0 ? aa * bb : 1.0);
}

/* Chunked prefill and one token at a time agree within this much of the
 * logits' scale (as test_deepseek2_unit). */
static constexpr double ROWS_TOL = 1e-4;

/* The fixture's logits on backend `name`; -1: no such backend. */
static int check_backend(const char *name, const struct mf_glm4moe *g, float out[static VOCAB]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("skip %s: not in this build\n", name);
        return -1;
    }
    struct tf_buf         b     = mf_glm4moe_gguf(g);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    float                 chunked[VOCAB], single[VOCAB], again[VOCAB];
    if (geist_model_load_from_memory(b.b, b.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK) {
        fprintf(stderr, "  %s: %s\n", name, geist_last_create_error());
        fails += geist_expect(false, "the glm4moe fixture loads");
    } else if (!logits_after(s, N_PROMPT, chunked) || !logits_after(s, 1, single) ||
               !logits_after(s, N_PROMPT, again)) {
        fprintf(stderr, "  %s: %s\n", name, geist_session_errmsg(s));
        fails += geist_expect(false, "prefill runs");
    } else {
        double scale = 0.0, diff = 0.0;
        for (size_t i = 0; i < VOCAB; i++) {
            scale = fmax(scale, fabs(chunked[i]));
            diff  = fmax(diff, fabs((double) chunked[i] - single[i]));
        }
        printf("%s: chunks vs one at a time: max|d| %.3g of scale %.3g\n", name, diff, scale);
        fails += geist_expect(diff <= ROWS_TOL * scale, "chunks and single tokens agree");
        fails += geist_expect(memcmp(chunked, again, sizeof chunked) == 0,
                              "a reset session reproduces its logits bit for bit");
        memcpy(out, chunked, sizeof chunked);
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(b.b);
    return fails;
}

static int check_shape(const char *what, const struct mf_glm4moe *g) {
    printf("== %s\n", what);
    struct tf_buf    b   = mf_glm4moe_gguf(g);
    const char      *err = nullptr;
    struct gguf_ctx *gc  = gguf_open_memory(b.b, b.n, &err);
    if (gc == nullptr) {
        fprintf(stderr, "cannot read the fixture: %s\n", err);
        free(b.b);
        return 1;
    }
    struct ref_model rm = {0};
    static double    ref[VOCAB], alt[VOCAB];
    ref_load(gc, &rm);
    ref_forward(&rm, g, (struct knobs) {0}, ref);
    int       fails = 0;
    float     scalar[VOCAB], other[VOCAB];
    const int sc = check_backend("cpu_scalar", g, scalar);
    fails += sc != 0 ? geist_expect(false, "cpu_scalar runs the fixture") : 0;
    if (sc == 0) {
        const double d = rel_l2_d(scalar, ref);
        printf("cpu_scalar vs the double reference: relative L2 %.3g\n", d);
        fails += geist_expect(d <= 1e-5, "cpu_scalar matches the reference within 1e-5");
        const struct {
            const char  *what;
            struct knobs k;
        } knobs[] = {
                {"the attention biases", {.no_bias = true}},
                {"partial RoPE (the whole head rotated instead)", {.full_rope = true}},
                {"the shared expert", {.no_shared = true}},
                {"the selection bias", {.no_sel_bias = true}},
        };
        for (size_t i = 0; i < sizeof knobs / sizeof knobs[0]; i++) {
            ref_forward(&rm, g, knobs[i].k, alt);
            const double moved = rel_l2_d(scalar, alt);
            printf("  without %s: relative L2 %.3g\n", knobs[i].what, moved);
            fails += geist_expect(moved > 1e-2, "each mechanism moves the logits");
        }
    }
    const char *const more[] = {"cpu_neon", "cpu_x86", "metal", "vulkan"};
    for (size_t i = 0; i < sizeof more / sizeof more[0]; i++) {
        const int r = check_backend(more[i], g, other);
        if (r < 0)
            continue;
        fails += r;
        if (r == 0 && sc == 0) {
            const double d = rel_l2_f(other, scalar), c = cosine(other, scalar);
            printf("%s vs cpu_scalar: relative L2 %.3g, cosine %.6f\n", more[i], d, c);
            const bool exact = strcmp(more[i], "cpu_neon") == 0 || strcmp(more[i], "metal") == 0;
            fails += geist_expect(exact ? d <= 1e-3 : c >= 0.999,
                                  "a backend matches cpu_scalar within its tolerance");
        }
        /* A GPU's experts (Metal; Vulkan, #499) against the reference and
         * against its own run with the experts on the CPU. */
        if (r == 0 && (strcmp(more[i], "metal") == 0 || strcmp(more[i], "vulkan") == 0)) {
            static float cpu_exp[VOCAB];
            const double dr = rel_l2_d(other, ref);
            printf("%s vs the double reference: relative L2 %.3g\n", more[i], dr);
            fails += geist_expect(dr <= 1e-5, "the GPU matches the reference within 1e-5");
            setenv("GEIST_MOE_HOT", "0", 1);
            const int rc = check_backend(more[i], g, cpu_exp);
            unsetenv("GEIST_MOE_HOT");
            fails += geist_expect(rc == 0, "the GPU runs with its experts on the CPU");
            if (rc == 0) {
                const double dc = rel_l2_f(other, cpu_exp);
                printf("%s, experts on the GPU vs on the CPU: relative L2 %.3g\n", more[i], dc);
                if (strcmp(more[i], "metal") == 0)
                    fails += geist_expect(dc <= 1e-5, "the GPU experts match the CPU experts");
            }
        }
    }
    ref_free(&rm);
    gguf_close(gc);
    free(b.b);
    return fails;
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    int               fails = check_shape("GLM-4.5 shape (QK norms)", &GLM45);
    struct mf_glm4moe air   = GLM45;
    air.no_qk_norm          = true;
    air.seed                = 53;
    fails += check_shape("GLM-4.5-Air shape (no QK norms)", &air);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
