/*
 * test_phimoe_unit - the Phi-MoE family on an in-memory model (#523).
 *
 * mf_phimoe_gguf writes a three-layer Phi-MoE with F32 weights: LayerNorms
 * with a bias before the attention, before the FFN and before the lm_head,
 * q/k/v/o biases, an lm_head bias, 4 query heads over 2 KV heads, NEOX RoPE
 * over the whole head, and 16 experts, 2 per token.
 *
 * The reference is this file's own forward pass in double, written from the
 * model's reference implementation (microsoft/Phi-mini-MoE-instruct,
 * modeling_slimmoe.py: nn.LayerNorm, sparsemixer in eval mode), not from
 * geist's code. llama.cpp's graph for the architecture (models/phi3.cpp at
 * d81235049) differs from it in two places, which the reference can switch
 * on: the norms as RMSNorm plus bias, and the experts weighted by the
 * softmax over all of them, renormalised.
 *
 *   - cpu_scalar within relative L2 1e-5 of the reference after a 21-token
 *     prompt (FP32 KV), with and without an lm_head bias;
 *   - each mechanism matters: without the attention biases, the lm_head
 *     bias or the output norm's bias, with RMSNorm-plus-bias norms, or with
 *     the softmax router, the reference moves by more than 1e-2; and the
 *     prompt meets SparseMixer's jitter band (a weight below 1);
 *   - every backend in the build: chunked and single-token prefill agree,
 *     a reset repeats its logits bit for bit, and the logits match
 *     cpu_scalar (cpu_neon and Metal within 1e-3, others cosine 0.999); a
 *     GPU backend matches the reference within 1e-5;
 *   - GPU mode with none, half and all of the experts hot (moe_hot_check.h);
 *   - a file with LongRoPE factors, or with another number of experts per
 *     token than two, is refused.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "gguf_dequant.h"
#include "gguf_reader.h"
#include "model_fixtures.h"
#include "moe_hot_check.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 96, N_PROMPT = 21, LAYERS = 3, D = 64, HEADS = 4, KV_HEADS = 2 };
enum { HD = D / HEADS, KO = KV_HEADS * HD, EXPERTS = 16, USED = 2, EFF = 16 };

static constexpr double NORM_EPS = 1e-5;
static constexpr double JITTER   = 0.01; /* config.router_jitter_noise */

static const struct mf_phimoe PHI = {.layers    = LAYERS,
                                     .d_model   = D,
                                     .heads     = HEADS,
                                     .kv_heads  = KV_HEADS,
                                     .experts   = EXPERTS,
                                     .used      = USED,
                                     .expert_ff = EFF,
                                     .vocab     = VOCAB,
                                     .seed      = 59};

static const struct geist_session_opts OPTS = {
        .top_p = 1.0f, .m_max = 8, .max_seq_len = 256, .kv_mode = GEIST_KV_FP32};

static geist_token_t prompt_id(size_t i) {
    return (geist_token_t) ((i * 37 + 5) % VOCAB);
}

/* ---- The reference --------------------------------------------------- */

struct knobs {
    bool no_attn_bias, no_lm_bias, no_out_norm_bias;
    bool rms_norm;       /* llama.cpp: RMSNorm, then weight and bias */
    bool softmax_router; /* llama.cpp: softmax over all, top 2 renormalised */
};

struct ref_layer {
    float *attn_norm, *attn_norm_b, *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;
    float *ffn_norm, *ffn_norm_b, *wr, *eg, *eu, *ed;
};

struct ref_model {
    float           *embd, *out, *out_b, *out_norm, *out_norm_b;
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
    m->embd       = tensor(g, "token_embd.weight");
    m->out        = tensor(g, "output.weight");
    m->out_b      = tensor(g, "output.bias");
    m->out_norm   = tensor(g, "output_norm.weight");
    m->out_norm_b = tensor(g, "output_norm.bias");
    for (int l = 0; l < LAYERS; l++) {
        struct ref_layer *r = &m->l[l];
        r->attn_norm        = tensor(g, "blk.%d.attn_norm.weight", l);
        r->attn_norm_b      = tensor(g, "blk.%d.attn_norm.bias", l);
        r->wq               = tensor(g, "blk.%d.attn_q.weight", l);
        r->bq               = tensor(g, "blk.%d.attn_q.bias", l);
        r->wk               = tensor(g, "blk.%d.attn_k.weight", l);
        r->bk               = tensor(g, "blk.%d.attn_k.bias", l);
        r->wv               = tensor(g, "blk.%d.attn_v.weight", l);
        r->bv               = tensor(g, "blk.%d.attn_v.bias", l);
        r->wo               = tensor(g, "blk.%d.attn_output.weight", l);
        r->bo               = tensor(g, "blk.%d.attn_output.bias", l);
        r->ffn_norm         = tensor(g, "blk.%d.ffn_norm.weight", l);
        r->ffn_norm_b       = tensor(g, "blk.%d.ffn_norm.bias", l);
        r->wr               = tensor(g, "blk.%d.ffn_gate_inp.weight", l);
        r->eg               = tensor(g, "blk.%d.ffn_gate_exps.weight", l);
        r->eu               = tensor(g, "blk.%d.ffn_up_exps.weight", l);
        r->ed               = tensor(g, "blk.%d.ffn_down_exps.weight", l);
    }
}

static void ref_free(struct ref_model *m) {
    free(m->embd);
    free(m->out);
    free(m->out_b);
    free(m->out_norm);
    free(m->out_norm_b);
    for (int l = 0; l < LAYERS; l++) {
        float **all = &m->l[l].attn_norm;
        for (size_t i = 0; i < sizeof m->l[l] / sizeof(float *); i++)
            free(all[i]);
    }
}

/* nn.LayerNorm(n, eps, elementwise_affine=True); with `rms` the mean stays
 * in (ggml_rms_norm, then the weight and the bias). b may be nullptr. */
static void norm(size_t n, bool rms, const double *x, const float *w, const float *b, double *y) {
    double sum = 0.0, ss = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += x[i];
    const double mean = rms ? 0.0 : sum / (double) n;
    for (size_t i = 0; i < n; i++)
        ss += (x[i] - mean) * (x[i] - mean);
    const double r = 1.0 / sqrt(ss / (double) n + NORM_EPS);
    for (size_t i = 0; i < n; i++)
        y[i] = (x[i] - mean) * r * w[i] + (b != nullptr ? (double) b[i] : 0.0);
}

/* y = w x (+ b when b is non-null). */
static void
matvec(size_t n_out, size_t n_in, const float *w, const float *b, const double *x, double *y) {
    for (size_t o = 0; o < n_out; o++) {
        double acc = b != nullptr ? (double) b[o] : 0.0;
        for (size_t i = 0; i < n_in; i++)
            acc += (double) w[o * n_in + i] * x[i];
        y[o] = acc;
    }
}

/* PhiMoEBlockSparseTop2MLP: w2(silu(w1 x) * w3 x), GGUF gate / up / down. */
static void swiglu(const float *wg, const float *wu, const float *wd, const double *x, double *y) {
    double g[EFF], u[EFF];
    matvec(EFF, D, wg, nullptr, x, g);
    matvec(EFF, D, wu, nullptr, x, u);
    for (size_t i = 0; i < EFF; i++)
        g[i] = g[i] / (1.0 + exp(-g[i])) * u[i];
    matvec(D, EFF, wd, nullptr, g, y);
}

/* ggml's NEOX rope over the whole head: pairs (i, i + HD/2). */
static void rope(double *x, size_t p) {
    for (size_t i = 0; i < HD / 2; i++) {
        const double th = (double) p * pow(10000.0, -2.0 * (double) i / (double) HD);
        const double x0 = x[i], x1 = x[i + HD / 2];
        x[i]          = x0 * cos(th) - x1 * sin(th);
        x[i + HD / 2] = x0 * sin(th) + x1 * cos(th);
    }
}

/* One pick of sparsemixer() in eval mode over `scores`, the experts in
 * `gone` already picked: the largest score left, and its softmax over the
 * scores left that the threshold does not mask,
 *   (top - s) / clamp(|s|, min = top) > 2 * jitter_eps. */
static size_t
mixer_pick(const double scores[static EXPERTS], const bool gone[static EXPERTS], double *w) {
    size_t best = EXPERTS;
    for (size_t e = 0; e < EXPERTS; e++)
        if (!gone[e] && (best == EXPERTS || scores[e] > scores[best]))
            best = e;
    const double top = scores[best];
    double       z   = 0.0;
    for (size_t e = 0; e < EXPERTS; e++) {
        const double factor = fmax(fabs(scores[e]), top);
        const bool   masked = factor != 0.0 && (top - scores[e]) / factor > 2.0 * JITTER;
        if (!gone[e] && !masked)
            z += exp(scores[e] - top);
    }
    *w = 1.0 / z;
    return best;
}

/* The two experts of x and their weights; returns whether a weight is
 * below 1 (the jitter band held more than the pick). */
static bool route(const struct ref_layer *r,
                  struct knobs            k,
                  const double           *x,
                  size_t                  ids[static USED],
                  double                  w[static USED]) {
    double logit[EXPERTS];
    bool   gone[EXPERTS] = {0};
    matvec(EXPERTS, D, r->wr, nullptr, x, logit);
    if (!k.softmax_router) {
        for (size_t j = 0; j < USED; j++) {
            ids[j]       = mixer_pick(logit, gone, &w[j]);
            gone[ids[j]] = true;
        }
        return w[0] < 1.0 - 1e-9 || w[1] < 1.0 - 1e-9;
    }
    /* build_moe_ffn, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, norm_w. */
    double sum = 0.0;
    for (size_t j = 0; j < USED; j++) {
        size_t best = EXPERTS;
        for (size_t e = 0; e < EXPERTS; e++)
            if (!gone[e] && (best == EXPERTS || logit[e] > logit[best]))
                best = e;
        gone[best] = true;
        ids[j]     = best;
        w[j]       = exp(logit[best] - logit[ids[0]]);
        sum += w[j];
    }
    for (size_t j = 0; j < USED; j++)
        w[j] /= sum;
    return false;
}

/* The logits after the prompt; returns the routings that met the band. */
static size_t ref_forward(const struct ref_model *m, struct knobs k, double logits[static VOCAB]) {
    static double kc[LAYERS][N_PROMPT][KO], vc[LAYERS][N_PROMPT][KO];
    double        h[D], x[D], q[D], kk[KO], v[KO], att[D], o[D], y[D], sc[N_PROMPT];
    size_t        banded = 0;
    for (size_t p = 0; p < N_PROMPT; p++) {
        const float *e = m->embd + (size_t) prompt_id(p) * D;
        for (size_t i = 0; i < D; i++)
            h[i] = e[i];
        for (size_t l = 0; l < LAYERS; l++) {
            const struct ref_layer *r = &m->l[l];
            norm(D, k.rms_norm, h, r->attn_norm, r->attn_norm_b, x);
            matvec(D, D, r->wq, k.no_attn_bias ? nullptr : r->bq, x, q);
            matvec(KO, D, r->wk, k.no_attn_bias ? nullptr : r->bk, x, kk);
            matvec(KO, D, r->wv, k.no_attn_bias ? nullptr : r->bv, x, v);
            for (size_t hh = 0; hh < HEADS; hh++)
                rope(q + hh * HD, p);
            for (size_t hh = 0; hh < KV_HEADS; hh++)
                rope(kk + hh * HD, p);
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
            matvec(D, D, r->wo, k.no_attn_bias ? nullptr : r->bo, att, o);
            for (size_t i = 0; i < D; i++)
                h[i] += o[i];
            norm(D, k.rms_norm, h, r->ffn_norm, r->ffn_norm_b, x);
            size_t ids[USED];
            double w[USED];
            banded += route(r, k, x, ids, w) ? 1 : 0;
            for (size_t j = 0; j < USED; j++) {
                const size_t ge = ids[j] * EFF * D;
                swiglu(r->eg + ge, r->eu + ge, r->ed + ge, x, y);
                for (size_t i = 0; i < D; i++)
                    h[i] += w[j] * y[i];
            }
        }
    }
    norm(D, k.rms_norm, h, m->out_norm, k.no_out_norm_bias ? nullptr : m->out_norm_b, x);
    matvec(VOCAB, D, m->out, k.no_lm_bias ? nullptr : m->out_b, x, logits);
    return banded;
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

static double rel_l2_dd(const double a[static VOCAB], const double b[static VOCAB]) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += b[i] * b[i];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

/* Chunked prefill and one token at a time agree within this much of the
 * logits' scale (as test_glm4moe_unit). */
static constexpr double ROWS_TOL = 1e-4;

/* The fixture's logits on backend `name`; -1: no such backend. */
static int check_backend(const char *name, const struct mf_phimoe *g, float out[static VOCAB]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("skip %s: not in this build\n", name);
        return -1;
    }
    struct tf_buf         b     = mf_phimoe_gguf(g);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    float                 chunked[VOCAB], single[VOCAB], again[VOCAB];
    if (geist_model_load_from_memory(b.b, b.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK) {
        fprintf(stderr, "  %s: %s\n", name, geist_last_create_error());
        fails += geist_expect(false, "the phimoe fixture loads");
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

/* The reference of fixture `g` into ref; the banded routings in *banded. */
static bool
reference(const struct mf_phimoe *g, struct knobs k, double ref[static VOCAB], size_t *banded) {
    struct tf_buf    b   = mf_phimoe_gguf(g);
    const char      *err = nullptr;
    struct gguf_ctx *gc  = gguf_open_memory(b.b, b.n, &err);
    if (gc == nullptr) {
        fprintf(stderr, "cannot read the fixture: %s\n", err);
        free(b.b);
        return false;
    }
    struct ref_model rm = {0};
    ref_load(gc, &rm);
    *banded = ref_forward(&rm, k, ref);
    ref_free(&rm);
    gguf_close(gc);
    free(b.b);
    return true;
}

static int check_family(void) {
    static double ref[VOCAB], alt[VOCAB];
    size_t        banded = 0, unused = 0;
    if (!reference(&PHI, (struct knobs) {0}, ref, &banded))
        return 1;
    int       fails = 0;
    float     scalar[VOCAB], other[VOCAB];
    const int sc = check_backend("cpu_scalar", &PHI, scalar);
    fails += sc != 0 ? geist_expect(false, "cpu_scalar runs the fixture") : 0;
    if (sc == 0) {
        const double d = rel_l2_d(scalar, ref);
        printf("cpu_scalar vs the double reference: relative L2 %.3g; %zu of %d routings met "
               "the jitter band\n",
               d,
               banded,
               N_PROMPT * LAYERS);
        fails += geist_expect(d <= 1e-5, "cpu_scalar matches the reference within 1e-5");
        fails += geist_expect(banded > 0, "the prompt meets SparseMixer's jitter band");
    }
    const struct {
        const char  *what;
        struct knobs k;
    } knobs[] = {
            {"the attention biases", {.no_attn_bias = true}},
            {"the lm_head bias", {.no_lm_bias = true}},
            {"the output norm's bias", {.no_out_norm_bias = true}},
            {"LayerNorm (RMSNorm plus bias, llama.cpp's norms)", {.rms_norm = true}},
            {"SparseMixer (softmax top-2 renormalised, llama.cpp's router)",
             {.softmax_router = true}},
            {"either (llama.cpp's graph)", {.rms_norm = true, .softmax_router = true}},
    };
    for (size_t i = 0; i < sizeof knobs / sizeof knobs[0]; i++) {
        if (!reference(&PHI, knobs[i].k, alt, &unused))
            return fails + 1;
        const double moved = rel_l2_dd(alt, ref);
        printf("  without %s: relative L2 %.3g\n", knobs[i].what, moved);
        fails += geist_expect(moved > 1e-2, "each mechanism moves the logits");
    }
    const char *const more[] = {"cpu_neon", "cpu_x86", "metal", "vulkan"};
    for (size_t i = 0; i < sizeof more / sizeof more[0]; i++) {
        const int r = check_backend(more[i], &PHI, other);
        if (r < 0)
            continue;
        fails += r;
        if (r == 0 && sc == 0) {
            const double d   = rel_l2_d(other, ref);
            double       num = 0.0, den = 0.0, ab = 0.0, aa = 0.0;
            for (size_t j = 0; j < VOCAB; j++) {
                num += ((double) other[j] - scalar[j]) * ((double) other[j] - scalar[j]);
                den += (double) scalar[j] * scalar[j];
                ab += (double) other[j] * scalar[j];
                aa += (double) other[j] * other[j];
            }
            const double rel = sqrt(num / (den > 0.0 ? den : 1.0));
            const double cs  = ab / sqrt(aa * den > 0.0 ? aa * den : 1.0);
            printf("%s vs cpu_scalar: relative L2 %.3g, cosine %.6f; vs the reference %.3g\n",
                   more[i],
                   rel,
                   cs,
                   d);
            const bool gpu   = strcmp(more[i], "metal") == 0 || strcmp(more[i], "vulkan") == 0;
            const bool exact = strcmp(more[i], "cpu_neon") == 0 || strcmp(more[i], "metal") == 0;
            fails += geist_expect(exact ? rel <= 1e-3 : cs >= 0.999,
                                  "a backend matches cpu_scalar within its tolerance");
            if (gpu)
                fails += geist_expect(d <= 1e-5, "the GPU matches the reference within 1e-5");
        }
    }
    /* GPU mode with hot experts (#523 item 5). */
    if (sc == 0) {
        geist_token_t ids[N_PROMPT];
        for (size_t i = 0; i < N_PROMPT; i++)
            ids[i] = prompt_id(i);
        struct tf_buf blob = mf_phimoe_gguf(&PHI);
        fails += moe_hot_checks(
                &blob, &OPTS, (size_t) LAYERS * EXPERTS, VOCAB, N_PROMPT, ids, scalar);
        free(blob.b);
    }
    return fails;
}

/* Without output.bias the head adds none (config.lm_head_bias off). */
static int check_no_lm_bias(void) {
    struct mf_phimoe g = PHI;
    g.no_lm_bias       = true;
    g.seed             = 61;
    static double ref[VOCAB];
    float         scalar[VOCAB];
    size_t        banded = 0;
    if (!reference(&g, (struct knobs) {.no_lm_bias = true}, ref, &banded))
        return 1;
    int fails = check_backend("cpu_scalar", &g, scalar) != 0;
    if (fails == 0) {
        const double d = rel_l2_d(scalar, ref);
        printf("no output.bias: cpu_scalar vs the reference: relative L2 %.3g\n", d);
        fails += geist_expect(d <= 1e-5, "a file without an lm_head bias matches too");
    }
    return fails;
}

static int refused(const struct mf_phimoe *g, const char *what) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return geist_expect(false, "cpu_scalar is in the build");
    struct tf_buf       b = mf_phimoe_gguf(g);
    struct geist_model *m = nullptr;
    const int           fails =
            geist_expect(geist_model_load_from_memory(b.b, b.n, be, &m) != GEIST_OK, what);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(b.b);
    return fails;
}

static int check_refusals(void) {
    struct mf_phimoe longrope = PHI, three = PHI;
    longrope.longrope = true;
    three.used        = 3;
    return refused(&longrope, "LongRoPE factors are refused") +
           refused(&three, "three experts per token are refused");
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    int fails = check_family();
    fails += check_no_lm_bias();
    fails += check_refusals();
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
