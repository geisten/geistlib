/*
 * test_llama4_unit - the Llama 4 family on an in-memory model (#244).
 *
 * mf_llama4_gguf writes a four-layer Llama 4 with F32 weights: layers 1 and
 * 3 (moe_step 2) route over 4 experts, top-1, plus a shared expert; 0 and 2
 * are dense. Layer 3 is the NoPE layer (no RoPE, whole context, q times the
 * temperature); 0-2 rope with rope_freqs factors 1, 1.5, 2, ..., normalise
 * q and k, and attend in chunks. attention.chunk_size 6 and
 * temperature_length 4 shrink llama.cpp's 8192, so a 21-token prompt
 * crosses three chunk boundaries and five temperature steps.
 *
 * The reference is this file's own forward pass in double, written from
 * llama.cpp's graph (llama4.cpp, build_moe_ffn with the sigmoid gate and
 * weight_before_ffn, the chunked mask, build_inp_attn_scale), in the file's
 * interleaved RoPE order; geist's load-time row reorder is not in it.
 *   - cpu_scalar within relative L2 1e-5 (FP32 KV);
 *   - each mechanism matters: the reference without the frequency factors,
 *     without chunks, without the temperature, without the q/k norm, with
 *     the gate weight after instead of before the expert, or without the
 *     shared expert moves the logits by more than 1e-2;
 *   - every backend: chunked and single-token prefill agree, a reset
 *     repeats its logits bit for bit, the logits match cpu_scalar.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 96, N_PROMPT = 21, LAYERS = 4, STEP = 2, D = 64, HEADS = 4, KVH = 2, HD = 16 };
enum { FF = 48, EXPERTS = 4, EFF = 32, CHUNK = 6, TEMP = 4 };

static const struct mf_llama4 GEOM = {.layers      = LAYERS,
                                      .moe_step    = STEP,
                                      .d_model     = D,
                                      .heads       = HEADS,
                                      .kv_heads    = KVH,
                                      .head_dim    = HD,
                                      .ff          = FF,
                                      .experts     = EXPERTS,
                                      .used        = 1,
                                      .expert_ff   = EFF,
                                      .vocab       = VOCAB,
                                      .chunk       = CHUNK,
                                      .temp_length = TEMP,
                                      .seed        = 47};

static const struct geist_session_opts OPTS = {
        .top_p = 1.0f, .m_max = 8, .max_seq_len = 256, .kv_mode = GEIST_KV_FP32};

static geist_token_t prompt_id(size_t i) {
    return (geist_token_t) ((i * 37 + 5) % VOCAB);
}

/* ---- The reference --------------------------------------------------- */

struct knobs {
    bool no_factors, no_chunks, no_temp, no_norm, weight_after, no_shared;
};

struct ref_layer {
    float *attn_norm, *wq, *wk, *wv, *wo, *ffn_norm, *wg, *wu, *wd, *wr, *eg, *eu, *ed, *sg, *su,
            *sd;
};

struct ref_model {
    float           *embd, *out, *out_norm, *ff;
    struct ref_layer l[LAYERS];
};

static float *tensor(struct gguf_ctx *g, const char *fmt, int layer) {
    char name[64];
    snprintf(name, sizeof name, fmt, layer);
    const struct gguf_tensor_t *t = gguf_get_tensor(g, name);
    return t != nullptr ? gguf_dequant_to_fp32(t) : nullptr;
}

static void ref_load(struct gguf_ctx *g, struct ref_model *m) {
    m->embd     = tensor(g, "token_embd.weight", 0);
    m->out      = tensor(g, "output.weight", 0);
    m->out_norm = tensor(g, "output_norm.weight", 0);
    m->ff       = tensor(g, "rope_freqs.weight", 0);
    for (int l = 0; l < LAYERS; l++) {
        struct ref_layer *r = &m->l[l];
        r->attn_norm        = tensor(g, "blk.%d.attn_norm.weight", l);
        r->wq               = tensor(g, "blk.%d.attn_q.weight", l);
        r->wk               = tensor(g, "blk.%d.attn_k.weight", l);
        r->wv               = tensor(g, "blk.%d.attn_v.weight", l);
        r->wo               = tensor(g, "blk.%d.attn_output.weight", l);
        r->ffn_norm         = tensor(g, "blk.%d.ffn_norm.weight", l);
        r->wg               = tensor(g, "blk.%d.ffn_gate.weight", l);
        r->wu               = tensor(g, "blk.%d.ffn_up.weight", l);
        r->wd               = tensor(g, "blk.%d.ffn_down.weight", l);
        r->wr               = tensor(g, "blk.%d.ffn_gate_inp.weight", l);
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
    free(m->ff);
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
    const double r = 1.0 / sqrt(ss / (double) n + 1e-5);
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] * r * (w != nullptr ? w[i] : 1.0);
}

static void matvec(size_t n_out, size_t n_in, const float *w, const double *x, double *y) {
    for (size_t o = 0; o < n_out; o++) {
        double acc = 0.0;
        for (size_t i = 0; i < n_in; i++)
            acc += (double) w[o * n_in + i] * x[i];
        y[o] = acc;
    }
}

static void
swiglu(size_t ff, const float *wg, const float *wu, const float *wd, const double *x, double *y) {
    double g[64], u[64];
    matvec(ff, D, wg, x, g);
    matvec(ff, D, wu, x, u);
    for (size_t i = 0; i < ff; i++)
        g[i] = g[i] / (1.0 + exp(-g[i])) * u[i];
    matvec(D, ff, wd, g, y);
}

/* NORM RoPE (pairs 2i, 2i+1) of each head of x at position p, pair i at
 * p * 500000^(-2i/HD) / ff[i]. */
static void rope(size_t heads, double *x, size_t p, const float *ff, bool no_factors) {
    for (size_t h = 0; h < heads; h++) {
        for (size_t i = 0; i < HD / 2; i++) {
            const double th =
                    (double) p * pow(500000.0, -2.0 * (double) i / HD) / (no_factors ? 1.0 : ff[i]);
            double      *xh = x + h * HD;
            const double a = xh[2 * i], b = xh[2 * i + 1];
            xh[2 * i]     = a * cos(th) - b * sin(th);
            xh[2 * i + 1] = a * sin(th) + b * cos(th);
        }
    }
}

static void ref_forward(const struct ref_model *m, struct knobs k, double logits[static VOCAB]) {
    static double kc[LAYERS][N_PROMPT][KVH * HD], vc[LAYERS][N_PROMPT][KVH * HD];
    double        h[D], x[D], q[HEADS * HD], att[HEADS * HD], o[D], y[D], xs[D];
    for (size_t p = 0; p < N_PROMPT; p++) {
        const float *e = m->embd + (size_t) prompt_id(p) * D;
        for (size_t i = 0; i < D; i++)
            h[i] = e[i];
        for (size_t l = 0; l < LAYERS; l++) {
            const struct ref_layer *r        = &m->l[l];
            const bool              use_rope = l % 4 != 3;
            rmsnorm(D, h, r->attn_norm, x);
            matvec(HEADS * HD, D, r->wq, x, q);
            matvec(KVH * HD, D, r->wk, x, kc[l][p]);
            matvec(KVH * HD, D, r->wv, x, vc[l][p]);
            if (use_rope) {
                rope(HEADS, q, p, m->ff, k.no_factors);
                rope(KVH, kc[l][p], p, m->ff, k.no_factors);
                if (!k.no_norm) {
                    for (size_t hh = 0; hh < HEADS; hh++)
                        rmsnorm(HD, q + hh * HD, nullptr, q + hh * HD);
                    for (size_t hh = 0; hh < KVH; hh++)
                        rmsnorm(HD, kc[l][p] + hh * HD, nullptr, kc[l][p] + hh * HD);
                }
            } else if (!k.no_temp) {
                const double t = log(floor((double) (p + 1) / TEMP) + 1.0) * 0.1 + 1.0;
                for (size_t i = 0; i < HEADS * HD; i++)
                    q[i] *= t;
            }
            const size_t s_lo = use_rope && !k.no_chunks ? p - p % CHUNK : 0;
            for (size_t hh = 0; hh < HEADS; hh++) {
                const size_t kvh = hh / (HEADS / KVH);
                double       sc[N_PROMPT], mx = -1e300, den = 0.0;
                for (size_t s = s_lo; s <= p; s++) {
                    double dot = 0.0;
                    for (size_t i = 0; i < HD; i++)
                        dot += q[hh * HD + i] * kc[l][s][kvh * HD + i];
                    sc[s] = dot / sqrt((double) HD);
                    mx    = fmax(mx, sc[s]);
                }
                for (size_t s = s_lo; s <= p; s++)
                    den += exp(sc[s] - mx);
                for (size_t i = 0; i < HD; i++) {
                    double acc = 0.0;
                    for (size_t s = s_lo; s <= p; s++)
                        acc += exp(sc[s] - mx) * vc[l][s][kvh * HD + i];
                    att[hh * HD + i] = acc / den;
                }
            }
            matvec(D, HEADS * HD, r->wo, att, o);
            for (size_t i = 0; i < D; i++)
                h[i] += o[i];
            rmsnorm(D, h, r->ffn_norm, x);
            if ((l + 1) % STEP != 0) {
                swiglu(FF, r->wg, r->wu, r->wd, x, y);
                for (size_t i = 0; i < D; i++)
                    h[i] += y[i];
                continue;
            }
            double lg[EXPERTS];
            matvec(EXPERTS, D, r->wr, x, lg);
            size_t best = 0;
            for (size_t e2 = 1; e2 < EXPERTS; e2++)
                if (lg[e2] > lg[best])
                    best = e2;
            const double w = 1.0 / (1.0 + exp(-lg[best]));
            for (size_t i = 0; i < D; i++)
                xs[i] = k.weight_after ? x[i] : x[i] * w;
            swiglu(EFF,
                   r->eg + best * EFF * D,
                   r->eu + best * EFF * D,
                   r->ed + best * D * EFF,
                   xs,
                   y);
            for (size_t i = 0; i < D; i++)
                h[i] += k.weight_after ? w * y[i] : y[i];
            if (!k.no_shared) {
                swiglu(EFF, r->sg, r->su, r->sd, x, y);
                for (size_t i = 0; i < D; i++)
                    h[i] += y[i];
            }
        }
    }
    rmsnorm(D, h, m->out_norm, x);
    matvec(VOCAB, D, m->out, x, logits);
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

static double cosine(const float a[static VOCAB], const float b[static VOCAB]) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / sqrt(aa * bb > 0.0 ? aa * bb : 1.0);
}

static int check_backend(const char *name, float out[static VOCAB]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("skip %s: not in this build\n", name);
        return -1;
    }
    struct tf_buf         b     = mf_llama4_gguf(&GEOM);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    float                 chunked[VOCAB], single[VOCAB], again[VOCAB];
    if (geist_model_load_from_memory(b.b, b.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK) {
        fprintf(stderr, "  %s: %s\n", name, geist_last_create_error());
        fails += geist_expect(false, "the llama4 fixture loads");
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
        fails += geist_expect(diff <= 1e-4 * scale, "chunks and single tokens agree");
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

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    struct tf_buf    b   = mf_llama4_gguf(&GEOM);
    const char      *err = nullptr;
    struct gguf_ctx *gc  = gguf_open_memory(b.b, b.n, &err);
    if (gc == nullptr) {
        fprintf(stderr, "cannot read the fixture: %s\n", err);
        return GEIST_TEST_ERROR;
    }
    struct ref_model rm = {0};
    static double    ref[VOCAB], alt[VOCAB];
    ref_load(gc, &rm);
    ref_forward(&rm, (struct knobs) {0}, ref);
    int       fails = 0;
    float     scalar[VOCAB], other[VOCAB];
    const int sc = check_backend("cpu_scalar", scalar);
    fails += sc != 0 ? geist_expect(false, "cpu_scalar runs the fixture") : 0;
    if (sc == 0) {
        const double d = rel_l2_d(scalar, ref);
        printf("cpu_scalar vs the double reference: relative L2 %.3g\n", d);
        fails += geist_expect(d <= 1e-5, "cpu_scalar matches the reference within 1e-5");
        const struct {
            const char  *what;
            struct knobs k;
        } knobs[] = {
                {"the frequency factors", {.no_factors = true}},
                {"chunks", {.no_chunks = true}},
                {"the temperature", {.no_temp = true}},
                {"the q/k norm", {.no_norm = true}},
                {"the weight before the expert", {.weight_after = true}},
                {"the shared expert", {.no_shared = true}},
        };
        for (size_t i = 0; i < sizeof knobs / sizeof knobs[0]; i++) {
            ref_forward(&rm, knobs[i].k, alt);
            const double moved = rel_l2_d(scalar, alt);
            printf("  without %s: relative L2 %.3g\n", knobs[i].what, moved);
            fails += geist_expect(moved > 1e-2, "each mechanism moves the logits");
        }
    }
    ref_free(&rm);
    gguf_close(gc);
    free(b.b);
    const char *const more[] = {"cpu_neon", "cpu_x86", "metal", "vulkan"};
    for (size_t i = 0; i < sizeof more / sizeof more[0]; i++) {
        const int r = check_backend(more[i], other);
        if (r < 0)
            continue;
        fails += r;
        if (r == 0 && sc == 0) {
            double num = 0.0, den = 0.0;
            for (size_t j = 0; j < VOCAB; j++) {
                num += ((double) other[j] - scalar[j]) * ((double) other[j] - scalar[j]);
                den += (double) scalar[j] * scalar[j];
            }
            const double d = sqrt(num / den), c = cosine(other, scalar);
            printf("%s vs cpu_scalar: relative L2 %.3g, cosine %.6f\n", more[i], d, c);
            const bool exact = strcmp(more[i], "cpu_neon") == 0 || strcmp(more[i], "metal") == 0;
            fails += geist_expect(exact ? d <= 1e-3 : c >= 0.999,
                                  "a backend matches cpu_scalar within its tolerance");
        }
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
