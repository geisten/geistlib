/*
 * test_gpt_oss_unit - the GPT-OSS family on an in-memory model (#243).
 *
 * mf_gpt_oss_gguf writes a two-layer GPT-OSS with F32 weights: q/k/v/o
 * biases, a sink logit per head, layer 0 sliding over WINDOW keys, YaRN
 * RoPE, and 8 experts (2 per token) with router and expert biases.
 *
 * The reference is this file's own forward pass in double precision,
 * written from llama.cpp's graph for the architecture (openai-moe.cpp,
 * build_moe_ffn with SOFTMAX_WEIGHT and SWIGLU_OAI, ggml's rope_yarn and
 * soft_max with sinks), not from geist's code. On cpu_scalar the
 * next-token logits after a 21-token prompt (FP32 KV) match it within
 * relative L2 1e-5. Each mechanism is shown to matter: the reference
 * without the sinks, without YaRN, with a window of all keys or with silu
 * instead of swiglu_oai moves the logits by more than 1e-2.
 *
 * On every backend in the build, as test_olmoe_unit: chunked prefill and
 * one token at a time agree within 1e-4 of the logits' scale, a reset
 * session repeats its logits bit for bit, and the logits match cpu_scalar
 * (cpu_neon and Metal within relative L2 1e-3, others by cosine 0.999).
 * Metal runs the experts on the GPU (#373) and matches the double
 * reference within 1e-5, as it does with GEIST_MOE_HOT=0 (experts on the
 * CPU).
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

enum { VOCAB = 96, N_PROMPT = 21, WINDOW = 6, LAYERS = 2, D = 64, HEADS = 4, KV_HEADS = 2 };
enum { HD = 16, EXPERTS = 8, USED = 2, FF = 32, Q_OUT = HEADS * HD, KV_OUT = KV_HEADS * HD };

static const struct mf_gpt_oss GEOM = {.layers     = LAYERS,
                                       .d_model    = D,
                                       .heads      = HEADS,
                                       .kv_heads   = KV_HEADS,
                                       .head_dim   = HD,
                                       .experts    = EXPERTS,
                                       .used       = USED,
                                       .expert_ff  = FF,
                                       .vocab      = VOCAB,
                                       .window     = WINDOW,
                                       .seed       = 29,
                                       .sink_scale = 20.0f};

/* FP32 KV: the family's default F16 cache alone moves the logits by ~1e-4
 * against the double reference. */
static const struct geist_session_opts OPTS = {
        .top_p = 1.0f, .m_max = 8, .max_seq_len = 256, .kv_mode = GEIST_KV_FP32};

static geist_token_t prompt_id(size_t i) {
    return (geist_token_t) ((i * 37 + 5) % VOCAB);
}

/* ---- The reference --------------------------------------------------- */

/* What the reference leaves out, to show each mechanism matters. */
struct ref_knobs {
    bool no_sinks, no_yarn, full_window, silu;
};

/* The fixture's tensors, as F32 from the GGUF. */
struct ref_model {
    float *embd, *out, *out_norm;
    struct {
        float *attn_norm, *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo, *sinks, *ffn_norm;
        float *wr, *br, *wg, *bg, *wu, *bu, *wd, *bd;
    } l[LAYERS];
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

static bool ref_load(struct gguf_ctx *g, struct ref_model *m) {
    m->embd     = tensor(g, "token_embd.weight");
    m->out      = tensor(g, "output.weight");
    m->out_norm = tensor(g, "output_norm.weight");
    bool ok     = m->embd && m->out && m->out_norm;
    for (int l = 0; l < LAYERS; l++) {
        m->l[l].attn_norm = tensor(g, "blk.%d.attn_norm.weight", l);
        m->l[l].wq        = tensor(g, "blk.%d.attn_q.weight", l);
        m->l[l].bq        = tensor(g, "blk.%d.attn_q.bias", l);
        m->l[l].wk        = tensor(g, "blk.%d.attn_k.weight", l);
        m->l[l].bk        = tensor(g, "blk.%d.attn_k.bias", l);
        m->l[l].wv        = tensor(g, "blk.%d.attn_v.weight", l);
        m->l[l].bv        = tensor(g, "blk.%d.attn_v.bias", l);
        m->l[l].wo        = tensor(g, "blk.%d.attn_output.weight", l);
        m->l[l].bo        = tensor(g, "blk.%d.attn_output.bias", l);
        m->l[l].sinks     = tensor(g, "blk.%d.attn_sinks.weight", l);
        m->l[l].ffn_norm  = tensor(g, "blk.%d.post_attention_norm.weight", l);
        m->l[l].wr        = tensor(g, "blk.%d.ffn_gate_inp.weight", l);
        m->l[l].br        = tensor(g, "blk.%d.ffn_gate_inp.bias", l);
        m->l[l].wg        = tensor(g, "blk.%d.ffn_gate_exps.weight", l);
        m->l[l].bg        = tensor(g, "blk.%d.ffn_gate_exps.bias", l);
        m->l[l].wu        = tensor(g, "blk.%d.ffn_up_exps.weight", l);
        m->l[l].bu        = tensor(g, "blk.%d.ffn_up_exps.bias", l);
        m->l[l].wd        = tensor(g, "blk.%d.ffn_down_exps.weight", l);
        m->l[l].bd        = tensor(g, "blk.%d.ffn_down_exps.bias", l);
        float **all       = &m->l[l].attn_norm;
        for (size_t i = 0; i < sizeof m->l[l] / sizeof(float *); i++)
            ok = ok && all[i] != nullptr;
    }
    return ok;
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

static void
rmsnorm(size_t n, const double x[static n], const float w[static n], double y[static n]) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++)
        ss += x[i] * x[i];
    const double r = 1.0 / sqrt(ss / (double) n + 1e-5);
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] * r * w[i];
}

/* y[n_out] = W[n_out][n_in] x + b (b may be nullptr). */
static void matvec(size_t       n_out,
                   size_t       n_in,
                   const float *w,
                   const double x[static n_in],
                   const float *b,
                   double       y[static n_out]) {
    for (size_t o = 0; o < n_out; o++) {
        double acc = b != nullptr ? b[o] : 0.0;
        for (size_t i = 0; i < n_in; i++)
            acc += (double) w[o * n_in + i] * x[i];
        y[o] = acc;
    }
}

/* NEOX RoPE over each head of x at position p, with ggml's rope_yarn
 * (ext_factor 1, attn_factor 1) unless no_yarn. */
static void rope(size_t n_heads, size_t p, bool no_yarn, double x[static n_heads * HD]) {
    const double base = 150000.0, factor = 32.0, orig = 64.0, fast = 32.0, slow = 1.0;
    const double pi     = 3.14159265358979323846;
    const double lo     = fmax(0.0, floor(HD * log(orig / (fast * 2 * pi)) / (2 * log(base))));
    const double hi     = fmin(HD - 1.0, ceil(HD * log(orig / (slow * 2 * pi)) / (2 * log(base))));
    const double mscale = no_yarn ? 1.0 : 1.0 + 0.1 * log(factor);
    for (size_t h = 0; h < n_heads; h++) {
        double *xh = x + h * HD;
        for (size_t i = 0; i < HD / 2; i++) {
            const double extrap = (double) p * pow(base, -2.0 * (double) i / HD);
            double       th     = extrap;
            if (!no_yarn) {
                const double ramp =
                        1.0 - fmin(1.0, fmax(0.0, ((double) i - lo) / fmax(0.001, hi - lo)));
                th = extrap / factor * (1.0 - ramp) + extrap * ramp;
            }
            const double c = cos(th) * mscale, s = sin(th) * mscale;
            const double a = xh[i], b = xh[i + HD / 2];
            xh[i]          = a * c - b * s;
            xh[i + HD / 2] = b * c + a * s;
        }
    }
}

/* Next-token logits after the prompt, in double. */
static void
ref_forward(const struct ref_model *m, struct ref_knobs k, double logits[static VOCAB]) {
    static double kc[LAYERS][N_PROMPT][KV_OUT], vc[LAYERS][N_PROMPT][KV_OUT];
    double        h[D], x[D], q[Q_OUT], att[Q_OUT], o[D];
    for (size_t p = 0; p < N_PROMPT; p++) {
        const float *e = m->embd + (size_t) prompt_id(p) * D;
        for (size_t i = 0; i < D; i++)
            h[i] = e[i];
        for (size_t l = 0; l < LAYERS; l++) {
            rmsnorm(D, h, m->l[l].attn_norm, x);
            matvec(Q_OUT, D, m->l[l].wq, x, m->l[l].bq, q);
            matvec(KV_OUT, D, m->l[l].wk, x, m->l[l].bk, kc[l][p]);
            matvec(KV_OUT, D, m->l[l].wv, x, m->l[l].bv, vc[l][p]);
            rope(HEADS, p, k.no_yarn, q);
            rope(KV_HEADS, p, k.no_yarn, kc[l][p]);
            const bool   slide = l % 2 == 0 && !k.full_window;
            const size_t s_lo  = slide && p + 1 > WINDOW ? p + 1 - WINDOW : 0;
            for (size_t hh = 0; hh < HEADS; hh++) {
                const size_t kvh = hh / (HEADS / KV_HEADS);
                double       sc[N_PROMPT], mx = k.no_sinks ? -1e300 : m->l[l].sinks[hh];
                for (size_t s = s_lo; s <= p; s++) {
                    double dot = 0.0;
                    for (size_t i = 0; i < HD; i++)
                        dot += q[hh * HD + i] * kc[l][s][kvh * HD + i];
                    sc[s] = dot / sqrt((double) HD);
                    mx    = fmax(mx, sc[s]);
                }
                double den = k.no_sinks ? 0.0 : exp(m->l[l].sinks[hh] - mx);
                for (size_t s = s_lo; s <= p; s++)
                    den += exp(sc[s] - mx);
                for (size_t i = 0; i < HD; i++) {
                    double acc = 0.0;
                    for (size_t s = s_lo; s <= p; s++)
                        acc += exp(sc[s] - mx) * vc[l][s][kvh * HD + i];
                    att[hh * HD + i] = acc / den;
                }
            }
            matvec(D, Q_OUT, m->l[l].wo, att, m->l[l].bo, o);
            for (size_t i = 0; i < D; i++)
                h[i] += o[i];
            /* MoE: the USED largest biased router logits, weighted by their
             * softmax; each expert biased, swiglu_oai, biased down. */
            rmsnorm(D, h, m->l[l].ffn_norm, x);
            double r[EXPERTS];
            matvec(EXPERTS, D, m->l[l].wr, x, m->l[l].br, r);
            size_t sel[USED];
            bool   taken[EXPERTS] = {false};
            for (size_t j = 0; j < USED; j++) {
                size_t best = EXPERTS;
                for (size_t e2 = 0; e2 < EXPERTS; e2++)
                    if (!taken[e2] && (best == EXPERTS || r[e2] > r[best]))
                        best = e2;
                taken[best] = true;
                sel[j]      = best;
            }
            double z = 0.0;
            for (size_t j = 0; j < USED; j++)
                z += exp(r[sel[j]] - r[sel[0]]);
            for (size_t j = 0; j < USED; j++) {
                const size_t e2 = sel[j];
                const double wt = exp(r[sel[j]] - r[sel[0]]) / z;
                double       g[FF], u[FF], y[D];
                matvec(FF, D, m->l[l].wg + e2 * FF * D, x, m->l[l].bg + e2 * FF, g);
                matvec(FF, D, m->l[l].wu + e2 * FF * D, x, m->l[l].bu + e2 * FF, u);
                for (size_t i = 0; i < FF; i++) {
                    if (k.silu) {
                        g[i] = g[i] / (1.0 + exp(-g[i])) * u[i];
                        continue;
                    }
                    const double gc = fmin(g[i], 7.0), uc = fmax(-7.0, fmin(u[i], 7.0));
                    g[i] = gc / (1.0 + exp(-1.702 * gc)) * (uc + 1.0);
                }
                matvec(D, FF, m->l[l].wd + e2 * D * FF, g, m->l[l].bd + e2 * D, y);
                for (size_t i = 0; i < D; i++)
                    h[i] += wt * y[i];
            }
        }
    }
    rmsnorm(D, h, m->out_norm, x);
    matvec(VOCAB, D, m->out, x, nullptr, logits);
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

static double rel_l2(const float a[static VOCAB], const float b[static VOCAB]) {
    double bd[VOCAB];
    for (size_t i = 0; i < VOCAB; i++)
        bd[i] = b[i];
    return rel_l2_d(a, bd);
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

/* One backend. Writes its chunked logits to `out`. Returns failures, or -1
 * when not built. */
/* `out`: the logits after the prompt; `out1`, when not nullptr, after one
 * more token (a decode step). */
static int check_backend(const char *name, float out[static VOCAB], float *out1) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("skip %s: not in this build\n", name);
        return -1;
    }
    struct tf_buf         g     = mf_gpt_oss_gguf(&GEOM);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    float                 chunked[VOCAB], single[VOCAB], again[VOCAB];
    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK) {
        fprintf(stderr,
                "  %s: %s\n",
                name,
                m != nullptr ? geist_model_errmsg(m) : geist_backend_errmsg(be));
        fails += geist_expect(false, "the gpt-oss fixture loads");
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
        if (out1 != nullptr)
            memcpy(out1, single, sizeof single);
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return fails;
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    int fails = 0;

    /* The reference, and that each mechanism moves it. */
    struct tf_buf    g   = mf_gpt_oss_gguf(&GEOM);
    const char      *err = nullptr;
    struct gguf_ctx *gc  = gguf_open_memory(g.b, g.n, &err);
    struct ref_model rm  = {0};
    static double    ref[VOCAB], alt[VOCAB];
    if (gc == nullptr || !ref_load(gc, &rm)) {
        fprintf(stderr, "cannot read the fixture: %s\n", err != nullptr ? err : "a tensor");
        return GEIST_TEST_ERROR;
    }
    ref_forward(&rm, (struct ref_knobs) {0}, ref);

    float     scalar[VOCAB], other[VOCAB];
    const int sc = check_backend("cpu_scalar", scalar, nullptr);
    fails += sc != 0 ? geist_expect(false, "cpu_scalar runs the fixture") : 0;
    if (sc == 0) {
        const double d = rel_l2_d(scalar, ref);
        printf("cpu_scalar vs the double reference: relative L2 %.3g\n", d);
        fails += geist_expect(d <= 1e-5, "cpu_scalar matches the reference within 1e-5");
        const struct {
            const char      *what;
            struct ref_knobs k;
        } knobs[] = {
                {"sinks", {.no_sinks = true}},
                {"YaRN", {.no_yarn = true}},
                {"the sliding window", {.full_window = true}},
                {"swiglu_oai", {.silu = true}},
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
    free(g.b);

    const char *const more[] = {"cpu_neon", "cpu_x86", "metal", "vulkan"};
    for (size_t b = 0; b < sizeof more / sizeof more[0]; b++) {
        const int r = check_backend(more[b], other, nullptr);
        if (r < 0)
            continue;
        fails += r;
        if (r == 0 && sc == 0) {
            const double d = rel_l2(other, scalar), c = cosine(other, scalar);
            printf("%s vs cpu_scalar: relative L2 %.3g, cosine %.6f\n", more[b], d, c);
            const bool exact = strcmp(more[b], "cpu_neon") == 0 || strcmp(more[b], "metal") == 0;
            fails += geist_expect(exact ? d <= 1e-3 : c >= 0.999,
                                  "a backend matches cpu_scalar within its tolerance");
        }
        /* #373: Metal (#499: and Vulkan) runs the experts itself (router
         * with its bias, softmax over the selected, expert biases,
         * swiglu_oai: prefill as GEMMs, decode on the device route); against
         * the double reference, and against its experts on the CPU
         * (GEIST_MOE_HOT=0). */
        if (r == 0 && (strcmp(more[b], "metal") == 0 || strcmp(more[b], "vulkan") == 0)) {
            float cpu_exp[VOCAB];
            setenv("GEIST_MOE_HOT", "0", 1);
            const int rc = check_backend(more[b], cpu_exp, nullptr);
            unsetenv("GEIST_MOE_HOT");
            fails += rc;
            const double dr = rel_l2_d(other, ref);
            printf("%s vs the double reference: relative L2 %.3g\n", more[b], dr);
            fails += geist_expect(dr <= 1e-5, "the GPU experts match the reference within 1e-5");
            /* The CPU's experts are exact on cpu_neon (Metal's host); on x86
             * cpu_x86 runs F32 weights as W8A8, so there the double
             * reference above is the measure. */
            if (rc == 0) {
                const double dc = rel_l2(other, cpu_exp);
                printf("%s, experts on the GPU vs on the CPU: relative L2 %.3g\n", more[b], dc);
                if (strcmp(more[b], "metal") == 0)
                    fails += geist_expect(dc <= 1e-5, "the GPU experts match the CPU experts");
            }
        }
        /* #499: 8 of the 16 experts on Vulkan, the rest on the CPU: a decode
         * step routes on the device (#162 with GPT-OSS's router bias and
         * softmax over the selected, the experts' biases in the GLU and the
         * down bias on the host's part), against the host routing the same
         * split (GEIST_MOE_GPU_ROUTE=0). */
        if (r == 0 && strcmp(more[b], "vulkan") == 0) {
            float routed[VOCAB], routed1[VOCAB], host[VOCAB], host1[VOCAB];
            setenv("GEIST_MOE_HOT", "8", 1);
            int rc = check_backend(more[b], routed, routed1);
            setenv("GEIST_MOE_GPU_ROUTE", "0", 1);
            rc += check_backend(more[b], host, host1);
            unsetenv("GEIST_MOE_GPU_ROUTE");
            unsetenv("GEIST_MOE_HOT");
            fails += geist_expect(rc == 0, "the experts split between the GPU and the CPU run");
            if (rc == 0) {
                const double dh = rel_l2(routed1, host1);
                printf("%s, 8 of 16 experts on the GPU, a decode step routed on the device vs on "
                       "the host: relative L2 %.3g\n",
                       more[b],
                       dh);
                fails += geist_expect(dh <= 1e-5, "routing on the device matches the host");
            }
        }
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
