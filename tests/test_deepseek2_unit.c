/*
 * test_deepseek2_unit - the DeepSeek-V2 family on in-memory models (#242,
 * #375).
 *
 * mf_deepseek2_gguf writes a three-layer DeepSeek-V2 with F32 weights in the
 * unsplit attn_kv_b layout: one dense lead layer, then two MoE layers of 8
 * experts (2 per token) plus two shared experts; multi-head latent attention
 * with a 32-dim latent, heads of 16 nope + 8 rope dims and 16-dim values;
 * YaRN with DeepSeek's log multiplier. With `split` the same weights come
 * as llama.cpp's converter writes them since 2025 (attn_k_b / attn_v_b, one
 * KV head of 40 dims), which geist runs in the absorbed form; the reference
 * rebuilds kv_b from k_b and v_b.
 *
 * The reference is this file's own forward pass in double, written from
 * llama.cpp's graph for the architecture (deepseek2.cpp, its non-absorbed
 * path; build_moe_ffn), not from geist's code: the interleaved (NORM) RoPE
 * on the trailing pe dims of each q head and of kv_a in the file's own row
 * order, k = [k_nope | k_pe], the q.k scale mscale^2 / sqrt(24). geist
 * reorders rows at load and pads V; none of that is in the reference.
 *
 *   - V2-Lite shape (attn_q, softmax router, no renormalisation): cpu_scalar
 *     within relative L2 1e-5 of the reference after a 21-token prompt (FP32
 *     KV);
 *   - V3 shape (q_a / q_a_norm / q_b; a sigmoid router with exp_probs_b,
 *     4 groups of which 2 stay, renormalised, scaled by 2.5): the same;
 *   - each mechanism matters: the reference with neighbouring instead of
 *     interleaved RoPE pairs, without the YaRN q scale, without the shared
 *     experts or (V3) without groups moves the logits by more than 1e-2;
 *   - every backend in the build: chunked and single-token prefill agree,
 *     a reset repeats its logits bit for bit, and the logits match
 *     cpu_scalar (cpu_neon and Metal within 1e-3, others cosine 0.999);
 *   - split (absorbed, #375), both shapes: the same within 1e-5 of the
 *     reference, and within 1e-5 of the unsplit file's logits on
 *     cpu_scalar. A third split shape with a 512-dim latent (a 520-dim KV
 *     head, past the quantized caches' 512) runs on every CPU backend, an
 *     INT8 KV request getting the F16 cache instead;
 *   - Metal (#376), every shape: the experts on the GPU (dense lead layer,
 *     DeepSeek's router on the device) and the latent attention on the
 *     device, within 1e-5 of the double reference and of its own run with
 *     the experts on the CPU (GEIST_MOE_HOT=0). A GPU without device MLA
 *     (Vulkan) refuses the split layout, saying so.
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

enum { VOCAB = 96, N_PROMPT = 21, LAYERS = 3, LEAD = 1, D = 64, HEADS = 4 };
enum { NOPE = 16, ROPE = 8, HK = NOPE + ROPE, VD = 16, RANK = 32, QRANK = 24, FF = 48 };
enum { RANK_WIDE = 512, RANK_MAX = RANK_WIDE }; /* the wide shape's latent */
enum { EXPERTS = 8, USED = 2, EFF = 16, SHARED = 2, SFF = SHARED * EFF };

static const struct mf_deepseek2 LITE = {.layers    = LAYERS,
                                         .lead      = LEAD,
                                         .d_model   = D,
                                         .heads     = HEADS,
                                         .nope      = NOPE,
                                         .rope      = ROPE,
                                         .v_dim     = VD,
                                         .kv_rank   = RANK,
                                         .ff        = FF,
                                         .experts   = EXPERTS,
                                         .used      = USED,
                                         .expert_ff = EFF,
                                         .shared    = SHARED,
                                         .vocab     = VOCAB,
                                         .seed      = 41};

static struct mf_deepseek2 v3_geom(void) {
    struct mf_deepseek2 g = LITE;
    g.q_rank              = QRANK;
    g.v3                  = true;
    g.n_group             = 4;
    g.topk_group          = 2;
    g.w_scale             = 2.5f;
    g.seed                = 43;
    return g;
}

static struct mf_deepseek2 split_of(struct mf_deepseek2 g) {
    g.split = true;
    return g;
}

static const struct geist_session_opts OPTS = {
        .top_p = 1.0f, .m_max = 8, .max_seq_len = 256, .kv_mode = GEIST_KV_FP32};

static geist_token_t prompt_id(size_t i) {
    return (geist_token_t) ((i * 37 + 5) % VOCAB);
}

/* ---- The reference --------------------------------------------------- */

struct knobs {
    bool neox, no_mscale, no_shared, no_groups;
};

struct ref_layer {
    float *attn_norm, *wq, *wq_a, *q_a_norm, *wq_b, *wkv_a, *kv_a_norm, *wkv_b, *wo, *ffn_norm;
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

/* kv_b [heads * (NOPE + VD), rank] from the split k_b [NOPE, rank, heads]
 * and v_b [rank, VD, heads] (GGUF order), as llama.cpp's converter split
 * it; nullptr when the file has neither. */
static float *kv_b_of_split(struct gguf_ctx *g, size_t rank, int l) {
    float *kb = tensor(g, "blk.%d.attn_k_b.weight", l),
          *vb = tensor(g, "blk.%d.attn_v_b.weight", l);
    float *kv = kb != nullptr && vb != nullptr ? malloc(HEADS * (NOPE + VD) * rank * sizeof *kv)
                                               : nullptr;
    for (size_t h = 0; kv != nullptr && h < HEADS; h++) {
        for (size_t r = 0; r < rank; r++) {
            for (size_t i = 0; i < NOPE; i++)
                kv[(h * (NOPE + VD) + i) * rank + r] = kb[i + NOPE * (r + rank * h)];
            for (size_t j = 0; j < VD; j++)
                kv[(h * (NOPE + VD) + NOPE + j) * rank + r] = vb[r + rank * (j + VD * h)];
        }
    }
    free(kb);
    free(vb);
    return kv;
}

static void ref_load(struct gguf_ctx *g, size_t rank, struct ref_model *m) {
    m->embd     = tensor(g, "token_embd.weight");
    m->out      = tensor(g, "output.weight");
    m->out_norm = tensor(g, "output_norm.weight");
    for (int l = 0; l < LAYERS; l++) {
        struct ref_layer *r = &m->l[l];
        r->attn_norm        = tensor(g, "blk.%d.attn_norm.weight", l);
        r->wq               = tensor(g, "blk.%d.attn_q.weight", l);
        r->wq_a             = tensor(g, "blk.%d.attn_q_a.weight", l);
        r->q_a_norm         = tensor(g, "blk.%d.attn_q_a_norm.weight", l);
        r->wq_b             = tensor(g, "blk.%d.attn_q_b.weight", l);
        r->wkv_a            = tensor(g, "blk.%d.attn_kv_a_mqa.weight", l);
        r->kv_a_norm        = tensor(g, "blk.%d.attn_kv_a_norm.weight", l);
        r->wkv_b            = tensor(g, "blk.%d.attn_kv_b.weight", l);
        if (r->wkv_b == nullptr)
            r->wkv_b = kv_b_of_split(g, rank, l);
        r->wo       = tensor(g, "blk.%d.attn_output.weight", l);
        r->ffn_norm = tensor(g, "blk.%d.ffn_norm.weight", l);
        r->wg       = tensor(g, "blk.%d.ffn_gate.weight", l);
        r->wu       = tensor(g, "blk.%d.ffn_up.weight", l);
        r->wd       = tensor(g, "blk.%d.ffn_down.weight", l);
        r->wr       = tensor(g, "blk.%d.ffn_gate_inp.weight", l);
        r->sel      = tensor(g, "blk.%d.exp_probs_b.bias", l);
        r->eg       = tensor(g, "blk.%d.ffn_gate_exps.weight", l);
        r->eu       = tensor(g, "blk.%d.ffn_up_exps.weight", l);
        r->ed       = tensor(g, "blk.%d.ffn_down_exps.weight", l);
        r->sg       = tensor(g, "blk.%d.ffn_gate_shexp.weight", l);
        r->su       = tensor(g, "blk.%d.ffn_up_shexp.weight", l);
        r->sd       = tensor(g, "blk.%d.ffn_down_shexp.weight", l);
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

/* ggml's rope_yarn on the ROPE dims at x (NORM: pairs (2i, 2i+1); with
 * k.neox, (i, i + ROPE/2) instead), magnitude 1. */
static void rope(double *x, size_t p, bool yarn, struct knobs k) {
    const double base = 10000.0, factor = 40.0, orig = 64.0, fast = 32.0, slow = 1.0;
    const double pi = 3.14159265358979323846;
    const double lo = fmax(0.0, floor(ROPE * log(orig / (fast * 2 * pi)) / (2 * log(base))));
    const double hi = fmin(ROPE - 1.0, ceil(ROPE * log(orig / (slow * 2 * pi)) / (2 * log(base))));
    for (size_t i = 0; i < ROPE / 2; i++) {
        const double extrap = (double) p * pow(base, -2.0 * (double) i / ROPE);
        double       th     = extrap;
        if (yarn) {
            const double ramp =
                    1.0 - fmin(1.0, fmax(0.0, ((double) i - lo) / fmax(0.001, hi - lo)));
            th = extrap / factor * (1.0 - ramp) + extrap * ramp;
        }
        const size_t a = k.neox ? i : 2 * i, b = k.neox ? i + ROPE / 2 : 2 * i + 1;
        const double x0 = x[a], x1 = x[b];
        x[a] = x0 * cos(th) - x1 * sin(th);
        x[b] = x0 * sin(th) + x1 * cos(th);
    }
}

static void ref_forward(const struct ref_model    *m,
                        const struct mf_deepseek2 *g,
                        struct knobs               k,
                        double                     logits[static VOCAB]) {
    static double kc[LAYERS][N_PROMPT][HEADS][HK], vc[LAYERS][N_PROMPT][HEADS][VD];
    const bool    yarn = !g->no_yarn;
    const double  ms   = yarn && !k.no_mscale ? 1.0 + 0.0707 * log(40.0) : 1.0;
    const double  kq   = ms * ms / sqrt((double) HK);
    const size_t  rank = g->kv_rank;
    double        h[D], x[D], q[HEADS * HK], qa[QRANK], kva[RANK_MAX + ROPE], lat[RANK_MAX];
    double        kv[HEADS * (NOPE + VD)], att[HEADS * VD], o[D], y[D];
    for (size_t p = 0; p < N_PROMPT; p++) {
        const float *e = m->embd + (size_t) prompt_id(p) * D;
        for (size_t i = 0; i < D; i++)
            h[i] = e[i];
        for (size_t l = 0; l < LAYERS; l++) {
            const struct ref_layer *r = &m->l[l];
            rmsnorm(D, h, r->attn_norm, x);
            if (g->q_rank > 0) {
                matvec(QRANK, D, r->wq_a, x, qa);
                rmsnorm(QRANK, qa, r->q_a_norm, qa);
                matvec(HEADS * HK, QRANK, r->wq_b, qa, q);
            } else {
                matvec(HEADS * HK, D, r->wq, x, q);
            }
            matvec(rank + ROPE, D, r->wkv_a, x, kva);
            rope(kva + rank, p, yarn, k); /* k_pe: the trailing dims */
            rmsnorm(rank, kva, r->kv_a_norm, lat);
            matvec(HEADS * (NOPE + VD), rank, r->wkv_b, lat, kv);
            for (size_t hh = 0; hh < HEADS; hh++) {
                rope(q + hh * HK + NOPE, p, yarn, k);
                memcpy(kc[l][p][hh], kv + hh * (NOPE + VD), NOPE * sizeof(double));
                memcpy(kc[l][p][hh] + NOPE, kva + rank, ROPE * sizeof(double));
                memcpy(vc[l][p][hh], kv + hh * (NOPE + VD) + NOPE, VD * sizeof(double));
            }
            for (size_t hh = 0; hh < HEADS; hh++) {
                double sc[N_PROMPT], mx = -1e300, den = 0.0;
                for (size_t s = 0; s <= p; s++) {
                    double dot = 0.0;
                    for (size_t i = 0; i < HK; i++)
                        dot += q[hh * HK + i] * kc[l][s][hh][i];
                    sc[s] = dot * kq;
                    mx    = fmax(mx, sc[s]);
                }
                for (size_t s = 0; s <= p; s++)
                    den += exp(sc[s] - mx);
                for (size_t i = 0; i < VD; i++) {
                    double acc = 0.0;
                    for (size_t s = 0; s <= p; s++)
                        acc += exp(sc[s] - mx) * vc[l][s][hh][i];
                    att[hh * VD + i] = acc / den;
                }
            }
            matvec(D, HEADS * VD, r->wo, att, o);
            for (size_t i = 0; i < D; i++)
                h[i] += o[i];
            rmsnorm(D, h, r->ffn_norm, x);
            if (l < LEAD) {
                swiglu(FF, r->wg, r->wu, r->wd, x, y);
                for (size_t i = 0; i < D; i++)
                    h[i] += y[i];
                continue;
            }
            /* Router: build_moe_ffn. */
            double lg[EXPERTS], sc[EXPERTS], sel[EXPERTS];
            matvec(EXPERTS, D, r->wr, x, lg);
            if (g->v3) {
                for (size_t e2 = 0; e2 < EXPERTS; e2++)
                    sc[e2] = 1.0 / (1.0 + exp(-lg[e2]));
            } else {
                double mx = lg[0], z = 0.0;
                for (size_t e2 = 1; e2 < EXPERTS; e2++)
                    mx = fmax(mx, lg[e2]);
                for (size_t e2 = 0; e2 < EXPERTS; e2++)
                    z += exp(lg[e2] - mx);
                for (size_t e2 = 0; e2 < EXPERTS; e2++)
                    sc[e2] = exp(lg[e2] - mx) / z;
            }
            for (size_t e2 = 0; e2 < EXPERTS; e2++)
                sel[e2] = sc[e2] + (r->sel != nullptr ? r->sel[e2] : 0.0);
            if (g->v3 && !k.no_groups) {
                const size_t per = EXPERTS / g->n_group;
                double       gs[EXPERTS];
                for (size_t q2 = 0; q2 < g->n_group; q2++) {
                    double a = -1e300, b = -1e300; /* finite: -ffast-math */
                    for (size_t i = q2 * per; i < (q2 + 1) * per; i++) {
                        if (sel[i] > a) {
                            b = a;
                            a = sel[i];
                        } else if (sel[i] > b) {
                            b = sel[i];
                        }
                    }
                    gs[q2] = a + b;
                }
                bool keep[EXPERTS] = {false};
                for (size_t j = 0; j < g->topk_group; j++) {
                    size_t best = EXPERTS;
                    for (size_t q2 = 0; q2 < g->n_group; q2++)
                        if (!keep[q2] && (best == EXPERTS || gs[q2] > gs[best]))
                            best = q2;
                    keep[best] = true;
                }
                for (size_t i = 0; i < EXPERTS; i++)
                    if (!keep[i / per])
                        sel[i] = -1e300;
            }
            size_t ids[USED];
            bool   taken[EXPERTS] = {false};
            for (size_t j = 0; j < USED; j++) {
                size_t best = EXPERTS;
                for (size_t e2 = 0; e2 < EXPERTS; e2++)
                    if (!taken[e2] && sel[e2] > -1e299 && (best == EXPERTS || sel[e2] > sel[best]))
                        best = e2;
                taken[best] = true;
                ids[j]      = best;
            }
            double wsum = 0.0;
            for (size_t j = 0; j < USED; j++)
                wsum += sc[ids[j]];
            for (size_t j = 0; j < USED; j++) {
                double wt = sc[ids[j]];
                if (g->v3)
                    wt = wt / wsum * g->w_scale;
                swiglu(EFF,
                       r->eg + ids[j] * EFF * D,
                       r->eu + ids[j] * EFF * D,
                       r->ed + ids[j] * D * EFF,
                       x,
                       y);
                for (size_t i = 0; i < D; i++)
                    h[i] += wt * y[i];
            }
            if (!k.no_shared) {
                swiglu(SFF, r->sg, r->su, r->sd, x, y);
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

/* Chunked prefill and one token at a time agree within 1e-4 of the
 * logits' scale on every backend, fixture and KV cache, the wide fixture's
 * included: cpu_x86 runs these F32 weights as W8A8, and its one-row and
 * many-row kernels give a row the same bits (#476), so no last-bit
 * difference reaches the int8 grid of a following quantization (it did,
 * 7e-3 of the scale, before; #475). */
static constexpr double ROWS_TOL = 1e-4;

/* The fixture's logits on backend `name` with KV cache `kv` (GEIST_KV_AUTO:
 * the session's default); -1: no such backend, or a GPU without device MLA
 * refusing the split layout (#376), as it must. */
static int check_backend(const char                *name,
                         const struct mf_deepseek2 *g,
                         enum geist_kv_mode         kv,
                         float                      out[static VOCAB]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("skip %s: not in this build\n", name);
        return -1;
    }
    struct tf_buf             b     = mf_deepseek2_gguf(g);
    struct geist_model       *m     = nullptr;
    struct geist_session     *s     = nullptr;
    int                       fails = 0;
    float                     chunked[VOCAB], single[VOCAB], again[VOCAB];
    struct geist_session_opts o = OPTS;
    o.kv_mode                   = kv;
    const bool gpu              = strncmp(name, "cpu_", 4) != 0;
    if (geist_model_load_from_memory(b.b, b.n, be, &m) != GEIST_OK && g->split && gpu &&
        strcmp(name, "metal") != 0) {
        const char *e = geist_backend_errmsg(be);
        printf("%s refuses the split layout: %s\n", name, e != nullptr ? e : "?");
        fails = geist_expect(e != nullptr && strstr(e, "absorbed") != nullptr,
                             "a GPU backend refuses the absorbed layout, saying so");
        fails = fails != 0 ? fails : -1;
    } else if (m == nullptr || geist_session_create(m, be, &o, &s) != GEIST_OK) {
        fprintf(stderr, "  %s: %s\n", name, geist_last_create_error());
        fails += geist_expect(false, "the deepseek2 fixture loads");
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

static double rel_l2_f(const float a[static VOCAB], const float b[static VOCAB]) {
    double num = 0.0, den = 0.0;
    for (size_t j = 0; j < VOCAB; j++) {
        num += ((double) a[j] - b[j]) * ((double) a[j] - b[j]);
        den += (double) b[j] * b[j];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

/* `scalar_out`, when non-null, receives cpu_scalar's logits. */
static int check_shape(const char *what, const struct mf_deepseek2 *g, float *scalar_out) {
    printf("== %s\n", what);
    struct tf_buf    b   = mf_deepseek2_gguf(g);
    const char      *err = nullptr;
    struct gguf_ctx *gc  = gguf_open_memory(b.b, b.n, &err);
    if (gc == nullptr) {
        fprintf(stderr, "cannot read the fixture: %s\n", err);
        free(b.b);
        return 1;
    }
    struct ref_model rm = {0};
    static double    ref[VOCAB], alt[VOCAB];
    ref_load(gc, g->kv_rank, &rm);
    ref_forward(&rm, g, (struct knobs) {0}, ref);
    int       fails = 0;
    float     scalar[VOCAB], other[VOCAB];
    const int sc = check_backend("cpu_scalar", g, OPTS.kv_mode, scalar);
    fails += sc != 0 ? geist_expect(false, "cpu_scalar runs the fixture") : 0;
    if (sc == 0 && scalar_out != nullptr)
        memcpy(scalar_out, scalar, sizeof scalar);
    if (sc == 0) {
        const double d = rel_l2_d(scalar, ref);
        printf("cpu_scalar vs the double reference: relative L2 %.3g\n", d);
        fails += geist_expect(d <= 1e-5, "cpu_scalar matches the reference within 1e-5");
        const struct {
            const char  *what;
            struct knobs k;
            bool         v3_only;
        } knobs[] = {
                {"interleaved RoPE pairs", {.neox = true}, false},
                {"the YaRN q scale", {.no_mscale = true}, false},
                {"the shared experts", {.no_shared = true}, false},
                {"the expert groups", {.no_groups = true}, true},
        };
        for (size_t i = 0; i < sizeof knobs / sizeof knobs[0]; i++) {
            if (knobs[i].v3_only && !g->v3)
                continue;
            ref_forward(&rm, g, knobs[i].k, alt);
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
        const int r = check_backend(more[i], g, OPTS.kv_mode, other);
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
        /* #376: Metal's GPU experts and device MLA (#499: Vulkan's GPU
         * experts) against the reference and against its experts on the CPU. */
        if (r == 0 && (strcmp(more[i], "metal") == 0 || strcmp(more[i], "vulkan") == 0)) {
            static float cpu_exp[VOCAB];
            const double dr = rel_l2_d(other, ref);
            printf("%s vs the double reference: relative L2 %.3g\n", more[i], dr);
            fails += geist_expect(dr <= 1e-5, "the GPU matches the reference within 1e-5");
            setenv("GEIST_MOE_HOT", "0", 1);
            const int rc = check_backend(more[i], g, OPTS.kv_mode, cpu_exp);
            unsetenv("GEIST_MOE_HOT");
            fails += geist_expect(rc == 0, "the GPU runs with its experts on the CPU");
            /* The CPU's experts are exact on cpu_neon (Metal's host); on x86
             * cpu_x86 runs F32 weights as W8A8, so there the double
             * reference above is the measure. */
            if (rc == 0) {
                const double dc = rel_l2_f(other, cpu_exp);
                printf("%s, experts on the GPU vs on the CPU: relative L2 %.3g\n", more[i], dc);
                if (strcmp(more[i], "metal") == 0)
                    fails += geist_expect(dc <= 1e-5, "the GPU experts match the CPU experts");
            }
        }
    }
    return fails;
}

/* The split file against the unsplit one with the same weights (#375). */
static int check_split(const char *what, const struct mf_deepseek2 *g) {
    static float plain[VOCAB], split[VOCAB];
    char         title[128];
    snprintf(title, sizeof title, "%s, split (absorbed MLA)", what);
    const struct mf_deepseek2 sg    = split_of(*g);
    int                       fails = check_shape(what, g, plain);
    fails += check_shape(title, &sg, split);
    const double d = rel_l2_f(split, plain);
    printf("cpu_scalar split vs unsplit: relative L2 %.3g\n", d);
    fails += geist_expect(d <= 1e-5, "the split and the unsplit file give the same logits");
    return fails;
}

/* A latent of RANK_WIDE (512, as DeepSeek-V3's): one KV head of 520 dims,
 * past the quantized caches; the default cache is FP32 (bit for bit the
 * FP32 request's logits), an INT8 request gets the F16 cache. The latent is
 * a multiple of 16: cpu_x86 runs F32 weights of such a width as W8A8 in
 * both its one-row and its many-row kernel (one rounding order, #476), and
 * others (520 did) on two F32 kernels of different summation order. */
static int check_wide(void) {
    struct mf_deepseek2 g = split_of(LITE);
    g.kv_rank             = RANK_WIDE;
    int fails = check_shape("V2-Lite shape, split, 512-dim latent (520-dim KV head)", &g, nullptr);
    static float f32[VOCAB], int8[VOCAB], dflt[VOCAB], scalar[VOCAB];
    const char  *cpus[] = {"cpu_scalar", "cpu_neon", "cpu_x86"};
    if (check_backend("cpu_scalar", &g, GEIST_KV_FP32, scalar) != 0)
        return fails + geist_expect(false, "cpu_scalar runs the wide head");
    for (size_t i = 0; i < sizeof cpus / sizeof cpus[0]; i++) {
        if (check_backend(cpus[i], &g, GEIST_KV_FP32, f32) != 0)
            continue;
        fails += geist_expect(check_backend(cpus[i], &g, GEIST_KV_AUTO, dflt) == 0 &&
                                      memcmp(dflt, f32, sizeof f32) == 0,
                              "a wide head's default KV cache is FP32");
        /* The backend's int8-grid reach: ~0 with F32 activations. */
        const double grid = 2.0 * rel_l2_f(f32, scalar);
        const int    r    = check_backend(cpus[i], &g, GEIST_KV_INT8, int8);
        fails += geist_expect(r == 0, "an INT8 KV request runs the wide head");
        if (r == 0) {
            const double d = rel_l2_f(int8, f32);
            printf("%s INT8 request (F16 cache) vs FP32 KV: relative L2 %.3g\n", cpus[i], d);
            /* F16 against FP32: 1.4e-4 with F32 activations; on W8A8
             * kernels the F16 rounding moves int8 values by a grid step
             * (cpu_x86 1.8e-2). */
            fails += geist_expect(d > 0.0 && d <= fmax(1e-3, grid),
                                  "the INT8 request gets the F16 cache");
        }
    }
    return fails;
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    int                       fails = check_split("V2-Lite shape", &LITE);
    const struct mf_deepseek2 v3    = v3_geom();
    fails += check_split("V3 shape (q_a/q_b, sigmoid router, groups, scale)", &v3);
    fails += check_wide();
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
