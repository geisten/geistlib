/*
 * src/archs/transformer/forward/layer_deltanet.c — gated-DeltaNet token
 * mixer (#281, qwen35 family).
 *
 * Layer: ARCHITECTURE.
 *
 * Replaces the attention block for layers with mixer == GEIST_MIXER_DELTANET.
 * Evaluation order per token (spec pinned in issue #281 from the HF
 * modeling code, cross-checked against llama.cpp):
 *
 *   x               = rmsnorm(h, attn_norm)
 *   qkv             = W_qkv x                    [conv_dim]
 *   z               = W_z x                      [value_dim]  (output gate)
 *   b, a            = W_b x, W_a x               [n_v_heads] each
 *   beta_h          = sigmoid(b_h)
 *   g_h             = ssm_a_h * softplus(a_h + dt_bias_h)      (<= 0)
 *   y               = silu(causal_depthwise_conv1d(qkv))       (kernel K,
 *                       state = last K-1 pre-conv qkv vectors)
 *   q,k,v           = split(y); q,k = l2norm per head; q *= head_k^-0.5
 *   per v-head:  S  = S*exp(g);  d = (v - S^T k)*beta;  S += k (x) d;
 *                o  = S^T q
 *   o_h             = rmsnorm(o_h, ssm_norm) * silu(z_h)
 *   out             = W_out concat(o)            [d_model]
 *   h_out           = h + out                    (plain residual add)
 *
 * Projections run through the backend's quantized linear kernels into
 * session scratch buffers; conv + recurrence + norms are host-side f32
 * loops over mapped pointers. Decode (seq == 1) runs the sequential
 * per-token recurrence; prefill (seq > 1) runs the chunked GEMM
 * formulation (dn_head_chunk), equivalence pinned by
 * test_deltanet_chunk_int. GEIST_DN_SEQ_PREFILL=1 forces the
 * sequential path everywhere (the test oracle).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_state.h"

#include "geist_gemm.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float silu_f(float v) {
    const float e = expf(-fabsf(v));
    return (v >= 0.0f) ? v / (1.0f + e) : (v * e) / (1.0f + e);
}

/* RMS-normalize `n` values of x in place with weight w (plain 1-centered
 * weight — the GGUF converter bakes any zero-centering in). */
static void rmsnorm_row(float *x, const float *w, size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++)
        ss += (double) x[i] * (double) x[i];
    const float inv = (float) (1.0 / sqrt(ss / (double) n + (double) eps));
    for (size_t i = 0; i < n; i++)
        x[i] = x[i] * inv * w[i];
}

static void l2norm_row(float *x, size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++)
        ss += (double) x[i] * (double) x[i];
    const float inv = (float) (1.0 / sqrt(ss + (double) eps));
    for (size_t i = 0; i < n; i++)
        x[i] *= inv;
}

/* One delta-rule step for one v-head (perf lever 2, #281):
 *   pass 1: S *= exp(g);  kv_mem = S^T k        (kv_mem stays L1-resident)
 *   delta  = (v - kv_mem) * beta
 *   pass 2: S += k (x) delta;  o = S^T q
 * The two passes are a real data dependency (delta needs the full
 * kv_mem). NEON: d_v-contiguous rows as float32x4 FMA chains; the
 * scalar tail covers d_v % 4 != 0 (none of the known variants). */
void transformer_dn_head_step(float       *S,  /* [d_k, d_v] */
                              const float *qh, /* [d_k] */
                              const float *kh, /* [d_k] */
                              const float *vh, /* [d_v] */
                              float        decay,
                              float        beta,
                              size_t       d_k,
                              size_t       d_v,
                              float       *kv_mem, /* scratch [d_v] */
                              float       *delta,  /* scratch [d_v] */
                              float       *o_h) {  /* out [d_v] */
#if defined(__ARM_NEON)
    const float32x4_t vdec = vdupq_n_f32(decay);
    for (size_t j = 0; j < d_v; j += 4)
        vst1q_f32(kv_mem + j, vdupq_n_f32(0.0f));
    for (size_t i = 0; i < d_k; i++) {
        float            *Srow = S + i * d_v;
        const float32x4_t vki  = vdupq_n_f32(kh[i]);
        for (size_t j = 0; j < d_v; j += 4) {
            float32x4_t sv = vmulq_f32(vld1q_f32(Srow + j), vdec);
            vst1q_f32(Srow + j, sv);
            vst1q_f32(kv_mem + j, vfmaq_f32(vld1q_f32(kv_mem + j), sv, vki));
        }
    }
    const float32x4_t vbeta = vdupq_n_f32(beta);
    for (size_t j = 0; j < d_v; j += 4) {
        const float32x4_t dv =
                vmulq_f32(vsubq_f32(vld1q_f32(vh + j), vld1q_f32(kv_mem + j)), vbeta);
        vst1q_f32(delta + j, dv);
        vst1q_f32(o_h + j, vdupq_n_f32(0.0f));
    }
    for (size_t i = 0; i < d_k; i++) {
        float            *Srow = S + i * d_v;
        const float32x4_t vki  = vdupq_n_f32(kh[i]);
        const float32x4_t vqi  = vdupq_n_f32(qh[i]);
        for (size_t j = 0; j < d_v; j += 4) {
            float32x4_t sv = vfmaq_f32(vld1q_f32(Srow + j), vld1q_f32(delta + j), vki);
            vst1q_f32(Srow + j, sv);
            vst1q_f32(o_h + j, vfmaq_f32(vld1q_f32(o_h + j), sv, vqi));
        }
    }
#else
    for (size_t j = 0; j < d_v; j++)
        kv_mem[j] = 0.0f;
    for (size_t i = 0; i < d_k; i++) {
        float      *Srow = S + i * d_v;
        const float ki   = kh[i];
        for (size_t j = 0; j < d_v; j++) {
            Srow[j] *= decay;
            kv_mem[j] += Srow[j] * ki;
        }
    }
    for (size_t j = 0; j < d_v; j++) {
        delta[j] = (vh[j] - kv_mem[j]) * beta;
        o_h[j]   = 0.0f;
    }
    for (size_t i = 0; i < d_k; i++) {
        float      *Srow = S + i * d_v;
        const float ki   = kh[i];
        const float qi   = qh[i];
        for (size_t j = 0; j < d_v; j++) {
            Srow[j] += ki * delta[j];
            o_h[j] += Srow[j] * qi;
        }
    }
#endif
}

/* Chunked delta-rule for C tokens of one v-head (#281 prefill phase).
 * Mathematically equivalent to C sequential dn_head_step calls — pinned
 * by test_deltanet_chunk_int against the sequential path. Formulation
 * per the HF/llama.cpp chunk recipe (issue #281 spec, section 5b), with
 * the (I - A_strict)^-1 forward substitution folded as (A + I) into the
 * single GEMM that consumes it. All exp arguments are <= 0 (g <= 0 and
 * gamma is non-increasing), so nothing overflows.
 *
 *   Q,K: [C, d_k] rows at stride sq/sk (already l2-normed; Q carries
 *   d_k^-1/2)   V: [C, d_v] rows at stride sv — strides let the rows
 *   live directly in the [seq, conv_dim] y buffer, no gather copies.
 *   beta,g: [C] at element stride sbg   S: [d_k, d_v] in/out
 *   o: [C, d_v] dense out
 *
 * ws layout (see dn_chunk_ws_floats): gamma C | eg C | Kb C*d_k |
 * KCe C*d_k | Qg C*d_k | Vb C*d_v | vnew C*d_v | A C*C | attn C*C |
 * row C (scratch for the substitution).
 */
size_t transformer_dn_chunk_ws_floats(size_t C, size_t d_k, size_t d_v) {
    return 2 * C + 3 * C * d_k + 2 * C * d_v + 2 * C * C + C;
}

void transformer_dn_head_chunk(float       *S,
                               const float *Q,
                               size_t       sq,
                               const float *K,
                               size_t       sk,
                               const float *V,
                               size_t       sv,
                               const float *beta,
                               const float *g,
                               size_t       sbg,
                               size_t       C,
                               size_t       d_k,
                               size_t       d_v,
                               float       *o,
                               float       *ws) {
    float *gamma = ws;
    float *eg    = gamma + C;
    float *Kb    = eg + C;
    float *KCe   = Kb + C * d_k;
    float *Qg    = KCe + C * d_k;
    float *Vb    = Qg + C * d_k;
    float *vnew  = Vb + C * d_v;
    float *A     = vnew + C * d_v;
    float *attn  = A + C * C;
    float *rowt  = attn + C * C;

    /* gamma = cumsum(g); scaled row variants. */
    float acc = 0.0f;
    for (size_t t = 0; t < C; t++) {
        acc += g[t * sbg];
        gamma[t] = acc;
        eg[t]    = expf(acc);
    }
    for (size_t t = 0; t < C; t++) {
        const float bt = beta[t * sbg];
        for (size_t i = 0; i < d_k; i++) {
            Kb[t * d_k + i]  = bt * K[t * sk + i];
            KCe[t * d_k + i] = Kb[t * d_k + i] * eg[t];
            Qg[t * d_k + i]  = Q[t * sq + i] * eg[t];
        }
        for (size_t j = 0; j < d_v; j++)
            Vb[t * d_v + j] = bt * V[t * sv + j];
    }

    /* A = -(Kb K^T) o D_strict, then forward-substitute (I-A)^-1 rows. */
    geist_sgemm(GEIST_OP_N,
                GEIST_OP_T,
                (int) C,
                (int) C,
                (int) d_k,
                1.0f,
                Kb,
                (int) d_k,
                K,
                (int) sk,
                0.0f,
                A,
                (int) C);
    for (size_t i = 0; i < C; i++) {
        for (size_t j = 0; j < C; j++) {
            A[i * C + j] = (i > j) ? -A[i * C + j] * expf(gamma[i] - gamma[j]) : 0.0f;
        }
    }
    for (size_t i = 1; i < C; i++) {
        memcpy(rowt, A + i * C, i * sizeof(float));
        for (size_t l = 0; l < i; l++) {
            float a = rowt[l];
            for (size_t j = l + 1; j < i; j++)
                a += rowt[j] * A[j * C + l];
            A[i * C + l] = a;
        }
    }

    /* v_new = (A+I)(Vb - KCe S): the spec transforms Vb and KCe through
     * (A+I) separately, but both feed the same difference, so fold the
     * substitution into it — one C x C GEMM instead of two. Stage A*vnew
     * in Vb, which is dead after the difference. */
    geist_sgemm(GEIST_OP_N,
                GEIST_OP_N,
                (int) C,
                (int) d_v,
                (int) d_k,
                -1.0f,
                KCe,
                (int) d_k,
                S,
                (int) d_v,
                0.0f,
                vnew,
                (int) d_v);
    for (size_t i = 0; i < C * d_v; i++)
        vnew[i] += Vb[i];
    geist_sgemm(GEIST_OP_N,
                GEIST_OP_N,
                (int) C,
                (int) d_v,
                (int) C,
                1.0f,
                A,
                (int) C,
                vnew,
                (int) d_v,
                0.0f,
                Vb,
                (int) d_v);
    for (size_t i = 0; i < C * d_v; i++)
        vnew[i] += Vb[i];

    /* attn = (Q K^T) o D_incl */
    geist_sgemm(GEIST_OP_N,
                GEIST_OP_T,
                (int) C,
                (int) C,
                (int) d_k,
                1.0f,
                Q,
                (int) sq,
                K,
                (int) sk,
                0.0f,
                attn,
                (int) C);
    for (size_t i = 0; i < C; i++)
        for (size_t j = 0; j < C; j++)
            attn[i * C + j] = (i >= j) ? attn[i * C + j] * expf(gamma[i] - gamma[j]) : 0.0f;

    /* O = Qg S + attn v_new. */
    geist_sgemm(GEIST_OP_N,
                GEIST_OP_N,
                (int) C,
                (int) d_v,
                (int) d_k,
                1.0f,
                Qg,
                (int) d_k,
                S,
                (int) d_v,
                0.0f,
                o,
                (int) d_v);
    geist_sgemm(GEIST_OP_N,
                GEIST_OP_N,
                (int) C,
                (int) d_v,
                (int) C,
                1.0f,
                attn,
                (int) C,
                vnew,
                (int) d_v,
                1.0f,
                o,
                (int) d_v);

    /* S = e^{gamma_last} S + (K o e^{gamma_last - gamma})^T v_new.
     * Reuse Kb as the decayed-K staging. */
    const float glast = gamma[C - 1];
    for (size_t t = 0; t < C; t++) {
        const float w = expf(glast - gamma[t]);
        for (size_t i = 0; i < d_k; i++)
            Kb[t * d_k + i] = K[t * sk + i] * w;
    }
    const float eglast = expf(glast);
    for (size_t i = 0; i < d_k * d_v; i++)
        S[i] *= eglast;
    geist_sgemm(GEIST_OP_T,
                GEIST_OP_N,
                (int) d_k,
                (int) d_v,
                (int) C,
                1.0f,
                Kb,
                (int) d_k,
                vnew,
                (int) d_v,
                1.0f,
                S,
                (int) d_v);
}

/* Chunked prefill (#281 phase 3). The engine batches prefill at m_max
 * (= 64) tokens per forward call, so the whole call is ONE chunk of
 * C = seq — no chunk loop, no remainder. Conv + norms + gating run as
 * whole-batch passes into a heap y buffer, then dn_head_chunk per
 * v-head (OMP). Returns false if scratch allocation fails; the caller
 * falls back to the sequential token loop. */
static bool dn_run_prefill_chunked(float       *qkv, /* [seq, convd] pre-conv, mapped */
                                   float       *zg,  /* [seq, vald] gate in / mix out */
                                   const float *bb,
                                   const float *baa,
                                   const float *convw,
                                   const float *aw,
                                   const float *dtb,
                                   const float *nrm,
                                   float       *cstate, /* [(K-1) * convd] */
                                   float       *S,      /* [n_vh * d_k * d_v] */
                                   size_t       seq,
                                   size_t       n_kh,
                                   size_t       n_vh,
                                   size_t       d_k,
                                   size_t       d_v,
                                   size_t       K,
                                   size_t       keyd,
                                   size_t       vald,
                                   size_t       convd,
                                   float        eps,
                                   float        qscale) {
    const size_t hist_rows = K - 1;
    const size_t y_f       = seq * convd;
    const size_t bg_f      = seq * n_vh;
    const size_t old_f     = hist_rows * convd;
    const size_t ws_f      = transformer_dn_chunk_ws_floats(seq, d_k, d_v) + seq * d_v;
#if defined(_OPENMP)
    const size_t nthr = (size_t) omp_get_max_threads();
#else
    const size_t nthr = 1;
#endif
    /* One allocation up front — no alloc inside the head loop, so a
     * failure here leaves state untouched and the sequential fallback
     * stays valid. */
    float *y = heap_alloc_aligned((y_f + 2 * bg_f + old_f + nthr * ws_f) * sizeof(float),
                                  OPTIMAL_ALIGNMENT);
    if (y == nullptr)
        return false;
    float *betas   = y + y_f;
    float *gs      = betas + bg_f;
    float *old_cst = gs + bg_f;
    float *ws_all  = old_cst + old_f;
    memcpy(old_cst, cstate, old_f * sizeof(float));

    /* Conv + silu for every token (reads only pre-conv qkv + old state,
     * so tokens are independent), then gating scalars. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t t = 0; t < seq; t++) {
        float *y_t = y + t * convd;
        for (size_t c = 0; c < convd; c++) {
            float acc = 0.0f;
            for (size_t j = 0; j < K; j++) {
                const ptrdiff_t src = (ptrdiff_t) (t + j) - (ptrdiff_t) hist_rows;
                const float     xv =
                        (src >= 0) ? qkv[(size_t) src * convd + c] : old_cst[(t + j) * convd + c];
                acc += convw[c * K + j] * xv;
            }
            y_t[c] = silu_f(acc);
        }
        float *q = y_t;
        float *k = y_t + keyd;
        for (size_t h = 0; h < n_kh; h++) {
            l2norm_row(q + h * d_k, d_k, eps);
            l2norm_row(k + h * d_k, d_k, eps);
            for (size_t i = 0; i < d_k; i++)
                q[h * d_k + i] *= qscale;
        }
        for (size_t hv = 0; hv < n_vh; hv++) {
            betas[t * n_vh + hv] = 1.0f / (1.0f + expf(-bb[t * n_vh + hv]));
            const float sp       = log1pf(expf(baa[t * n_vh + hv] + dtb[hv]));
            gs[t * n_vh + hv]    = aw[hv] * sp;
        }
    }

    /* Roll the conv state forward by seq tokens: the new last K-1
     * pre-conv rows, taking from old state when seq < K-1. */
    for (size_t j = 0; j < hist_rows; j++) {
        const ptrdiff_t src = (ptrdiff_t) (seq + j) - (ptrdiff_t) hist_rows;
        const float    *row = (src >= 0) ? qkv + (size_t) src * convd : old_cst + (seq + j) * convd;
        memcpy(cstate + j * convd, row, convd * sizeof(float));
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t hv = 0; hv < n_vh; hv++) {
#if defined(_OPENMP)
        float *ws = ws_all + (size_t) omp_get_thread_num() * ws_f;
#else
        float *ws = ws_all;
#endif
        float       *o  = ws + ws_f - seq * d_v;
        const size_t hk = hv % n_kh; /* tiled v-head order, see decode path */
        transformer_dn_head_chunk(S + hv * d_k * d_v,
                                  y + hk * d_k,
                                  convd,
                                  y + keyd + hk * d_k,
                                  convd,
                                  y + 2 * keyd + hv * d_v,
                                  convd,
                                  betas + hv,
                                  gs + hv,
                                  n_vh,
                                  seq,
                                  d_k,
                                  d_v,
                                  o,
                                  ws);
        for (size_t t = 0; t < seq; t++) {
            float *o_t = o + t * d_v;
            float *z_t = zg + t * vald + hv * d_v;
            rmsnorm_row(o_t, nrm, d_v, eps);
            for (size_t j = 0; j < d_v; j++)
                z_t[j] = o_t[j] * silu_f(z_t[j]); /* gate read, then slot reused as mix */
        }
    }
    safe_free((void **) &y);
    return true;
}

[[nodiscard]] enum geist_status
transformer_layer_run_deltanet_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session       *sess  = ctx->sess;
    struct transformer_arch_state         *st    = ctx->st;
    struct transformer_layer_weights      *L     = ctx->L;
    struct geist_backend                  *be    = ctx->be;
    const struct geist_backend_vtbl       *v     = ctx->v;
    const struct geist_backend_primitives *prims = ctx->prims;
    const size_t                           seq   = ctx->seq;

    const size_t n_kh   = st->config.dn_n_k_heads;
    const size_t n_vh   = st->config.dn_n_v_heads;
    const size_t d_k    = st->config.dn_head_k;
    const size_t d_v    = st->config.dn_head_v;
    const size_t K      = st->config.dn_conv_kernel;
    const size_t keyd   = n_kh * d_k;
    const size_t vald   = n_vh * d_v;
    const size_t convd  = 2 * keyd + vald;
    const float  eps    = ctx->eps;
    const float  qscale = 1.0f / sqrtf((float) d_k);
    /* 0.8B has n_kh == n_vh; larger variants (27B: 16 k-heads, 48
     * v-heads) share each k-head across n_vh/n_kh v-heads. */
    const size_t vh_per_kh = n_vh / n_kh;

    if (sess->dn_conv_state == nullptr || sess->dn_S == nullptr ||
        sess->dn_conv_state[ctx->layer_idx] == nullptr || sess->dn_S[ctx->layer_idx] == nullptr ||
        vh_per_kh == 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_STATE, "deltanet: state not allocated");
        return GEIST_E_INVALID_STATE;
    }

    /* 1. Input norm into scratch_normed [seq, d_model]. */
    struct geist_tensor t_h_2d  = view_2d(ctx->h_in_buf, ctx->SEQ, st->d_model);
    struct geist_tensor t_x_2d  = view_2d(sess->scratch_normed, ctx->SEQ, st->d_model);
    struct geist_tensor t_gamma = view_1d(L->attn_norm.buffer, st->d_model);
    enum geist_status   s       = prims->rmsnorm(be, &t_h_2d, &t_gamma, eps, &t_x_2d);
    if (s != GEIST_OK)
        return s;

    /* 2. Projections through the backend linear kernels. */
    /* qkv + z as one pair call (#294): shared activation quantization
     * and a single thread-pool wake when the backend installs a pair
     * kernel (Q4_0 x8 on Mac); falls back to two calls otherwise. */
    struct geist_tensor t_qkv = view_2d(sess->dn_scratch_qkv, ctx->SEQ, (int64_t) convd);
    struct geist_tensor t_z   = view_2d(sess->dn_scratch_z, ctx->SEQ, (int64_t) vald);
    s                         = linear_w_pair_or_legacy(be,
                                                        v,
                                                        sess->scratch_normed,
                                                        sess->dn_scratch_qkv,
                                                        sess->dn_scratch_z,
                                                        &L->dn_qkv_w,
                                                        &L->dn_z_w,
                                                        seq,
                                                        &t_x_2d,
                                                        &L->dn_qkv,
                                                        &L->dn_z,
                                                        &t_qkv,
                                                        &t_z);
    if (s != GEIST_OK)
        return s;
    /* beta and alpha each in their own scratch — the resolved CPU
     * kernels write at buffer_map(y_buf) start, so a shared buffer with
     * offset views would alias. */
    struct geist_tensor t_b = view_2d(sess->dn_scratch_b, ctx->SEQ, (int64_t) n_vh);
    s                       = linear_w_or_legacy(be,
                                                 v,
                                                 sess->scratch_normed,
                                                 sess->dn_scratch_b,
                                                 &L->dn_beta_w,
                                                 seq,
                                                 &t_x_2d,
                                                 &L->dn_beta,
                                                 &t_b);
    if (s != GEIST_OK)
        return s;
    struct geist_tensor t_a = view_2d(sess->dn_scratch_a, ctx->SEQ, (int64_t) n_vh);
    s                       = linear_w_or_legacy(be,
                                                 v,
                                                 sess->scratch_normed,
                                                 sess->dn_scratch_a,
                                                 &L->dn_alpha_w,
                                                 seq,
                                                 &t_x_2d,
                                                 &L->dn_alpha,
                                                 &t_a);
    if (s != GEIST_OK)
        return s;

    /* 3..7 host-side: conv + gating + recurrence + gated norm. */
    float       *qkv    = (float *) v->buffer_map(sess->dn_scratch_qkv);
    float       *zg     = (float *) v->buffer_map(sess->dn_scratch_z);
    float       *bb     = (float *) v->buffer_map(sess->dn_scratch_b);
    float       *baa    = (float *) v->buffer_map(sess->dn_scratch_a);
    const float *convw  = (const float *) v->buffer_map(L->dn_conv.buffer);
    const float *aw     = (const float *) v->buffer_map(L->dn_a.buffer);
    const float *dtb    = (const float *) v->buffer_map(L->dn_dt_bias.buffer);
    const float *nrm    = (const float *) v->buffer_map(L->dn_norm.buffer);
    float       *cstate = sess->dn_conv_state[ctx->layer_idx]; /* [(K-1) * convd] */
    float       *S      = sess->dn_S[ctx->layer_idx];          /* [n_vh * d_k * d_v] */
    if (qkv == nullptr || zg == nullptr || bb == nullptr || baa == nullptr || convw == nullptr ||
        aw == nullptr || dtb == nullptr || nrm == nullptr || d_v > 512 || d_k > 512) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "deltanet: buffer_map failed");
        return GEIST_E_BACKEND;
    }

    /* mixer output rows accumulate here, then one linear to d_model */
    float *mix = zg; /* reuse the z buffer in place: gate consumed per row */

    /* Prefill (seq > 1): chunked delta rule — GEMM work instead of seq
     * sequential state passes. Decode and the (alloc-failure) fallback
     * take the exact sequential loop below. */
    const bool chunked = seq > 1 && !st->runtime_flags.dn_seq_prefill &&
                         dn_run_prefill_chunked(qkv,
                                                zg,
                                                bb,
                                                baa,
                                                convw,
                                                aw,
                                                dtb,
                                                nrm,
                                                cstate,
                                                S,
                                                seq,
                                                n_kh,
                                                n_vh,
                                                d_k,
                                                d_v,
                                                K,
                                                keyd,
                                                vald,
                                                convd,
                                                eps,
                                                qscale);

    for (size_t t = 0; chunked == false && t < seq; t++) {
        float *qkv_t = qkv + t * convd;

        /* 3. causal depthwise conv over [cstate | current]: y[c] =
         * silu(sum_j w[c][j] * hist[j][c]) where hist is the last K
         * pre-conv vectors ending at t. Conv weight layout from GGUF
         * [K, convd] row-major C: convw[c*K + j]. Update the rolling
         * state BEFORE overwriting qkv_t with the conv result. */
        float y[16384]; /* convd <= 16384 for all known variants (27B: 10240) */
        if (convd > sizeof y / sizeof y[0]) {
            geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "deltanet: conv_dim too large");
            return GEIST_E_UNSUPPORTED;
        }
#if defined(__ARM_NEON)
        if (K == 4) {
            /* Kernel-4 fast path: vld4 deinterleaves w[c][4] into 4
             * per-tap vectors of 4 channels; history rows are channel-
             * contiguous. silu stays scalar (expf). */
            const float *h0 = cstate;             /* t-3 */
            const float *h1 = cstate + convd;     /* t-2 */
            const float *h2 = cstate + 2 * convd; /* t-1 */
            size_t       c  = 0;
            for (; c + 4 <= convd; c += 4) {
                const float32x4x4_t wv  = vld4q_f32(convw + c * 4);
                float32x4_t         acc = vmulq_f32(wv.val[0], vld1q_f32(h0 + c));
                acc                     = vfmaq_f32(acc, wv.val[1], vld1q_f32(h1 + c));
                acc                     = vfmaq_f32(acc, wv.val[2], vld1q_f32(h2 + c));
                acc                     = vfmaq_f32(acc, wv.val[3], vld1q_f32(qkv_t + c));
                float av[4];
                vst1q_f32(av, acc);
                y[c]     = silu_f(av[0]);
                y[c + 1] = silu_f(av[1]);
                y[c + 2] = silu_f(av[2]);
                y[c + 3] = silu_f(av[3]);
            }
            for (; c < convd; c++) {
                float acc = convw[c * 4] * h0[c] + convw[c * 4 + 1] * h1[c] +
                            convw[c * 4 + 2] * h2[c] + convw[c * 4 + 3] * qkv_t[c];
                y[c]      = silu_f(acc);
            }
        } else
#endif
        {
            for (size_t c = 0; c < convd; c++) {
                float acc = 0.0f;
                for (size_t j = 0; j < K; j++) {
                    /* history index: j=0..K-2 from cstate, j=K-1 is current */
                    const float xv = (j + 1 < K) ? cstate[j * convd + c] : qkv_t[c];
                    acc += convw[c * K + j] * xv;
                }
                y[c] = silu_f(acc);
            }
        }
        /* roll conv state: drop oldest, append current pre-conv vector */
        if (K >= 2) {
            memmove(cstate, cstate + convd, (K - 2) * convd * sizeof(float));
            memcpy(cstate + (K - 2) * convd, qkv_t, convd * sizeof(float));
        }

        /* 4. split + per-head l2 norm + q scale */
        float *q   = y;            /* [n_kh, d_k] */
        float *k   = y + keyd;     /* [n_kh, d_k] */
        float *val = y + 2 * keyd; /* [n_vh, d_v] */
        for (size_t h = 0; h < n_kh; h++) {
            l2norm_row(q + h * d_k, d_k, eps);
            l2norm_row(k + h * d_k, d_k, eps);
            for (size_t i = 0; i < d_k; i++)
                q[h * d_k + i] *= qscale;
        }

        /* 5. gating scalars + delta-rule recurrence per v-head */
        const float *b_t = bb + t * n_vh;
        const float *a_t = baa + t * n_vh;
        float       *z_t = zg + t * vald;
        float       *o_t = mix + t * vald; /* == z_t buffer, overwritten below */

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (size_t hv = 0; hv < n_vh; hv++) {
            const float beta = 1.0f / (1.0f + expf(-b_t[hv]));
            /* softplus in f32; aw = -exp(A_log) already */
            const float sp    = log1pf(expf(a_t[hv] + dtb[hv]));
            const float g     = aw[hv] * sp;
            const float decay = expf(g);

            /* GGUF v-heads are in TILED order [G0_v0, G1_v0, ...] (the
             * converter's _reorder_v_heads): v-head position hv maps to
             * k-head hv % n_kh. Identity when n_vh == n_kh (0.8B);
             * load-bearing for the 27B's 16:48 split. */
            const size_t hk = hv % n_kh;
            const float *qh = q + hk * d_k;
            const float *kh = k + hk * d_k;
            const float *vh = val + hv * d_v;
            float       *Sh = S + hv * d_k * d_v;

            float kv_mem[512], delta[512], o_h[512];
            transformer_dn_head_step(Sh, qh, kh, vh, decay, beta, d_k, d_v, kv_mem, delta, o_h);

            /* 6. gated per-head RMSNorm + silu(z) gate. o_t aliases z_t,
             * so read the gate BEFORE overwriting. */
            rmsnorm_row(o_h, nrm, d_v, eps);
            for (size_t j = 0; j < d_v; j++) {
                const float gate  = silu_f(z_t[hv * d_v + j]);
                o_t[hv * d_v + j] = o_h[j] * gate;
            }
        }
    }

    v->buffer_unmap(sess->dn_scratch_qkv);
    v->buffer_unmap(sess->dn_scratch_z);
    v->buffer_unmap(sess->dn_scratch_b);
    v->buffer_unmap(sess->dn_scratch_a);
    v->buffer_unmap(L->dn_conv.buffer);
    v->buffer_unmap(L->dn_a.buffer);
    v->buffer_unmap(L->dn_dt_bias.buffer);
    v->buffer_unmap(L->dn_norm.buffer);

    /* 7. out projection [seq, vald] -> [seq, d_model] into scratch_o,
     * then plain residual add into h_out (h_post_attn slot, so the FFN
     * block reads the same place as after an attention block). */
    struct geist_tensor t_mix  = view_2d(sess->dn_scratch_z, ctx->SEQ, (int64_t) vald);
    struct geist_tensor t_o_2d = view_2d(sess->scratch_o, ctx->SEQ, st->d_model);
    s                          = linear_w_or_legacy(be,
                                                    v,
                                                    sess->dn_scratch_z,
                                                    sess->scratch_o,
                                                    &L->dn_out_w,
                                                    seq,
                                                    &t_mix,
                                                    &L->dn_out,
                                                    &t_o_2d);
    if (s != GEIST_OK)
        return s;
    struct geist_tensor t_h_post = view_2d(sess->scratch_h_post_attn, ctx->SEQ, st->d_model);
    s                            = prims->add(be, &t_h_2d, &t_o_2d, &t_h_post);
    if (s != GEIST_OK)
        return s;
    /* Recurrent state advanced — there is no KV to advance, but the
     * shared kv_len bookkeeping still drives positions for the attn
     * layers; ctx->kv_len_now is applied by the caller as usual. */
    return GEIST_OK;
}
