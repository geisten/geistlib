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
 * scalar loops over mapped pointers.
 * ponytail: sequential per-token recurrence (exact); the chunked prefill
 * formulation is a later speed phase, NEON/x86 kernels likewise.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_state.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <string.h>

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
    struct geist_tensor t_qkv = view_2d(sess->dn_scratch_qkv, ctx->SEQ, (int64_t) convd);
    s                         = linear_w_or_legacy(be,
                                                   v,
                                                   sess->scratch_normed,
                                                   sess->dn_scratch_qkv,
                                                   &L->dn_qkv_w,
                                                   seq,
                                                   &t_x_2d,
                                                   &L->dn_qkv,
                                                   &t_qkv);
    if (s != GEIST_OK)
        return s;
    struct geist_tensor t_z = view_2d(sess->dn_scratch_z, ctx->SEQ, (int64_t) vald);
    s                       = linear_w_or_legacy(be,
                                                 v,
                                                 sess->scratch_normed,
                                                 sess->dn_scratch_z,
                                                 &L->dn_z_w,
                                                 seq,
                                                 &t_x_2d,
                                                 &L->dn_z,
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

    for (size_t t = 0; t < seq; t++) {
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
        for (size_t c = 0; c < convd; c++) {
            float acc = 0.0f;
            for (size_t j = 0; j < K; j++) {
                /* history index: j=0..K-2 from cstate, j=K-1 is current */
                const float xv = (j + 1 < K) ? cstate[j * convd + c] : qkv_t[c];
                acc += convw[c * K + j] * xv;
            }
            y[c] = silu_f(acc);
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

            /* S *= exp(g); kv_mem = S^T k; delta = (v - kv_mem)*beta;
             * S += k (x) delta; o = S^T q — fused loops. */
            float kv_mem[512], delta[512], o_h[512];
            for (size_t j = 0; j < d_v; j++)
                kv_mem[j] = 0.0f;
            for (size_t i = 0; i < d_k; i++) {
                float      *Srow = Sh + i * d_v;
                const float ki   = kh[i];
                for (size_t j = 0; j < d_v; j++) {
                    Srow[j] *= decay;
                    kv_mem[j] += Srow[j] * ki;
                }
            }
            for (size_t j = 0; j < d_v; j++)
                delta[j] = (vh[j] - kv_mem[j]) * beta;
            for (size_t j = 0; j < d_v; j++)
                o_h[j] = 0.0f;
            for (size_t i = 0; i < d_k; i++) {
                float      *Srow = Sh + i * d_v;
                const float ki   = kh[i];
                const float qi   = qh[i];
                for (size_t j = 0; j < d_v; j++) {
                    Srow[j] += ki * delta[j];
                    o_h[j] += Srow[j] * qi;
                }
            }

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
