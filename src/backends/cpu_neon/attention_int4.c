/*
 * src/backends/cpu_neon/attention_int4.c — attention over the packed INT4
 * KV cache with FEAT_DotProd: cpu_neon's fused->attention_kv_int4.
 *
 * Layer: BACKEND (cpu_neon).
 *
 * The architecture's attention_int4_via_buffers (forward/attention.c)
 * dotted each query head against the unpacked K rows with vdotq_s32 on
 * ARM. That loop is here now, unchanged: each cache row unpacked from
 * head_dim / 2 bytes (int4_kv.h), the query quantized to int8 (amax / 127,
 * round to nearest), exact int32 dots, an online softmax over blocks of
 * 512 positions, the V sums in fp32. The architecture keeps the portable
 * loop as the decomposed twin; built without FEAT_DotProd, this file is
 * empty and cpu_neon leaves the slot null.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "gemma4_kernels.h" /* ATTN_EXP_FLOOR */
#include "int4_kv.h"
#include "tensor_view.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>

/* The architecture refuses a larger head_dim at load
 * (TRANSFORMER_HEAD_DIM_MAX); the stack arrays below are sized by it. */
constexpr size_t NI4_HEAD_DIM_MAX = 512;

/* Context positions per online-softmax block. */
constexpr size_t NI4_BLOCK = 512;

/* scores[j] = exp(scores[j] - max_score) for j < n, the exponent clamped at
 * ATTN_EXP_FLOOR (gemma4_kernels.h); returns their sum. Out of line: inlined
 * into the loop below, the vector exp call changed the register
 * allocation of their whole loop nest, and INT4 decode ran 2-8 % slower
 * though the loop itself gained one vmaxps (measured on x86, where this
 * loop came from). */
[[gnu::noinline]] static double attn_exp_block(size_t n, float scores[static n], float max_score) {
    double sum = 0.0;
    for (size_t j = 0; j < n; j++) {
        scores[j] = expf(fmaxf(scores[j] - max_score, ATTN_EXP_FLOOR));
        sum += scores[j];
    }
    return sum;
}

/* The kernel on host pointers, validated by the caller: q and out
 * [n_q, n_q_heads, head_dim], k_q4 and v_q4 [n_kv, n_kv_heads, head_dim / 2]
 * bytes, the scales [n_kv, n_kv_heads].
 *
 * int4_unpack_row fully writes [0, head_dim), and nothing reads past it, so
 * GCC's -Wmaybe-uninitialized on the unpack buffers is a false positive,
 * suppressed here rather than paid for with a zeroed row. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
void cpu_neon_attention_kv_int4_run(size_t         n_q,
                                    size_t         n_q_heads,
                                    size_t         head_dim,
                                    size_t         n_kv,
                                    size_t         n_kv_heads,
                                    size_t         q_offset,
                                    size_t         sliding_window,
                                    const float   *q,
                                    const uint8_t *k_q4,
                                    const float   *k_scale,
                                    const uint8_t *v_q4,
                                    const float   *v_scale,
                                    float         *out) {

    const size_t kv_group_size = n_q_heads / n_kv_heads;
    const size_t packed        = head_dim / 2; /* bytes per cache row */
/* collapse(2), not a loop over t alone: decode passes n_q == 1, so only
 * the head axis has width to parallelize over. */
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (size_t t = 0; t < n_q; t++) {
        for (size_t h = 0; h < n_q_heads; h++) {
            const size_t q_pos = q_offset + t;
            const size_t s_lo  = (sliding_window > 0 && q_pos + 1 > sliding_window)
                                         ? q_pos + 1 - sliding_window
                                         : 0;
            const size_t s_hi  = q_pos < n_kv ? q_pos : n_kv - 1;
            float        scores[NI4_BLOCK]; /* one block of the context */

            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            int8_t q_q8[NI4_HEAD_DIM_MAX];
            float  amax = 0.0f;
            for (size_t i = 0; i < head_dim; i++) {
                float a = fabsf(qv[i]);
                if (a > amax) {
                    amax = a;
                }
            }
            float scale_q = amax / 127.0f;
            if (scale_q == 0.0f) {
                scale_q = 1.0f;
            }
            const float inv_q = 1.0f / scale_q;
            for (size_t i = 0; i < head_dim; i++) {
                q_q8[i] = (int8_t) lrintf(qv[i] * inv_q);
            }

            float *outv = out + (t * n_q_heads + h) * head_dim;
            for (size_t i = 0; i < head_dim; i++) {
                outv[i] = 0.0f;
            }
            float  max_score = 0.0f;
            double sum_exp   = 0.0;
            for (size_t b0 = s_lo; b0 <= s_hi; b0 += NI4_BLOCK) {
                const size_t n = s_hi - b0 < NI4_BLOCK ? s_hi - b0 + 1 : NI4_BLOCK;
                for (size_t j = 0; j < n; j++) {
                    const size_t s = b0 + j;
                    int8_t       k[NI4_HEAD_DIM_MAX];
                    int4_unpack_row(head_dim, k_q4 + (s * n_kv_heads + kv_h) * packed, k);
                    const float ks      = k_scale[s * n_kv_heads + kv_h];
                    int32_t     int_dot = 0;
                    int32x4_t   acc     = vdupq_n_s32(0);
                    size_t      i       = 0;
                    for (; i + 16 <= head_dim; i += 16) {
                        acc = vdotq_s32(acc, vld1q_s8(q_q8 + i), vld1q_s8(k + i));
                    }
                    int_dot = vaddvq_s32(acc);
                    for (; i < head_dim; i++) {
                        int_dot += (int32_t) q_q8[i] * (int32_t) k[i];
                    }
                    scores[j] = (float) int_dot * scale_q * ks;
                }

                float block_max = scores[0];
                for (size_t j = 1; j < n; j++) {
                    if (scores[j] > block_max) {
                        block_max = scores[j];
                    }
                }
                if (b0 == s_lo) {
                    max_score = block_max;
                } else if (block_max > max_score) {
                    const float d = max_score - block_max;
                    const float c = d < ATTN_EXP_FLOOR ? 0.0f : expf(d);
                    sum_exp *= c;
                    for (size_t i = 0; i < head_dim; i++) {
                        outv[i] *= c;
                    }
                    max_score = block_max;
                }
                sum_exp += attn_exp_block(n, scores, max_score);

                for (size_t j = 0; j < n; j++) {
                    const size_t s = b0 + j;
                    int8_t       vv[NI4_HEAD_DIM_MAX];
                    int4_unpack_row(head_dim, v_q4 + (s * n_kv_heads + kv_h) * packed, vv);
                    const float vs  = v_scale[s * n_kv_heads + kv_h];
                    const float wvs = scores[j] * vs;
                    for (size_t i = 0; i < head_dim; i++) {
                        outv[i] += wvs * (float) vv[i];
                    }
                }
            }
            const float inv_sum = (float) (1.0 / sum_exp);
            for (size_t i = 0; i < head_dim; i++) {
                outv[i] *= inv_sum;
            }
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

/* ---- The fused op -------------------------------------------------------- */

/* The first element of a contiguous DENSE view of `ndim` dimensions and
 * `dtype` (elements of `elem` bytes), or nullptr if the view is malformed or
 * runs past its buffer; *out_n gets its element count. */
static const void *ni4_view(const struct geist_tensor *t,
                            enum geist_dtype           dtype,
                            size_t                     elem,
                            int                        ndim,
                            size_t                    *out_n) {
    return t != nullptr && t->buffer != nullptr
                   ? geist_tensor_dense(
                             t, dtype, elem, ndim, t->buffer->host, t->buffer->bytes, out_n)
                   : nullptr;
}

bool cpu_neon_attention_kv_int4_supported(const struct geist_fusion_query *q) {
    return q != nullptr && q->head_dim >= 2 && q->head_dim % 2 == 0 &&
           q->head_dim <= NI4_HEAD_DIM_MAX && q->n_kv_heads >= 1 && q->n_q_heads >= q->n_kv_heads &&
           q->n_q_heads % q->n_kv_heads == 0;
}

enum geist_status cpu_neon_attention_kv_int4(struct geist_backend                      *be,
                                             const struct geist_attention_kv_int4_args *args) {
    if (be == nullptr || args == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t         nq = 0, nk = 0, nv = 0, nks = 0, nvs = 0, no = 0;
    const float   *q   = ni4_view(args->q, GEIST_DTYPE_F32, sizeof(float), 3, &nq);
    const uint8_t *k   = ni4_view(args->k, GEIST_DTYPE_U8, 1, 3, &nk);
    const uint8_t *v   = ni4_view(args->v, GEIST_DTYPE_U8, 1, 3, &nv);
    const float   *ks  = ni4_view(args->k_scale, GEIST_DTYPE_F32, sizeof(float), 2, &nks);
    const float   *vs  = ni4_view(args->v_scale, GEIST_DTYPE_F32, sizeof(float), 2, &nvs);
    float         *out = (float *) ni4_view(args->out, GEIST_DTYPE_F32, sizeof(float), 3, &no);
    if (q == nullptr || k == nullptr || v == nullptr || ks == nullptr || vs == nullptr ||
        out == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_neon attention_kv_int4: q, scales and out must be F32, k "
                                "and v U8, DENSE views of the rank given within their buffers");
        return GEIST_E_INVALID_ARG;
    }
    const size_t                    n_q        = (size_t) args->q->shape[0];
    const size_t                    n_q_heads  = (size_t) args->q->shape[1];
    const size_t                    head_dim   = (size_t) args->q->shape[2];
    const size_t                    n_kv       = (size_t) args->k->shape[0];
    const size_t                    n_kv_heads = (size_t) args->k->shape[1];
    const struct geist_fusion_query shape      = {.op         = GEIST_FUSED_ATTN_KV_INT4,
                                                  .head_dim   = head_dim,
                                                  .n_q_heads  = n_q_heads,
                                                  .n_kv_heads = n_kv_heads};
    if (!cpu_neon_attention_kv_int4_supported(&shape)) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "cpu_neon attention_kv_int4: head_dim %zu, %zu query heads on %zu "
                                "KV heads",
                                head_dim,
                                n_q_heads,
                                n_kv_heads);
        return GEIST_E_UNSUPPORTED;
    }
    /* Matching shapes (half a byte per K and V value), and every query's
     * position in the cache: n_q and n_kv are >= 1 (ni4_view refuses an
     * empty dimension), so the subtraction cannot wrap. */
    if ((size_t) args->k->shape[2] != head_dim / 2 || nv != nk ||
        (size_t) args->v->shape[0] != n_kv || (size_t) args->v->shape[1] != n_kv_heads ||
        no != nq || (size_t) args->out->shape[0] != n_q ||
        (size_t) args->out->shape[1] != n_q_heads || nks != n_kv * n_kv_heads ||
        nvs != n_kv * n_kv_heads || (size_t) args->k_scale->shape[0] != n_kv ||
        (size_t) args->v_scale->shape[0] != n_kv || n_q > n_kv || args->q_offset > n_kv - n_q) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_neon attention_kv_int4: shapes do not match, or %zu "
                                "queries at %zu run past %zu cached positions",
                                n_q,
                                args->q_offset,
                                n_kv);
        return GEIST_E_INVALID_ARG;
    }
    cpu_neon_attention_kv_int4_run(n_q,
                                   n_q_heads,
                                   head_dim,
                                   n_kv,
                                   n_kv_heads,
                                   args->q_offset,
                                   args->sliding_window,
                                   q,
                                   k,
                                   ks,
                                   v,
                                   vs,
                                   out);
    return GEIST_OK;
}

#endif /* __ARM_NEON && __ARM_FEATURE_DOTPROD */
