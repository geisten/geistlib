/*
 * src/backends/cpu_neon/attention_kv.h — the attention loop and the
 * argument checks shared by attention_int8.c and attention_int4.c.
 *
 * Layer: BACKEND (cpu_neon). Included only by those two files, under
 * __ARM_NEON && __ARM_FEATURE_DOTPROD.
 *
 * The two caches differ only in how a K or V row reaches the int8 dot:
 * the INT8 cache is read in place, the INT4 one unpacked into a stack row
 * first (int4_kv.h). The including file says which with a
 *
 *     constexpr bool NKV_INT4 = ...;
 *
 * before the #include. A constant rather than a parameter: the parallel
 * region is outlined before a parameter could be propagated into it, and
 * only a constant visible there specializes the body per cache, so neither
 * pays for the other's branch.
 */
#ifndef GEIST_CPU_NEON_ATTENTION_KV_H
#define GEIST_CPU_NEON_ATTENTION_KV_H

#include "internal.h"

#include "gemma4_kernels.h" /* ATTN_EXP_FLOOR, ATTN_F32_BLOCK */
#include "int4_kv.h"
#include "tensor_view.h"

#include <geist.h>
#include <geist_backend.h>

#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* The architecture refuses a larger head_dim at load
 * (TRANSFORMER_HEAD_DIM_MAX); the stack arrays below are sized by it. */
constexpr size_t NKV_HEAD_DIM_MAX = 512;

/* scores[j] = exp(scores[j] - max_score) for j < n, the exponent clamped at
 * ATTN_EXP_FLOOR (gemma4_kernels.h); returns their sum. Out of line: inlined,
 * the vector exp call perturbs register allocation of the whole loop nest
 * (INT4 decode 2-8 % slower, measured on x86). */
[[gnu::noinline]] static double attn_exp_block(size_t n, float scores[static n], float max_score) {
    double sum = 0.0;
    for (size_t j = 0; j < n; j++) {
        scores[j] = expf(fmaxf(scores[j] - max_score, ATTN_EXP_FLOOR));
        sum += scores[j];
    }
    return sum;
}

/* Row `row` of a cache as head_dim int8 values: in place for INT8,
 * unpacked from head_dim / 2 bytes into `buf` for INT4. */
static inline const int8_t *
nkv_row(size_t head_dim, const void *cache, size_t row, int8_t buf[static head_dim]) {
    if (NKV_INT4) {
        int4_unpack_row(head_dim, (const uint8_t *) cache + row * (head_dim / 2), buf);
        return buf;
    }
    return (const int8_t *) cache + row * head_dim;
}

/* The kernel on host pointers, validated by nkv_attention: q and out
 * [n_q, n_q_heads, head_dim], k and v [n_kv, n_kv_heads, head_dim] int8
 * or [n_kv, n_kv_heads, head_dim / 2] packed INT4 bytes, the scales
 * [n_kv, n_kv_heads].
 *
 * int4_unpack_row fully writes [0, head_dim), and nothing reads past it, so
 * GCC's -Wmaybe-uninitialized on the unpack buffers is a false positive,
 * suppressed here rather than paid for with a zeroed row. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
static void nkv_run(size_t       n_q,
                    size_t       n_q_heads,
                    size_t       head_dim,
                    size_t       n_kv,
                    size_t       n_kv_heads,
                    size_t       q_offset,
                    size_t       sliding_window,
                    const float *q,
                    const void  *k_cache,
                    const float *k_scale,
                    const void  *v_cache,
                    const float *v_scale,
                    float       *out) {

    const size_t kv_group_size = n_q_heads / n_kv_heads;
    /* The O(n^2) attention core. Every (t,h) is independent: it reads the
     * shared Q/K/V and writes only its own out[(t*n_q_heads+h)*head_dim..]
     * slice plus a private `scores` scratch, so this parallelizes with no
     * change to any per-(t,h) reduction order — bit-exact vs serial.
     *
     * collapse(2), NOT a plain loop over t: decode passes n_q == 1, and a
     * parallel loop over one iteration runs on one thread; the heads are the
     * axis that still has width when t does not.

     *
     * Causal + sliding-window masking makes per-t work uneven (later positions
     * attend to more keys), so schedule(dynamic). Guarded so a non-OpenMP build
     * (no -fopenmp) skips the pragma cleanly under -Wunknown-pragmas -Werror. */
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (size_t t = 0; t < n_q; t++) {
        for (size_t h = 0; h < n_q_heads; h++) {
            /* Recomputed per (t,h) rather than hoisted to the t loop: three
             * scalar ops, and perfect nesting is what collapse(2) requires. */
            const size_t q_pos = q_offset + t;
            const size_t s_lo  = (sliding_window > 0 && q_pos + 1 > sliding_window)
                                         ? q_pos + 1 - sliding_window
                                         : 0;
            const size_t s_hi  = q_pos < n_kv ? q_pos : n_kv - 1;
            /* One block of the context at a time, private per (t,h): the
             * softmax runs online, so the stack does not hold n_kv scores. */
            float scores[ATTN_F32_BLOCK];

            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            /* Per-head INT8 quant of Q[t,h,:]; head_dim <= NKV_HEAD_DIM_MAX
             * (the entry refuses more). */
            int8_t q_q8[NKV_HEAD_DIM_MAX];
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
            for (size_t b0 = s_lo; b0 <= s_hi; b0 += ATTN_F32_BLOCK) {
                const size_t n = s_hi - b0 < ATTN_F32_BLOCK ? s_hi - b0 + 1 : ATTN_F32_BLOCK;
                for (size_t j = 0; j < n; j++) {
                    const size_t  s = b0 + j;
                    const size_t  r = s * n_kv_heads + kv_h;
                    int8_t        k_buf[NKV_HEAD_DIM_MAX];
                    const int8_t *k       = nkv_row(head_dim, k_cache, r, k_buf);
                    const float   ks      = k_scale[r];
                    int32_t       int_dot = 0;
                    int32x4_t     acc     = vdupq_n_s32(0);
                    size_t        i       = 0;
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
                    /* The sum and the V sums so far were taken against the
                     * lower max: scale them down to the new one (to 0 below
                     * ATTN_EXP_FLOOR). */
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
                    const size_t  s = b0 + j;
                    const size_t  r = s * n_kv_heads + kv_h;
                    int8_t        v_buf[NKV_HEAD_DIM_MAX];
                    const int8_t *vv  = nkv_row(head_dim, v_cache, r, v_buf);
                    const float   vs  = v_scale[r];
                    const float   wvs = scores[j] * vs;
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

/* The first element of a contiguous DENSE view of `ndim` dimensions and
 * `dtype` (elements of `elem` bytes), or nullptr if the view is malformed or
 * runs past its buffer; *out_n gets its element count. */
static inline const void *nkv_view(const struct geist_tensor *t,
                                   enum geist_dtype           dtype,
                                   size_t                     elem,
                                   int                        ndim,
                                   size_t                    *out_n) {
    return t != nullptr && t->buffer != nullptr
                   ? geist_tensor_dense(
                             t, dtype, elem, ndim, t->buffer->host, t->buffer->bytes, out_n)
                   : nullptr;
}

/* fused->attention_kv_int8 or fused->attention_kv_int4, per NKV_INT4, its
 * argument struct unpacked: checks the views and the shapes, then runs. */
static inline enum geist_status nkv_attention(struct geist_backend      *be,
                                              const struct geist_tensor *q_t,
                                              const struct geist_tensor *k_t,
                                              const struct geist_tensor *k_scale_t,
                                              const struct geist_tensor *v_t,
                                              const struct geist_tensor *v_scale_t,
                                              struct geist_tensor       *out_t,
                                              size_t                     q_offset,
                                              size_t                     sliding_window) {
    const char            *name  = NKV_INT4 ? "int4" : "int8";
    const enum geist_dtype kv_dt = NKV_INT4 ? GEIST_DTYPE_U8 : GEIST_DTYPE_I8;
    size_t                 nq = 0, nk = 0, nv = 0, nks = 0, nvs = 0, no = 0;
    const float           *q   = nkv_view(q_t, GEIST_DTYPE_F32, sizeof(float), 3, &nq);
    const void            *k   = nkv_view(k_t, kv_dt, 1, 3, &nk);
    const void            *v   = nkv_view(v_t, kv_dt, 1, 3, &nv);
    const float           *ks  = nkv_view(k_scale_t, GEIST_DTYPE_F32, sizeof(float), 2, &nks);
    const float           *vs  = nkv_view(v_scale_t, GEIST_DTYPE_F32, sizeof(float), 2, &nvs);
    float                 *out = (float *) nkv_view(out_t, GEIST_DTYPE_F32, sizeof(float), 3, &no);
    if (q == nullptr || k == nullptr || v == nullptr || ks == nullptr || vs == nullptr ||
        out == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_neon attention_kv_%s: q, scales and out must be F32, k "
                                "and v %s, DENSE views of the rank given within their buffers",
                                name,
                                NKV_INT4 ? "U8" : "I8");
        return GEIST_E_INVALID_ARG;
    }
    const size_t                    n_q        = (size_t) q_t->shape[0];
    const size_t                    n_q_heads  = (size_t) q_t->shape[1];
    const size_t                    head_dim   = (size_t) q_t->shape[2];
    const size_t                    n_kv       = (size_t) k_t->shape[0];
    const size_t                    n_kv_heads = (size_t) k_t->shape[1];
    const struct geist_fusion_query shape      = {.op         = NKV_INT4 ? GEIST_FUSED_ATTN_KV_INT4
                                                                         : GEIST_FUSED_ATTN_KV_INT8,
                                                  .head_dim   = head_dim,
                                                  .n_q_heads  = n_q_heads,
                                                  .n_kv_heads = n_kv_heads};
    if (!(NKV_INT4 ? cpu_neon_attention_kv_int4_supported(&shape)
                   : cpu_neon_attention_kv_int8_supported(&shape))) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "cpu_neon attention_kv_%s: head_dim %zu, %zu query heads on %zu "
                                "KV heads",
                                name,
                                head_dim,
                                n_q_heads,
                                n_kv_heads);
        return GEIST_E_UNSUPPORTED;
    }
    /* Matching shapes (half a byte per INT4 K and V value), and every
     * query's position in the cache: n_q and n_kv are >= 1 (nkv_view
     * refuses an empty dimension), so the subtraction cannot wrap. */
    if ((size_t) k_t->shape[2] != (NKV_INT4 ? head_dim / 2 : head_dim) || nv != nk ||
        (size_t) v_t->shape[0] != n_kv || (size_t) v_t->shape[1] != n_kv_heads || no != nq ||
        (size_t) out_t->shape[0] != n_q || (size_t) out_t->shape[1] != n_q_heads ||
        nks != n_kv * n_kv_heads || nvs != n_kv * n_kv_heads ||
        (size_t) k_scale_t->shape[0] != n_kv || (size_t) v_scale_t->shape[0] != n_kv ||
        n_q > n_kv || q_offset > n_kv - n_q) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_neon attention_kv_%s: shapes do not match, or %zu "
                                "queries at %zu run past %zu cached positions",
                                name,
                                n_q,
                                q_offset,
                                n_kv);
        return GEIST_E_INVALID_ARG;
    }
    nkv_run(n_q,
            n_q_heads,
            head_dim,
            n_kv,
            n_kv_heads,
            q_offset,
            sliding_window,
            q,
            k,
            ks,
            v,
            vs,
            out);
    return GEIST_OK;
}

#endif /* GEIST_CPU_NEON_ATTENTION_KV_H */
