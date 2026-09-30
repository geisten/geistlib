/*
 * src/backends/cpu_x86/attention.c — AVX2/FMA causal MQA/GQA attention.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * cpu_x86 inherited attention_mqa_causal_kv (src/backends/common/
 * gemma4_kernels.c) verbatim through the `cpu_x86_vtbl = cpu_scalar_vtbl`
 * copy in backend.c — a plain scalar QK/PV dot product per thread, no SIMD.
 * cpu_neon has had a NEON version of this since #281 (transformer_ops.c);
 * this is the x86 equivalent (#499), same structure: an MQA fast path
 * (n_kv_heads == 1, the common case) plus a general GQA path, both
 * FMA-vectorized 8-wide (__m256) with 2-4 independent accumulators, OpenMP
 * over (q_pos, q_head) pairs. Falls back to attention_mqa_causal_kv for
 * whatever it does not cover (n_q * n_q_heads < 16). Like that reference,
 * the softmax runs online over stack-sized blocks of the context
 * (attn_row_avx2), so neither allocates.
 *
 * -march=x86-64-v3 (this backend's floor) guarantees AVX2 + FMA + F16C
 * unconditionally, so this file needs no runtime ISA dispatch (unlike the
 * quantized GEMM kernels, which vary support across x86-64-v3 hosts) —
 * see mk/backend-cpu_x86.mk.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "attention.h"

#include "gemma4_kernels.h"
#include "tensor_view.h"

#include <geist.h>
#include <geist_backend.h>

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* CPU buffer layout, owned by cpu_scalar's buffer_create (cpu_x86 inherits
 * its buffer vtable). Mirrored here as elementwise.c does — geist.h keeps
 * the struct opaque, but the host pointer is needed for the dense fast path. */
struct geist_buffer {
    void                  *host;
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
};

static float *get_f32_dense_ptr_full(const struct geist_tensor *t, size_t *out_n) {
    if (t == nullptr || t->buffer == nullptr) {
        return nullptr;
    }
    return geist_tensor_f32_dense(t, t->buffer->host, t->buffer->bytes, out_n);
}

static inline float hsum256(__m256 v) {
    __m128 lo   = _mm256_castps256_ps128(v);
    __m128 hi   = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4        = _mm_hadd_ps(sum4, sum4);
    sum4        = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

static inline float dot_f32_avx2(const float *a, const float *b, size_t n) {
    size_t i    = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    float out = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) {
        out += a[i] * b[i];
    }
    return out;
}

/* head_dim == 256, the qwen35/Bonsai full-attention shape: fully unrolled,
 * 4 accumulators (32-wide/iteration). */
static inline float dot_f32_avx2_256(const float *a, const float *b) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (size_t i = 0; i < 256; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    return hsum256(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
}

static inline void zero_f32_avx2(float *x, size_t n) {
    size_t       i = 0;
    const __m256 z = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(x + i, z);
    }
    for (; i < n; i++) {
        x[i] = 0.0f;
    }
}

static inline void zero_f32_avx2_256(float *x) {
    const __m256 z = _mm256_setzero_ps();
    for (size_t i = 0; i < 256; i += 8) {
        _mm256_storeu_ps(x + i, z);
    }
}

static inline void add_scaled_f32_avx2(float *dst, const float *src, float scale, size_t n) {
    size_t       i = 0;
    const __m256 s = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), s, _mm256_loadu_ps(dst + i)));
    }
    for (; i < n; i++) {
        dst[i] += scale * src[i];
    }
}

static inline void add_scaled_f32_avx2_256(float *dst, const float *src, float scale) {
    const __m256 s = _mm256_set1_ps(scale);
    for (size_t i = 0; i < 256; i += 8) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), s, _mm256_loadu_ps(dst + i)));
    }
}

static inline void scale_f32_avx2(float *x, float scale, size_t n) {
    size_t       i = 0;
    const __m256 s = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), s));
    }
    for (; i < n; i++) {
        x[i] *= scale;
    }
}

/* One (query, head): outv = softmax_s(qv . k[s]) v[s] over s in [s_lo, s_hi],
 * where k[s] = k + s * row and v[s] = v + s * row (the KV head applied, row
 * the floats between positions). The scores go a block at a time through
 * the stack (ATTN_F32_BLOCK); the running max and sum carry across blocks,
 * outv is accumulated unnormalized, rescaled when a block raises the max
 * and divided by the sum last. Nothing grows with the context, nothing is
 * allocated. */
static inline void attn_row_avx2(const float *qv,
                                 const float *k,
                                 const float *v,
                                 size_t       row,
                                 size_t       s_lo,
                                 size_t       s_hi,
                                 size_t       head_dim,
                                 bool         hd256,
                                 float       *outv) {
    float  scores[ATTN_F32_BLOCK];
    float  max_score = 0.0f;
    double sum_exp   = 0.0;
    if (hd256) {
        zero_f32_avx2_256(outv);
    } else {
        zero_f32_avx2(outv, head_dim);
    }
    for (size_t b0 = s_lo; b0 <= s_hi; b0 += ATTN_F32_BLOCK) {
        const size_t n         = s_hi - b0 < ATTN_F32_BLOCK ? s_hi - b0 + 1 : ATTN_F32_BLOCK;
        float        block_max = 0.0f;
        for (size_t j = 0; j < n; j++) {
            const float *kv = k + (b0 + j) * row;
            const float  sc = hd256 ? dot_f32_avx2_256(qv, kv) : dot_f32_avx2(qv, kv, head_dim);
            scores[j]       = sc;
            if (j == 0 || sc > block_max) {
                block_max = sc;
            }
        }
        if (b0 == s_lo) {
            max_score = block_max;
        } else if (block_max > max_score) {
            const float d = max_score - block_max;
            const float c = d < ATTN_EXP_FLOOR ? 0.0f : expf(d);
            sum_exp *= c;
            scale_f32_avx2(outv, c, head_dim);
            max_score = block_max;
        }
        for (size_t j = 0; j < n; j++) {
            /* The exponent clamped at ATTN_EXP_FLOOR (gemma4_kernels.h). */
            scores[j] = expf(fmaxf(scores[j] - max_score, ATTN_EXP_FLOOR));
            sum_exp += scores[j];
        }
        for (size_t j = 0; j < n; j++) {
            const float *vv = v + (b0 + j) * row;
            if (hd256) {
                add_scaled_f32_avx2_256(outv, vv, scores[j]);
            } else {
                add_scaled_f32_avx2(outv, vv, scores[j], head_dim);
            }
        }
    }
    if (sum_exp > 0.0) { /* else an empty window: outv stays zero */
        scale_f32_avx2(outv, (float) (1.0 / sum_exp), head_dim);
    }
}

/* MQA fast path: n_kv_heads == 1 (every q_head reads the same K/V row). */
static bool attention_mqa1_causal_kv_avx2(const float *q,
                                          const float *k,
                                          const float *v,
                                          size_t       n_q,
                                          size_t       n_kv,
                                          size_t       q_offset,
                                          size_t       n_q_heads,
                                          size_t       head_dim,
                                          size_t       sliding_window,
                                          float       *out) {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || n_q == 0 || n_kv == 0 ||
        n_q_heads == 0 || head_dim == 0) {
        return false;
    }
    const size_t total = n_q * n_q_heads;
    if (total < 16) {
        return false;
    }
    const bool hd256 = head_dim == 256;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (size_t idx = 0; idx < total; idx++) {
        const size_t t     = idx / n_q_heads;
        const size_t h     = idx - t * n_q_heads;
        const size_t q_pos = q_offset + t;
        const size_t s_lo =
                (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
        const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;
        attn_row_avx2(q + (t * n_q_heads + h) * head_dim,
                      k,
                      v,
                      head_dim,
                      s_lo,
                      s_hi,
                      head_dim,
                      hd256,
                      out + (t * n_q_heads + h) * head_dim);
    }
    return true;
}

/* General GQA path (n_kv_heads > 1, arbitrary group size). */
static bool attention_mqa_causal_kv_avx2(const float *q,
                                         const float *k,
                                         const float *v,
                                         size_t       n_q,
                                         size_t       n_kv,
                                         size_t       q_offset,
                                         size_t       n_q_heads,
                                         size_t       n_kv_heads,
                                         size_t       head_dim,
                                         size_t       sliding_window,
                                         float       *out) {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || n_q == 0 || n_kv == 0 ||
        n_q_heads == 0 || n_kv_heads == 0 || head_dim == 0 || n_q_heads % n_kv_heads != 0) {
        return false;
    }
    const size_t total = n_q * n_q_heads;
    if (total < 16) {
        return false;
    }
    const size_t kv_group_size = n_q_heads / n_kv_heads;
    const bool   hd256         = head_dim == 256;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (size_t idx = 0; idx < total; idx++) {
        const size_t t     = idx / n_q_heads;
        const size_t h     = idx - t * n_q_heads;
        const size_t q_pos = q_offset + t;
        const size_t s_lo =
                (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
        const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;
        const size_t kv_h = h / kv_group_size;
        attn_row_avx2(q + (t * n_q_heads + h) * head_dim,
                      k + kv_h * head_dim,
                      v + kv_h * head_dim,
                      n_kv_heads * head_dim,
                      s_lo,
                      s_hi,
                      head_dim,
                      hd256,
                      out + (t * n_q_heads + h) * head_dim);
    }
    return true;
}

[[nodiscard]] enum geist_status cpu_x86_attention(struct geist_backend      *be,
                                                  const struct geist_tensor *q,
                                                  const struct geist_tensor *k,
                                                  const struct geist_tensor *v,
                                                  size_t                     q_offset,
                                                  size_t                     sliding_window,
                                                  struct geist_tensor       *out) {
    if (be == nullptr || q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nq, nk, nv, no;
    const float *qp = get_f32_dense_ptr_full(q, &nq);
    const float *kp = get_f32_dense_ptr_full(k, &nk);
    const float *vp = get_f32_dense_ptr_full(v, &nv);
    float       *op = get_f32_dense_ptr_full(out, &no);
    if (qp == nullptr || kp == nullptr || vp == nullptr || op == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "cpu_x86 attention: tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || out->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 attention: bad ranks");
        return GEIST_E_INVALID_ARG;
    }
    size_t n_q        = (size_t) q->shape[0];
    size_t n_q_heads  = (size_t) q->shape[1];
    size_t head_dim   = (size_t) q->shape[2];
    size_t n_kv       = (size_t) k->shape[0];
    size_t n_kv_heads = (size_t) k->shape[1];
    if (k->shape[2] != (int64_t) head_dim || v->shape[2] != (int64_t) head_dim ||
        v->shape[0] != (int64_t) n_kv || v->shape[1] != (int64_t) n_kv_heads ||
        out->shape[0] != (int64_t) n_q || out->shape[1] != (int64_t) n_q_heads ||
        out->shape[2] != (int64_t) head_dim) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 attention: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (n_kv_heads == 1 &&
        attention_mqa1_causal_kv_avx2(
                qp, kp, vp, n_q, n_kv, q_offset, n_q_heads, head_dim, sliding_window, op)) {
        return GEIST_OK;
    }
    if (attention_mqa_causal_kv_avx2(qp,
                                     kp,
                                     vp,
                                     n_q,
                                     n_kv,
                                     q_offset,
                                     n_q_heads,
                                     n_kv_heads,
                                     head_dim,
                                     sliding_window,
                                     op)) {
        return GEIST_OK;
    }
    attention_mqa_causal_kv(
            n_q, n_kv, q_offset, n_q_heads, n_kv_heads, head_dim, sliding_window, qp, kp, vp, op);
    return GEIST_OK;
}
