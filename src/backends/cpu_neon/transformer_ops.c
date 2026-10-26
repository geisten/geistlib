/*
 * src/backends/cpu_neon/transformer_ops.c — rope_apply and
 * FP32 attention (NEON dot products).
 *
 * Layer: BACKEND.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"
#include "par.h"

#include "gemma4_kernels.h"
#include "host_buffer.h"
#include "quant.h"
#include "heap.h"
#include "checked.h"
#include "env.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

[[nodiscard]] enum geist_status cpu_neon_rope_apply(struct geist_backend      *be,
                                                    struct geist_tensor       *x,
                                                    const struct geist_tensor *cos,
                                                    const struct geist_tensor *sin) {
    if (be == nullptr || x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, nc = 0, ns = 0;
    float       *xp   = geist_host_tensor_f32(x, &nx);
    const float *cosp = geist_host_tensor_f32(cos, &nc);
    const float *sinp = geist_host_tensor_f32(sin, &ns);
    if (xp == nullptr || cosp == nullptr || sinp == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "cpu_neon rope_apply: tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (x->ndim != 3 || cos->ndim != 2 || sin->ndim != 2) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_neon rope_apply: bad ranks (x=%d cos=%d sin=%d)",
                                x->ndim,
                                cos->ndim,
                                sin->ndim);
        return GEIST_E_INVALID_ARG;
    }
    size_t seq_len  = (size_t) x->shape[0];
    size_t n_heads  = (size_t) x->shape[1];
    size_t head_dim = (size_t) x->shape[2];
    if (cos->shape[0] != (int64_t) seq_len || sin->shape[0] != (int64_t) seq_len) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_neon rope_apply: cos/sin seq_len mismatch");
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_rot = (size_t) cos->shape[cos->ndim - 1];
    if (!rope_apply_shape_ok(seq_len, head_dim, n_rot, nc, ns)) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_neon rope_apply: cos/sin do not cover x");
        return GEIST_E_INVALID_ARG;
    }
    rope_apply(seq_len, n_heads, head_dim, n_rot, xp, cosp, sinp);
    return GEIST_OK;
}

/* ---- attention (sliding-window MQA causal) ---- */

#if defined(__ARM_NEON)
static inline float dot_f32_neon(const float *a, const float *b, size_t n) {
    size_t      i    = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float out = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; i < n; i++) {
        out += a[i] * b[i];
    }
    return out;
}

static inline float dot_f32_neon_256(const float *a, const float *b) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < 256; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    return vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
}

static inline void zero_f32_neon(float *x, size_t n) {
    size_t            i = 0;
    const float32x4_t z = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(x + i, z);
        vst1q_f32(x + i + 4, z);
        vst1q_f32(x + i + 8, z);
        vst1q_f32(x + i + 12, z);
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(x + i, z);
    }
    for (; i < n; i++) {
        x[i] = 0.0f;
    }
}

static inline void add_scaled_f32_neon(float *dst, const float *src, float scale, size_t n) {
    size_t            i = 0;
    const float32x4_t s = vdupq_n_f32(scale);
    for (; i + 16 <= n; i += 16) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t d2 = vld1q_f32(dst + i + 8);
        float32x4_t d3 = vld1q_f32(dst + i + 12);
        d0             = vfmaq_f32(d0, vld1q_f32(src + i), s);
        d1             = vfmaq_f32(d1, vld1q_f32(src + i + 4), s);
        d2             = vfmaq_f32(d2, vld1q_f32(src + i + 8), s);
        d3             = vfmaq_f32(d3, vld1q_f32(src + i + 12), s);
        vst1q_f32(dst + i, d0);
        vst1q_f32(dst + i + 4, d1);
        vst1q_f32(dst + i + 8, d2);
        vst1q_f32(dst + i + 12, d3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vld1q_f32(dst + i);
        d             = vfmaq_f32(d, vld1q_f32(src + i), s);
        vst1q_f32(dst + i, d);
    }
    for (; i < n; i++) {
        dst[i] += scale * src[i];
    }
}

static inline void zero_f32_neon_256(float *x) {
    const float32x4_t z = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < 256; i += 16) {
        vst1q_f32(x + i, z);
        vst1q_f32(x + i + 4, z);
        vst1q_f32(x + i + 8, z);
        vst1q_f32(x + i + 12, z);
    }
}

static inline void add_scaled_f32_neon_256(float *dst, const float *src, float scale) {
    const float32x4_t s = vdupq_n_f32(scale);
    for (size_t i = 0; i < 256; i += 16) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t d2 = vld1q_f32(dst + i + 8);
        float32x4_t d3 = vld1q_f32(dst + i + 12);
        d0             = vfmaq_f32(d0, vld1q_f32(src + i), s);
        d1             = vfmaq_f32(d1, vld1q_f32(src + i + 4), s);
        d2             = vfmaq_f32(d2, vld1q_f32(src + i + 8), s);
        d3             = vfmaq_f32(d3, vld1q_f32(src + i + 12), s);
        vst1q_f32(dst + i, d0);
        vst1q_f32(dst + i + 4, d1);
        vst1q_f32(dst + i + 8, d2);
        vst1q_f32(dst + i + 12, d3);
    }
}

/* Runs fn over the `total` attention items (query, head), handed out 8 at
 * a time, or one at a time below 16 items: a decode step of a model with
 * 8 heads (Gemma 4 E2B) then spreads its rows over 8 threads instead of
 * one, where it used to fall back to the scalar reference (#16). */
static void attention_par(size_t total, geist_par_fn fn, void *ctx) {
    geist_par_for_dynamic(total, total < 16 ? 1 : 8, fn, ctx);
}

static inline void scale_f32_neon(float *x, float scale, size_t n) {
    size_t            i = 0;
    const float32x4_t s = vdupq_n_f32(scale);
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), s));
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
static inline void attn_row_neon(const float *qv,
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
        zero_f32_neon_256(outv);
    } else {
        zero_f32_neon(outv, head_dim);
    }
    for (size_t b0 = s_lo; b0 <= s_hi; b0 += ATTN_F32_BLOCK) {
        const size_t n         = s_hi - b0 < ATTN_F32_BLOCK ? s_hi - b0 + 1 : ATTN_F32_BLOCK;
        float        block_max = 0.0f;
        for (size_t j = 0; j < n; j++) {
            const float *kv = k + (b0 + j) * row;
            const float  sc = hd256 ? dot_f32_neon_256(qv, kv) : dot_f32_neon(qv, kv, head_dim);
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
            scale_f32_neon(outv, c, head_dim);
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
                add_scaled_f32_neon_256(outv, vv, scores[j]);
            } else {
                add_scaled_f32_neon(outv, vv, scores[j], head_dim);
            }
        }
    }
    if (sum_exp > 0.0) { /* else an empty window: outv stays zero */
        scale_f32_neon(outv, (float) (1.0 / sum_exp), head_dim);
    }
}

/* One FP32 attention call, for the item bodies below. */
struct attn_job {
    const float *q, *k, *v;
    size_t       n_kv, q_offset, n_q_heads, n_kv_heads, head_dim, sliding_window;
    bool         hd256;
    float       *out;
};

/* Items [i0, i1) of attention_mqa1_causal_kv_neon, item idx the query
 * position idx / n_q_heads and head idx % n_q_heads. */
static void attn_mqa1_items(void *ctx, size_t i0, size_t i1) {
    const struct attn_job *job            = ctx;
    const float           *q              = job->q;
    const float           *k              = job->k;
    const float           *v              = job->v;
    const size_t           n_kv           = job->n_kv;
    const size_t           q_offset       = job->q_offset;
    const size_t           n_q_heads      = job->n_q_heads;
    const size_t           head_dim       = job->head_dim;
    const size_t           sliding_window = job->sliding_window;
    const bool             hd256          = job->hd256;
    float                 *out            = job->out;
    for (size_t idx = i0; idx < i1; idx++) {
        const size_t t     = idx / n_q_heads;
        const size_t h     = idx - t * n_q_heads;
        const size_t q_pos = q_offset + t;
        const size_t s_lo =
                (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
        const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;
        attn_row_neon(q + (t * n_q_heads + h) * head_dim,
                      k,
                      v,
                      head_dim,
                      s_lo,
                      s_hi,
                      head_dim,
                      hd256,
                      out + (t * n_q_heads + h) * head_dim);
    }
}

/* Items [i0, i1) of attention_mqa_causal_kv_neon, as attn_mqa1_items. */
static void attn_mqa_items(void *ctx, size_t i0, size_t i1) {
    const struct attn_job *job            = ctx;
    const float           *q              = job->q;
    const float           *k              = job->k;
    const float           *v              = job->v;
    const size_t           n_kv           = job->n_kv;
    const size_t           q_offset       = job->q_offset;
    const size_t           n_q_heads      = job->n_q_heads;
    const size_t           n_kv_heads     = job->n_kv_heads;
    const size_t           head_dim       = job->head_dim;
    const size_t           sliding_window = job->sliding_window;
    float                 *out            = job->out;
    const size_t           kv_group_size  = n_q_heads / n_kv_heads;
    for (size_t idx = i0; idx < i1; idx++) {
        const size_t t     = idx / n_q_heads;
        const size_t h     = idx - t * n_q_heads;
        const size_t q_pos = q_offset + t;
        const size_t s_lo =
                (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
        const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;
        const size_t kv_h = h / kv_group_size;
        attn_row_neon(q + (t * n_q_heads + h) * head_dim,
                      k + kv_h * head_dim,
                      v + kv_h * head_dim,
                      n_kv_heads * head_dim,
                      s_lo,
                      s_hi,
                      head_dim,
                      false, /* general dot, also at head_dim 256 */
                      out + (t * n_q_heads + h) * head_dim);
    }
}

static bool attention_mqa1_causal_kv_neon(const float *q,
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
    const bool   hd256 = head_dim == 256;

    struct attn_job job = {.q              = q,
                           .k              = k,
                           .v              = v,
                           .n_kv           = n_kv,
                           .q_offset       = q_offset,
                           .n_q_heads      = n_q_heads,
                           .head_dim       = head_dim,
                           .sliding_window = sliding_window,
                           .hd256          = hd256,
                           .out            = out};
    attention_par(total, attn_mqa1_items, &job);
    return true;
}

static bool attention_mqa_causal_kv_neon(const float *q,
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
    const size_t    total = n_q * n_q_heads;
    struct attn_job job   = {.q              = q,
                             .k              = k,
                             .v              = v,
                             .n_kv           = n_kv,
                             .q_offset       = q_offset,
                             .n_q_heads      = n_q_heads,
                             .n_kv_heads     = n_kv_heads,
                             .head_dim       = head_dim,
                             .sliding_window = sliding_window,
                             .out            = out};
    attention_par(total, attn_mqa_items, &job);
    return true;
}
#endif

#if defined(__ARM_NEON)
/* ---- Query heads that share a KV head, worked together (#16) ----------
 * The rows above read every K and V row once per query head. A grouped-
 * query model's heads of one KV head and one position share both the rows
 * and the key range, so up to ATTN_HEADS_MAX of them now go together: each
 * K row is loaded once for all of their dots, and each V row once for all
 * of their output updates, which also stay in registers across a block of
 * keys instead of going through outv per key.
 *
 * Per head the arithmetic is attn_row_neon's, in its order: the same dot
 * (the hd256 shape where that row uses it), the same block max, rescale
 * and exponentials, and every output element takes the same fused
 * multiply-adds over the keys. The bytes do not change. */
enum { ATTN_HEADS_MAX = 4 };

/* qv[h] . kv for g heads, dot_f32_neon_256's or dot_f32_neon's shape per
 * head (head_dim a multiple of 16). */
static inline __attribute__((always_inline)) void attn_heads_dots(size_t             g,
                                                                  const float *const qv[],
                                                                  const float       *kv,
                                                                  size_t             head_dim,
                                                                  bool               hd256,
                                                                  float              out[]) {
    if (hd256) {
        float32x4_t a0[ATTN_HEADS_MAX], a1[ATTN_HEADS_MAX], a2[ATTN_HEADS_MAX], a3[ATTN_HEADS_MAX];
        for (size_t h = 0; h < g; h++) {
            a0[h] = a1[h] = a2[h] = a3[h] = vdupq_n_f32(0.0f);
        }
        for (size_t i = 0; i < 256; i += 16) {
            const float32x4_t k0 = vld1q_f32(kv + i);
            const float32x4_t k1 = vld1q_f32(kv + i + 4);
            const float32x4_t k2 = vld1q_f32(kv + i + 8);
            const float32x4_t k3 = vld1q_f32(kv + i + 12);
            for (size_t h = 0; h < g; h++) {
                a0[h] = vfmaq_f32(a0[h], vld1q_f32(qv[h] + i), k0);
                a1[h] = vfmaq_f32(a1[h], vld1q_f32(qv[h] + i + 4), k1);
                a2[h] = vfmaq_f32(a2[h], vld1q_f32(qv[h] + i + 8), k2);
                a3[h] = vfmaq_f32(a3[h], vld1q_f32(qv[h] + i + 12), k3);
            }
        }
        for (size_t h = 0; h < g; h++) {
            out[h] = vaddvq_f32(a0[h]) + vaddvq_f32(a1[h]) + vaddvq_f32(a2[h]) + vaddvq_f32(a3[h]);
        }
        return;
    }
    float32x4_t a0[ATTN_HEADS_MAX], a1[ATTN_HEADS_MAX];
    for (size_t h = 0; h < g; h++) {
        a0[h] = a1[h] = vdupq_n_f32(0.0f);
    }
    for (size_t i = 0; i < head_dim; i += 8) {
        const float32x4_t k0 = vld1q_f32(kv + i);
        const float32x4_t k1 = vld1q_f32(kv + i + 4);
        for (size_t h = 0; h < g; h++) {
            a0[h] = vfmaq_f32(a0[h], vld1q_f32(qv[h] + i), k0);
            a1[h] = vfmaq_f32(a1[h], vld1q_f32(qv[h] + i + 4), k1);
        }
    }
    for (size_t h = 0; h < g; h++) {
        out[h] = vaddvq_f32(a0[h]) + vaddvq_f32(a1[h]);
    }
}

/* attn_row_neon for g <= ATTN_HEADS_MAX heads over one K/V head and one
 * key range [s_lo, s_hi]; inlined per g, so the head loops unroll. */
static inline __attribute__((always_inline)) void attn_heads_neon(size_t             g,
                                                                  const float *const qv[],
                                                                  const float       *k,
                                                                  const float       *v,
                                                                  size_t             row,
                                                                  size_t             s_lo,
                                                                  size_t             s_hi,
                                                                  size_t             head_dim,
                                                                  bool               hd256,
                                                                  float *const       outv[],
                                                                  float             *part_max,
                                                                  double            *part_sum) {
    float  scores[ATTN_HEADS_MAX][ATTN_F32_BLOCK];
    float  max_score[ATTN_HEADS_MAX] = {};
    double sum_exp[ATTN_HEADS_MAX]   = {};
    for (size_t h = 0; h < g; h++) {
        zero_f32_neon(outv[h], head_dim);
    }
    for (size_t b0 = s_lo; b0 <= s_hi; b0 += ATTN_F32_BLOCK) {
        const size_t n = s_hi - b0 < ATTN_F32_BLOCK ? s_hi - b0 + 1 : ATTN_F32_BLOCK;
        float        block_max[ATTN_HEADS_MAX] = {};
        for (size_t j = 0; j < n; j++) {
            float sc[ATTN_HEADS_MAX];
            attn_heads_dots(g, qv, k + (b0 + j) * row, head_dim, hd256, sc);
            for (size_t h = 0; h < g; h++) {
                scores[h][j] = sc[h];
                if (j == 0 || sc[h] > block_max[h]) {
                    block_max[h] = sc[h];
                }
            }
        }
        for (size_t h = 0; h < g; h++) {
            if (b0 == s_lo) {
                max_score[h] = block_max[h];
            } else if (block_max[h] > max_score[h]) {
                const float d = max_score[h] - block_max[h];
                const float c = d < ATTN_EXP_FLOOR ? 0.0f : expf(d);
                sum_exp[h] *= c;
                scale_f32_neon(outv[h], c, head_dim);
                max_score[h] = block_max[h];
            }
            for (size_t j = 0; j < n; j++) {
                scores[h][j] = expf(fmaxf(scores[h][j] - max_score[h], ATTN_EXP_FLOOR));
                sum_exp[h] += scores[h][j];
            }
        }
        /* outv += scores[j] * v[j] over the block, 16 elements of every
         * head at a time held in registers; per element the same FMAs in
         * the same key order as add_scaled_f32_neon. */
        for (size_t c = 0; c < head_dim; c += 16) {
            float32x4_t d[ATTN_HEADS_MAX][4];
            for (size_t h = 0; h < g; h++) {
                for (size_t e = 0; e < 4; e++) {
                    d[h][e] = vld1q_f32(outv[h] + c + 4 * e);
                }
            }
            for (size_t j = 0; j < n; j++) {
                const float      *vv = v + (b0 + j) * row + c;
                const float32x4_t v0 = vld1q_f32(vv);
                const float32x4_t v1 = vld1q_f32(vv + 4);
                const float32x4_t v2 = vld1q_f32(vv + 8);
                const float32x4_t v3 = vld1q_f32(vv + 12);
                for (size_t h = 0; h < g; h++) {
                    const float32x4_t s = vdupq_n_f32(scores[h][j]);
                    d[h][0]             = vfmaq_f32(d[h][0], v0, s);
                    d[h][1]             = vfmaq_f32(d[h][1], v1, s);
                    d[h][2]             = vfmaq_f32(d[h][2], v2, s);
                    d[h][3]             = vfmaq_f32(d[h][3], v3, s);
                }
            }
            for (size_t h = 0; h < g; h++) {
                for (size_t e = 0; e < 4; e++) {
                    vst1q_f32(outv[h] + c + 4 * e, d[h][e]);
                }
            }
        }
    }
    if (part_sum != nullptr) { /* a split range's part: left unnormalized */
        for (size_t h = 0; h < g; h++) {
            part_max[h] = max_score[h];
            part_sum[h] = sum_exp[h];
        }
        return;
    }
    for (size_t h = 0; h < g; h++) {
        if (sum_exp[h] > 0.0) { /* else an empty window: outv stays zero */
            scale_f32_neon(outv[h], (float) (1.0 / sum_exp[h]), head_dim);
        }
    }
}

/* Items [i0, i1) of attention_heads_grouped_neon: item idx is query
 * position t, KV head kv_h and the chunk of up to ATTN_HEADS_MAX of that
 * KV head's query heads, t-major. */
static void attn_group_items(void *ctx, size_t i0, size_t i1) {
    const struct attn_job *job      = ctx;
    const size_t           group    = job->n_q_heads / job->n_kv_heads;
    const size_t           chunks   = (group + ATTN_HEADS_MAX - 1) / ATTN_HEADS_MAX;
    const size_t           per_t    = job->n_kv_heads * chunks;
    const size_t           head_dim = job->head_dim;
    const size_t           row      = job->n_kv_heads * head_dim;
    for (size_t idx = i0; idx < i1; idx++) {
        const size_t t     = idx / per_t;
        const size_t kv_h  = (idx - t * per_t) / chunks;
        const size_t ch    = idx - t * per_t - kv_h * chunks;
        const size_t h0    = kv_h * group + ch * ATTN_HEADS_MAX;
        const size_t left  = group - ch * ATTN_HEADS_MAX;
        const size_t g     = left < ATTN_HEADS_MAX ? left : ATTN_HEADS_MAX;
        const size_t q_pos = job->q_offset + t;
        const size_t s_lo  = (job->sliding_window > 0 && q_pos + 1 > job->sliding_window)
                                     ? q_pos + 1 - job->sliding_window
                                     : 0;
        const size_t s_hi  = q_pos < job->n_kv ? q_pos : job->n_kv - 1;
        const float *qv[ATTN_HEADS_MAX]   = {};
        float       *outv[ATTN_HEADS_MAX] = {};
        for (size_t h = 0; h < g; h++) {
            qv[h]   = job->q + (t * job->n_q_heads + h0 + h) * head_dim;
            outv[h] = job->out + (t * job->n_q_heads + h0 + h) * head_dim;
        }
        const float *k = job->k + kv_h * head_dim;
        const float *v = job->v + kv_h * head_dim;
        switch (g) {
        case 1:
            attn_heads_neon(
                    1, qv, k, v, row, s_lo, s_hi, head_dim, job->hd256, outv, nullptr, nullptr);
            break;
        case 2:
            attn_heads_neon(
                    2, qv, k, v, row, s_lo, s_hi, head_dim, job->hd256, outv, nullptr, nullptr);
            break;
        case 3:
            attn_heads_neon(
                    3, qv, k, v, row, s_lo, s_hi, head_dim, job->hd256, outv, nullptr, nullptr);
            break;
        default:
            attn_heads_neon(
                    4, qv, k, v, row, s_lo, s_hi, head_dim, job->hd256, outv, nullptr, nullptr);
            break;
        }
    }
}

/* ---- A decode step's key range split across threads (#16) -------------
 * A decode step has one grouped item per KV head (and head chunk): Llama
 * 3.2 3B has 8 of them for 7 decode threads, so at depth one thread runs
 * two while six wait, and the region is ~0.57 busy; Gemma 4 E2B has 2.
 * With at most ATTN_SPLIT_MAX_ITEMS items and a key range of at least
 * 2 * ATTN_SPLIT_CHUNK keys, each item's range is cut at the multiples of
 * the part length (flash-decoding): every part runs attn_heads_neon on its
 * keys and leaves its output unnormalized with its max and sum, and
 * attn_split_merge rescales the parts to their common max, adds them and
 * divides by the total in double, rounding once. The cut points depend on
 * the key positions only, not on the thread count, so a step computes the
 * same bits on any number of threads. Each part's arithmetic is the row
 * kernel's; the merge adds the parts instead of carrying one running sum
 * through all keys, so the result differs from the unsplit one in rounding
 * only (in a model, by about as much as one ulp on the attention output
 * does). Short ranges and prefill keep the unsplit items, bit for bit.
 *
 * Opt-in (GEIST_ATTN_SPLIT_DECODE=1), off by default: decode at depth 1024
 * is ~9-11 % faster with it on the M1 Max, but cpu_neon's int8 decode
 * activations turn its rounding-sized difference into a quantization-sized
 * one, so greedy output departs from the default's at near-ties (#178). */
enum {
    ATTN_SPLIT_CHUNK        = 128,
    ATTN_SPLIT_MAX_PARTS    = 16,
    ATTN_SPLIT_MAX_ITEMS    = 16,
    ATTN_SPLIT_MAX_HEAD_DIM = 512,
};

/* The split is opt-in: GEIST_ATTN_SPLIT_DECODE=1 (docs/BACKENDS.md). Read
 * once. Off, decode attention is the unsplit kernel, bit for bit. */
static bool attn_split_decode_on(void) {
    static _Atomic int on = -1;
    int                v  = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        v = geist_env_flag("GEIST_ATTN_SPLIT_DECODE", false);
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v != 0;
}

/* One split call: the grouped job, the parts per item, where they start
 * (base, a multiple of len) and their scratch, one [head_dim] output and
 * one max and sum per (query, head, part). Part p of every item covers
 * keys [base + p * len, base + (p + 1) * len) within the item's range. */
struct attn_split_job {
    struct attn_job job;
    size_t          parts, base, len;
    float          *part_out;
    float          *part_max;
    double         *part_sum;
};

/* Items [i0, i1) of the split: item idx is attn_group_items' item
 * idx / parts and part idx % parts of its key range. */
static void attn_split_items(void *ctx, size_t i0, size_t i1) {
    const struct attn_split_job *sj       = ctx;
    const struct attn_job       *job      = &sj->job;
    const size_t                 parts    = sj->parts;
    const size_t                 group    = job->n_q_heads / job->n_kv_heads;
    const size_t                 chunks   = (group + ATTN_HEADS_MAX - 1) / ATTN_HEADS_MAX;
    const size_t                 per_t    = job->n_kv_heads * chunks;
    const size_t                 head_dim = job->head_dim;
    const size_t                 row      = job->n_kv_heads * head_dim;
    for (size_t idx = i0; idx < i1; idx++) {
        const size_t it    = idx / parts;
        const size_t p     = idx - it * parts;
        const size_t t     = it / per_t;
        const size_t kv_h  = (it - t * per_t) / chunks;
        const size_t ch    = it - t * per_t - kv_h * chunks;
        const size_t h0    = kv_h * group + ch * ATTN_HEADS_MAX;
        const size_t left  = group - ch * ATTN_HEADS_MAX;
        const size_t g     = left < ATTN_HEADS_MAX ? left : ATTN_HEADS_MAX;
        const size_t q_pos = job->q_offset + t;
        const size_t s_lo  = (job->sliding_window > 0 && q_pos + 1 > job->sliding_window)
                                     ? q_pos + 1 - job->sliding_window
                                     : 0;
        const size_t s_hi  = q_pos < job->n_kv ? q_pos : job->n_kv - 1;
        const size_t c0 = sj->base + p * sj->len, c1 = c0 + sj->len;
        const size_t a     = c0 > s_lo ? c0 : s_lo; /* this part: keys [a, b) */
        const size_t b     = c1 < s_hi + 1 ? c1 : s_hi + 1;
        const size_t slot0 = (t * job->n_q_heads + h0) * parts + p; /* head h: + h * parts */
        if (a >= b) { /* the part lies outside this query's range */
            for (size_t h = 0; h < g; h++) {
                sj->part_sum[slot0 + h * parts] = 0.0;
                sj->part_max[slot0 + h * parts] = 0.0f;
            }
            continue;
        }
        const float *qv[ATTN_HEADS_MAX]   = {};
        float       *outv[ATTN_HEADS_MAX] = {};
        float        pm[ATTN_HEADS_MAX]   = {};
        double       ps[ATTN_HEADS_MAX]   = {};
        for (size_t h = 0; h < g; h++) {
            qv[h]   = job->q + (t * job->n_q_heads + h0 + h) * head_dim;
            outv[h] = sj->part_out + (slot0 + h * parts) * head_dim;
        }
        const float *k = job->k + kv_h * head_dim;
        const float *v = job->v + kv_h * head_dim;
        switch (g) {
        case 1:
            attn_heads_neon(1, qv, k, v, row, a, b - 1, head_dim, job->hd256, outv, pm, ps);
            break;
        case 2:
            attn_heads_neon(2, qv, k, v, row, a, b - 1, head_dim, job->hd256, outv, pm, ps);
            break;
        case 3:
            attn_heads_neon(3, qv, k, v, row, a, b - 1, head_dim, job->hd256, outv, pm, ps);
            break;
        default:
            attn_heads_neon(4, qv, k, v, row, a, b - 1, head_dim, job->hd256, outv, pm, ps);
            break;
        }
        for (size_t h = 0; h < g; h++) {
            sj->part_max[slot0 + h * parts] = pm[h];
            sj->part_sum[slot0 + h * parts] = ps[h];
        }
    }
}

/* out[r] = sum_p c_p * part_out[r][p] / sum_p c_p * part_sum[r][p] for
 * every (query, head) r, c_p = exp(part_max - the parts' max), floored as
 * the row kernel's rescale is; accumulated in double and rounded once.
 * Parts without keys (sum 0) drop out. */
static void attn_split_merge(const struct attn_split_job *sj, size_t n_rows) {
    const size_t parts    = sj->parts;
    const size_t head_dim = sj->job.head_dim;
    for (size_t r = 0; r < n_rows; r++) {
        const float  *pm   = sj->part_max + r * parts;
        const double *ps   = sj->part_sum + r * parts;
        const float  *po   = sj->part_out + r * parts * head_dim;
        float        *outv = sj->job.out + r * head_dim;
        bool          any  = false;
        float         m    = 0.0f;
        for (size_t p = 0; p < parts; p++) {
            if (ps[p] > 0.0 && (!any || pm[p] > m)) {
                m   = pm[p];
                any = true;
            }
        }
        double acc[ATTN_SPLIT_MAX_HEAD_DIM];
        for (size_t i = 0; i < head_dim; i++) {
            acc[i] = 0.0;
        }
        double sum = 0.0;
        for (size_t p = 0; p < parts; p++) {
            if (!(ps[p] > 0.0)) {
                continue;
            }
            const double d = (double) pm[p] - (double) m;
            const double c = d < (double) ATTN_EXP_FLOOR ? 0.0 : exp(d);
            sum += c * ps[p];
            const float *o = po + p * head_dim;
            for (size_t i = 0; i < head_dim; i++) {
                acc[i] += c * (double) o[i];
            }
        }
        const double inv = sum > 0.0 ? 1.0 / sum : 0.0; /* 0: an empty window, out zero */
        for (size_t i = 0; i < head_dim; i++) {
            outv[i] = (float) (acc[i] * inv);
        }
    }
}

/* The grouped rows where they apply: at least two query heads per KV head,
 * head_dim a multiple of 16, and enough items to give every thread one (a
 * Gemma 4 E2B decode step has one KV head for eight query heads, two items;
 * its eight single-head rows spread wider). hd256 as the caller's row
 * kernel decides it. False: the caller's per-head rows. */
static bool attention_heads_grouped_neon(const float               *q,
                                         const float               *k,
                                         const float               *v,
                                         size_t                     n_q,
                                         size_t                     n_kv,
                                         size_t                     q_offset,
                                         size_t                     n_q_heads,
                                         size_t                     n_kv_heads,
                                         size_t                     head_dim,
                                         size_t                     sliding_window,
                                         bool                       hd256,
                                         float                     *out,
                                         struct cpu_neon_workspace *ws) {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || n_q == 0 || n_kv == 0 ||
        n_kv_heads == 0 || n_q_heads % n_kv_heads != 0 || head_dim == 0 || head_dim % 16 != 0) {
        return false;
    }
    const size_t group = n_q_heads / n_kv_heads;
    const size_t items = n_q * n_kv_heads * ((group + ATTN_HEADS_MAX - 1) / ATTN_HEADS_MAX);
    if (group < 2) {
        return false;
    }
    /* Parts cut at multiples of len from base: base is the chunk holding
     * the first query's first key, the parts reach the last query's last
     * key; len doubles until at most ATTN_SPLIT_MAX_PARTS parts do. */
    const size_t lo0   = (sliding_window > 0 && q_offset + 1 > sliding_window)
                                 ? q_offset + 1 - sliding_window
                                 : 0;
    const size_t q_end = q_offset + n_q - 1;
    const size_t hi1   = q_end < n_kv ? q_end : n_kv - 1;
    const size_t hi0   = q_offset < n_kv ? q_offset : n_kv - 1;
    size_t       len = ATTN_SPLIT_CHUNK, base = 0, parts = 1;
    if (items <= ATTN_SPLIT_MAX_ITEMS && head_dim <= ATTN_SPLIT_MAX_HEAD_DIM && hi0 >= lo0 &&
        attn_split_decode_on() && hi0 - lo0 + 1 >= 2 * ATTN_SPLIT_CHUNK) {
        for (;;) {
            base  = lo0 / len * len;
            parts = (hi1 - base) / len + 1;
            if (parts <= ATTN_SPLIT_MAX_PARTS) {
                break;
            }
            len *= 2;
        }
    }
    size_t n_rows = 0, n_slots = 0, n_out = 0;
    if (parts > 1 && ws != nullptr && !ckd_mul(&n_rows, n_q, n_q_heads) &&
        !ckd_mul(&n_slots, n_rows, parts) && !ckd_mul(&n_out, n_slots, head_dim)) {
        if (heap_scratch_grow_array(ws->attn_part_out, ws->attn_part_out_cap, n_out) &&
            heap_scratch_grow_array(ws->attn_part_max, ws->attn_part_max_cap, n_slots) &&
            heap_scratch_grow_array(ws->attn_part_sum, ws->attn_part_sum_cap, n_slots)) {
            struct attn_split_job sj = {.job      = {.q              = q,
                                                     .k              = k,
                                                     .v              = v,
                                                     .n_kv           = n_kv,
                                                     .q_offset       = q_offset,
                                                     .n_q_heads      = n_q_heads,
                                                     .n_kv_heads     = n_kv_heads,
                                                     .head_dim       = head_dim,
                                                     .sliding_window = sliding_window,
                                                     .hd256          = hd256,
                                                     .out            = out},
                                        .parts    = parts,
                                        .base     = base,
                                        .len      = len,
                                        .part_out = ws->attn_part_out,
                                        .part_max = ws->attn_part_max,
                                        .part_sum = ws->attn_part_sum};
            geist_par_for_dynamic(items * parts, 1, attn_split_items, &sj);
            attn_split_merge(&sj, n_rows);
            return true;
        }
    }
    if (items < geist_par_max_threads()) {
        return false;
    }
    struct attn_job job = {.q              = q,
                           .k              = k,
                           .v              = v,
                           .n_kv           = n_kv,
                           .q_offset       = q_offset,
                           .n_q_heads      = n_q_heads,
                           .n_kv_heads     = n_kv_heads,
                           .head_dim       = head_dim,
                           .sliding_window = sliding_window,
                           .hd256          = hd256,
                           .out            = out};
    geist_par_for_dynamic(items, 1, attn_group_items, &job);
    return true;
}
#endif

#if defined(__ARM_NEON)
/* ---- The same row kernel over an F16 KV cache (#48): K and V rows are
 * IEEE half, widened eight at a time by vcvt as they are loaded; the
 * arithmetic is attn_row_neon's. */
static inline float dot_f16_neon(const float *a, const uint16_t *b, size_t n) {
    size_t      i    = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(b + i));
        acc0                = vfmaq_f32(acc0, vld1q_f32(a + i), vcvt_f32_f16(vget_low_f16(h)));
        acc1                = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vcvt_high_f32_f16(h));
    }
    float out = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; i < n; i++) {
        out += a[i] * fp16_to_fp32(b[i]);
    }
    return out;
}

static inline void add_scaled_f16_neon(float *dst, const uint16_t *src, float scale, size_t n) {
    size_t            i = 0;
    const float32x4_t s = vdupq_n_f32(scale);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src + i));
        vst1q_f32(dst + i, vfmaq_f32(vld1q_f32(dst + i), vcvt_f32_f16(vget_low_f16(h)), s));
        vst1q_f32(dst + i + 4, vfmaq_f32(vld1q_f32(dst + i + 4), vcvt_high_f32_f16(h), s));
    }
    for (; i < n; i++) {
        dst[i] += scale * fp16_to_fp32(src[i]);
    }
}

static inline void attn_row_neon_f16(const float    *qv,
                                     const uint16_t *k,
                                     const uint16_t *v,
                                     size_t          row,
                                     size_t          s_lo,
                                     size_t          s_hi,
                                     size_t          head_dim,
                                     float          *outv) {
    float  scores[ATTN_F32_BLOCK];
    float  max_score = 0.0f;
    double sum_exp   = 0.0;
    zero_f32_neon(outv, head_dim);
    for (size_t b0 = s_lo; b0 <= s_hi; b0 += ATTN_F32_BLOCK) {
        const size_t n         = s_hi - b0 < ATTN_F32_BLOCK ? s_hi - b0 + 1 : ATTN_F32_BLOCK;
        float        block_max = 0.0f;
        for (size_t j = 0; j < n; j++) {
            const float sc = dot_f16_neon(qv, k + (b0 + j) * row, head_dim);
            scores[j]      = sc;
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
            scale_f32_neon(outv, c, head_dim);
            max_score = block_max;
        }
        for (size_t j = 0; j < n; j++) {
            scores[j] = expf(fmaxf(scores[j] - max_score, ATTN_EXP_FLOOR));
            sum_exp += scores[j];
        }
        for (size_t j = 0; j < n; j++) {
            add_scaled_f16_neon(outv, v + (b0 + j) * row, scores[j], head_dim);
        }
    }
    if (sum_exp > 0.0) {
        scale_f32_neon(outv, (float) (1.0 / sum_exp), head_dim);
    }
}

struct attn_job_f16 {
    const float    *q;
    const uint16_t *k, *v;
    size_t          n_kv, q_offset, n_q_heads, n_kv_heads, head_dim, sliding_window;
    float          *out;
};

static void attn_f16_items(void *ctx, size_t i0, size_t i1) {
    const struct attn_job_f16 j     = *(const struct attn_job_f16 *) ctx;
    const size_t              group = j.n_q_heads / j.n_kv_heads;
    for (size_t idx = i0; idx < i1; idx++) {
        const size_t t     = idx / j.n_q_heads;
        const size_t h     = idx - t * j.n_q_heads;
        const size_t q_pos = j.q_offset + t;
        const size_t s_lo  = (j.sliding_window > 0 && q_pos + 1 > j.sliding_window)
                                     ? q_pos + 1 - j.sliding_window
                                     : 0;
        const size_t s_hi  = q_pos < j.n_kv ? q_pos : j.n_kv - 1;
        const size_t kv_h  = h / group;
        attn_row_neon_f16(j.q + (t * j.n_q_heads + h) * j.head_dim,
                          j.k + kv_h * j.head_dim,
                          j.v + kv_h * j.head_dim,
                          j.n_kv_heads * j.head_dim,
                          s_lo,
                          s_hi,
                          j.head_dim,
                          j.out + (t * j.n_q_heads + h) * j.head_dim);
    }
}
#endif

/* Attention over an F16 KV cache (#48): the NEON rows from 16 (query,
 * head) pairs, the shared reference below. */
[[nodiscard]] static enum geist_status cpu_neon_attention_f16(struct geist_backend      *be,
                                                              const struct geist_tensor *q,
                                                              const struct geist_tensor *k,
                                                              const struct geist_tensor *v,
                                                              size_t                     q_offset,
                                                              size_t               sliding_window,
                                                              struct geist_tensor *out) {
    size_t          nq = 0, nk = 0, nv = 0, no = 0;
    const float    *qp = geist_host_tensor_f32(q, &nq);
    const uint16_t *kp = geist_host_tensor_f16_3d(k, &nk);
    const uint16_t *vp = geist_host_tensor_f16_3d(v, &nv);
    float          *op = geist_host_tensor_f32(out, &no);
    if (qp == nullptr || kp == nullptr || vp == nullptr || op == nullptr || q->ndim != 3 ||
        out->ndim != 3 || k->shape[1] != v->shape[1] || k->shape[0] != v->shape[0] ||
        k->shape[2] != q->shape[2] || v->shape[2] != q->shape[2] ||
        q->shape[1] % k->shape[1] != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon attention: bad F16 KV views");
        return GEIST_E_INVALID_ARG;
    }
#if defined(__ARM_NEON)
    /* 16+ (query, head) pairs: the NEON rows, as for the FP32 cache. */
    if ((size_t) q->shape[0] * (size_t) q->shape[1] >= 16) {
        struct attn_job_f16 job = {.q              = qp,
                                   .k              = kp,
                                   .v              = vp,
                                   .n_kv           = (size_t) k->shape[0],
                                   .q_offset       = q_offset,
                                   .n_q_heads      = (size_t) q->shape[1],
                                   .n_kv_heads     = (size_t) k->shape[1],
                                   .head_dim       = (size_t) q->shape[2],
                                   .sliding_window = sliding_window,
                                   .out            = op};
        attention_par((size_t) q->shape[0] * (size_t) q->shape[1], attn_f16_items, &job);
        return GEIST_OK;
    }
#endif
    if (!attention_mqa_causal_kv_f16((size_t) q->shape[0],
                                     (size_t) k->shape[0],
                                     q_offset,
                                     (size_t) q->shape[1],
                                     (size_t) k->shape[1],
                                     (size_t) q->shape[2],
                                     sliding_window,
                                     qp,
                                     kp,
                                     vp,
                                     op)) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "cpu_neon attention: F16 head_dim too large");
        return GEIST_E_UNSUPPORTED;
    }
    return GEIST_OK;
}

enum geist_status cpu_neon_kv_append_f16(struct geist_backend      *be,
                                         const struct geist_tensor *k_src,
                                         const struct geist_tensor *v_src,
                                         size_t                     q_position,
                                         struct geist_tensor       *k_cache,
                                         struct geist_tensor       *v_cache) {
    size_t       nks = 0, nvs = 0, nkc = 0, nvc = 0;
    const float *ks = geist_host_tensor_f32(k_src, &nks);
    const float *vs = geist_host_tensor_f32(v_src, &nvs);
    uint16_t    *kc = geist_host_tensor_f16_3d(k_cache, &nkc);
    uint16_t    *vc = geist_host_tensor_f16_3d(v_cache, &nvc);
    /* The source rows land at row q_position of the caches, whose views
     * end with them: the subtraction cannot wrap once both hold. */
    if (ks == nullptr || vs == nullptr || kc == nullptr || vc == nullptr || nks != nvs ||
        nkc != nvc || nks > nkc || k_src->shape[1] != k_cache->shape[1] ||
        k_src->shape[2] != k_cache->shape[2] ||
        q_position * (size_t) k_cache->shape[1] * (size_t) k_cache->shape[2] != nkc - nks) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon kv_append_f16: bad views");
        return GEIST_E_INVALID_ARG;
    }
    kv_rows_to_f16(nks, ks, kc + (nkc - nks));
    kv_rows_to_f16(nvs, vs, vc + (nvc - nvs));
    return GEIST_OK;
}

/* fused->attn_qkv_prep (Gemma 4) and attn_rope_append (Llama, Qwen3), the
 * attention prep after the q/k/v projections (#195): per token, the q
 * heads' rmsnorm (when a weight is given), RoPE and q_scale, and with k the
 * k heads' rmsnorm, the v heads' rmsnorm (Gemma 4's v_norm_w, ones), the k
 * heads' RoPE and both rows appended to the caches. Every row goes through
 * the rmsnorm_fp32, rope_apply, scale multiply and kv_rows_to_f16 the
 * separate prims and the arch run (an F32 cache takes a memcpy, as the
 * arch's mapped append does), so the bytes match the decomposed ops; the
 * tokens are split across threads from QKV_PREP_PAR_ROWS on, where these
 * passes ran on the main thread (Gemma 4 E2B: 64 tokens x 8 q heads x 256
 * per prefill chunk). */
enum { QKV_PREP_PAR_ROWS = 8 };

struct qkv_prep_job {
    float       *q, *k, *v;
    const float *qw, *kw, *vw, *cos, *sin;
    float       *kc32, *vc32;
    uint16_t    *kc16, *vc16;
    size_t       n_q_heads, n_kv_heads, hd, n_rot, q_position;
    float        eps, q_scale;
};

static void qkv_prep_rows(void *ctx, size_t t0, size_t t1) {
    const struct qkv_prep_job *j  = ctx;
    const size_t               qr = j->n_q_heads * j->hd;
    const size_t               kr = j->n_kv_heads * j->hd;
    for (size_t t = t0; t < t1; t++) {
        const float *cos_t = j->cos + t * j->n_rot;
        const float *sin_t = j->sin + t * j->n_rot;
        float       *qt    = j->q + t * qr;
        if (j->qw != nullptr) {
            rmsnorm_fp32(j->n_q_heads, j->hd, qt, j->qw, j->eps, qt);
        }
        rope_apply(1, j->n_q_heads, j->hd, j->n_rot, qt, cos_t, sin_t);
        if (j->q_scale != 1.0f) {
            for (size_t i = 0; i < qr; i++) {
                qt[i] *= j->q_scale;
            }
        }
        if (j->k == nullptr) {
            continue;
        }
        float       *kt  = j->k + t * kr;
        const float *vt  = j->v + t * kr;
        const size_t row = (j->q_position + t) * kr;
        if (j->kw != nullptr) {
            rmsnorm_fp32(j->n_kv_heads, j->hd, kt, j->kw, j->eps, kt);
        }
        if (j->vw != nullptr) {
            rmsnorm_fp32(j->n_kv_heads, j->hd, vt, j->vw, j->eps, j->v + t * kr);
        }
        rope_apply(1, j->n_kv_heads, j->hd, j->n_rot, kt, cos_t, sin_t);
        if (j->kc16 != nullptr) {
            kv_rows_to_f16(kr, kt, j->kc16 + row);
            kv_rows_to_f16(kr, vt, j->vc16 + row);
        } else {
            memcpy(j->kc32 + row, kt, kr * sizeof(float));
            memcpy(j->vc32 + row, vt, kr * sizeof(float));
        }
    }
}

/* Validates the views into *job and runs it. k, v and the caches are all
 * nullptr (q only: a KV-shared layer) or all present; the norm weights are
 * nullptr or [hd]. */
static enum geist_status qkv_prep_run(struct geist_backend      *be,
                                      const char                *op,
                                      struct geist_tensor       *q,
                                      struct geist_tensor       *k,
                                      const struct geist_tensor *v,
                                      const struct geist_tensor *q_norm_w,
                                      const struct geist_tensor *k_norm_w,
                                      const struct geist_tensor *v_norm_w,
                                      const struct geist_tensor *cos,
                                      const struct geist_tensor *sin,
                                      float                      eps,
                                      float                      q_scale,
                                      size_t                     q_position,
                                      struct geist_tensor       *k_cache,
                                      struct geist_tensor       *v_cache) {
    size_t       nq = 0, nqw = 0, nkw = 0, nvw = 0, nc = 0, ns = 0;
    float       *qp   = geist_host_tensor_f32(q, &nq);
    const float *qwp  = q_norm_w != nullptr ? geist_host_tensor_f32(q_norm_w, &nqw) : nullptr;
    const float *kwp  = k_norm_w != nullptr ? geist_host_tensor_f32(k_norm_w, &nkw) : nullptr;
    const float *vwp  = v_norm_w != nullptr ? geist_host_tensor_f32(v_norm_w, &nvw) : nullptr;
    const float *cosp = geist_host_tensor_f32(cos, &nc);
    const float *sinp = geist_host_tensor_f32(sin, &ns);
    if (qp == nullptr || cosp == nullptr || sinp == nullptr || q->ndim != 3 || cos->ndim != 2 ||
        sin->ndim != 2 || (q_norm_w != nullptr && qwp == nullptr) ||
        (k_norm_w != nullptr && kwp == nullptr) || (v_norm_w != nullptr && vwp == nullptr)) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon %s: bad q views", op);
        return GEIST_E_INVALID_ARG;
    }
    const size_t seq = (size_t) q->shape[0], n_q_heads = (size_t) q->shape[1];
    const size_t hd = (size_t) q->shape[2], n_rot = (size_t) cos->shape[1];
    if ((qwp != nullptr && nqw != hd) || (kwp != nullptr && nkw != hd) ||
        (vwp != nullptr && nvw != hd) || cos->shape[0] != (int64_t) seq ||
        sin->shape[0] != (int64_t) seq || sin->shape[1] != (int64_t) n_rot ||
        !rope_apply_shape_ok(seq, hd, n_rot, nc, ns)) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon %s: bad shapes", op);
        return GEIST_E_INVALID_ARG;
    }
    struct qkv_prep_job job = {.q          = qp,
                               .qw         = qwp,
                               .kw         = kwp,
                               .vw         = vwp,
                               .cos        = cosp,
                               .sin        = sinp,
                               .n_q_heads  = n_q_heads,
                               .hd         = hd,
                               .n_rot      = n_rot,
                               .q_position = q_position,
                               .eps        = eps,
                               .q_scale    = q_scale};
    if (k != nullptr || v != nullptr || k_cache != nullptr || v_cache != nullptr) {
        size_t nk = 0, nv = 0, nkc = 0, nvc = 0;
        job.k = k != nullptr ? geist_host_tensor_f32(k, &nk) : nullptr;
        job.v = v != nullptr ? geist_host_tensor_f32(v, &nv) : nullptr;
        if (k_cache != nullptr && k_cache->dtype == GEIST_DTYPE_F16) {
            job.kc16 = geist_host_tensor_f16_3d(k_cache, &nkc);
            job.vc16 = v_cache != nullptr ? geist_host_tensor_f16_3d(v_cache, &nvc) : nullptr;
        } else {
            job.kc32 = k_cache != nullptr ? geist_host_tensor_f32(k_cache, &nkc) : nullptr;
            job.vc32 = v_cache != nullptr ? geist_host_tensor_f32(v_cache, &nvc) : nullptr;
        }
        const bool caches = (job.kc16 != nullptr && job.vc16 != nullptr) ||
                            (job.kc32 != nullptr && job.vc32 != nullptr);
        /* The new rows land at row q_position of the caches, whose views
         * end with them (kv_append_f16's check). */
        if (job.k == nullptr || job.v == nullptr || !caches || k->ndim != 3 || v->ndim != 3 ||
            k_cache->ndim != 3 || v_cache->ndim != 3 || k->shape[0] != (int64_t) seq ||
            k->shape[2] != (int64_t) hd || nv != nk || nkc != nvc || nk > nkc ||
            k_cache->shape[1] != k->shape[1] || k_cache->shape[2] != (int64_t) hd ||
            q_position * (size_t) k_cache->shape[1] * hd != nkc - nk) {
            geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon %s: bad k/v views", op);
            return GEIST_E_INVALID_ARG;
        }
        job.n_kv_heads = (size_t) k->shape[1];
    }
    if (seq < QKV_PREP_PAR_ROWS) {
        qkv_prep_rows(&job, 0, seq);
    } else {
        geist_par_for(seq, qkv_prep_rows, &job);
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_attn_qkv_prep(struct geist_backend      *be,
                                                       struct geist_tensor       *q,
                                                       struct geist_tensor       *k,
                                                       struct geist_tensor       *v,
                                                       const struct geist_tensor *q_norm_w,
                                                       const struct geist_tensor *k_norm_w,
                                                       const struct geist_tensor *v_norm_w,
                                                       const struct geist_tensor *cos,
                                                       const struct geist_tensor *sin,
                                                       float                      eps,
                                                       size_t                     q_position,
                                                       struct geist_tensor       *k_cache,
                                                       struct geist_tensor       *v_cache) {
    /* Gemma 4: every norm present (k's and v's with k). */
    if (q_norm_w == nullptr || (k != nullptr && (k_norm_w == nullptr || v_norm_w == nullptr))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon attn_qkv_prep: null norm");
        return GEIST_E_INVALID_ARG;
    }
    return qkv_prep_run(be,
                        "attn_qkv_prep",
                        q,
                        k,
                        v,
                        q_norm_w,
                        k != nullptr ? k_norm_w : nullptr,
                        k != nullptr ? v_norm_w : nullptr,
                        cos,
                        sin,
                        eps,
                        1.0f,
                        q_position,
                        k_cache,
                        v_cache);
}

[[nodiscard]] enum geist_status cpu_neon_attn_rope_append(struct geist_backend      *be,
                                                          struct geist_tensor       *q,
                                                          struct geist_tensor       *k,
                                                          const struct geist_tensor *v,
                                                          const struct geist_tensor *q_norm_w,
                                                          const struct geist_tensor *k_norm_w,
                                                          const struct geist_tensor *cos,
                                                          const struct geist_tensor *sin,
                                                          float                      eps,
                                                          float                      q_scale,
                                                          size_t                     q_position,
                                                          struct geist_tensor       *k_cache,
                                                          struct geist_tensor       *v_cache) {
    if (k == nullptr || v == nullptr || k_cache == nullptr || v_cache == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon attn_rope_append: null k/v");
        return GEIST_E_INVALID_ARG;
    }
    return qkv_prep_run(be,
                        "attn_rope_append",
                        q,
                        k,
                        v,
                        q_norm_w,
                        k_norm_w,
                        nullptr,
                        cos,
                        sin,
                        eps,
                        q_scale,
                        q_position,
                        k_cache,
                        v_cache);
}

[[nodiscard]] enum geist_status cpu_neon_attention(struct geist_backend      *be,
                                                   const struct geist_tensor *q,
                                                   const struct geist_tensor *k,
                                                   const struct geist_tensor *v,
                                                   size_t                     q_offset,
                                                   size_t                     sliding_window,
                                                   struct geist_tensor       *out) {
    if (be == nullptr || q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (k->dtype == GEIST_DTYPE_F16) {
        return cpu_neon_attention_f16(be, q, k, v, q_offset, sliding_window, out);
    }
    size_t       nq, nk, nv, no;
    const float *qp = geist_host_tensor_f32(q, &nq);
    const float *kp = geist_host_tensor_f32(k, &nk);
    const float *vp = geist_host_tensor_f32(v, &nv);
    float       *op = geist_host_tensor_f32(out, &no);
    if (qp == nullptr || kp == nullptr || vp == nullptr || op == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "cpu_neon attention: tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || out->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon attention: bad ranks");
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
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon attention: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
#if defined(__ARM_NEON)
    /* hd256: the shape attention_mqa1_causal_kv_neon's rows use. */
    if (attention_heads_grouped_neon(qp,
                                     kp,
                                     vp,
                                     n_q,
                                     n_kv,
                                     q_offset,
                                     n_q_heads,
                                     n_kv_heads,
                                     head_dim,
                                     sliding_window,
                                     n_kv_heads == 1 && head_dim == 256,
                                     op,
                                     be->state != nullptr
                                             ? cpu_neon_ws((struct cpu_neon_state *) be->state)
                                             : nullptr)) {
        return GEIST_OK;
    }
    if (n_kv_heads == 1 &&
        attention_mqa1_causal_kv_neon(
                qp, kp, vp, n_q, n_kv, q_offset, n_q_heads, head_dim, sliding_window, op)) {
        return GEIST_OK;
    }
    if (attention_mqa_causal_kv_neon(qp,
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
#endif
    attention_mqa_causal_kv(
            n_q, n_kv, q_offset, n_q_heads, n_kv_heads, head_dim, sliding_window, qp, kp, vp, op);
    return GEIST_OK;
}
