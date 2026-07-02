/*
 * src/backends/cpu_neon/transformer_ops.c — rope_apply, embedding_lookup,
 * attention wrappers over gemma4_kernels.c.
 *
 * Layer: BACKEND.
 *
 * These are the heavyweight transformer ops the engine needs to fully
 * route lm.c's forward pass through the backend vtable. The
 * gemma4_kernels.c implementations are FP32 reference kernels with
 * partial NEON acceleration; cpu_scalar uses the same code path
 * (no SIMD difference for these particular ops today).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "gemma4_kernels.h"
#include "quant.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static float *get_f32_dense_ptr_full(const struct geist_tensor *t, size_t *out_n) {
    if (t == nullptr || t->dtype != GEIST_DTYPE_F32 || t->layout != GEIST_LAYOUT_DENSE ||
        t->buffer == nullptr || t->ndim < 1) {
        return nullptr;
    }
    size_t n = 1;
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0)
            return nullptr;
        n *= (size_t) t->shape[d];
    }
    *out_n = n;
    return (float *) ((uint8_t *) t->buffer->host + t->offset);
}

static bool grow_f32(float **p, size_t *cap, size_t need) {
    if (*cap >= need)
        return true;
    safe_free((void **) p);
    *p = heap_alloc_array_aligned(float, need);
    if (*p == nullptr) {
        *cap = 0;
        return false;
    }
    *cap = need;
    return true;
}

static bool grow_i8(int8_t **p, size_t *cap, size_t need) {
    if (*cap >= need)
        return true;
    safe_free((void **) p);
    *p = heap_alloc_array_aligned(int8_t, need);
    if (*p == nullptr) {
        *cap = 0;
        return false;
    }
    *cap = need;
    return true;
}

static bool grow_i32(int32_t **p, size_t *cap, size_t need) {
    if (*cap >= need)
        return true;
    safe_free((void **) p);
    *p = heap_alloc_array_aligned(int32_t, need);
    if (*p == nullptr) {
        *cap = 0;
        return false;
    }
    *cap = need;
    return true;
}

static size_t ffn_tile_blocks(void) {
    static size_t cached = 0; /* 0 = env not parsed yet */
    if (cached == 0) {
        size_t      v   = 2;
        const char *env = getenv("GEIST_FFN_TILE_BLOCKS");
        if (env != nullptr && env[0] != '\0') {
            char               *end = nullptr;
            const unsigned long u   = strtoul(env, &end, 10);
            if (end != env && u >= 1 && u <= 8)
                v = (size_t) u;
        }
        cached = v;
    }
    return cached;
}

[[nodiscard]] enum geist_status cpu_neon_ffn_geglu_q4q6_mN(struct geist_backend      *be,
                                                           const float               *x,
                                                           size_t                     m,
                                                           size_t                     d_model,
                                                           size_t                     inter,
                                                           const struct geist_weight *gate,
                                                           const struct geist_weight *up,
                                                           const struct geist_weight *down,
                                                           const float               *down_scale,
                                                           float                     *y) {
    if (be == nullptr || be->state == nullptr || x == nullptr || y == nullptr || gate == nullptr ||
        up == nullptr || down == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (m == 0 || m > GEIST_QUANT_M_CAP || d_model == 0 || inter == 0 ||
        d_model % Q4_K_BLOCK_ELEMS != 0 || inter % Q6_K_BLOCK_ELEMS != 0 ||
        gate->dtype != GEIST_DTYPE_Q4_K || up->dtype != GEIST_DTYPE_Q4_K ||
        down->dtype != GEIST_DTYPE_Q6_K || (size_t) gate->n_in != d_model ||
        (size_t) up->n_in != d_model || (size_t) down->n_in != inter ||
        (size_t) gate->n_out != inter || (size_t) up->n_out != inter ||
        (size_t) down->n_out != d_model) {
        return GEIST_E_UNSUPPORTED;
    }

    struct cpu_neon_state     *st                  = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws                  = &st->workspace;
    const size_t               xq_need             = m * d_model;
    const size_t               sum_need            = m * (d_model / 32);
    const size_t               xsc_need            = m;
    const size_t               blocks_per_gate_row = d_model / Q4_K_BLOCK_ELEMS;
    const size_t               total_inter_blocks  = inter / Q6_K_BLOCK_ELEMS;
    const size_t               tile_blocks_req     = ffn_tile_blocks();

    if (!grow_i8(&ws->qk_mN_xq, &ws->qk_mN_xq_cap, xq_need) ||
        !grow_i32(&ws->qk_mN_sum32, &ws->qk_mN_sum32_cap, sum_need) ||
        !grow_f32(&ws->qk_mN_sc, &ws->qk_mN_sc_cap, xsc_need) ||
        !grow_f32(&ws->ffn_mid_sc, &ws->ffn_mid_sc_cap, m)) {
        return GEIST_E_OOM;
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (m >= 4)
#endif
    for (size_t i = 0; i < m; i++) {
        ws->qk_mN_sc[i] = quantize_x_for_q4k(x + i * d_model,
                                             d_model,
                                             ws->qk_mN_xq + i * d_model,
                                             ws->qk_mN_sum32 + i * (d_model / 32));
    }

    memset(y, 0, m * d_model * sizeof(float));
    const float kAlpha = 0.7978845608028654f;
    const float kBeta  = 0.044715f;

    for (size_t b0 = 0; b0 < total_inter_blocks; b0 += tile_blocks_req) {
        const size_t nb = (total_inter_blocks - b0 < tile_blocks_req) ? (total_inter_blocks - b0)
                                                                      : tile_blocks_req;
        const size_t tile_n     = nb * Q6_K_BLOCK_ELEMS;
        const size_t tile_elems = m * tile_n;
        if (!grow_f32(&ws->ffn_gate, &ws->ffn_gate_cap, tile_elems) ||
            !grow_f32(&ws->ffn_up, &ws->ffn_up_cap, tile_elems) ||
            !grow_f32(&ws->ffn_mid, &ws->ffn_mid_cap, tile_elems) ||
            !grow_i8(&ws->ffn_mid_q8, &ws->ffn_mid_q8_cap, tile_elems)) {
            return GEIST_E_OOM;
        }

        const uint8_t *gate_tile = (const uint8_t *) gate->raw +
                                   b0 * Q6_K_BLOCK_ELEMS * blocks_per_gate_row * Q4_K_BLOCK_BYTES;
        const uint8_t *up_tile   = (const uint8_t *) up->raw +
                                   b0 * Q6_K_BLOCK_ELEMS * blocks_per_gate_row * Q4_K_BLOCK_BYTES;

        linear_q4k_w4a8_prefill_pre(ws->qk_mN_xq,
                                    ws->qk_mN_sc,
                                    ws->qk_mN_sum32,
                                    m,
                                    gate_tile,
                                    d_model,
                                    tile_n,
                                    ws->ffn_gate);
        linear_q4k_w4a8_prefill_pre(ws->qk_mN_xq,
                                    ws->qk_mN_sc,
                                    ws->qk_mN_sum32,
                                    m,
                                    up_tile,
                                    d_model,
                                    tile_n,
                                    ws->ffn_up);

        for (size_t i = 0; i < tile_elems; i++) {
            const size_t j     = i % tile_n;
            const float  g     = ws->ffn_gate[i];
            const float  inner = kAlpha * (g + kBeta * g * g * g);
            float        v     = (0.5f * g * (1.0f + tanhf(inner))) * ws->ffn_up[i];
            if (down_scale != nullptr) {
                v *= down_scale[b0 * Q6_K_BLOCK_ELEMS + j];
            }
            ws->ffn_mid[i] = v;
        }

        for (size_t i = 0; i < m; i++) {
            ws->ffn_mid_sc[i] = quantize_x_int8_sym(
                    ws->ffn_mid + i * tile_n, tile_n, ws->ffn_mid_q8 + i * tile_n);
        }

        linear_q6k_w6a8_prefill_pre_accum_blocks(
                ws->ffn_mid_q8, ws->ffn_mid_sc, m, down->raw, inter, d_model, b0, nb, y);
    }
    return GEIST_OK;
}

/* ---- rope_apply ---- */

[[nodiscard]] enum geist_status cpu_neon_rope_apply(struct geist_backend      *be,
                                                    struct geist_tensor       *x,
                                                    const struct geist_tensor *cos,
                                                    const struct geist_tensor *sin) {
    if (be == nullptr || x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx, nc, ns;
    float       *xp   = get_f32_dense_ptr_full(x, &nx);
    const float *cosp = get_f32_dense_ptr_full(cos, &nc);
    const float *sinp = get_f32_dense_ptr_full(sin, &ns);
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
    rope_apply(xp, cosp, sinp, seq_len, n_heads, head_dim);
    return GEIST_OK;
}

/* ---- embedding_lookup ---- */

[[nodiscard]] enum geist_status cpu_neon_embedding_lookup(struct geist_backend      *be,
                                                          const struct geist_tensor *embed_table,
                                                          geist_token_t              token_id,
                                                          struct geist_tensor       *out) {
    if (be == nullptr || embed_table == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       n_table, n_out;
    const float *tablep = get_f32_dense_ptr_full(embed_table, &n_table);
    float       *outp   = get_f32_dense_ptr_full(out, &n_out);
    if (tablep == nullptr || outp == nullptr || embed_table->ndim != 2) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon embedding_lookup: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    int64_t vocab_size = embed_table->shape[0];
    int64_t d_model    = embed_table->shape[1];
    if (token_id < 0 || (int64_t) token_id >= vocab_size) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_neon embedding_lookup: token_id %d out of range [0, %lld)",
                                (int) token_id,
                                (long long) vocab_size);
        return GEIST_E_INVALID_ARG;
    }
    if (n_out != (size_t) d_model) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_neon embedding_lookup: out size %zu != d_model %lld",
                                n_out,
                                (long long) d_model);
        return GEIST_E_INVALID_ARG;
    }
    memcpy(outp, tablep + (size_t) token_id * (size_t) d_model, (size_t) d_model * sizeof(float));
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

#ifdef _OPENMP
static void attention_set_omp_schedule(void) {
    static int initialized = 0;
    if (initialized) {
        return;
    }
    initialized = 1;

    const char *env   = getenv("GEIST_ATTENTION_OMP_SCHEDULE");
    omp_sched_t kind  = omp_sched_dynamic;
    int         chunk = 8;
    if (env != nullptr && env[0] != '\0') {
        if (strncmp(env, "dynamic", 7) == 0) {
            kind  = omp_sched_dynamic;
            chunk = 8;
        } else if (strncmp(env, "guided", 6) == 0) {
            kind  = omp_sched_guided;
            chunk = 8;
        } else if (strncmp(env, "static", 6) == 0) {
            kind  = omp_sched_static;
            chunk = 0;
        }
        const char *comma = strchr(env, ',');
        if (comma != nullptr && comma[1] != '\0') {
            const long parsed = strtol(comma + 1, nullptr, 10);
            if (parsed > 0 && parsed < (1L << 20)) {
                chunk = (int) parsed;
            }
        }
    }
    omp_set_schedule(kind, chunk);
}
#endif

/* Score scratch for the SDPA kernels below — grown on demand and kept for
 * the thread's lifetime instead of a heap round-trip per attention call
 * (one per layer per forward pass). */
static _Thread_local float *tls_score_arena     = nullptr;
static _Thread_local size_t tls_score_arena_cap = 0;

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
    if (total < 16) {
        return false;
    }

    size_t n_threads = 1;
#ifdef _OPENMP
    n_threads = (size_t) omp_get_max_threads();
#endif
    if (!grow_f32(&tls_score_arena, &tls_score_arena_cap, n_threads * n_kv)) {
        return false;
    }
    float     *score_arena = tls_score_arena;
    const bool hd256       = head_dim == 256;

#ifdef _OPENMP
    attention_set_omp_schedule();
#pragma omp parallel
    {
        const int tid    = omp_get_thread_num();
        float    *scores = score_arena + (size_t) tid * n_kv;
#pragma omp for schedule(runtime)
        for (size_t idx = 0; idx < total; idx++) {
            const size_t t     = idx / n_q_heads;
            const size_t h     = idx - t * n_q_heads;
            const size_t q_pos = q_offset + t;
            const size_t s_lo  = (sliding_window > 0 && q_pos + 1 > sliding_window)
                                         ? q_pos + 1 - sliding_window
                                         : 0;
            size_t       s_hi  = q_pos;
            if (s_hi >= n_kv) {
                s_hi = n_kv - 1;
            }

            const float *qv        = q + (t * n_q_heads + h) * head_dim;
            float        max_score = -INFINITY;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *kv = k + s * head_dim;
                const float  sc = hd256 ? dot_f32_neon_256(qv, kv) : dot_f32_neon(qv, kv, head_dim);
                scores[s]       = sc;
                if (sc > max_score) {
                    max_score = sc;
                }
            }

            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float e = expf(scores[s] - max_score);
                scores[s]     = e;
                sum_exp += e;
            }
            const float inv_sum = (float) (1.0 / sum_exp);

            float *outv = out + (t * n_q_heads + h) * head_dim;
            if (hd256) {
                zero_f32_neon_256(outv);
                for (size_t s = s_lo; s <= s_hi; s++) {
                    add_scaled_f32_neon_256(outv, v + s * head_dim, scores[s] * inv_sum);
                }
            } else {
                zero_f32_neon(outv, head_dim);
                for (size_t s = s_lo; s <= s_hi; s++) {
                    add_scaled_f32_neon(outv, v + s * head_dim, scores[s] * inv_sum, head_dim);
                }
            }
        }
    }
#else
    {
        float *scores = score_arena;
        for (size_t idx = 0; idx < total; idx++) {
            const size_t t     = idx / n_q_heads;
            const size_t h     = idx - t * n_q_heads;
            const size_t q_pos = q_offset + t;
            const size_t s_lo  = (sliding_window > 0 && q_pos + 1 > sliding_window)
                                         ? q_pos + 1 - sliding_window
                                         : 0;
            size_t       s_hi  = q_pos;
            if (s_hi >= n_kv) {
                s_hi = n_kv - 1;
            }

            const float *qv        = q + (t * n_q_heads + h) * head_dim;
            float        max_score = -INFINITY;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *kv = k + s * head_dim;
                const float  sc = hd256 ? dot_f32_neon_256(qv, kv) : dot_f32_neon(qv, kv, head_dim);
                scores[s]       = sc;
                if (sc > max_score) {
                    max_score = sc;
                }
            }

            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float e = expf(scores[s] - max_score);
                scores[s]     = e;
                sum_exp += e;
            }
            const float inv_sum = (float) (1.0 / sum_exp);

            float *outv = out + (t * n_q_heads + h) * head_dim;
            if (hd256) {
                zero_f32_neon_256(outv);
                for (size_t s = s_lo; s <= s_hi; s++) {
                    add_scaled_f32_neon_256(outv, v + s * head_dim, scores[s] * inv_sum);
                }
            } else {
                zero_f32_neon(outv, head_dim);
                for (size_t s = s_lo; s <= s_hi; s++) {
                    add_scaled_f32_neon(outv, v + s * head_dim, scores[s] * inv_sum, head_dim);
                }
            }
        }
    }
#endif

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
    const size_t total = n_q * n_q_heads;
    if (total < 16) {
        return false;
    }
    const size_t kv_group_size = n_q_heads / n_kv_heads;

    size_t n_threads = 1;
#ifdef _OPENMP
    n_threads = (size_t) omp_get_max_threads();
#endif
    if (!grow_f32(&tls_score_arena, &tls_score_arena_cap, n_threads * n_kv)) {
        return false;
    }
    float *score_arena = tls_score_arena;

#ifdef _OPENMP
    attention_set_omp_schedule();
#pragma omp parallel
    {
        const int tid    = omp_get_thread_num();
        float    *scores = score_arena + (size_t) tid * n_kv;
#pragma omp for schedule(runtime)
        for (size_t idx = 0; idx < total; idx++) {
            const size_t t     = idx / n_q_heads;
            const size_t h     = idx - t * n_q_heads;
            const size_t q_pos = q_offset + t;
            const size_t s_lo  = (sliding_window > 0 && q_pos + 1 > sliding_window)
                                         ? q_pos + 1 - sliding_window
                                         : 0;
            size_t       s_hi  = q_pos;
            if (s_hi >= n_kv) {
                s_hi = n_kv - 1;
            }
            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            float max_score = -INFINITY;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *kv = k + (s * n_kv_heads + kv_h) * head_dim;
                const float  sc = dot_f32_neon(qv, kv, head_dim);
                scores[s]       = sc;
                if (sc > max_score) {
                    max_score = sc;
                }
            }
            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float e = expf(scores[s] - max_score);
                scores[s]     = e;
                sum_exp += e;
            }
            const float inv_sum = (float) (1.0 / sum_exp);

            float *outv = out + (t * n_q_heads + h) * head_dim;
            zero_f32_neon(outv, head_dim);
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *vv = v + (s * n_kv_heads + kv_h) * head_dim;
                add_scaled_f32_neon(outv, vv, scores[s] * inv_sum, head_dim);
            }
        }
    }
#endif
#if !defined(_OPENMP)
    {
        float *scores = score_arena;
        for (size_t idx = 0; idx < total; idx++) {
            const size_t t     = idx / n_q_heads;
            const size_t h     = idx - t * n_q_heads;
            const size_t q_pos = q_offset + t;
            const size_t s_lo  = (sliding_window > 0 && q_pos + 1 > sliding_window)
                                         ? q_pos + 1 - sliding_window
                                         : 0;
            size_t       s_hi  = q_pos;
            if (s_hi >= n_kv) {
                s_hi = n_kv - 1;
            }
            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            float max_score = -INFINITY;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *kv = k + (s * n_kv_heads + kv_h) * head_dim;
                const float  sc = dot_f32_neon(qv, kv, head_dim);
                scores[s]       = sc;
                if (sc > max_score) {
                    max_score = sc;
                }
            }
            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float e = expf(scores[s] - max_score);
                scores[s]     = e;
                sum_exp += e;
            }
            const float inv_sum = (float) (1.0 / sum_exp);

            float *outv = out + (t * n_q_heads + h) * head_dim;
            zero_f32_neon(outv, head_dim);
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *vv = v + (s * n_kv_heads + kv_h) * head_dim;
                add_scaled_f32_neon(outv, vv, scores[s] * inv_sum, head_dim);
            }
        }
    }
#endif

    return true;
}
#endif

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
    size_t       nq, nk, nv, no;
    const float *qp = get_f32_dense_ptr_full(q, &nq);
    const float *kp = get_f32_dense_ptr_full(k, &nk);
    const float *vp = get_f32_dense_ptr_full(v, &nv);
    float       *op = get_f32_dense_ptr_full(out, &no);
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
            qp, kp, vp, n_q, n_kv, q_offset, n_q_heads, n_kv_heads, head_dim, sliding_window, op);
    return GEIST_OK;
}
