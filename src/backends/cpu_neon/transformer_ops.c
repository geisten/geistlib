/*
 * src/backends/cpu_neon/transformer_ops.c — rope_apply, embedding_lookup,
 * FP32 attention (NEON dot products) and the fused Q4_K/Q6_K GEGLU FFN.
 *
 * Layer: BACKEND.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"
#include "parallel.h"

#include "gemma4_kernels.h"
#include "host_buffer.h"
#include "quant.h"
#include "heap.h"

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
    const char *env = getenv("GEIST_FFN_TILE_BLOCKS");
    if (env != nullptr && env[0] != '\0') {
        char               *end = nullptr;
        const unsigned long v   = strtoul(env, &end, 10);
        if (end != env && v >= 1 && v <= 8)
            return (size_t) v;
    }
    return 2;
}

/* One activation quantization of the fused FFN, for ffn_quant_rows. */
struct ffn_quant_job {
    const float *x;
    size_t       d_model, n_groups;
    int8_t      *xq;
    float       *sc;
    int32_t     *sum32;
};

/* Rows [i0, i1) of the fused FFN's activation quantization. */
static void ffn_quant_rows(void *ctx, size_t i0, size_t i1) {
    const struct ffn_quant_job *job      = ctx;
    const float                *x        = job->x;
    const size_t                d_model  = job->d_model;
    const size_t                n_groups = job->n_groups;
    int8_t                     *xq       = job->xq;
    float                      *sc       = job->sc;
    int32_t                    *sum32    = job->sum32;
    for (size_t i = i0; i < i1; i++) {
        quantize_x_q8_groups(d_model,
                             GEIST_ACT_Q8K_ELEMS,
                             x + i * d_model,
                             xq + i * d_model,
                             sc + i * n_groups,
                             sum32 + i * (d_model / 32));
    }
}

[[nodiscard]] enum geist_status cpu_neon_ffn_geglu_q4q6_mN(struct geist_backend      *be,
                                                           size_t                     m,
                                                           size_t                     d_model,
                                                           size_t                     inter,
                                                           const float               *x,
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
    struct cpu_neon_workspace *ws                  = cpu_neon_ws(st);
    const size_t               xq_need             = m * d_model;
    const size_t               sum_need            = m * (d_model / 32);
    const size_t               xsc_need            = m * (d_model / Q4_K_BLOCK_ELEMS);
    const size_t               blocks_per_gate_row = d_model / Q4_K_BLOCK_ELEMS;
    const size_t               total_inter_blocks  = inter / Q6_K_BLOCK_ELEMS;
    const size_t               tile_blocks_req     = ffn_tile_blocks();

    if (!grow_i8(&ws->qk_mN_xq, &ws->qk_mN_xq_cap, xq_need) ||
        !grow_i32(&ws->qk_mN_sum32, &ws->qk_mN_sum32_cap, sum_need) ||
        !grow_f32(&ws->qk_mN_sc, &ws->qk_mN_sc_cap, xsc_need) ||
        !grow_f32(&ws->ffn_mid_sc, &ws->ffn_mid_sc_cap, m * tile_blocks_req)) {
        return GEIST_E_OOM;
    }

    struct ffn_quant_job qj = {.x        = x,
                               .d_model  = d_model,
                               .xq       = ws->qk_mN_xq,
                               .sc       = ws->qk_mN_sc,
                               .sum32    = ws->qk_mN_sum32,
                               .n_groups = blocks_per_gate_row};
    if (m >= 4) {
        geist_par_for(m, ffn_quant_rows, &qj);
    } else {
        ffn_quant_rows(&qj, 0, m);
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

        linear_q4k_w4a8_prefill_pre(m,
                                    d_model,
                                    tile_n,
                                    ws->qk_mN_xq,
                                    ws->qk_mN_sc,
                                    ws->qk_mN_sum32,
                                    gate_tile,
                                    ws->ffn_gate);
        linear_q4k_w4a8_prefill_pre(m,
                                    d_model,
                                    tile_n,
                                    ws->qk_mN_xq,
                                    ws->qk_mN_sc,
                                    ws->qk_mN_sum32,
                                    up_tile,
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

        /* One scale per 256-element Q6_K block of the tile row (#698). */
        for (size_t i = 0; i < m; i++) {
            quantize_x_q8_groups(tile_n,
                                 GEIST_ACT_Q8K_ELEMS,
                                 ws->ffn_mid + i * tile_n,
                                 ws->ffn_mid_q8 + i * tile_n,
                                 ws->ffn_mid_sc + i * nb,
                                 nullptr);
        }

        linear_q6k_w6a8_prefill_pre_accum_blocks(
                m, inter, d_model, b0, nb, ws->ffn_mid_q8, ws->ffn_mid_sc, down->raw, y);
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_rope_apply(struct geist_backend      *be,
                                                    struct geist_tensor       *x,
                                                    const struct geist_tensor *cos,
                                                    const struct geist_tensor *sin) {
    if (be == nullptr || x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx, nc, ns;
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
    rope_apply(seq_len, n_heads, head_dim, (size_t) cos->shape[cos->ndim - 1], xp, cosp, sinp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_embedding_lookup(struct geist_backend      *be,
                                                          const struct geist_tensor *embed_table,
                                                          geist_token_t              token_id,
                                                          struct geist_tensor       *out) {
    if (be == nullptr || embed_table == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       n_table, n_out;
    const float *tablep = geist_host_tensor_f32(embed_table, &n_table);
    float       *outp   = geist_host_tensor_f32(out, &n_out);
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

/* The FP32 attention's chunk: items (query, head) are handed out `chunk`
 * at a time, or split statically for 0. GEIST_ATTENTION_OMP_SCHEDULE,
 * read once: "dynamic[,c]" (the default, c = 8), "guided[,c]" (taken as
 * dynamic: geist_par_for has no guided schedule), "static" (a chunk after
 * it is ignored: the split is static either way and the output the same). */
static size_t attention_chunk(void) {
    static _Atomic long cached = -1;
    long                chunk  = atomic_load_explicit(&cached, memory_order_relaxed);
    if (chunk >= 0) {
        return (size_t) chunk;
    }
    chunk           = 8;
    const char *env = getenv("GEIST_ATTENTION_OMP_SCHEDULE");
    if (env != nullptr && env[0] != '\0') {
        const bool  is_static = strncmp(env, "static", 6) == 0;
        const char *comma     = strchr(env, ',');
        if (comma != nullptr && comma[1] != '\0') {
            const long parsed = strtol(comma + 1, nullptr, 10);
            if (parsed > 0 && parsed < (1L << 20)) {
                chunk = parsed;
            }
        }
        if (is_static) {
            chunk = 0;
        }
    }
    atomic_store_explicit(&cached, chunk, memory_order_relaxed);
    return (size_t) chunk;
}

/* Runs fn over the `total` attention items on attention_chunk()'s schedule. */
static void attention_par(size_t total, geist_par_fn fn, void *ctx) {
    const size_t chunk = attention_chunk();
    if (chunk == 0) {
        geist_par_for(total, fn, ctx);
    } else {
        cpu_neon_par_for_dynamic(total, chunk, fn, ctx);
    }
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
    if (total < 16) {
        return false;
    }
    const bool hd256 = head_dim == 256;

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
    const size_t total = n_q * n_q_heads;
    if (total < 16) {
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
                           .out            = out};
    attention_par(total, attn_mqa_items, &job);
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
