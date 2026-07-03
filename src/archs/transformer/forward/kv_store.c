/*
 * src/archs/transformer/forward/kv_store.c - KV-cache append and
 * attention dispatch for transformer layers.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include <geist_types.h>

#include "kivi.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float kv_row_absmax(const float *x, size_t n) {
#if defined(__ARM_NEON)
    size_t      i    = 0;
    float32x4_t vmax = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vabsq_f32(vld1q_f32(x + i));
        vmax                = vmaxq_f32(vmax, v);
    }
    float out = vmaxvq_f32(vmax);
    for (; i < n; i++) {
        const float a = fabsf(x[i]);
        if (a > out) {
            out = a;
        }
    }
    return out;
#else
    float out = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float a = fabsf(x[i]);
        if (a > out) {
            out = a;
        }
    }
    return out;
#endif
}

enum geist_status transformer_kv_store_append(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_state   *st         = ctx->st;
    const struct geist_backend_vtbl *v          = ctx->v;
    const size_t                     seq        = ctx->seq;
    const size_t                     q_position = ctx->q_position;
    const size_t                     hd         = ctx->hd;
    const size_t                     kv_out     = ctx->kv_out;

    /* Plain f32 cache + device-copy-capable backend: append on-device so
     * batched GPU backends need no host round-trip (mapping scratch_k/v
     * here would force them to flush the pending pipeline). */
    if (!ctx->kv_kivi_enabled && !ctx->kv_int8_enabled &&
        v->buffer_copy != nullptr) {
        const size_t      row_bytes  = kv_out * sizeof(float);
        const size_t      span_bytes = seq * row_bytes;
        enum geist_status cs         = v->buffer_copy(
                ctx->k_cache_buf, q_position * row_bytes, st->sess->scratch_k, 0, span_bytes);
        if (cs == GEIST_OK) {
            cs = v->buffer_copy(
                    ctx->v_cache_buf, q_position * row_bytes, st->sess->scratch_v, 0, span_bytes);
        }
        if (cs == GEIST_OK) {
            return GEIST_OK;
        }
        /* fall through to the host path on failure */
    }

    const float *k_src = (const float *) v->buffer_map(st->sess->scratch_k);
    const float *v_src = (const float *) v->buffer_map(st->sess->scratch_v);
    if (ctx->kv_kivi_enabled) {
        float       *k_res     = (float *) v->buffer_map(ctx->k_residual_buf);
        float       *v_res     = (float *) v->buffer_map(ctx->v_residual_buf);
        const size_t row_elems = kv_out;
        for (size_t t = 0; t < seq; t++) {
            const size_t res_idx = (q_position + t) - st->sess->kivi_drained_count;
            memcpy(k_res + res_idx * row_elems, k_src + t * row_elems, row_elems * sizeof(float));
            memcpy(v_res + res_idx * row_elems, v_src + t * row_elems, row_elems * sizeof(float));
        }
        v->buffer_unmap(ctx->k_residual_buf);
        v->buffer_unmap(ctx->v_residual_buf);
    } else if (ctx->kv_int8_enabled) {
        int8_t      *k_dst          = (int8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        int8_t      *v_dst          = (int8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        float       *k_sca          = (float *) v->buffer_map(ctx->k_cache_scale_buf);
        float       *v_sca          = (float *) v->buffer_map(ctx->v_cache_scale_buf);
        const size_t row_elems      = kv_out;
        const size_t scales_per_row = st->n_kv_heads;
        for (size_t t = 0; t < seq; t++) {
            const size_t slot = q_position + t;
            for (size_t h = 0; h < st->n_kv_heads; h++) {
                const float *k_row   = k_src + t * row_elems + h * hd;
                const float *v_row   = v_src + t * row_elems + h * hd;
                const float  k_amax  = kv_row_absmax(k_row, hd);
                const float  v_amax  = kv_row_absmax(v_row, hd);
                float        k_scale = k_amax / 127.0f;
                if (k_scale == 0.0f) {
                    k_scale = 1.0f;
                }
                float v_scale = v_amax / 127.0f;
                if (v_scale == 0.0f) {
                    v_scale = 1.0f;
                }
                const float k_inv  = 1.0f / k_scale;
                const float v_inv  = 1.0f / v_scale;
                int8_t     *k_drow = k_dst + slot * row_elems + h * hd;
                int8_t     *v_drow = v_dst + slot * row_elems + h * hd;
                for (size_t i = 0; i < hd; i++) {
                    k_drow[i] = (int8_t) lrintf(k_row[i] * k_inv);
                    v_drow[i] = (int8_t) lrintf(v_row[i] * v_inv);
                }
                k_sca[slot * scales_per_row + h] = k_scale;
                v_sca[slot * scales_per_row + h] = v_scale;
            }
        }
        v->buffer_unmap(ctx->k_cache_q8_buf);
        v->buffer_unmap(ctx->v_cache_q8_buf);
        v->buffer_unmap(ctx->k_cache_scale_buf);
        v->buffer_unmap(ctx->v_cache_scale_buf);
    } else {
        uint8_t     *k_dst      = (uint8_t *) v->buffer_map(ctx->k_cache_buf);
        uint8_t     *v_dst      = (uint8_t *) v->buffer_map(ctx->v_cache_buf);
        const size_t row_bytes  = kv_out * sizeof(float);
        const size_t span_bytes = seq * row_bytes;
        memcpy(k_dst + q_position * row_bytes, (const uint8_t *) k_src, span_bytes);
        memcpy(v_dst + q_position * row_bytes, (const uint8_t *) v_src, span_bytes);
        v->buffer_unmap(ctx->k_cache_buf);
        v->buffer_unmap(ctx->v_cache_buf);
    }
    v->buffer_unmap(st->sess->scratch_k);
    v->buffer_unmap(st->sess->scratch_v);
    return GEIST_OK;
}

enum geist_status transformer_kv_store_attention(struct transformer_layer_forward_ctx *ctx,
                                                 const struct geist_tensor            *t_q_3d,
                                                 struct geist_tensor                  *t_attn_3d) {

    struct transformer_arch_state    *st         = ctx->st;
    struct transformer_layer_weights *L          = ctx->L;
    struct geist_backend             *be         = ctx->be;
    const struct geist_backend_vtbl  *v          = ctx->v;
    const size_t                      kv_len_now = ctx->kv_len_now;

    if (ctx->kv_kivi_enabled) {
        const float   *qp     = (const float *) v->buffer_map(st->sess->scratch_q);
        const uint8_t *kqp    = (const uint8_t *) v->buffer_map(ctx->k_kivi_q_buf);
        const uint8_t *vqp    = (const uint8_t *) v->buffer_map(ctx->v_kivi_q_buf);
        const float   *kscp   = (const float *) v->buffer_map(ctx->k_kivi_scales_buf);
        const float   *kzep   = (const float *) v->buffer_map(ctx->k_kivi_zeros_buf);
        const float   *vscp   = (const float *) v->buffer_map(ctx->v_kivi_scales_buf);
        const float   *vzep   = (const float *) v->buffer_map(ctx->v_kivi_zeros_buf);
        const float   *krp    = (const float *) v->buffer_map(ctx->k_residual_buf);
        const float   *vrp    = (const float *) v->buffer_map(ctx->v_residual_buf);
        float         *outp   = (float *) v->buffer_map(st->sess->scratch_attn);
        float         *scores = (float *) frame_arena_alloc(
                &st->sess->scratch_arena, kv_len_now * sizeof(float), 16);
        if (scores == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_OOM,
                                    "transformer: scratch arena exhausted for "
                                    "KIVI scores (kv_len_now=%zu)",
                                    kv_len_now);
            return GEIST_E_OOM;
        }
        attention_kivi_via_buffers(qp,
                                   ctx->seq,
                                   st->n_q_heads,
                                   ctx->hd,
                                   kqp,
                                   kscp,
                                   kzep,
                                   vqp,
                                   vscp,
                                   vzep,
                                   krp,
                                   vrp,
                                   kv_len_now,
                                   st->n_kv_heads,
                                   ctx->q_position,
                                   L->sliding_window,
                                   st->sess->kivi_drained_count,
                                   KIVI_K_GROUP_SIZE,
                                   scores,
                                   outp);
        v->buffer_unmap(st->sess->scratch_q);
        v->buffer_unmap(ctx->k_kivi_q_buf);
        v->buffer_unmap(ctx->v_kivi_q_buf);
        v->buffer_unmap(ctx->k_kivi_scales_buf);
        v->buffer_unmap(ctx->k_kivi_zeros_buf);
        v->buffer_unmap(ctx->v_kivi_scales_buf);
        v->buffer_unmap(ctx->v_kivi_zeros_buf);
        v->buffer_unmap(ctx->k_residual_buf);
        v->buffer_unmap(ctx->v_residual_buf);
        v->buffer_unmap(st->sess->scratch_attn);
    } else if (ctx->kv_int8_enabled) {
        const float  *qp       = (const float *) v->buffer_map(st->sess->scratch_q);
        const int8_t *k_q8p    = (const int8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        const int8_t *v_q8p    = (const int8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        const float  *k_scalep = (const float *) v->buffer_map(ctx->k_cache_scale_buf);
        const float  *v_scalep = (const float *) v->buffer_map(ctx->v_cache_scale_buf);
        float        *outp     = (float *) v->buffer_map(st->sess->scratch_attn);
        /* `scores` scratch is now private per query position inside the kernel
         * (the loop is parallelized), so no shared arena buffer is needed. */
        attention_int8_via_buffers(qp,
                                   ctx->seq,
                                   st->n_q_heads,
                                   ctx->hd,
                                   k_q8p,
                                   k_scalep,
                                   v_q8p,
                                   v_scalep,
                                   kv_len_now,
                                   st->n_kv_heads,
                                   ctx->q_position,
                                   L->sliding_window,
                                   outp);
        v->buffer_unmap(st->sess->scratch_q);
        v->buffer_unmap(ctx->k_cache_q8_buf);
        v->buffer_unmap(ctx->v_cache_q8_buf);
        v->buffer_unmap(ctx->k_cache_scale_buf);
        v->buffer_unmap(ctx->v_cache_scale_buf);
        v->buffer_unmap(st->sess->scratch_attn);
    } else {
        struct geist_tensor t_kcache_3d =
                view_3d(ctx->k_cache_buf, (int64_t) kv_len_now, st->n_kv_heads, (int64_t) ctx->hd);
        struct geist_tensor t_vcache_3d =
                view_3d(ctx->v_cache_buf, (int64_t) kv_len_now, st->n_kv_heads, (int64_t) ctx->hd);
        enum geist_status s = v->attention(be,
                                           t_q_3d,
                                           &t_kcache_3d,
                                           &t_vcache_3d,
                                           ctx->q_position,
                                           L->sliding_window,
                                           t_attn_3d);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}
