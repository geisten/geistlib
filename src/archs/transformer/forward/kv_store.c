/*
 * src/archs/transformer/forward/kv_store.c - KV-cache append and
 * attention dispatch for transformer layers.
 *
 * The Hadamard rotation (GEIST_KV_ROT) and packed INT4 store K in the cache
 * after RoPE, rotated. This is safe only because geist
 * never re-bases cached positions — the sliding window merely masks and does
 * not re-RoPE the cache. Any context-shift implementation must unpack,
 * un-rotate, apply RoPE again, rotate and repack cached rows; otherwise the
 * cache is corrupted (issue #71; llama.cpp#21038 had to add explicit
 * cache-shift support for exactly this reason).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../forward.h"
#include <geist_types.h>

#include "fwht.h"
#include "int4_kv.h"
#include "kivi.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* max |x[i]|; 0 for a row of zeros. Taken on the bit patterns: with the
 * sign cleared, the unsigned integers order as the values do (NaN aside),
 * and an integer max vectorizes under any floating-point flags, where
 * clang keeps a float compare-and-select scalar unless it may assume no
 * NaNs (not under -fno-finite-math-only). gcc 14 (x86-64-v3, aarch64) and
 * clang 19 (aarch64) all give vector code; on x86-64 it is as fast as the
 * float compare a row of 64 and 12-17 % faster at 128 and 256. */
static inline float kv_row_absmax(size_t n, const float x[static n]) {
    uint32_t m = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t b;
        memcpy(&b, &x[i], sizeof b);
        b &= 0x7FFFFFFFu;
        m = b > m ? b : m;
    }
    float out;
    memcpy(&out, &m, sizeof out);
    return out;
}

enum geist_status transformer_kv_store_append(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_state    *st         = ctx->st;
    struct transformer_arch_session  *sess       = ctx->sess;
    const struct geist_backend_vtbl  *v          = ctx->v;
    const struct geist_backend_fused *fused      = ctx->fused;
    const size_t                      seq        = ctx->seq;
    const size_t                      q_position = ctx->q_position;
    const size_t                      hd         = ctx->hd;
    const size_t                      kv_out     = ctx->kv_out;

    /* F16 cache: fused f32→f16 converting append on-device. The mode is
     * only enabled when the backend provides the slot (arch_state.c). */
    if (ctx->kv_f16_enabled) {
        struct geist_tensor t_k_src =
                view_3d(sess->scratch_k, (int64_t) seq, st->n_kv_heads, (int64_t) hd);
        struct geist_tensor t_v_src =
                view_3d(sess->scratch_v, (int64_t) seq, st->n_kv_heads, (int64_t) hd);
        struct geist_tensor t_k_dst = view_3d_f16(
                ctx->k_cache_buf, (int64_t) (q_position + seq), st->n_kv_heads, (int64_t) hd);
        struct geist_tensor t_v_dst = view_3d_f16(
                ctx->v_cache_buf, (int64_t) (q_position + seq), st->n_kv_heads, (int64_t) hd);
        return fused->kv_append_f16(ctx->be, &t_k_src, &t_v_src, q_position, &t_k_dst, &t_v_dst);
    }

    /* Plain f32 cache + device-copy-capable backend: append on-device so
     * batched GPU backends need no host round-trip (mapping scratch_k/v
     * here would force them to flush the pending pipeline). */
    if (!ctx->kv_kivi_enabled && !ctx->kv_int8_enabled &&
        ctx->st->model_fusions.backend_buffer_copy) {
        const size_t      row_bytes  = kv_out * sizeof(float);
        const size_t      span_bytes = seq * row_bytes;
        enum geist_status cs         = v->buffer_copy(
                ctx->k_cache_buf, q_position * row_bytes, sess->scratch_k, 0, span_bytes);
        if (cs == GEIST_OK) {
            cs = v->buffer_copy(
                    ctx->v_cache_buf, q_position * row_bytes, sess->scratch_v, 0, span_bytes);
        }
        /* Bound (#352): the capability is decided at plan build, so a
         * non-OK result here is a device error, not "this backend cannot
         * do it" — returning it beats a host path that hides it. */
        return cs;
    }

    const float *k_src = (const float *) v->buffer_map(sess->scratch_k);
    const float *v_src = (const float *) v->buffer_map(sess->scratch_v);
    if (ctx->kv_kivi_enabled) {
        float       *k_res     = (float *) v->buffer_map(ctx->k_residual_buf);
        float       *v_res     = (float *) v->buffer_map(ctx->v_residual_buf);
        const size_t row_elems = kv_out;
        for (size_t t = 0; t < seq; t++) {
            const size_t res_idx = (q_position + t) - sess->kivi_drained_count;
            memcpy(k_res + res_idx * row_elems, k_src + t * row_elems, row_elems * sizeof(float));
            memcpy(v_res + res_idx * row_elems, v_src + t * row_elems, row_elems * sizeof(float));
        }
        v->buffer_unmap(ctx->k_residual_buf);
        v->buffer_unmap(ctx->v_residual_buf);
    } else if (sess->kv_int4_packed_enabled) {
        /* Packed 4-bit: 2 values/byte into the half-size int8 slots. Same
         * per-token per-head scale + optional rotation as INT8. denom 7 →
         * scale = amax/7, values in [-7,7]. */
        uint8_t     *k_dst          = (uint8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        uint8_t     *v_dst          = (uint8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        float       *k_sca          = (float *) v->buffer_map(ctx->k_cache_scale_buf);
        float       *v_sca          = (float *) v->buffer_map(ctx->v_cache_scale_buf);
        const size_t row_elems      = kv_out;
        const size_t scales_per_row = st->n_kv_heads;
        const bool   rot            = sess->kv_rot_enabled && fwht_supported(hd) && hd <= 512;
        float        krot[512];
        float        vrot[512];
        for (size_t t = 0; t < seq; t++) {
            const size_t slot = q_position + t;
            for (size_t h = 0; h < st->n_kv_heads; h++) {
                const float *k_row = k_src + t * row_elems + h * hd;
                const float *v_row = v_src + t * row_elems + h * hd;
                if (rot) {
                    memcpy(krot, k_row, hd * sizeof(float));
                    memcpy(vrot, v_row, hd * sizeof(float));
                    fwht_orthonormal(hd, krot);
                    fwht_orthonormal(hd, vrot);
                    k_row = krot;
                    v_row = vrot;
                }
                float k_scale = kv_row_absmax(hd, k_row) / 7.0f;
                float v_scale = kv_row_absmax(hd, v_row) / 7.0f;
                if (k_scale == 0.0f)
                    k_scale = 1.0f;
                if (v_scale == 0.0f)
                    v_scale = 1.0f;
                const size_t byte_off = (slot * row_elems + h * hd) / 2;
                int4_pack_row(hd, k_row, 1.0f / k_scale, k_dst + byte_off);
                int4_pack_row(hd, v_row, 1.0f / v_scale, v_dst + byte_off);
                k_sca[slot * scales_per_row + h] = k_scale;
                v_sca[slot * scales_per_row + h] = v_scale;
            }
        }
        v->buffer_unmap(ctx->k_cache_q8_buf);
        v->buffer_unmap(ctx->v_cache_q8_buf);
        v->buffer_unmap(ctx->k_cache_scale_buf);
        v->buffer_unmap(ctx->v_cache_scale_buf);
    } else if (ctx->kv_int8_enabled) {
        int8_t      *k_dst          = (int8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        int8_t      *v_dst          = (int8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        float       *k_sca          = (float *) v->buffer_map(ctx->k_cache_scale_buf);
        float       *v_sca          = (float *) v->buffer_map(ctx->v_cache_scale_buf);
        const size_t row_elems      = kv_out;
        const size_t scales_per_row = st->n_kv_heads;
        /* Issue #61: rotate each K/V head row before quantizing. Q is
         * rotated symmetrically at attention time (kv_store_attention). */
        const bool rot = sess->kv_rot_enabled && fwht_supported(hd) && hd <= 512;
        float      krot[512];
        float      vrot[512];
        /* Issue #61: low-bit quality-sim quantizes on an N-bit grid
         * (amax / (2^(N-1)-1)); the int8 container then holds values in
         * [-(2^(N-1)-1), +...]. Rounding stays in range (|x| <= amax), so no
         * clamp is needed. qbits==0 is the native 8-bit path (denom 127). */
        const int   qbits = sess->kv_sim_qbits;
        const float denom = qbits != 0 ? (float) ((1 << (qbits - 1)) - 1) : 127.0f;
        for (size_t t = 0; t < seq; t++) {
            const size_t slot = q_position + t;
            for (size_t h = 0; h < st->n_kv_heads; h++) {
                const float *k_row = k_src + t * row_elems + h * hd;
                const float *v_row = v_src + t * row_elems + h * hd;
                if (rot) {
                    memcpy(krot, k_row, hd * sizeof(float));
                    memcpy(vrot, v_row, hd * sizeof(float));
                    fwht_orthonormal(hd, krot);
                    fwht_orthonormal(hd, vrot);
                    k_row = krot;
                    v_row = vrot;
                }
                const float k_amax  = kv_row_absmax(hd, k_row);
                const float v_amax  = kv_row_absmax(hd, v_row);
                float       k_scale = k_amax / denom;
                if (k_scale == 0.0f) {
                    k_scale = 1.0f;
                }
                float v_scale = v_amax / denom;
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
    v->buffer_unmap(sess->scratch_k);
    v->buffer_unmap(sess->scratch_v);
    return GEIST_OK;
}

/* fwht_orthonormal on each of the first n_rows rows (hd floats) of `b`:
 * Q before a backend's attention kernel and its output after, where the
 * cache holds rotated rows (GEIST_KV_ROT). H is its own inverse. */
static void kv_rotate_rows(const struct geist_backend_vtbl *v,
                           size_t                           n_rows,
                           size_t                           hd,
                           struct geist_buffer             *b) {
    float *p = (float *) v->buffer_map(b);
    for (size_t r = 0; r < n_rows; r++) {
        fwht_orthonormal(hd, p + r * hd);
    }
    v->buffer_unmap(b);
}

enum geist_status transformer_kv_store_attention(struct transformer_layer_forward_ctx *ctx,
                                                 const struct geist_tensor            *t_q_3d,
                                                 struct geist_tensor                  *t_attn_3d) {

    struct transformer_arch_state         *st         = ctx->st;
    struct transformer_arch_session       *sess       = ctx->sess;
    struct transformer_layer_weights      *L          = ctx->L;
    struct geist_backend                  *be         = ctx->be;
    const struct geist_backend_vtbl       *v          = ctx->v;
    const struct geist_backend_primitives *prims      = ctx->prims;
    const size_t                           kv_len_now = ctx->kv_len_now;

    if (ctx->kv_kivi_enabled) {
        const float   *qp   = (const float *) v->buffer_map(sess->scratch_q);
        const uint8_t *kqp  = (const uint8_t *) v->buffer_map(ctx->k_kivi_q_buf);
        const uint8_t *vqp  = (const uint8_t *) v->buffer_map(ctx->v_kivi_q_buf);
        const float   *kscp = (const float *) v->buffer_map(ctx->k_kivi_scales_buf);
        const float   *kzep = (const float *) v->buffer_map(ctx->k_kivi_zeros_buf);
        const float   *vscp = (const float *) v->buffer_map(ctx->v_kivi_scales_buf);
        const float   *vzep = (const float *) v->buffer_map(ctx->v_kivi_zeros_buf);
        const float   *krp  = (const float *) v->buffer_map(ctx->k_residual_buf);
        const float   *vrp  = (const float *) v->buffer_map(ctx->v_residual_buf);
        float         *outp = (float *) v->buffer_map(sess->scratch_attn);
        float         *scores =
                (float *) frame_arena_alloc(&sess->scratch_arena, kv_len_now * sizeof(float), 16);
        if (scores == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_OOM,
                                    "transformer: scratch arena exhausted for "
                                    "KIVI scores (kv_len_now=%zu)",
                                    kv_len_now);
            return GEIST_E_OOM;
        }
        attention_kivi_via_buffers(ctx->seq,
                                   st->n_q_heads,
                                   ctx->hd,
                                   kv_len_now,
                                   st->n_kv_heads,
                                   ctx->q_position,
                                   L->sliding_window,
                                   sess->kivi_drained_count,
                                   KIVI_K_GROUP_SIZE,
                                   qp,
                                   kqp,
                                   kscp,
                                   kzep,
                                   vqp,
                                   vscp,
                                   vzep,
                                   krp,
                                   vrp,
                                   scores,
                                   outp);
        v->buffer_unmap(sess->scratch_q);
        v->buffer_unmap(ctx->k_kivi_q_buf);
        v->buffer_unmap(ctx->v_kivi_q_buf);
        v->buffer_unmap(ctx->k_kivi_scales_buf);
        v->buffer_unmap(ctx->k_kivi_zeros_buf);
        v->buffer_unmap(ctx->v_kivi_scales_buf);
        v->buffer_unmap(ctx->v_kivi_zeros_buf);
        v->buffer_unmap(ctx->k_residual_buf);
        v->buffer_unmap(ctx->v_residual_buf);
        v->buffer_unmap(sess->scratch_attn);
    } else if (sess->kv_int4_packed_enabled && ctx->P != nullptr && ctx->P->fuse_attn_kv_int4) {
        /* The backend's kernel (plan-bound) reads the packed cache through
         * views; the rotation of Q and of the output stays here. */
        const bool   rot    = sess->kv_rot_enabled && fwht_supported(ctx->hd) && ctx->hd <= 512;
        const size_t n_rows = ctx->seq * st->n_q_heads;
        if (rot) {
            kv_rotate_rows(v, n_rows, ctx->hd, sess->scratch_q);
        }
        const int64_t       n_kv   = (int64_t) kv_len_now;
        const int64_t       n_kh   = st->n_kv_heads;
        const int64_t       packed = (int64_t) (ctx->hd / 2);
        struct geist_tensor t_k    = view_3d_u8(ctx->k_cache_q8_buf, n_kv, n_kh, packed);
        struct geist_tensor t_v    = view_3d_u8(ctx->v_cache_q8_buf, n_kv, n_kh, packed);
        struct geist_tensor t_ks   = view_2d(ctx->k_cache_scale_buf, n_kv, n_kh);
        struct geist_tensor t_vs   = view_2d(ctx->v_cache_scale_buf, n_kv, n_kh);
        const struct geist_attention_kv_int4_args args = {.q              = t_q_3d,
                                                          .k              = &t_k,
                                                          .k_scale        = &t_ks,
                                                          .v              = &t_v,
                                                          .v_scale        = &t_vs,
                                                          .out            = t_attn_3d,
                                                          .q_offset       = ctx->q_position,
                                                          .sliding_window = L->sliding_window};
        const enum geist_status                   s    = ctx->fused->attention_kv_int4(be, &args);
        if (s != GEIST_OK) {
            return s;
        }
        if (rot) {
            kv_rotate_rows(v, n_rows, ctx->hd, sess->scratch_attn);
        }
    } else if (sess->kv_int4_packed_enabled) {
        float         *qp       = (float *) v->buffer_map(sess->scratch_q);
        const uint8_t *k_q4p    = (const uint8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        const uint8_t *v_q4p    = (const uint8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        const float   *k_scalep = (const float *) v->buffer_map(ctx->k_cache_scale_buf);
        const float   *v_scalep = (const float *) v->buffer_map(ctx->v_cache_scale_buf);
        float         *outp     = (float *) v->buffer_map(sess->scratch_attn);
        const bool     rot      = sess->kv_rot_enabled && fwht_supported(ctx->hd) && ctx->hd <= 512;
        const size_t   n_rows   = ctx->seq * st->n_q_heads;
        if (rot) {
            for (size_t r = 0; r < n_rows; r++) {
                fwht_orthonormal(ctx->hd, qp + r * ctx->hd);
            }
        }
        attention_int4_via_buffers(ctx->seq,
                                   st->n_q_heads,
                                   ctx->hd,
                                   kv_len_now,
                                   st->n_kv_heads,
                                   ctx->q_position,
                                   L->sliding_window,
                                   qp,
                                   k_q4p,
                                   k_scalep,
                                   v_q4p,
                                   v_scalep,
                                   outp);
        if (rot) {
            for (size_t r = 0; r < n_rows; r++) {
                fwht_orthonormal(ctx->hd, outp + r * ctx->hd);
            }
        }
        v->buffer_unmap(sess->scratch_q);
        v->buffer_unmap(ctx->k_cache_q8_buf);
        v->buffer_unmap(ctx->v_cache_q8_buf);
        v->buffer_unmap(ctx->k_cache_scale_buf);
        v->buffer_unmap(ctx->v_cache_scale_buf);
        v->buffer_unmap(sess->scratch_attn);
    } else if (ctx->kv_int8_enabled && ctx->P != nullptr && ctx->P->fuse_attn_kv_int8) {
        /* The backend's kernel (plan-bound) reads the cache through views;
         * the rotation of Q and of the output stays here. */
        const bool   rot    = sess->kv_rot_enabled && fwht_supported(ctx->hd) && ctx->hd <= 512;
        const size_t n_rows = ctx->seq * st->n_q_heads;
        if (rot) {
            kv_rotate_rows(v, n_rows, ctx->hd, sess->scratch_q);
        }
        const int64_t       n_kv = (int64_t) kv_len_now;
        const int64_t       n_kh = st->n_kv_heads;
        struct geist_tensor t_k  = view_3d_i8(ctx->k_cache_q8_buf, n_kv, n_kh, (int64_t) ctx->hd);
        struct geist_tensor t_v  = view_3d_i8(ctx->v_cache_q8_buf, n_kv, n_kh, (int64_t) ctx->hd);
        struct geist_tensor t_ks = view_2d(ctx->k_cache_scale_buf, n_kv, n_kh);
        struct geist_tensor t_vs = view_2d(ctx->v_cache_scale_buf, n_kv, n_kh);
        const struct geist_attention_kv_int8_args args = {.q              = t_q_3d,
                                                          .k              = &t_k,
                                                          .k_scale        = &t_ks,
                                                          .v              = &t_v,
                                                          .v_scale        = &t_vs,
                                                          .out            = t_attn_3d,
                                                          .q_offset       = ctx->q_position,
                                                          .sliding_window = L->sliding_window};
        const enum geist_status                   s    = ctx->fused->attention_kv_int8(be, &args);
        if (s != GEIST_OK) {
            return s;
        }
        if (rot) {
            kv_rotate_rows(v, n_rows, ctx->hd, sess->scratch_attn);
        }
    } else if (ctx->kv_int8_enabled) {
        float        *qp       = (float *) v->buffer_map(sess->scratch_q);
        const int8_t *k_q8p    = (const int8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        const int8_t *v_q8p    = (const int8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        const float  *k_scalep = (const float *) v->buffer_map(ctx->k_cache_scale_buf);
        const float  *v_scalep = (const float *) v->buffer_map(ctx->v_cache_scale_buf);
        float        *outp     = (float *) v->buffer_map(sess->scratch_attn);
        /* Issue #61: rotate Q by the same H used on K/V so QK scores are
         * unchanged; the kernel then quantizes rotated Q, and we rotate the
         * (V-rotated) output back below. H is its own inverse. */
        const bool   rot    = sess->kv_rot_enabled && fwht_supported(ctx->hd) && ctx->hd <= 512;
        const size_t n_rows = ctx->seq * st->n_q_heads;
        if (rot) {
            for (size_t r = 0; r < n_rows; r++) {
                fwht_orthonormal(ctx->hd, qp + r * ctx->hd);
            }
        }
        /* The softmax scratch is private per work item inside the kernel;
         * the arena holds the partial results of a split decode (sized for
         * them at session create). Without it decode runs unsplit. */
        size_t n_scratch = attention_int8_scratch_floats(st->n_q_heads, ctx->hd);
        float *scratch =
                (float *) frame_arena_alloc(&sess->scratch_arena, n_scratch * sizeof(float), 64);
        if (scratch == nullptr) {
            n_scratch = 0;
        }
        attention_int8_via_buffers(ctx->seq,
                                   st->n_q_heads,
                                   ctx->hd,
                                   kv_len_now,
                                   st->n_kv_heads,
                                   n_scratch,
                                   ctx->q_position,
                                   L->sliding_window,
                                   qp,
                                   k_q8p,
                                   k_scalep,
                                   v_q8p,
                                   v_scalep,
                                   outp,
                                   scratch);
        if (rot) {
            for (size_t r = 0; r < n_rows; r++) {
                fwht_orthonormal(ctx->hd, outp + r * ctx->hd);
            }
        }
        v->buffer_unmap(sess->scratch_q);
        v->buffer_unmap(ctx->k_cache_q8_buf);
        v->buffer_unmap(ctx->v_cache_q8_buf);
        v->buffer_unmap(ctx->k_cache_scale_buf);
        v->buffer_unmap(ctx->v_cache_scale_buf);
        v->buffer_unmap(sess->scratch_attn);
    } else {
        struct geist_tensor t_kcache_3d = ctx->kv_f16_enabled ? view_3d_f16(ctx->k_cache_buf,
                                                                            (int64_t) kv_len_now,
                                                                            st->n_kv_heads,
                                                                            (int64_t) ctx->hd)
                                                              : view_3d(ctx->k_cache_buf,
                                                                        (int64_t) kv_len_now,
                                                                        st->n_kv_heads,
                                                                        (int64_t) ctx->hd);
        struct geist_tensor t_vcache_3d = ctx->kv_f16_enabled ? view_3d_f16(ctx->v_cache_buf,
                                                                            (int64_t) kv_len_now,
                                                                            st->n_kv_heads,
                                                                            (int64_t) ctx->hd)
                                                              : view_3d(ctx->v_cache_buf,
                                                                        (int64_t) kv_len_now,
                                                                        st->n_kv_heads,
                                                                        (int64_t) ctx->hd);
        enum geist_status   s           = prims->attention(be,
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
