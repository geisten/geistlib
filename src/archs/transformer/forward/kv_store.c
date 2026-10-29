/*
 * src/archs/transformer/forward/kv_store.c - KV-cache append and
 * attention dispatch for transformer layers.
 *
 * The Hadamard rotation (GEIST_KV_ROT) and packed INT4 store K in the cache
 * after RoPE, rotated. This is safe only because geist
 * never re-bases cached positions — the sliding window merely masks and does
 * not re-RoPE the cache. Any context-shift implementation must unpack,
 * un-rotate, apply RoPE again, rotate and repack cached rows; otherwise the
 * cache is corrupted (see #71).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../forward.h"
#include <geist_types.h>

#include "fwht.h"
#include "int4_kv.h"
#include "kivi.h"
#include "par.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* max |x[i]|; 0 for a row of zeros. Taken on the bit patterns: with the
 * sign cleared, the unsigned integers order as the values do (NaN aside),
 * and an integer max vectorizes under any floating-point flags, where
 * clang keeps a float compare-and-select scalar unless it may assume no
 * NaNs (not under -fno-finite-math-only). */
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

/* Rows from which a rotation or a quantizing append runs across threads.
 * Fewer, as in decode (one token: n_q_heads rows to rotate, n_kv_heads to
 * append), are a microsecond of work, less than starting the threads. */
constexpr size_t KV_PAR_MIN_ROWS = 64;

/* fwht_orthonormal on rows [r0, r1) of hd floats at `p`. */
struct kv_rot_job {
    size_t hd;
    float *p;
};

static void kv_rot_items(void *ctx, size_t r0, size_t r1) {
    const struct kv_rot_job *job = ctx;
    for (size_t r = r0; r < r1; r++) {
        fwht_orthonormal(job->hd, job->p + r * job->hd);
    }
}

/* fwht_orthonormal on each of n_rows rows of hd floats at p; each row on
 * its own, so the result is the same bytes on any number of threads. */
static void kv_rotate_host(size_t n_rows, size_t hd, float *p) {
    struct kv_rot_job job = {.hd = hd, .p = p};
    if (n_rows < KV_PAR_MIN_ROWS) {
        kv_rot_items(&job, 0, n_rows);
    } else {
        geist_par_for(n_rows, kv_rot_items, &job);
    }
}

/* The INT8 (and INT4) append of rows (token t, KV head h), item
 * t * n_kv_heads + h, into cache slot q_position + t. */
struct kv_quant_job {
    size_t       hd, n_kv_heads, row_elems, q_position;
    bool         rot, v_int4;
    float        k_den, v_den;
    const float *k_src;
    const float *v_src;
    int8_t      *k_dst;
    int8_t      *v_dst;
    float       *k_sca;
    float       *v_sca;
};

static void kv_quant_items(void *ctx, size_t i0, size_t i1) {
    const struct kv_quant_job *job = ctx;
    const size_t               hd  = job->hd;
    float                      krot[512];
    float                      vrot[512];
    for (size_t it = i0; it < i1; it++) {
        const size_t t     = it / job->n_kv_heads;
        const size_t h     = it % job->n_kv_heads;
        const size_t slot  = job->q_position + t;
        const float *k_row = job->k_src + t * job->row_elems + h * hd;
        const float *v_row = job->v_src + t * job->row_elems + h * hd;
        if (job->rot) {
            memcpy(krot, k_row, hd * sizeof(float));
            memcpy(vrot, v_row, hd * sizeof(float));
            fwht_orthonormal(hd, krot);
            fwht_orthonormal(hd, vrot);
            k_row = krot;
            v_row = vrot;
        }
        float k_scale = kv_row_absmax(hd, k_row) / job->k_den;
        if (k_scale == 0.0f) {
            k_scale = 1.0f;
        }
        float v_scale = kv_row_absmax(hd, v_row) / job->v_den;
        if (v_scale == 0.0f) {
            v_scale = 1.0f;
        }
        const float k_inv  = 1.0f / k_scale;
        const float v_inv  = 1.0f / v_scale;
        int8_t     *k_drow = job->k_dst + slot * job->row_elems + h * hd;
        for (size_t i = 0; i < hd; i++) {
            k_drow[i] = (int8_t) lrintf(k_row[i] * k_inv);
        }
        if (job->v_int4) {
            int4_pack_row(hd,
                          v_row,
                          v_inv,
                          (uint8_t *) job->v_dst + (slot * job->row_elems + h * hd) / 2);
        } else {
            int8_t *v_drow = job->v_dst + slot * job->row_elems + h * hd;
            for (size_t i = 0; i < hd; i++) {
                v_drow[i] = (int8_t) lrintf(v_row[i] * v_inv);
            }
        }
        job->k_sca[slot * job->n_kv_heads + h] = k_scale;
        job->v_sca[slot * job->n_kv_heads + h] = v_scale;
    }
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
        /* The capability is decided at plan build, so a non-OK result here
         * is a device error, not "this backend cannot do it" (#352). */
        return cs;
    }

    const float *k_src = (const float *) v->buffer_map(sess->scratch_k);
    const float *v_src = (const float *) v->buffer_map(sess->scratch_v);
    if (k_src == nullptr || v_src == nullptr) {
        return GEIST_E_BACKEND; /* the backend said why */
    }
    if (ctx->kv_kivi_enabled) {
        float *k_res = (float *) v->buffer_map(ctx->k_residual_buf);
        float *v_res = (float *) v->buffer_map(ctx->v_residual_buf);
        if (k_res == nullptr || v_res == nullptr) {
            return GEIST_E_BACKEND;
        }
        const size_t row_elems = kv_out;
        for (size_t t = 0; t < seq; t++) {
            const size_t res_idx = (q_position + t) - sess->kivi_drained_count;
            memcpy(k_res + res_idx * row_elems, k_src + t * row_elems, row_elems * sizeof(float));
            memcpy(v_res + res_idx * row_elems, v_src + t * row_elems, row_elems * sizeof(float));
        }
        v->buffer_unmap(ctx->k_residual_buf);
        v->buffer_unmap(ctx->v_residual_buf);
    } else if (ctx->kv_int8_enabled) {
        /* INT8, and INT4: K at INT8 as well, V packed two values a byte at
         * amax / 7 (int4_kv.h). K is what attention is sharp on: QK-norm
         * families (Qwen3, Gemma) carry outlier channels that a 4-bit K
         * grid cannot hold even rotated, while a 4-bit V costs next to
         * nothing (#53). */
        int8_t *k_dst = (int8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        int8_t *v_dst = (int8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        float  *k_sca = (float *) v->buffer_map(ctx->k_cache_scale_buf);
        float  *v_sca = (float *) v->buffer_map(ctx->v_cache_scale_buf);
        if (k_dst == nullptr || v_dst == nullptr || k_sca == nullptr || v_sca == nullptr) {
            return GEIST_E_BACKEND;
        }
        /* Rotate each K/V head row before quantizing (#61). Q is
         * rotated symmetrically at attention time (kv_store_attention).
         * Low-bit quality-sim quantizes on an N-bit grid
         * (amax / (2^(N-1)-1)); the int8 container then holds values in
         * [-(2^(N-1)-1), +...]. Rounding stays in range (|x| <= amax), so no
         * clamp is needed. qbits==0 is the native 8-bit path (denom 127).
         * Every row is quantized on its own, so the threads change no
         * byte. */
        const int           qbits = sess->kv_sim_qbits;
        const float         denom = qbits != 0 ? (float) ((1 << (qbits - 1)) - 1) : 127.0f;
        struct kv_quant_job job   = {
                .hd         = hd,
                .n_kv_heads = st->n_kv_heads,
                .row_elems  = kv_out,
                .q_position = q_position,
                .rot        = sess->kv_rot_enabled && fwht_supported(hd) && hd <= 512,
                .v_int4     = sess->kv_int4_packed_enabled,
                .k_den      = denom,
                .v_den      = sess->kv_int4_packed_enabled ? 7.0f : denom,
                .k_src      = k_src,
                .v_src      = v_src,
                .k_dst      = k_dst,
                .v_dst      = v_dst,
                .k_sca      = k_sca,
                .v_sca      = v_sca,
        };
        const size_t n_items = seq * st->n_kv_heads;
        if (n_items < KV_PAR_MIN_ROWS) {
            kv_quant_items(&job, 0, n_items);
        } else {
            geist_par_for(n_items, kv_quant_items, &job);
        }
        v->buffer_unmap(ctx->k_cache_q8_buf);
        v->buffer_unmap(ctx->v_cache_q8_buf);
        v->buffer_unmap(ctx->k_cache_scale_buf);
        v->buffer_unmap(ctx->v_cache_scale_buf);
    } else {
        uint8_t *k_dst = (uint8_t *) v->buffer_map(ctx->k_cache_buf);
        uint8_t *v_dst = (uint8_t *) v->buffer_map(ctx->v_cache_buf);
        if (k_dst == nullptr || v_dst == nullptr) {
            return GEIST_E_BACKEND;
        }
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
[[nodiscard]] static enum geist_status kv_rotate_rows(const struct geist_backend_vtbl *v,
                                                      size_t                           n_rows,
                                                      size_t                           hd,
                                                      struct geist_buffer             *b) {
    float *p = (float *) v->buffer_map(b);
    if (p == nullptr) {
        return GEIST_E_BACKEND; /* the backend said why */
    }
    kv_rotate_host(n_rows, hd, p);
    v->buffer_unmap(b);
    return GEIST_OK;
}

/* Batched decode (#26): transformer_kv_store_append's F16 and device-copy
 * paths for one row taken at byte `off` of k_src / v_src (a carrier's
 * rows) instead of the session's scratch_k / scratch_v. UNSUPPORTED for the
 * quantised caches and the host path. */
enum geist_status transformer_kv_store_append_row(struct transformer_layer_forward_ctx *ctx,
                                                  struct geist_buffer                  *k_src,
                                                  struct geist_buffer                  *v_src,
                                                  size_t                                off) {
    struct transformer_arch_state *st = ctx->st;
    const size_t                   qp = ctx->q_position;
    if (ctx->seq != 1 || ctx->kv_kivi_enabled || ctx->kv_int8_enabled) {
        return GEIST_E_UNSUPPORTED;
    }
    if (ctx->kv_f16_enabled) {
        struct geist_tensor t_k = view_3d(k_src, 1, st->n_kv_heads, (int64_t) ctx->hd);
        struct geist_tensor t_v = view_3d(v_src, 1, st->n_kv_heads, (int64_t) ctx->hd);
        t_k.offset = t_v.offset  = off;
        struct geist_tensor t_kc = view_3d_f16(
                ctx->k_cache_buf, (int64_t) (qp + 1), st->n_kv_heads, (int64_t) ctx->hd);
        struct geist_tensor t_vc = view_3d_f16(
                ctx->v_cache_buf, (int64_t) (qp + 1), st->n_kv_heads, (int64_t) ctx->hd);
        return ctx->fused->kv_append_f16(ctx->be, &t_k, &t_v, qp, &t_kc, &t_vc);
    }
    if (!st->model_fusions.backend_buffer_copy) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t      row_bytes = ctx->kv_out * sizeof(float);
    enum geist_status s =
            ctx->v->buffer_copy(ctx->k_cache_buf, qp * row_bytes, k_src, off, row_bytes);
    if (s == GEIST_OK) {
        s = ctx->v->buffer_copy(ctx->v_cache_buf, qp * row_bytes, v_src, off, row_bytes);
    }
    return s;
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
        if (qp == nullptr || kqp == nullptr || vqp == nullptr || kscp == nullptr ||
            kzep == nullptr || vscp == nullptr || vzep == nullptr || krp == nullptr ||
            vrp == nullptr || outp == nullptr) {
            return GEIST_E_BACKEND; /* the backend said why */
        }
        float *scores =
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
    } else if (sess->kv_int4_packed_enabled && ctx->P->fuse_attn_kv_int4) {
        /* The backend's kernel (plan-bound) reads the packed cache through
         * views; the rotation of Q and of the output stays here. */
        const bool   rot    = sess->kv_rot_enabled && fwht_supported(ctx->hd) && ctx->hd <= 512;
        const size_t n_rows = ctx->seq * st->n_q_heads;
        if (rot) {
            const enum geist_status rs = kv_rotate_rows(v, n_rows, ctx->hd, sess->scratch_q);
            if (rs != GEIST_OK) {
                return rs;
            }
        }
        const int64_t       n_kv   = (int64_t) kv_len_now;
        const int64_t       n_kh   = st->n_kv_heads;
        const int64_t       packed = (int64_t) (ctx->hd / 2);
        struct geist_tensor t_k    = view_3d_i8(ctx->k_cache_q8_buf, n_kv, n_kh, (int64_t) ctx->hd);
        struct geist_tensor t_v    = view_3d_u8(ctx->v_cache_q8_buf, n_kv, n_kh, packed);
        struct geist_tensor t_ks   = view_2d(ctx->k_cache_scale_buf, n_kv, n_kh);
        struct geist_tensor t_vs   = view_2d(ctx->v_cache_scale_buf, n_kv, n_kh);
        const struct geist_attention_kv_args args = {.q              = t_q_3d,
                                                     .k              = &t_k,
                                                     .k_scale        = &t_ks,
                                                     .v              = &t_v,
                                                     .v_scale        = &t_vs,
                                                     .out            = t_attn_3d,
                                                     .q_offset       = ctx->q_position,
                                                     .sliding_window = L->sliding_window};
        const enum geist_status              s    = ctx->fused->attention_kv_int4(be, &args);
        if (s != GEIST_OK) {
            return s;
        }
        if (rot) {
            return kv_rotate_rows(v, n_rows, ctx->hd, sess->scratch_attn);
        }
    } else if (sess->kv_int4_packed_enabled) {
        float         *qp       = (float *) v->buffer_map(sess->scratch_q);
        const int8_t  *k_q8p    = (const int8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        const uint8_t *v_q4p    = (const uint8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        const float   *k_scalep = (const float *) v->buffer_map(ctx->k_cache_scale_buf);
        const float   *v_scalep = (const float *) v->buffer_map(ctx->v_cache_scale_buf);
        float         *outp     = (float *) v->buffer_map(sess->scratch_attn);
        if (qp == nullptr || k_q8p == nullptr || v_q4p == nullptr || k_scalep == nullptr ||
            v_scalep == nullptr || outp == nullptr) {
            return GEIST_E_BACKEND; /* the backend said why */
        }
        const bool   rot    = sess->kv_rot_enabled && fwht_supported(ctx->hd) && ctx->hd <= 512;
        const size_t n_rows = ctx->seq * st->n_q_heads;
        if (rot) {
            kv_rotate_host(n_rows, ctx->hd, qp);
        }
        attention_int4_via_buffers(ctx->seq,
                                   st->n_q_heads,
                                   ctx->hd,
                                   kv_len_now,
                                   st->n_kv_heads,
                                   ctx->q_position,
                                   L->sliding_window,
                                   qp,
                                   k_q8p,
                                   k_scalep,
                                   v_q4p,
                                   v_scalep,
                                   outp);
        if (rot) {
            kv_rotate_host(n_rows, ctx->hd, outp);
        }
        v->buffer_unmap(sess->scratch_q);
        v->buffer_unmap(ctx->k_cache_q8_buf);
        v->buffer_unmap(ctx->v_cache_q8_buf);
        v->buffer_unmap(ctx->k_cache_scale_buf);
        v->buffer_unmap(ctx->v_cache_scale_buf);
        v->buffer_unmap(sess->scratch_attn);
    } else if (ctx->kv_int8_enabled && ctx->P->fuse_attn_kv_int8) {
        /* The backend's kernel (plan-bound) reads the cache through views;
         * the rotation of Q and of the output stays here. */
        const bool   rot    = sess->kv_rot_enabled && fwht_supported(ctx->hd) && ctx->hd <= 512;
        const size_t n_rows = ctx->seq * st->n_q_heads;
        if (rot) {
            const enum geist_status rs = kv_rotate_rows(v, n_rows, ctx->hd, sess->scratch_q);
            if (rs != GEIST_OK) {
                return rs;
            }
        }
        const int64_t       n_kv = (int64_t) kv_len_now;
        const int64_t       n_kh = st->n_kv_heads;
        struct geist_tensor t_k  = view_3d_i8(ctx->k_cache_q8_buf, n_kv, n_kh, (int64_t) ctx->hd);
        struct geist_tensor t_v  = view_3d_i8(ctx->v_cache_q8_buf, n_kv, n_kh, (int64_t) ctx->hd);
        struct geist_tensor t_ks = view_2d(ctx->k_cache_scale_buf, n_kv, n_kh);
        struct geist_tensor t_vs = view_2d(ctx->v_cache_scale_buf, n_kv, n_kh);
        const struct geist_attention_kv_args args = {.q              = t_q_3d,
                                                     .k              = &t_k,
                                                     .k_scale        = &t_ks,
                                                     .v              = &t_v,
                                                     .v_scale        = &t_vs,
                                                     .out            = t_attn_3d,
                                                     .q_offset       = ctx->q_position,
                                                     .sliding_window = L->sliding_window};
        const enum geist_status              s    = ctx->fused->attention_kv_int8(be, &args);
        if (s != GEIST_OK) {
            return s;
        }
        if (rot) {
            return kv_rotate_rows(v, n_rows, ctx->hd, sess->scratch_attn);
        }
    } else if (ctx->kv_int8_enabled) {
        float        *qp       = (float *) v->buffer_map(sess->scratch_q);
        const int8_t *k_q8p    = (const int8_t *) v->buffer_map(ctx->k_cache_q8_buf);
        const int8_t *v_q8p    = (const int8_t *) v->buffer_map(ctx->v_cache_q8_buf);
        const float  *k_scalep = (const float *) v->buffer_map(ctx->k_cache_scale_buf);
        const float  *v_scalep = (const float *) v->buffer_map(ctx->v_cache_scale_buf);
        float        *outp     = (float *) v->buffer_map(sess->scratch_attn);
        if (qp == nullptr || k_q8p == nullptr || v_q8p == nullptr || k_scalep == nullptr ||
            v_scalep == nullptr || outp == nullptr) {
            return GEIST_E_BACKEND; /* the backend said why */
        }
        /* Rotate Q by the same H used on K/V so QK scores are
         * unchanged; the kernel then quantizes rotated Q, and we rotate the
         * (V-rotated) output back below. H is its own inverse. */
        const bool   rot    = sess->kv_rot_enabled && fwht_supported(ctx->hd) && ctx->hd <= 512;
        const size_t n_rows = ctx->seq * st->n_q_heads;
        if (rot) {
            kv_rotate_host(n_rows, ctx->hd, qp);
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
            kv_rotate_host(n_rows, ctx->hd, outp);
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
