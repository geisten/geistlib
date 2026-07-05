/*
 * src/archs/transformer/forward/internal.h — file-private declarations
 * shared by the forward/{attention,layer,layer_attn,layer_ffn,layer_ple,
 * probes,head,step}.c translation units.
 *
 * Layer: ARCHITECTURE (private). Not part of the public ABI.
 *
 * Contains:
 *   1. Tiny tensor-view builders as `static inline` so each TU inlines
 *      them locally without an extra symbol or call.
 *   2. Per-token activation-quant helpers (`static inline`).
 *   3. extern declarations for the cross-TU attention and dequant
 *      helpers that need to be visible across the forward/ TUs
 *      without escaping to the public arch ABI.
 */
#pragma once

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "forward/internal.h is a private architecture-layer header"
#endif

#include "../arch_state.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ---- Per-layer forward context ---------------------------------------- *
 *
 * Private, per-call bundle used to keep transformer_forward_one_layer()
 * as orchestration while allowing attention/FFN/PLE stages to live in
 * separate translation units. Model-family switches come from the layer
 * exec plan when available; session-dependent KV representation remains
 * read from st->sess until the architecture has a per-session plan.
 */
struct transformer_layer_forward_ctx {
    struct transformer_arch_state              *st;
    struct geist_backend                       *be;
    const struct geist_backend_vtbl            *v;
    struct transformer_layer_weights           *L;
    const struct transformer_layer_exec_plan   *P;
    const struct transformer_session_exec_plan *SP;

    int                  layer_idx;
    size_t               q_position;
    size_t               seq;
    bool                 advance_kv;
    struct geist_buffer *h_in_buf;
    struct geist_buffer *per_layer_input_buf;
    struct geist_buffer *h_out_buf;

    int                            kv_src;
    bool                           compute_kv;
    bool                           apply_bitnet_input_quant;
    bool                           apply_sub_ln;
    bool                           apply_gemma_attn_norms;
    bool                           rope_interleaved;
    bool                           apply_ple;
    bool                           kv_int8_enabled;
    bool                           kv_kivi_enabled;
    bool                           kv_f16_enabled;
    enum geist_ffn_activation_kind ffn_activation;

    float   eps;
    size_t  hd;
    size_t  q_out;
    size_t  kv_out;
    size_t  inter;
    int64_t SEQ;
    size_t  kv_len_now;

    struct geist_buffer *k_cache_buf;
    struct geist_buffer *v_cache_buf;
    struct geist_buffer *k_cache_q8_buf;
    struct geist_buffer *v_cache_q8_buf;
    struct geist_buffer *k_cache_scale_buf;
    struct geist_buffer *v_cache_scale_buf;
    struct geist_buffer *k_kivi_q_buf;
    struct geist_buffer *v_kivi_q_buf;
    struct geist_buffer *k_kivi_scales_buf;
    struct geist_buffer *k_kivi_zeros_buf;
    struct geist_buffer *v_kivi_scales_buf;
    struct geist_buffer *v_kivi_zeros_buf;
    struct geist_buffer *k_residual_buf;
    struct geist_buffer *v_residual_buf;
};

/* ---- Tensor view builders --------------------------------------------- *
 *
 * Build a geist_tensor describing a row-major F32 DENSE view onto the
 * given backend buffer. All views start at offset 0 except view_2d_at,
 * which exposes an explicit byte offset (used by attention scratch
 * partitioning).
 */
static inline struct geist_tensor view_1d(struct geist_buffer *b, int64_t n) {
    struct geist_tensor t = {
            .buffer = b,
            .offset = 0,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 1,
            .shape  = {n, 0, 0, 0, 0, 0, 0, 0},
            .stride = {1, 0, 0, 0, 0, 0, 0, 0},
    };
    return t;
}

static inline struct geist_tensor
view_2d_at(struct geist_buffer *b, size_t off_bytes, int64_t s0, int64_t s1) {
    struct geist_tensor t = {
            .buffer = b,
            .offset = off_bytes,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {s0, s1, 0, 0, 0, 0, 0, 0},
            .stride = {s1, 1, 0, 0, 0, 0, 0, 0},
    };
    return t;
}

static inline struct geist_tensor view_2d(struct geist_buffer *b, int64_t s0, int64_t s1) {
    return view_2d_at(b, 0, s0, s1);
}

static inline struct geist_tensor
view_3d(struct geist_buffer *b, int64_t s0, int64_t s1, int64_t s2) {
    struct geist_tensor t = {
            .buffer = b,
            .offset = 0,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 3,
            .shape  = {s0, s1, s2, 0, 0, 0, 0, 0},
            .stride = {s1 * s2, s2, 1, 0, 0, 0, 0, 0},
    };
    return t;
}

/* F16 variant — used for the half-float KV cache views. */
static inline struct geist_tensor
view_3d_f16(struct geist_buffer *b, int64_t s0, int64_t s1, int64_t s2) {
    struct geist_tensor t = view_3d(b, s0, s1, s2);
    t.dtype               = GEIST_DTYPE_F16;
    return t;
}

/* ---- Per-row activation-quant helpers --------------------------------- *
 *
 * AWQ inverse scale: y[t, j] *= scale[j]. No-op if scale == nullptr.
 * BitNet INT8 absmax round-trip: per-row [-128..127] quant + dequant
 *   in place, mirroring HF 1bitLLM/bitnet_b1_58 utils_quant.py.
 *
 * Both run on caller-mapped buffers. Hot path; kept `static inline` so
 * the compiler can fold them into the layer body.
 */
static inline void apply_per_channel_inv_scale_inplace(const struct geist_backend_vtbl *v,
                                                       struct geist_buffer             *buf,
                                                       size_t                           seq,
                                                       size_t                           n,
                                                       const float                     *scale) {
    if (scale == nullptr)
        return;
    float *p = (float *) v->buffer_map(buf);
    for (size_t t = 0; t < seq; t++) {
        float *row = p + t * n;
        for (size_t j = 0; j < n; j++)
            row[j] *= scale[j];
    }
    v->buffer_unmap(buf);
}

static inline void apply_bitnet_input_quant_inplace(const struct geist_backend_vtbl *v,
                                                    struct geist_buffer             *buf,
                                                    size_t                           seq,
                                                    size_t                           n) {
    float *p = (float *) v->buffer_map(buf);
    for (size_t t = 0; t < seq; t++) {
        float *row    = p + t * n;
        float  maxabs = 1e-5f;
        for (size_t j = 0; j < n; j++) {
            float a = row[j] < 0.0f ? -row[j] : row[j];
            if (a > maxabs)
                maxabs = a;
        }
        const float s     = 127.0f / maxabs;
        const float inv_s = 1.0f / s;
        for (size_t j = 0; j < n; j++) {
            float q = row[j] * s;
            q       = q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q);
            int qi  = (int) (q < 0.0f ? q - 0.5f : q + 0.5f);
            if (qi > 127)
                qi = 127;
            if (qi < -128)
                qi = -128;
            row[j] = (float) qi * inv_s;
        }
    }
    v->buffer_unmap(buf);
}

/* ---- Cross-TU function declarations ----------------------------------- *
 *
 * forward/attention.c — both static-helper bodies promoted to extern so
 * forward/layer.c can call them via the same names they had pre-split.
 */
void kivi_drain_one_layer(float   *k_residual,
                          float   *v_residual,
                          uint8_t *k_q4,
                          uint8_t *v_q4,
                          float   *k_scales,
                          float   *k_zeros,
                          float   *v_scales,
                          float   *v_zeros,
                          size_t   drained_count,
                          size_t   residual_count,
                          size_t   R,
                          size_t   head_dim,
                          size_t   n_kv_heads);

void attention_kivi_via_buffers(const float   *q,
                                size_t         n_q,
                                size_t         n_q_heads,
                                size_t         head_dim,
                                const uint8_t *k_q4,
                                const float   *k_scales,
                                const float   *k_zeros,
                                const uint8_t *v_q4,
                                const float   *v_scales,
                                const float   *v_zeros,
                                const float   *k_residual,
                                const float   *v_residual,
                                size_t         n_kv,
                                size_t         n_kv_heads,
                                size_t         q_offset,
                                size_t         sliding_window,
                                size_t         drained_count,
                                size_t         R,
                                float         *scores,
                                float         *out);

/* Parallelized over query positions; `scores` scratch is now private per
 * iteration, so it is no longer a caller-supplied buffer. */
void attention_int8_via_buffers(const float  *q,
                                size_t        n_q,
                                size_t        n_q_heads,
                                size_t        head_dim,
                                const int8_t *k_q8,
                                const float  *k_scale,
                                const int8_t *v_q8,
                                const float  *v_scale,
                                size_t        n_kv,
                                size_t        n_kv_heads,
                                size_t        q_offset,
                                size_t        sliding_window,
                                float        *out);

/* Packed-INT4 variant (issue #61): k_q4/v_q4 hold two 4-bit values per byte
 * (head_dim/2 bytes per row); otherwise identical to the INT8 kernel. */
void attention_int4_via_buffers(const float   *q,
                                size_t         n_q,
                                size_t         n_q_heads,
                                size_t         head_dim,
                                const uint8_t *k_q4,
                                const float   *k_scale,
                                const uint8_t *v_q4,
                                const float   *v_scale,
                                size_t         n_kv,
                                size_t         n_kv_heads,
                                size_t         q_offset,
                                size_t         sliding_window,
                                float         *out);

/* forward/layer_attn.c */
[[nodiscard]] enum geist_status
transformer_layer_run_attention_block(struct transformer_layer_forward_ctx *ctx);

/* forward/kv_store.c */
[[nodiscard]] enum geist_status
transformer_kv_store_append(struct transformer_layer_forward_ctx *ctx);
[[nodiscard]] enum geist_status
transformer_kv_store_attention(struct transformer_layer_forward_ctx *ctx,
                               const struct geist_tensor            *t_q_3d,
                               struct geist_tensor                  *t_attn_3d);

/* forward/layer_ffn.c */
[[nodiscard]] enum geist_status
transformer_layer_run_ffn_block(struct transformer_layer_forward_ctx *ctx);

/* forward/layer_ple.c */
[[nodiscard]] enum geist_status
transformer_layer_run_ple_or_copy(struct transformer_layer_forward_ctx *ctx);
void transformer_layer_scale_output(struct transformer_layer_forward_ctx *ctx);

/* forward/probes.c */
void transformer_probe_ffn_sparsity(const struct geist_backend_vtbl *v,
                                    bool                             enabled,
                                    int                              layer_idx,
                                    struct geist_buffer             *buf,
                                    size_t                           n_elems);

/* forward/layer.c — exported helper used across forward/. */
[[nodiscard]] enum geist_status linear_w_or_legacy(struct geist_backend            *be,
                                                   const struct geist_backend_vtbl *v,
                                                   struct geist_buffer             *x_buf,
                                                   struct geist_buffer             *y_buf,
                                                   const struct geist_weight       *w,
                                                   size_t                           seq,
                                                   const struct geist_tensor       *t_x,
                                                   const struct geist_tensor       *t_w,
                                                   struct geist_tensor             *t_y);
[[nodiscard]] enum geist_status linear_w_scaled_input_or_legacy(struct geist_backend            *be,
                                                                const struct geist_backend_vtbl *v,
                                                                struct geist_buffer       *x_buf,
                                                                struct geist_buffer       *y_buf,
                                                                const struct geist_weight *w,
                                                                size_t                     seq,
                                                                size_t                     scale_n,
                                                                const float               *scale,
                                                                const struct geist_tensor *t_x,
                                                                const struct geist_tensor *t_w,
                                                                struct geist_tensor       *t_y);
[[nodiscard]] enum geist_status linear_w_pair_or_legacy(struct geist_backend            *be,
                                                        const struct geist_backend_vtbl *v,
                                                        struct geist_buffer             *x_buf,
                                                        struct geist_buffer             *y0_buf,
                                                        struct geist_buffer             *y1_buf,
                                                        const struct geist_weight       *w0,
                                                        const struct geist_weight       *w1,
                                                        size_t                           seq,
                                                        const struct geist_tensor       *t_x,
                                                        const struct geist_tensor       *t_w0,
                                                        const struct geist_tensor       *t_w1,
                                                        struct geist_tensor             *t_y0,
                                                        struct geist_tensor             *t_y1);
[[nodiscard]] enum geist_status linear_w_triple_or_legacy(struct geist_backend            *be,
                                                          const struct geist_backend_vtbl *v,
                                                          struct geist_buffer             *x_buf,
                                                          struct geist_buffer             *y0_buf,
                                                          struct geist_buffer             *y1_buf,
                                                          struct geist_buffer             *y2_buf,
                                                          const struct geist_weight       *w0,
                                                          const struct geist_weight       *w1,
                                                          const struct geist_weight       *w2,
                                                          size_t                           seq,
                                                          const struct geist_tensor       *t_x,
                                                          const struct geist_tensor       *t_w0,
                                                          const struct geist_tensor       *t_w1,
                                                          const struct geist_tensor       *t_w2,
                                                          struct geist_tensor             *t_y0,
                                                          struct geist_tensor             *t_y1,
                                                          struct geist_tensor             *t_y2);

/* finalize_logits_* and dequant_one_row are declared in ../forward.h —
 * the public arch-internal header. Don't re-declare here. */
