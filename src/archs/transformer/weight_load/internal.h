/*
 * src/archs/transformer/weight_load/internal.h — file-private
 * declarations shared by the weight_load/ TUs.
 *
 * Layer: ARCHITECTURE (private). Not part of the public ABI.
 *
 * Contains the GGUF dtype map entry struct, small inline view + arena
 * helpers, and extern decls for the larger cross-TU primitives.
 */
#pragma once

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "weight_load/internal.h is a private architecture-layer header"
#endif

#include "../arch_state.h"

#include "gguf_reader.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- GGUF dtype mapping ----------------------------------------------- */

struct dtype_map_entry {
    enum geist_dtype  dtype;
    enum geist_layout layout;
    bool              supported;
};

/* weight_load/dtype_map.c */
struct dtype_map_entry map_gguf_dtype(gguf_dtype_t gd);

/* ---- Tensor view builders (inline; tiny pure functions) --------------- */

static inline struct geist_tensor make_view_2d(struct geist_buffer *buf,
                                               enum geist_dtype     dtype,
                                               enum geist_layout    layout,
                                               int64_t              shape0,
                                               int64_t              shape1) {
    struct geist_tensor t = {
            .buffer = buf,
            .offset = 0,
            .dtype  = dtype,
            .layout = layout,
            .ndim   = 2,
            .shape  = {shape0, shape1, 0, 0, 0, 0, 0, 0},
    };
    if (layout == GEIST_LAYOUT_DENSE) {
        t.stride[0] = shape1;
        t.stride[1] = 1;
    }
    return t;
}

static inline struct geist_tensor make_view_1d(struct geist_buffer *buf,
                                               enum geist_dtype     dtype,
                                               enum geist_layout    layout,
                                               int64_t              shape0) {
    struct geist_tensor t = {
            .buffer = buf,
            .offset = 0,
            .dtype  = dtype,
            .layout = layout,
            .ndim   = 1,
            .shape  = {shape0, 0, 0, 0, 0, 0, 0, 0},
    };
    if (layout == GEIST_LAYOUT_DENSE) {
        t.stride[0] = 1;
    }
    return t;
}

/* ---- Storage intent --------------------------------------------------- *
 *
 * What the model does with a tensor, which decides where its bytes live.
 * The loader knows it from the call site; it is the only input to the
 * arena decision (weight_off_arena), so a new tensor needs no rule of its
 * own (#468).
 */
enum weight_storage {
    /* Kernels bind the buffer itself: norm gammas, biases, conv taps, the
     * F32 widening of a small half-precision matrix. In the arena in β
     * mode. */
    WEIGHT_BIND,
    /* A matrix behind a geist_weight: resolve_weight runs on it, and the
     * backend reads it through what that returns. */
    WEIGHT_LINEAR,
    /* A table read one row per token (an untied token_embd, the PLE table).
     * On a weights_device_copy backend the host gathers its rows. */
    WEIGHT_LOOKUP,
};

/* True when a tensor of this intent stays out of the β-mode arena: the
 * backend copies its matrices to the device itself
 * (caps.weights_device_copy) and gathers lookup tables on the host, so
 * neither needs a bindable host copy and the GGUF mmap page range is
 * aliased instead of duplicated. */
static inline bool weight_off_arena(const struct geist_backend *be, enum weight_storage storage) {
    return be->desc->caps.weights_device_copy && storage != WEIGHT_BIND;
}

/* ---- Weight arena allocator ------------------------------------------ *
 *
 * A 64-byte-aligned (or `align`-aligned) slice of the weight arena, bump-
 * allocated from its newest chunk; a request that does not fit opens a
 * new chunk (weight_load/tensor_views.c). nullptr, with the backend error
 * set, when that allocation fails. Only in β mode (st->weight_arena set).
 */
[[nodiscard]] void *arena_alloc(struct transformer_arch_state *st, size_t bytes, size_t align);

/* ---- Cross-TU functions ----------------------------------------------- *
 *
 * weight_load/tensor_views.c — GGUF tensor → backend buffer staging.
 */
/* Stage a norm vector (an rmsnorm gamma) as an F32 backend buffer,
 * converting when the GGUF stores it narrower. Kernels read gammas as F32
 * only. */
[[nodiscard]] enum geist_status load_norm_to_f32_buffer(struct transformer_arch_state *st,
                                                        struct gguf_ctx               *gguf,
                                                        const char                    *name,
                                                        size_t                expected_elems,
                                                        struct geist_buffer **out_buf);

/* Copy n_rows rows of row_bytes from src to dst, each head of head_dim rows
 * reordered from interleaved to half-split RoPE pairs: row 2i to i, row
 * 2i + 1 to i + head_dim/2. head_dim is even and divides n_rows; src and
 * dst do not overlap. */
void permute_rope_rows(
        size_t n_rows, size_t row_bytes, size_t head_dim, const uint8_t *src, uint8_t *dst);

/* load_norm_to_f32_buffer for a 2D tensor, with the rows of a llama-family
 * attn_q / attn_k permuted as load_tensor_to_buffer_rope_il does when
 * rope_il_head_dim > 0: the F32-widened q/k of a backend without a
 * half-precision linear (#464). */
[[nodiscard]] enum geist_status load_f32_buffer_rope_il(struct transformer_arch_state *st,
                                                        struct gguf_ctx               *gguf,
                                                        const char                    *name,
                                                        size_t                expected_elems,
                                                        size_t                rope_il_head_dim,
                                                        enum weight_storage   storage,
                                                        struct geist_buffer **out_buf);

/* Hand `p` (a heap_alloc_aligned block holding a weight the backend reads
 * through an aliased buffer) to the state, which frees it at destroy. On
 * failure `p` is freed here. */
[[nodiscard]] enum geist_status keep_host_weight(struct transformer_arch_state *st, void *p);

/* As load_tensor_to_buffer, with the rows of a llama-family attn_q /
 * attn_k reordered from interleaved to half-split RoPE pairs, head by head,
 * when rope_il_head_dim > 0 (#464). */
[[nodiscard]] enum geist_status load_tensor_to_buffer_rope_il(struct transformer_arch_state *st,
                                                              struct gguf_ctx               *gguf,
                                                              const char                    *name,
                                                              size_t              expected_elems,
                                                              size_t              rope_il_head_dim,
                                                              enum weight_storage storage,
                                                              const struct gguf_tensor_t **out_t,
                                                              struct geist_buffer        **out_buf);

[[nodiscard]] enum geist_status load_tensor_to_buffer(struct transformer_arch_state *st,
                                                      struct gguf_ctx               *gguf,
                                                      const char                    *name,
                                                      size_t                         expected_elems,
                                                      enum weight_storage            storage,
                                                      const struct gguf_tensor_t   **out_t,
                                                      struct geist_buffer          **out_buf);
