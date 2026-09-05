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

/* ---- Weight arena allocator (inline; bump on caller-owned bytes) ------ *
 *
 * Returns aligned slices from st->weight_arena. nullptr when the arena
 * is exhausted (programmer error: undersized initial allocation).
 */
static inline void *arena_alloc(struct transformer_arch_state *st, size_t bytes, size_t align) {
    if (align < 64)
        align = 64;
    const size_t mask         = align - 1;
    const size_t aligned_used = (st->weight_arena_used + mask) & ~mask;
    if (aligned_used + bytes > st->weight_arena_capacity) {
        return nullptr;
    }
    void *p               = (uint8_t *) st->weight_arena + aligned_used;
    st->weight_arena_used = aligned_used + bytes;
    return p;
}

/* ---- Cross-TU functions ----------------------------------------------- *
 *
 * weight_load/tensor_views.c — GGUF tensor → backend buffer staging.
 */
/* Stage a norm vector (rmsnorm gamma) as an F32 backend buffer, converting
 * when the GGUF stores it narrower. Kernels read gammas as F32 only. */
[[nodiscard]] enum geist_status load_norm_to_f32_buffer(struct transformer_arch_state *st,
                                                        struct gguf_ctx               *gguf,
                                                        const char                    *name,
                                                        size_t                expected_elems,
                                                        struct geist_buffer **out_buf);

[[nodiscard]] enum geist_status load_tensor_to_buffer(struct transformer_arch_state *st,
                                                      struct gguf_ctx               *gguf,
                                                      const char                    *name,
                                                      size_t                         expected_elems,
                                                      const struct gguf_tensor_t   **out_t,
                                                      struct geist_buffer          **out_buf);
