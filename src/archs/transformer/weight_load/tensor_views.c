/*
 * src/archs/transformer/weight_load/tensor_views.c — GGUF tensor →
 * backend buffer staging and arena capacity computation.
 *
 * Layer: ARCHITECTURE.
 *
 * Extracted from weight_load.c during R5 of the C23/AGENT.md cleanup.
 * Contains:
 *
 *   compute_weight_arena_capacity — sum tensor bytes for arena sizing
 *   load_tensor_to_buffer         — bump-alloc + memcpy, or mmap-alias
 *
 * make_view_2d / make_view_1d / arena_alloc live as static inline in
 * internal.h.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"

#include "gguf_reader.h"
#include "gguf_dequant.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

[[nodiscard]] enum geist_status compute_weight_arena_capacity(struct gguf_ctx *gguf,
                                                              size_t          *out_bytes) {

    size_t       total = 0;
    const size_t n     = gguf_tensor_count(gguf);
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(gguf, i);
        if (t == nullptr)
            continue;
        const size_t aligned = (t->nbytes + 63u) & ~((size_t) 63u);
        total += aligned;
    }
    /* Headroom for derived buffers: per_layer_model_proj FP32 (2× the
     * F16 source, ~28 MB extra on Gemma 4 E2B). Round up to 64 MB to
     * absorb any other small dequant'd globals. */
    total += 64ULL * 1024 * 1024;
    *out_bytes = total;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status load_norm_to_f32_buffer(struct transformer_arch_state *st,
                                                        struct gguf_ctx               *gguf,
                                                        const char                    *name,
                                                        size_t                expected_elems,
                                                        struct geist_buffer **out_buf) {
    struct geist_backend *be = st->backend;
    *out_buf                 = nullptr;

    const struct gguf_tensor_t *t   = nullptr;
    struct geist_buffer        *buf = nullptr;
    enum geist_status           s = load_tensor_to_buffer(st, gguf, name, expected_elems, &t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    if (t->dtype == GGUF_TYPE_F32) {
        *out_buf = buf;
        return GEIST_OK;
    }
    /* Not F32 in the file. Every kernel that consumes a norm gamma reads
     * F32, so convert once here rather than teaching rmsnorm a dtype —
     * the microsoft/bitnet-embedding GGUFs store every gamma as F16, the
     * first models in tree to do so. The staged buffer is dropped: its
     * bytes are the file's F16, not the F32 we need. */
    be->desc->vtbl->buffer_destroy(be, buf);
    buf = nullptr;

    float *fp32 = gguf_dequant_to_fp32(t);
    if (fp32 == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: '%s' is %s and dequant to F32 failed",
                                name,
                                gguf_dtype_name(t->dtype));
        return GEIST_E_FORMAT;
    }
    const size_t bytes = expected_elems * sizeof(float);
    if (st->weight_arena != nullptr) {
        void *arena_ptr = arena_alloc(st, bytes, 64);
        if (arena_ptr == nullptr) {
            void *p = fp32;
            safe_free(&p);
            geist_backend_set_error(
                    be, GEIST_E_OOM, "transformer: weight arena exhausted at '%s' fp32", name);
            return GEIST_E_OOM;
        }
        memcpy(arena_ptr, fp32, bytes);
        void *p = fp32;
        safe_free(&p);
        return be->desc->vtbl->buffer_create_aliased(
                be, arena_ptr, bytes, GEIST_BUFFER_WEIGHT, out_buf);
    }
    /* mmap-alias mode has no arena: the F32 form is not a slice of the
     * file, so the backend owns this one outright. */
    s = be->desc->vtbl->buffer_create(be, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &buf);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(buf, bytes, (const uint8_t *) fp32);
        if (s != GEIST_OK) {
            be->desc->vtbl->buffer_destroy(be, buf);
            buf = nullptr;
        }
    }
    void *p = fp32;
    safe_free(&p);
    *out_buf = buf;
    return s;
}

[[nodiscard]] enum geist_status load_tensor_to_buffer(struct transformer_arch_state *st,
                                                      struct gguf_ctx               *gguf,
                                                      const char                    *name,
                                                      size_t                         expected_elems,
                                                      const struct gguf_tensor_t   **out_t,
                                                      struct geist_buffer          **out_buf) {

    struct geist_backend *be = st->backend;

    *out_buf = nullptr;
    *out_t   = nullptr;

    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, name);
    if (t == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_NOT_FOUND, "transformer: tensor '%s' not found in GGUF", name);
        return GEIST_E_NOT_FOUND;
    }
    size_t actual = gguf_tensor_elem_count(t);
    if (actual != expected_elems) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: '%s' has %zu elements, expected %zu",
                                name,
                                actual,
                                expected_elems);
        return GEIST_E_FORMAT;
    }

    /* Two storage modes, picked at state-create time:
     *
     *   β mode (default, post-P1.1.f): weight bytes are copied from
     *   the GGUF mmap into a backend-owned arena via bump-allocation;
     *   gguf_close runs after all loads. Backend has full ownership.
     *   Cost: 2.8 GB upfront disk read + memcpy on Pi 5 IQ2_M.
     *
     *   mmap-alias mode (GEIST_WEIGHT_MMAP=1): weight bytes are NOT
     *   copied; we wrap the mmap pointer in an aliased buffer (the
     *   P0.3 path). gguf_ctx is retained for state lifetime; kernels
     *   read directly from mmap pages. Disk reads happen on demand
     *   during attention. Pi 5 IQ2_M cold-load ~1.7 s.
     *
     * The two modes share the same hot path because both expose a
     * GEIST_MEMORY_ALIASED buffer to the kernel layer. Only the
     * underlying ownership differs. */
    struct geist_buffer             *buf = nullptr;
    enum geist_status                s;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    void                            *raw_ptr;
    if (st->weight_arena != nullptr) {
        /* β: bump-allocate + memcpy. */
        raw_ptr = arena_alloc(st, t->nbytes, 64);
        if (raw_ptr == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_OOM,
                                    "transformer: weight arena exhausted at '%s' "
                                    "(used %zu, capacity %zu, need %zu)",
                                    name,
                                    st->weight_arena_used,
                                    st->weight_arena_capacity,
                                    t->nbytes);
            return GEIST_E_OOM;
        }
        memcpy(raw_ptr, t->data, t->nbytes);
    } else {
        /* mmap-alias: zero-copy; gguf mmap retained by caller. */
        raw_ptr = (void *) t->data;
    }
    s = v->buffer_create_aliased(be, raw_ptr, t->nbytes, GEIST_BUFFER_WEIGHT, &buf);
    if (s != GEIST_OK) {
        return s;
    }

    *out_t   = t;
    *out_buf = buf;
    return GEIST_OK;
}
