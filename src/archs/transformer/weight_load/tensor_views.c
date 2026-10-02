/*
 * src/archs/transformer/weight_load/tensor_views.c — GGUF tensor →
 * backend buffer staging and arena capacity computation.
 *
 * Layer: ARCHITECTURE.
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

#include "gguf_dequant.h"
#include "gguf_reader.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

bool weight_skips_arena(const struct geist_backend *be, const struct gguf_tensor_t *t) {
    /* 1 MiB: norms, biases and small mixer tensors keep bindable arena
     * storage; only the big matrices go through the device copy. The PLE
     * projection is widened to F32 into the arena (its capacity share is
     * the source size), so it is not one of them. */
    return be->desc->caps.weights_device_copy && t->n_dims == 2 && t->nbytes >= (1u << 20) &&
           strcmp(t->name, "per_layer_model_proj.weight") != 0;
}

[[nodiscard]] enum geist_status compute_weight_arena_capacity(const struct geist_backend *be,
                                                              struct gguf_ctx            *gguf,
                                                              size_t *out_bytes) {

    size_t       total = 0;
    const size_t n     = gguf_tensor_count(gguf);
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(gguf, i);
        if (t == nullptr || weight_skips_arena(be, t))
            continue;
        const size_t aligned = (t->nbytes + 63u) & ~((size_t) 63u);
        total += aligned;
        /* A small half-precision matrix a GPU backend refuses to resolve is
         * widened to F32 (load_layer_proj): load_norm_to_f32_buffer stages the
         * source in the arena a second time and adds the F32 copy — the bump
         * allocator frees nothing. Bonsai: 96 BF16 ssm_alpha/ssm_beta. */
        const size_t elems = gguf_tensor_elem_count(t);
        if (be->desc->caps.weights_need_backend_arena && t->n_dims == 2 &&
            (t->dtype == GGUF_TYPE_F16 || t->dtype == GGUF_TYPE_BF16) && elems <= (4u << 20)) {
            total += aligned + ((elems * sizeof(float) + 63u) & ~((size_t) 63u));
        }
    }
    /* Headroom for derived buffers: per_layer_model_proj FP32 (2× the
     * F16 source, ~28 MB extra on Gemma 4 E2B). Round up to 64 MB to
     * absorb any other small dequant'd globals. */
    total += 64ULL * 1024 * 1024;
    *out_bytes = total;
    return GEIST_OK;
}

/* Reorder the rows of a llama-family attn_q / attn_k from the GGUF's
 * interleaved RoPE pair order to the half-split order the runtime rotates:
 * within each head, output row i comes from row 2i and row half + i from
 * row 2i + 1. That is the permutation the forward pass used to apply to the
 * q/k activations on every layer and token; applied to the weight rows it
 * yields the same activations, bit for bit, since every output row is its
 * own dot product. Quantized blocks never straddle rows, so a row is a
 * plain byte range for every dtype. */
void permute_rope_rows(
        size_t n_rows, size_t row_bytes, size_t head_dim, const uint8_t *src, uint8_t *dst) {
    const size_t half = head_dim / 2;
    for (size_t h = 0; h < n_rows; h += head_dim) {
        for (size_t i = 0; i < half; i++) {
            memcpy(dst + (h + i) * row_bytes, src + (h + 2 * i) * row_bytes, row_bytes);
            memcpy(dst + (h + half + i) * row_bytes, src + (h + 2 * i + 1) * row_bytes, row_bytes);
        }
    }
}

/* Bytes every attn_q / attn_k of the file takes, 64-byte aligned each: the
 * one host allocation for permuted rows outside the arena. */
static size_t rope_il_rows_capacity(struct gguf_ctx *gguf) {
    static const char *const suffixes[] = {".attn_q.weight", ".attn_k.weight"};
    size_t                   total      = 0;
    const size_t             n          = gguf_tensor_count(gguf);
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(gguf, i);
        if (t == nullptr) {
            continue;
        }
        const size_t len = strlen(t->name);
        for (size_t k = 0; k < sizeof suffixes / sizeof suffixes[0]; k++) {
            const size_t sl = strlen(suffixes[k]);
            if (len >= sl && memcmp(t->name + len - sl, suffixes[k], sl) == 0) {
                total += (t->nbytes + 63u) & ~((size_t) 63u);
            }
        }
    }
    return total;
}

/* A 64-byte-aligned slice of st->rope_il_rows, allocated on first use;
 * nullptr when it cannot be allocated or is exhausted. */
static void *
rope_il_rows_alloc(struct transformer_arch_state *st, struct gguf_ctx *gguf, size_t bytes) {
    if (st->rope_il_rows == nullptr) {
        const size_t cap = rope_il_rows_capacity(gguf);
        st->rope_il_rows = cap > 0 ? heap_alloc_aligned(cap, 64) : nullptr;
        if (st->rope_il_rows == nullptr) {
            return nullptr;
        }
        st->rope_il_rows_capacity = cap;
        st->rope_il_rows_used     = 0;
    }
    const size_t need = (bytes + 63u) & ~((size_t) 63u); /* as the capacity counts it */
    if (need > st->rope_il_rows_capacity - st->rope_il_rows_used) {
        return nullptr;
    }
    void *p = (uint8_t *) st->rope_il_rows + st->rope_il_rows_used;
    st->rope_il_rows_used += need;
    return p;
}

[[nodiscard]] enum geist_status load_norm_to_f32_buffer(struct transformer_arch_state *st,
                                                        struct gguf_ctx               *gguf,
                                                        const char                    *name,
                                                        size_t                expected_elems,
                                                        struct geist_buffer **out_buf) {
    return load_f32_buffer_rope_il(st, gguf, name, expected_elems, 0, out_buf);
}

[[nodiscard]] enum geist_status load_f32_buffer_rope_il(struct transformer_arch_state *st,
                                                        struct gguf_ctx               *gguf,
                                                        const char                    *name,
                                                        size_t                expected_elems,
                                                        size_t                rope_il_head_dim,
                                                        struct geist_buffer **out_buf) {
    struct geist_backend *be = st->backend;
    *out_buf                 = nullptr;

    /* Only an F32 tensor keeps its staged buffer, so only that one is staged
     * permuted; a narrower one is permuted after widening, below. */
    const struct gguf_tensor_t *t0  = gguf_get_tensor(gguf, name);
    const bool                  f32 = t0 != nullptr && t0->dtype == GGUF_TYPE_F32;
    const struct gguf_tensor_t *t   = nullptr;
    struct geist_buffer        *buf = nullptr;
    enum geist_status           s   = load_tensor_to_buffer_rope_il(
            st, gguf, name, expected_elems, f32 ? rope_il_head_dim : 0, &t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    if (t->dtype == GGUF_TYPE_F32) {
        *out_buf = buf;
        return GEIST_OK;
    }
    const size_t n_rows = t->n_dims == 2 ? (size_t) t->dims[1] : 0;
    if (rope_il_head_dim > 0 &&
        (rope_il_head_dim % 2 != 0 || n_rows == 0 || n_rows % rope_il_head_dim != 0)) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: '%s' cannot be permuted for RoPE "
                                "(%zu rows, head_dim %zu)",
                                name,
                                n_rows,
                                rope_il_head_dim);
        return GEIST_E_FORMAT;
    }

    /* Not F32 in the file. Every kernel that consumes a norm gamma reads
     * F32, so convert once here rather than teaching rmsnorm a dtype.
     * Microsoft's published bitnet-embedding GGUFs store every gamma as
     * F16 -- the first models in tree to do so, and the reason a norm now
     * goes through this instead of a flat dtype check. The staged buffer is
     * dropped: its bytes are the file's F16, not the F32 we need. */
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
    if (rope_il_head_dim > 0) {
        float *permuted = heap_alloc_aligned(bytes, 64);
        if (permuted == nullptr) {
            void *p = fp32;
            safe_free(&p);
            geist_backend_set_error(
                    be, GEIST_E_OOM, "transformer: no memory to permute '%s'", name);
            return GEIST_E_OOM;
        }
        permute_rope_rows(n_rows,
                          bytes / n_rows,
                          rope_il_head_dim,
                          (const uint8_t *) fp32,
                          (uint8_t *) permuted);
        void *p = fp32;
        safe_free(&p);
        fp32 = permuted;
    }
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
    return load_tensor_to_buffer_rope_il(st, gguf, name, expected_elems, 0, out_t, out_buf);
}

[[nodiscard]] enum geist_status load_tensor_to_buffer_rope_il(struct transformer_arch_state *st,
                                                              struct gguf_ctx               *gguf,
                                                              const char                    *name,
                                                              size_t expected_elems,
                                                              size_t rope_il_head_dim,
                                                              const struct gguf_tensor_t **out_t,
                                                              struct geist_buffer **out_buf) {

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

    /* Interleaved-RoPE rows (rope_il_head_dim > 0): the permuted copy needs
     * storage of its own, since the mmap is read-only: the arena slot in
     * β mode; on a weights_device_copy backend a slice of st->rope_il_rows,
     * host memory aliased like the mmap pages it replaces, so the backend
     * uploads it at resolve_weight and none of it lands in a device heap
     * the backend budgets for scratch (Vulkan's BAR window); else a
     * backend-owned buffer. */
    const size_t n_rows = t->n_dims == 2 ? (size_t) t->dims[1] : 0;
    if (rope_il_head_dim > 0) {
        if (rope_il_head_dim % 2 != 0 || n_rows == 0 || n_rows % rope_il_head_dim != 0 ||
            t->nbytes % n_rows != 0) {
            geist_backend_set_error(be,
                                    GEIST_E_FORMAT,
                                    "transformer: '%s' cannot be permuted for RoPE "
                                    "(%zu rows, head_dim %zu)",
                                    name,
                                    n_rows,
                                    rope_il_head_dim);
            return GEIST_E_FORMAT;
        }
        if (st->weight_arena == nullptr || weight_skips_arena(be, t)) {
            const size_t row_bytes = t->nbytes / n_rows;
            if (be->desc->caps.weights_device_copy) {
                void *rows = rope_il_rows_alloc(st, gguf, t->nbytes);
                if (rows == nullptr) {
                    geist_backend_set_error(
                            be, GEIST_E_OOM, "transformer: no memory to permute '%s'", name);
                    return GEIST_E_OOM;
                }
                permute_rope_rows(
                        n_rows, row_bytes, rope_il_head_dim, (const uint8_t *) t->data, rows);
                s = v->buffer_create_aliased(be, rows, t->nbytes, GEIST_BUFFER_WEIGHT, &buf);
            } else {
                void *tmp = heap_alloc_aligned(t->nbytes, 64);
                if (tmp == nullptr) {
                    geist_backend_set_error(
                            be, GEIST_E_OOM, "transformer: no memory to permute '%s'", name);
                    return GEIST_E_OOM;
                }
                permute_rope_rows(
                        n_rows, row_bytes, rope_il_head_dim, (const uint8_t *) t->data, tmp);
                s = v->buffer_create(be, t->nbytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &buf);
                if (s == GEIST_OK) {
                    s = v->buffer_upload(buf, t->nbytes, (const uint8_t *) tmp);
                    if (s != GEIST_OK) {
                        v->buffer_destroy(be, buf);
                        buf = nullptr;
                    }
                }
                safe_free(&tmp);
            }
            if (s != GEIST_OK) {
                return s;
            }
            *out_t   = t;
            *out_buf = buf;
            return GEIST_OK;
        }
    }
    if (st->weight_arena != nullptr && !weight_skips_arena(be, t)) {
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
        if (rope_il_head_dim > 0) {
            permute_rope_rows(n_rows,
                              t->nbytes / n_rows,
                              rope_il_head_dim,
                              (const uint8_t *) t->data,
                              (uint8_t *) raw_ptr);
        } else {
            memcpy(raw_ptr, t->data, t->nbytes);
        }
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
