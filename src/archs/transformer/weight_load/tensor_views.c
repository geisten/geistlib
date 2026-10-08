/*
 * src/archs/transformer/weight_load/tensor_views.c — GGUF tensor →
 * backend buffer staging and arena capacity computation.
 *
 * Layer: ARCHITECTURE.
 * Contains:
 *
 *   compute_weight_arena_capacity — first arena chunk from the tensor table
 *   arena_alloc / weight_arena_open — the chunked β-mode weight arena
 *   load_tensor_to_buffer         — bump-alloc + memcpy, or mmap-alias,
 *                                   by the caller's storage intent
 *
 * make_view_2d / make_view_1d / weight_off_arena live as static inline in
 * internal.h.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"

#include "gguf_dequant.h"
#include "gguf_reader.h"
#include "checked.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Growth step of the weight arena: a tensor that does not fit the newest
 * chunk opens one of this size, or of its own size when larger. A device-
 * copy backend keeps only bindable tensors there (tens of MiB on the
 * models in tree), and a chunk is the most a model leaves unused. */
constexpr size_t WEIGHT_ARENA_CHUNK = 32u << 20;

[[nodiscard]] enum geist_status compute_weight_arena_capacity(const struct geist_backend *be,
                                                              struct gguf_ctx            *gguf,
                                                              size_t *out_bytes) {
    if (be->desc->caps.weights_device_copy) {
        *out_bytes = WEIGHT_ARENA_CHUNK;
        return GEIST_OK;
    }
    size_t       total = 0;
    const size_t n     = gguf_tensor_count(gguf);
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(gguf, i);
        if (t == nullptr) {
            continue;
        }
        total += (t->nbytes + 63u) & ~((size_t) 63u);
        /* A small half-precision matrix a backend refuses to resolve is
         * widened to F32 (load_layer_proj), which adds the F32 copy to the
         * arena beside the file's bytes. Counted whatever the caps say: the
         * refusal comes from resolve_weight, not from a cap (#561). A
         * backend that resolves the matrix natively leaves it unused. */
        const size_t elems = gguf_tensor_elem_count(t);
        if (t->n_dims == 2 && (t->dtype == GGUF_TYPE_F16 || t->dtype == GGUF_TYPE_BF16) &&
            elems <= (4u << 20)) {
            total += (elems * sizeof(float) + 63u) & ~((size_t) 63u);
        }
    }
    /* Headroom for derived buffers: per_layer_model_proj FP32 (2× the
     * F16 source, ~28 MB extra on Gemma 4 E2B) and the F32 form of a
     * narrower norm gamma. Round up to 64 MB to absorb any other small
     * dequant'd globals; past it the arena grows a chunk. */
    total += 64ULL * 1024 * 1024;
    *out_bytes = total;
    return GEIST_OK;
}

/* Append a mapped backend buffer of `bytes` as the arena's newest chunk. */
[[nodiscard]] static enum geist_status arena_add_chunk(struct transformer_arch_state *st,
                                                       size_t                         bytes) {
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    if (st->n_weight_arena_chunks == st->cap_weight_arena_chunks) {
        const size_t cap = st->cap_weight_arena_chunks == 0 ? 4 : 2 * st->cap_weight_arena_chunks;
        struct geist_buffer **nc = heap_alloc_array_aligned(struct geist_buffer *, cap);
        if (nc == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "transformer: weight arena chunk list");
            return GEIST_E_OOM;
        }
        if (st->n_weight_arena_chunks > 0) {
            memcpy(nc, st->weight_arena_chunks, st->n_weight_arena_chunks * sizeof *nc);
        }
        void *old = st->weight_arena_chunks;
        safe_free(&old);
        st->weight_arena_chunks     = nc;
        st->cap_weight_arena_chunks = cap;
    }
    struct geist_buffer *buf = nullptr;
    enum geist_status    s =
            v->buffer_create(be, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_MAPPED, &buf);
    void *base = s == GEIST_OK ? v->buffer_map(buf) : nullptr;
    if (base == nullptr) {
        if (buf != nullptr) {
            v->buffer_destroy(be, buf);
        }
        geist_backend_set_error(
                be, GEIST_E_OOM, "transformer: weight arena alloc failed (%zu bytes)", bytes);
        return GEIST_E_OOM;
    }
    st->weight_arena_chunks[st->n_weight_arena_chunks++] = buf;
    st->weight_arena                                     = base;
    st->weight_arena_capacity                            = bytes;
    st->weight_arena_used                                = 0;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status keep_host_weight(struct transformer_arch_state *st, void *p) {
    if (st->n_host_weights == st->cap_host_weights) {
        const size_t cap = st->cap_host_weights == 0 ? 16 : 2 * st->cap_host_weights;
        void       **nl  = heap_alloc_array_aligned(void *, cap);
        if (nl == nullptr) {
            safe_free(&p);
            geist_backend_set_error(st->backend, GEIST_E_OOM, "transformer: host weight list");
            return GEIST_E_OOM;
        }
        if (st->n_host_weights > 0) {
            memcpy(nl, st->host_weights, st->n_host_weights * sizeof *nl);
        }
        void *old = st->host_weights;
        safe_free(&old);
        st->host_weights     = nl;
        st->cap_host_weights = cap;
    }
    st->host_weights[st->n_host_weights++] = p;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status weight_arena_open(struct transformer_arch_state *st, size_t bytes) {
    return arena_add_chunk(st, bytes);
}

void *arena_alloc(struct transformer_arch_state *st, size_t bytes, size_t align) {
    if (align < 64) {
        align = 64;
    }
    const size_t mask         = align - 1;
    size_t       aligned_used = (st->weight_arena_used + mask) & ~mask;
    if (aligned_used > st->weight_arena_capacity ||
        bytes > st->weight_arena_capacity - aligned_used) {
        /* A chunk's mapping is at least 64-byte aligned, so a fresh one
         * starts aligned for any align <= 64; a wider one pads. */
        size_t need;
        if (ckd_add(&need, bytes, align)) {
            geist_backend_set_error(st->backend, GEIST_E_OOM, "transformer: weight arena overflow");
            return nullptr;
        }
        if (arena_add_chunk(st, need > WEIGHT_ARENA_CHUNK ? need : WEIGHT_ARENA_CHUNK) !=
            GEIST_OK) {
            return nullptr;
        }
        const uintptr_t base = (uintptr_t) st->weight_arena;
        aligned_used         = (size_t) (((base + mask) & ~(uintptr_t) mask) - base);
    }
    void *p               = (uint8_t *) st->weight_arena + aligned_used;
    st->weight_arena_used = aligned_used + bytes;
    return p;
}

/* Reorder the rows of a llama-family attn_q / attn_k from the GGUF's
 * interleaved RoPE pair order to the half-split order the runtime rotates:
 * within each head, output row i comes from row 2i and row half + i from
 * row 2i + 1. Applied to the weight rows it yields the same activations as
 * permuting q/k per token, bit for bit, since every output row is its own
 * dot product. Quantized blocks never straddle rows, so a row is a plain
 * byte range for every dtype. */
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
    return load_f32_buffer_rope_il(st, gguf, name, expected_elems, 0, WEIGHT_BIND, out_buf);
}

[[nodiscard]] enum geist_status load_f32_buffer_rope_il(struct transformer_arch_state *st,
                                                        struct gguf_ctx               *gguf,
                                                        const char                    *name,
                                                        size_t                expected_elems,
                                                        size_t                rope_il_head_dim,
                                                        enum weight_storage   storage,
                                                        struct geist_buffer **out_buf) {
    struct geist_backend *be = st->backend;
    *out_buf                 = nullptr;

    /* An F32 tensor is the buffer itself, staged (and permuted) as kernels
     * bind it. A narrower one is widened straight from the file's bytes,
     * below, and permuted after widening; nothing of it is staged. */
    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, name);
    if (t != nullptr && t->dtype == GGUF_TYPE_F32) {
        const struct gguf_tensor_t *t_staged = nullptr;
        return load_tensor_to_buffer_rope_il(
                st, gguf, name, expected_elems, rope_il_head_dim, storage, &t_staged, out_buf);
    }
    if (t == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_NOT_FOUND, "transformer: tensor '%s' not found in GGUF", name);
        return GEIST_E_NOT_FOUND;
    }
    if (gguf_tensor_elem_count(t) != expected_elems) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: '%s' has %zu elements, expected %zu",
                                name,
                                gguf_tensor_elem_count(t),
                                expected_elems);
        return GEIST_E_FORMAT;
    }
    const size_t n_rows = t->n_dims == 2 ? (size_t) t->dims[1] : 0;
    if (rope_il_head_dim > 0 &&
        (rope_il_head_dim % 2 != 0 || n_rows == 0 || n_rows % rope_il_head_dim != 0)) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: '%s' cannot be permuted for RoPE "
                                "(%zu rows, head_dim %zu)",
                                name,
                                n_rows,
                                rope_il_head_dim);
        return GEIST_E_FORMAT;
    }

    /* Not F32 in the file (e.g. the F16 gammas of bitnet-embedding GGUFs).
     * Every kernel that consumes a norm gamma reads F32, so convert once
     * here rather than teaching rmsnorm a dtype. */
    struct geist_buffer *buf = nullptr;
    enum geist_status    s;
    float               *fp32 = gguf_dequant_to_fp32(t);
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
    if (st->weight_arena != nullptr && weight_off_arena(be, storage)) {
        /* A matrix the backend copies to the device: the F32 form is host
         * memory of the state's own, aliased like the mmap pages of an F32
         * matrix would be. */
        s = keep_host_weight(st, fp32);
        return s != GEIST_OK ? s
                             : be->desc->vtbl->buffer_create_aliased(
                                       be, fp32, bytes, GEIST_BUFFER_WEIGHT, out_buf);
    }
    if (st->weight_arena != nullptr) {
        void *arena_ptr = arena_alloc(st, bytes, 64);
        if (arena_ptr == nullptr) {
            void *p = fp32;
            safe_free(&p);
            return GEIST_E_OOM; /* arena_alloc said why */
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
                                                      enum weight_storage            storage,
                                                      const struct gguf_tensor_t   **out_t,
                                                      struct geist_buffer          **out_buf) {
    return load_tensor_to_buffer_rope_il(
            st, gguf, name, expected_elems, 0, storage, out_t, out_buf);
}

[[nodiscard]] enum geist_status load_tensor_to_buffer_rope_il(struct transformer_arch_state *st,
                                                              struct gguf_ctx               *gguf,
                                                              const char                    *name,
                                                              size_t              expected_elems,
                                                              size_t              rope_il_head_dim,
                                                              enum weight_storage storage,
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
     *   β mode (the default where caps.weights_need_backend_arena is
     *   set, i.e. Vulkan; GEIST_WEIGHT_MMAP=0 elsewhere): weight bytes
     *   are copied from the GGUF mmap into a backend-owned arena via
     *   bump-allocation; gguf_close runs after all loads. On a
     *   weights_device_copy backend only WEIGHT_BIND tensors are; the rest
     *   alias the mmap as below (weight_off_arena).
     *
     *   mmap-alias mode (the CPU and Metal default): weight bytes are NOT
     *   copied; the mmap pointer is wrapped in an aliased buffer.
     *   gguf_ctx is retained for state lifetime; kernels read directly
     *   from mmap pages, faulted in on demand.
     *
     * Both expose a GEIST_MEMORY_ALIASED buffer to the kernel layer; only
     * the ownership differs. */
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
        if (st->weight_arena == nullptr || weight_off_arena(be, storage)) {
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
    if (st->weight_arena != nullptr && !weight_off_arena(be, storage)) {
        /* β: bump-allocate + memcpy. */
        raw_ptr = arena_alloc(st, t->nbytes, 64);
        if (raw_ptr == nullptr) {
            return GEIST_E_OOM; /* arena_alloc said why */
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
