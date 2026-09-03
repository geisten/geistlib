/*
 * src/backends/metal/resources.c — buffers, the host-pointer registry, and tensor validators.
 *
 * Layer: BACKEND (metal). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "metal_internal.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>

/* Intra-module forward decl (definition order preserved from the split). */
static bool metal_tensor_is_dense_3d_dtype(const struct geist_tensor *t,
                                           enum geist_dtype           dtype,
                                           size_t                     elem_size,
                                           size_t                    *out_d0,
                                           size_t                    *out_d1,
                                           size_t                    *out_d2,
                                           size_t                    *out_offset_elems);

static bool metal_buffer_wants_host_visible(enum geist_buffer_role role,
                                            unsigned int           memory_flags) {
    (void) role;
    (void) memory_flags;
    /* main's contract maps weights and scratch to host pointers (resolve_
     * weight reads w->raw; linear_w_or_legacy buffer_maps x/y). On Apple
     * unified memory SHARED storage is the same physical memory as PRIVATE
     * with zero perf cost, so make every buffer host-visible — otherwise
     * buffer_map returns null and state_create fails. */
    return true;
}

/* ---- Buffer registry (host contents pointer -> geist_buffer) ---------- */

static void metal_buf_reg_add(struct metal_state *st, struct geist_buffer *buf) {
    if (st == nullptr || buf == nullptr || buf->mapped == nullptr) {
        return;
    }
    if (st->buf_reg_count == st->buf_reg_cap) {
        const size_t ncap  = st->buf_reg_cap != 0 ? st->buf_reg_cap * 2 : 64;
        void        *grown = realloc(st->buf_reg, ncap * sizeof(*st->buf_reg));
        if (grown == nullptr) {
            return; /* lookup will miss and report loudly */
        }
        st->buf_reg     = grown;
        st->buf_reg_cap = ncap;
    }
    st->buf_reg[st->buf_reg_count++] = (struct metal_buf_reg_entry) {
            .base  = buf->mapped,
            .bytes = buf->bytes,
            .buf   = buf,
    };
}

static void metal_buf_reg_remove(struct metal_state *st, const struct geist_buffer *buf) {
    if (st == nullptr || st->buf_reg == nullptr) {
        return;
    }
    for (size_t i = 0; i < st->buf_reg_count; i++) {
        if (st->buf_reg[i].buf == buf) {
            st->buf_reg[i] = st->buf_reg[--st->buf_reg_count];
            return;
        }
    }
}

struct geist_buffer *metal_buf_reg_find(struct metal_state *st, const void *p, size_t *out_off) {
    if (st == nullptr || p == nullptr) {
        return nullptr;
    }
    const uint8_t *q = p;
    for (size_t i = 0; i < st->buf_reg_count; i++) {
        const struct metal_buf_reg_entry *e = &st->buf_reg[i];
        if (q >= e->base && q < e->base + e->bytes) {
            *out_off = (size_t) (q - e->base);
            return e->buf;
        }
    }
    return nullptr;
}

[[nodiscard]] enum geist_status metal_new_buffer(struct geist_backend  *be,
                                                 size_t                 bytes,
                                                 enum geist_buffer_role role,
                                                 unsigned int           memory_flags,
                                                 bool                   host_visible,
                                                 struct geist_buffer  **out) {

    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (be == nullptr || be->state == nullptr || bytes == 0) {
        if (be != nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_INVALID_ARG, "metal: invalid buffer create request");
        }
        return GEIST_E_INVALID_ARG;
    }

    struct metal_state  *st  = be->state;
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "metal: failed to allocate buffer handle");
        return GEIST_E_OOM;
    }

    const unsigned long options =
            host_visible ? METAL_RESOURCE_STORAGE_MODE_SHARED : METAL_RESOURCE_STORAGE_MODE_PRIVATE;
    void *mtl_buffer = metal_msg_send_id_size_uint(
            st, st->device, "newBufferWithLength:options:", bytes, options);
    if (mtl_buffer == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal: failed to allocate %zu-byte buffer", bytes);
        return GEIST_E_BACKEND;
    }

    void *mapped = host_visible ? metal_msg_send_id0(st, mtl_buffer, "contents") : nullptr;
    if (host_visible && mapped == nullptr) {
        metal_msg_send_void0(st, mtl_buffer, "release");
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: host-visible buffer has no contents");
        return GEIST_E_BACKEND;
    }

    const unsigned int actual_flags =
            memory_flags | (host_visible ? (GEIST_MEMORY_HOST_VISIBLE | GEIST_MEMORY_MAPPED)
                                         : GEIST_MEMORY_DEVICE);
    *buf = (struct geist_buffer) {
            .owner        = st,
            .buffer       = mtl_buffer,
            .mapped       = mapped,
            .bytes        = bytes,
            .role         = role,
            .memory_flags = actual_flags,
            .host_visible = host_visible,
    };
    metal_buf_reg_add(st, buf);
    *out = buf;
    return GEIST_OK;
}

void metal_buffer_destroy_internal(struct geist_backend *be, struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    struct metal_state *st = buf->owner;
    metal_buf_reg_remove(st, buf);
    if (be == nullptr && st != nullptr) {
        be = st->backend;
    }
    if (st != nullptr && st->objc_msgSend != nullptr && st->sel_registerName != nullptr) {
        metal_msg_send_void0(st, buf->buffer, "release");
    }
    if (be != nullptr) {
        geist_backend_free(be, buf);
    }
}

[[nodiscard]] static enum geist_status metal_submit_copy(struct metal_state *st,
                                                         void               *src,
                                                         size_t              src_offset,
                                                         void               *dst,
                                                         size_t              dst_offset,
                                                         size_t              n_bytes) {

    if (st == nullptr || src == nullptr || dst == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }

    /* In-sequence fast path: when a command sequence is recording, encode the
     * copy as a compute dispatch (copy_u32) on the SEQUENCE's existing compute
     * encoder instead of committing a standalone buffer + waitUntilCompleted.
     * The per-copy GPU round-trip was ~53% of prefill wall (per-layer PLE/KV
     * copies). Dispatches run in order on a serial compute encoder, so the copy
     * correctly sees prior writes; all commits once at sequence_end. Using a
     * compute dispatch (not a blit encoder) avoids exhausting the per-command-
     * buffer encoder limit at long context. Requires 4-byte alignment; other
     * copies fall through to the standalone blit below. */
    if (st->sequence_active && st->sequence_compute_encoder != nullptr &&
        st->copy_u32_pipeline != nullptr && (src_offset % 4u) == 0 && (dst_offset % 4u) == 0 &&
        (n_bytes % 4u) == 0) {
        void *enc = metal_sequence_encoder(st);
        struct {
            uint32_t so, dof, n;
        } cp = {(uint32_t) (src_offset / 4u),
                (uint32_t) (dst_offset / 4u),
                (uint32_t) (n_bytes / 4u)};
        metal_msg_send_set_pipeline(st, enc, st->copy_u32_pipeline);
        metal_msg_send_set_buffer(st, enc, src, 0, 0);
        metal_msg_send_set_buffer(st, enc, dst, 0, 1);
        metal_msg_send_set_bytes(st, enc, &cp, sizeof(cp), 2);
        const struct metal_size groups  = {(cp.n + 255u) / 256u, 1, 1};
        const struct metal_size threads = {256, 1, 1};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_COPY_U32, groups);
        metal_msg_send_dispatch(st, enc, groups, threads);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(
                st->backend, GEIST_E_BACKEND, "metal: failed to create command buffer");
        return GEIST_E_BACKEND;
    }
    void *blit = metal_msg_send_id0(st, cmd, "blitCommandEncoder");
    if (blit == nullptr) {
        geist_backend_set_error(
                st->backend, GEIST_E_BACKEND, "metal: failed to create blit encoder");
        return GEIST_E_BACKEND;
    }

    metal_msg_send_copy_buffer(st,
                               blit,
                               "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
                               src,
                               src_offset,
                               dst,
                               dst_offset,
                               n_bytes);
    metal_msg_send_void0(st, blit, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_BACKEND, "metal: blit command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status metal_buffer_create(struct geist_backend  *be,
                                                    size_t                 bytes,
                                                    enum geist_buffer_role role,
                                                    unsigned int           memory_flags,
                                                    struct geist_buffer  **out) {

    const bool host_visible = metal_buffer_wants_host_visible(role, memory_flags);
    return metal_new_buffer(be, bytes, role, memory_flags, host_visible, out);
}

/* #357: is [p, p+n) inside one file-backed VM region, and what is the
 * enclosing page range? A weight aliased straight out of the loader's mmap
 * needs no copy — unified memory lets the GPU read the file pages in place.
 * The mmap is page-granular but the tensor inside it is only 32-byte
 * aligned, so the wrapper covers whole pages and the caller keeps the
 * in-page offset. Neighbouring wrappers overlap on a boundary page; that is
 * fine for read-only file pages (llama.cpp's Metal backend does the same).
 * A heap pointer — load-from-memory, an arena slice — fails the region
 * check and takes the copy path, the only one safe for memory whose extent
 * the backend does not know. */
static bool
metal_host_range_file_backed(const void *p, size_t n, uint8_t **base_out, size_t *len_out) {
    const uintptr_t page = (uintptr_t) vm_page_size;
    const uintptr_t lo   = (uintptr_t) p & ~(page - 1u);
    if (n > UINTPTR_MAX - (uintptr_t) p - page) {
        return false;
    }
    const uintptr_t hi = ((uintptr_t) p + n + page - 1u) & ~(page - 1u);

    mach_vm_address_t              addr  = lo;
    mach_vm_size_t                 size  = 0;
    vm_region_extended_info_data_t info  = {0};
    mach_msg_type_number_t         count = VM_REGION_EXTENDED_INFO_COUNT;
    mach_port_t                    obj   = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(),
                       &addr,
                       &size,
                       VM_REGION_EXTENDED_INFO,
                       (vm_region_info_t) &info,
                       &count,
                       &obj) != KERN_SUCCESS) {
        return false;
    }
    if (obj != MACH_PORT_NULL) {
        mach_port_deallocate(mach_task_self(), obj);
    }
    /* mach_vm_region returns the first region at or after `addr`: a start
     * past `lo` means `lo` itself is unmapped. */
    if (addr > lo || size < hi - addr || !info.external_pager) {
        return false;
    }
    *base_out = (uint8_t *) lo;
    *len_out  = (size_t) (hi - lo);
    return true;
}

/* Alias a host-resident region (mmap'd weight or an arena sub-range) as a
 * device buffer. A region that lives in a file-backed mapping is wrapped in
 * place (newBufferWithBytesNoCopy over its page range, base_off pointing at
 * the bytes) — the model is then resident once, as file pages, instead of
 * once as file pages and once more as device copies (#357). Anything else
 * arrives as an arbitrary 64-byte-aligned sub-pointer that NoCopy cannot
 * wrap, and is copied into a SHARED MTLBuffer (unified memory, host+GPU
 * coherent). Weights are read-only; arena scratch is always accessed via
 * this handle, so a per-buffer copy stays coherent. */
[[nodiscard]] enum geist_status metal_buffer_create_aliased(struct geist_backend  *be,
                                                            void                  *host_ptr,
                                                            size_t                 n_bytes,
                                                            enum geist_buffer_role role,
                                                            struct geist_buffer  **out) {

    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (be == nullptr || be->state == nullptr || host_ptr == nullptr || n_bytes == 0) {
        if (be != nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_INVALID_ARG, "metal: invalid aliased buffer request");
        }
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state  *st  = be->state;
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "metal: failed to allocate buffer handle");
        return GEIST_E_OOM;
    }
    uint8_t *base       = nullptr;
    size_t   base_len   = 0;
    size_t   base_off   = 0;
    void    *mtl_buffer = nullptr;
    if (metal_host_range_file_backed(host_ptr, n_bytes, &base, &base_len)) {
        mtl_buffer = metal_msg_send_id_ptr_size_uint_ptr(
                st,
                st->device,
                "newBufferWithBytesNoCopy:length:options:deallocator:",
                base,
                base_len,
                METAL_RESOURCE_STORAGE_MODE_SHARED,
                nullptr);
        base_off = (size_t) ((const uint8_t *) host_ptr - base);
    }
    if (mtl_buffer == nullptr) {
        base_off   = 0;
        mtl_buffer = metal_msg_send_id_ptr_size_uint(st,
                                                     st->device,
                                                     "newBufferWithBytes:length:options:",
                                                     host_ptr,
                                                     n_bytes,
                                                     METAL_RESOURCE_STORAGE_MODE_SHARED);
    }
    if (mtl_buffer == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: failed to alias %zu bytes", n_bytes);
        return GEIST_E_BACKEND;
    }
    void *mapped = metal_msg_send_id0(st, mtl_buffer, "contents");
    if (mapped == nullptr) {
        metal_msg_send_void0(st, mtl_buffer, "release");
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: aliased buffer has no contents");
        return GEIST_E_BACKEND;
    }
    *buf = (struct geist_buffer) {
            .owner        = st,
            .buffer       = mtl_buffer,
            .mapped       = (uint8_t *) mapped + base_off,
            .bytes        = n_bytes,
            .base_off     = base_off,
            .role         = role,
            .memory_flags = GEIST_MEMORY_HOST_VISIBLE | GEIST_MEMORY_MAPPED | GEIST_MEMORY_ALIASED,
            .host_visible = true,
    };
    metal_buf_reg_add(st, buf);
    *out = buf;
    return GEIST_OK;
}

void metal_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf) {
    metal_buffer_destroy_internal(be, buf);
}

[[nodiscard]] enum geist_status metal_buffer_copy(struct geist_buffer       *dst,
                                                  size_t                     dst_offset,
                                                  const struct geist_buffer *src,
                                                  size_t                     src_offset,
                                                  size_t                     n_bytes) {

    if (dst == nullptr || src == nullptr || dst->owner == nullptr || src->owner == nullptr ||
        dst->owner != src->owner) {
        return GEIST_E_INVALID_ARG;
    }
    if (dst_offset > dst->bytes || src_offset > src->bytes || n_bytes > dst->bytes - dst_offset ||
        n_bytes > src->bytes - src_offset) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }

    struct metal_state *st = dst->owner;
    if (dst == src && metal_ranges_overlap(dst_offset, src_offset, n_bytes)) {
        struct geist_buffer *tmp = nullptr;
        enum geist_status    s   = metal_new_buffer(
                st->backend, n_bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, false, &tmp);
        if (s != GEIST_OK) {
            return s;
        }
        s = metal_submit_copy(st, src->buffer, src->base_off + src_offset, tmp->buffer, 0, n_bytes);
        if (s == GEIST_OK) {
            s = metal_submit_copy(
                    st, tmp->buffer, 0, dst->buffer, dst->base_off + dst_offset, n_bytes);
        }
        metal_buffer_destroy_internal(st->backend, tmp);
        return s;
    }

    return metal_submit_copy(st,
                             src->buffer,
                             src->base_off + src_offset,
                             dst->buffer,
                             dst->base_off + dst_offset,
                             n_bytes);
}

[[nodiscard]] enum geist_status
metal_buffer_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t *src) {

    if (buf == nullptr || n_bytes > buf->bytes || (n_bytes > 0 && src == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }
    metal_flush_if_referenced(buf->owner, buf->buffer);

    if (buf->host_visible) {
        memcpy(buf->mapped, src, n_bytes);
        return GEIST_OK;
    }

    struct metal_state  *st      = buf->owner;
    struct geist_buffer *staging = nullptr;
    enum geist_status    s       = metal_new_buffer(
            st->backend, n_bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE, true, &staging);
    if (s != GEIST_OK) {
        return s;
    }
    memcpy(staging->mapped, src, n_bytes);
    s = metal_submit_copy(st, staging->buffer, 0, buf->buffer, buf->base_off, n_bytes);
    metal_buffer_destroy_internal(st->backend, staging);
    return s;
}

[[nodiscard]] enum geist_status
metal_buffer_download(size_t n_bytes, uint8_t *dst, const struct geist_buffer *buf) {

    if (buf == nullptr || n_bytes > buf->bytes || (n_bytes > 0 && dst == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }
    metal_flush_if_referenced(buf->owner, buf->buffer);

    if (buf->host_visible) {
        memcpy(dst, buf->mapped, n_bytes);
        return GEIST_OK;
    }

    struct metal_state  *st      = buf->owner;
    struct geist_buffer *staging = nullptr;
    enum geist_status    s       = metal_new_buffer(
            st->backend, n_bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE, true, &staging);
    if (s != GEIST_OK) {
        return s;
    }
    s = metal_submit_copy(st, buf->buffer, buf->base_off, staging->buffer, 0, n_bytes);
    if (s == GEIST_OK) {
        memcpy(dst, staging->mapped, n_bytes);
    }
    metal_buffer_destroy_internal(st->backend, staging);
    return s;
}

void *metal_buffer_map(struct geist_buffer *buf) {
    if (buf == nullptr || !buf->host_visible) {
        return nullptr;
    }
    /* The engine reads/writes through this pointer; if pending batched GPU
     * work references the buffer, submit it first (read: results must be
     * visible; write: encoded ops must not observe the new contents). */
    {
        struct metal_state *st = buf->owner;
        if (metal_seq_references(st, buf->buffer)) {
            if (metal_env_enabled("GEIST_METAL_STRICT_BATCH")) {
                geist_backend_set_error(st->backend,
                                        GEIST_E_BACKEND,
                                        "metal: host map of active sequence buffer (%zu bytes)",
                                        buf->bytes);
                return nullptr;
            }
            static _Atomic int dbg = -1;
            if (dbg < 0) {
                const char *e = getenv("GEIST_SEQ_COUNT");
                dbg           = (e && e[0]) ? 1 : 0;
            }
            if (dbg) {
                fprintf(stderr, "[flushmap] bytes=%zu role=%d\n", buf->bytes, (int) buf->role);
            }
            metal_batch_flush(st);
        }
    }
    return buf->mapped;
}

void metal_buffer_unmap(struct geist_buffer *buf) {
    (void) buf;
}

bool metal_tensor_is_f32_vector(const struct geist_tensor *t,
                                size_t                    *out_n,
                                size_t                    *out_offset_floats) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE || (t->ndim != 1 && t->ndim != 2) ||
        t->offset % sizeof(float) != 0) {
        return false;
    }
    size_t n = 0;
    if (t->ndim == 1) {
        if (t->shape[0] <= 0 || t->stride[0] != 1) {
            return false;
        }
        n = (size_t) t->shape[0];
    } else {
        if (t->shape[0] != 1 || t->shape[1] <= 0 || t->stride[0] != t->shape[1] ||
            t->stride[1] != 1) {
            return false;
        }
        n = (size_t) t->shape[1];
    }
    if (t->offset > t->buffer->bytes || n > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_n             = n;
    *out_offset_floats = t->offset / sizeof(float);
    return true;
}

bool metal_tensor_is_f32_matrix(const struct geist_tensor *t,
                                size_t                    *out_rows,
                                size_t                    *out_cols,
                                size_t                    *out_offset_floats,
                                size_t                    *out_row_stride) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE || t->ndim != 2 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->stride[0] != t->shape[1] || t->stride[1] != 1 || t->offset % sizeof(float) != 0) {
        return false;
    }
    const size_t rows = (size_t) t->shape[0];
    const size_t cols = (size_t) t->shape[1];
    if (rows > SIZE_MAX / cols) {
        return false;
    }
    const size_t elems = rows * cols;
    if (t->offset > t->buffer->bytes || elems > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_rows          = rows;
    *out_cols          = cols;
    *out_offset_floats = t->offset / sizeof(float);
    *out_row_stride    = cols;
    return true;
}

/* Like metal_tensor_is_f32_matrix but accepts a row stride wider than the
 * column count (a strided view into a larger slab, e.g. the per-layer PLE
 * slice). Kernels take the row stride from their params. */
static bool metal_tensor_is_f32_matrix_strided(const struct geist_tensor *t,
                                               size_t                    *out_rows,
                                               size_t                    *out_cols,
                                               size_t                    *out_offset_floats,
                                               size_t                    *out_row_stride) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE || t->ndim != 2 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->stride[0] < t->shape[1] || t->stride[1] != 1 || t->offset % sizeof(float) != 0) {
        return false;
    }
    const size_t rows   = (size_t) t->shape[0];
    const size_t cols   = (size_t) t->shape[1];
    const size_t stride = (size_t) t->stride[0];
    if (rows > 1 && stride > (SIZE_MAX - cols) / (rows - 1)) {
        return false;
    }
    const size_t elems = (rows - 1) * stride + cols;
    if (t->offset > t->buffer->bytes || elems > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_rows          = rows;
    *out_cols          = cols;
    *out_offset_floats = t->offset / sizeof(float);
    *out_row_stride    = stride;
    return true;
}

bool metal_tensor_is_f32_rows(const struct geist_tensor *t,
                              size_t                    *out_rows,
                              size_t                    *out_cols,
                              size_t                    *out_offset_floats,
                              size_t                    *out_row_stride) {
    if (t == nullptr || out_rows == nullptr || out_cols == nullptr ||
        out_offset_floats == nullptr || out_row_stride == nullptr) {
        return false;
    }
    if (t->ndim == 1) {
        size_t n   = 0;
        size_t off = 0;
        if (!metal_tensor_is_f32_vector(t, &n, &off)) {
            return false;
        }
        *out_rows          = 1;
        *out_cols          = n;
        *out_offset_floats = off;
        *out_row_stride    = n;
        return true;
    }
    /* The elementwise kernels take per-tensor row strides from their
     * params, so a strided 2D view (e.g. the per-layer PLE slice of the
     * [seq, n_layers*hpl] slab) is fine here. */
    return metal_tensor_is_f32_matrix_strided(
            t, out_rows, out_cols, out_offset_floats, out_row_stride);
}

bool metal_tensor_is_q4k_matrix(const struct geist_tensor *t,
                                size_t                    *out_rows,
                                size_t                    *out_cols,
                                size_t                    *out_offset_bytes) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_Q4_K ||
        t->layout != GEIST_LAYOUT_BLOCK_QUANTIZED || t->ndim != 2 || t->shape[0] <= 0 ||
        t->shape[1] <= 0 || ((size_t) t->shape[1] % METAL_Q4K_BLOCK_ELEMS) != 0) {
        return false;
    }
    const size_t rows           = (size_t) t->shape[0];
    const size_t cols           = (size_t) t->shape[1];
    const size_t blocks_per_row = cols / METAL_Q4K_BLOCK_ELEMS;
    if (rows > SIZE_MAX / blocks_per_row ||
        rows * blocks_per_row > SIZE_MAX / METAL_Q4K_BLOCK_BYTES) {
        return false;
    }
    const size_t bytes = rows * blocks_per_row * METAL_Q4K_BLOCK_BYTES;
    if (t->offset > t->buffer->bytes || bytes > t->buffer->bytes - t->offset) {
        return false;
    }
    *out_rows         = rows;
    *out_cols         = cols;
    *out_offset_bytes = t->offset;
    return true;
}

bool metal_tensor_is_q6k_matrix(const struct geist_tensor *t,
                                size_t                    *out_rows,
                                size_t                    *out_cols,
                                size_t                    *out_offset_bytes) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_Q6_K ||
        t->layout != GEIST_LAYOUT_BLOCK_QUANTIZED || t->ndim != 2 || t->shape[0] <= 0 ||
        t->shape[1] <= 0 || ((size_t) t->shape[1] % METAL_Q6K_BLOCK_ELEMS) != 0) {
        return false;
    }
    const size_t rows           = (size_t) t->shape[0];
    const size_t cols           = (size_t) t->shape[1];
    const size_t blocks_per_row = cols / METAL_Q6K_BLOCK_ELEMS;
    if (rows > SIZE_MAX / blocks_per_row ||
        rows * blocks_per_row > SIZE_MAX / METAL_Q6K_BLOCK_BYTES) {
        return false;
    }
    const size_t bytes = rows * blocks_per_row * METAL_Q6K_BLOCK_BYTES;
    if (t->offset > t->buffer->bytes || bytes > t->buffer->bytes - t->offset) {
        return false;
    }
    *out_rows         = rows;
    *out_cols         = cols;
    *out_offset_bytes = t->offset;
    return true;
}

bool metal_tensor_is_q5k_matrix(const struct geist_tensor *t,
                                size_t                    *out_rows,
                                size_t                    *out_cols,
                                size_t                    *out_offset_bytes) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_Q5_K ||
        t->layout != GEIST_LAYOUT_BLOCK_QUANTIZED || t->ndim != 2 || t->shape[0] <= 0 ||
        t->shape[1] <= 0 || ((size_t) t->shape[1] % METAL_Q5K_BLOCK_ELEMS) != 0) {
        return false;
    }
    const size_t rows           = (size_t) t->shape[0];
    const size_t cols           = (size_t) t->shape[1];
    const size_t blocks_per_row = cols / METAL_Q5K_BLOCK_ELEMS;
    if (rows > SIZE_MAX / blocks_per_row ||
        rows * blocks_per_row > SIZE_MAX / METAL_Q5K_BLOCK_BYTES) {
        return false;
    }
    const size_t bytes = rows * blocks_per_row * METAL_Q5K_BLOCK_BYTES;
    if (t->offset > t->buffer->bytes || bytes > t->buffer->bytes - t->offset) {
        return false;
    }
    *out_rows         = rows;
    *out_cols         = cols;
    *out_offset_bytes = t->offset;
    return true;
}

bool metal_tensor_is_q40_q80_matrix(const struct geist_tensor *t,
                                    enum geist_dtype           dtype,
                                    size_t                    *out_rows,
                                    size_t                    *out_cols,
                                    size_t                    *out_offset_bytes) {
    const size_t blk_elems =
            (dtype == GEIST_DTYPE_IQ4_XS || dtype == GEIST_DTYPE_Q3_K || dtype == GEIST_DTYPE_IQ3_S)
                    ? METAL_IQ4XS_BLOCK_ELEMS
                    : METAL_Q40_Q80_BLOCK_ELEMS;
    if ((dtype != GEIST_DTYPE_Q4_0 && dtype != GEIST_DTYPE_Q4_1 && dtype != GEIST_DTYPE_Q8_0 &&
         dtype != GEIST_DTYPE_IQ4_NL && dtype != GEIST_DTYPE_IQ4_XS && dtype != GEIST_DTYPE_Q3_K &&
         dtype != GEIST_DTYPE_IQ3_S) ||
        t == nullptr || t->buffer == nullptr || t->dtype != dtype ||
        t->layout != GEIST_LAYOUT_BLOCK_QUANTIZED || t->ndim != 2 || t->shape[0] <= 0 ||
        t->shape[1] <= 0 || ((size_t) t->shape[1] % blk_elems) != 0) {
        return false;
    }
    const size_t rows           = (size_t) t->shape[0];
    const size_t cols           = (size_t) t->shape[1];
    const size_t blocks_per_row = cols / blk_elems;
    const size_t block_bytes    = dtype == GEIST_DTYPE_Q4_0     ? METAL_Q40_BLOCK_BYTES
                                  : dtype == GEIST_DTYPE_Q4_1   ? METAL_Q41_BLOCK_BYTES
                                  : dtype == GEIST_DTYPE_IQ4_NL ? METAL_IQ4NL_BLOCK_BYTES
                                  : dtype == GEIST_DTYPE_IQ4_XS ? METAL_IQ4XS_BLOCK_BYTES
                                  : dtype == GEIST_DTYPE_Q3_K   ? METAL_Q3K_BLOCK_BYTES
                                  : dtype == GEIST_DTYPE_IQ3_S  ? METAL_IQ3S_BLOCK_BYTES
                                                                : METAL_Q80_BLOCK_BYTES;
    if (rows > SIZE_MAX / blocks_per_row || rows * blocks_per_row > SIZE_MAX / block_bytes) {
        return false;
    }
    const size_t bytes = rows * blocks_per_row * block_bytes;
    if (t->offset > t->buffer->bytes || bytes > t->buffer->bytes - t->offset) {
        return false;
    }
    *out_rows         = rows;
    *out_cols         = cols;
    *out_offset_bytes = t->offset;
    return true;
}

bool metal_tensor_is_f32_3d(const struct geist_tensor *t,
                            size_t                    *out_d0,
                            size_t                    *out_d1,
                            size_t                    *out_d2,
                            size_t                    *out_offset_floats) {
    return metal_tensor_is_dense_3d_dtype(
            t, GEIST_DTYPE_F32, sizeof(float), out_d0, out_d1, out_d2, out_offset_floats);
}

bool metal_tensor_is_f16_3d(const struct geist_tensor *t,
                            size_t                    *out_d0,
                            size_t                    *out_d1,
                            size_t                    *out_d2,
                            size_t                    *out_offset_halfs) {
    return metal_tensor_is_dense_3d_dtype(
            t, GEIST_DTYPE_F16, sizeof(uint16_t), out_d0, out_d1, out_d2, out_offset_halfs);
}

static bool metal_tensor_is_dense_3d_dtype(const struct geist_tensor *t,
                                           enum geist_dtype           dtype,
                                           size_t                     elem_size,
                                           size_t                    *out_d0,
                                           size_t                    *out_d1,
                                           size_t                    *out_d2,
                                           size_t                    *out_offset_elems) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != dtype ||
        t->layout != GEIST_LAYOUT_DENSE || t->ndim != 3 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->shape[2] <= 0 || t->stride[2] != 1 || t->stride[1] != t->shape[2] ||
        t->stride[0] != t->shape[1] * t->shape[2] || elem_size == 0 || t->offset % elem_size != 0) {
        return false;
    }
    const size_t d0 = (size_t) t->shape[0];
    const size_t d1 = (size_t) t->shape[1];
    const size_t d2 = (size_t) t->shape[2];
    if (d0 > SIZE_MAX / d1 || d0 * d1 > SIZE_MAX / d2) {
        return false;
    }
    const size_t elems = d0 * d1 * d2;
    if (t->offset > t->buffer->bytes || elems > (t->buffer->bytes - t->offset) / elem_size) {
        return false;
    }
    *out_d0           = d0;
    *out_d1           = d1;
    *out_d2           = d2;
    *out_offset_elems = t->offset / elem_size;
    return true;
}
