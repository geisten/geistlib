/*
 * src/backends/common/host_buffer.h — the host-memory geist_buffer the CPU
 * backends share, its vtable entry points, and checked access to the
 * tensors that live in it.
 *
 * Layer: BACKEND (shared by cpu_scalar, cpu_neon, cpu_x86). cpu_x86 takes
 * its buffer ops from cpu_scalar_vtbl, so all three see this one layout.
 * GPU backends define their own struct geist_buffer and must not include
 * this; engine code sees the struct only as the opaque forward declaration
 * in <geist.h>.
 */
#pragma once

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "host_buffer.h is internal to the backend layer."
#endif

#include "tensor_view.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_types.h>

#include <stddef.h>
#include <stdint.h>

/* CPU buffer = host pointer + size + role. GEIST_MEMORY_ALIASED in
 * memory_flags: `host` is not owned (typically the GGUF mmap). */
struct geist_buffer {
    void                  *host;
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
};

/* geist_backend_vtbl buffer entry points, for any backend whose buffers
 * are plain host memory. Errors are reported under be->desc->name. */
[[nodiscard]] enum geist_status geist_host_buffer_create(struct geist_backend  *be,
                                                         size_t                 bytes,
                                                         enum geist_buffer_role role,
                                                         unsigned int           memory_flags,
                                                         struct geist_buffer  **out);
[[nodiscard]] enum geist_status geist_host_buffer_create_aliased(struct geist_backend  *be,
                                                                 void                  *host_ptr,
                                                                 size_t                 n_bytes,
                                                                 enum geist_buffer_role role,
                                                                 struct geist_buffer  **out);
void geist_host_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf);
[[nodiscard]] enum geist_status geist_host_buffer_upload(struct geist_buffer *buf,
                                                         size_t               n_bytes,
                                                         const uint8_t        src[static n_bytes]);
[[nodiscard]] enum geist_status geist_host_buffer_download(size_t  n_bytes,
                                                           uint8_t dst[static n_bytes],
                                                           const struct geist_buffer *buf);
void                           *geist_host_buffer_map(struct geist_buffer *buf);
void                            geist_host_buffer_unmap(struct geist_buffer *buf);

/* geist_tensor_dense (tensor_view.h) on the buffer the view points into:
 * the first element of a contiguous DENSE view of exactly `ndim`
 * dimensions and `dtype`, or nullptr when the view or its buffer is
 * missing, malformed, misaligned or runs past the buffer. *out_n gets the
 * element count on success. */
[[nodiscard]] static inline void *geist_host_tensor(const struct geist_tensor *t,
                                                    enum geist_dtype           dtype,
                                                    size_t                     elem,
                                                    int                        ndim,
                                                    size_t                    *out_n) {
    if (t == nullptr || t->buffer == nullptr) {
        return nullptr;
    }
    return geist_tensor_dense(t, dtype, elem, ndim, t->buffer->host, t->buffer->bytes, out_n);
}

/* geist_host_tensor for an F32 view of any rank. */
[[nodiscard]] static inline float *geist_host_tensor_f32(const struct geist_tensor *t,
                                                         size_t                    *out_n) {
    if (t == nullptr || t->buffer == nullptr) {
        return nullptr;
    }
    return geist_tensor_f32_dense(t, t->buffer->host, t->buffer->bytes, out_n);
}
