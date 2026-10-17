/*
 * src/backends/common/host_buffer.c — geist_buffer ops for host memory,
 * shared by the CPU backends (see host_buffer.h).
 *
 * Layer: BACKEND. Handles come from the backend's geist_allocator; the
 * data from heap.h at OPTIMAL_ALIGNMENT, which the SIMD kernels rely on.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "host_buffer.h"

#include "heap.h"

#include <string.h>

static const char *backend_name(const struct geist_backend *be) {
    return be != nullptr && be->desc != nullptr && be->desc->name != nullptr ? be->desc->name
                                                                             : "cpu";
}

enum geist_status geist_host_buffer_create(struct geist_backend  *be,
                                           size_t                 bytes,
                                           enum geist_buffer_role role,
                                           unsigned int           memory_flags,
                                           struct geist_buffer  **out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (bytes == 0) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "%s: zero-byte buffer requested", backend_name(be));
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "%s: failed to allocate buffer handle", backend_name(be));
        return GEIST_E_OOM;
    }

    /* SIMD-friendly alignment for activation/weight data. */
    void *host = heap_alloc_aligned(bytes, OPTIMAL_ALIGNMENT);
    if (host == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "%s: failed to allocate %zu-byte host buffer",
                                backend_name(be),
                                bytes);
        return GEIST_E_OOM;
    }

    *buf = (struct geist_buffer) {
            .host         = host,
            .bytes        = bytes,
            .role         = role,
            .memory_flags = memory_flags,
    };
    *out = buf;
    return GEIST_OK;
}

enum geist_status geist_host_buffer_create_aliased(struct geist_backend  *be,
                                                   void                  *host_ptr,
                                                   size_t                 n_bytes,
                                                   enum geist_buffer_role role,
                                                   struct geist_buffer  **out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (host_ptr == nullptr || n_bytes == 0) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "%s: aliased buffer needs host_ptr + bytes",
                                backend_name(be));
        return GEIST_E_INVALID_ARG;
    }
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "%s: buffer handle alloc", backend_name(be));
        return GEIST_E_OOM;
    }
    *buf = (struct geist_buffer) {
            .host         = host_ptr,
            .bytes        = n_bytes,
            .role         = role,
            .memory_flags = GEIST_MEMORY_ALIASED,
    };
    *out = buf;
    return GEIST_OK;
}

void geist_host_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    /* Aliased host memory is owned externally (typically the gguf_reader's
     * mmap); free only the handle. */
    if ((buf->memory_flags & GEIST_MEMORY_ALIASED) == 0 && buf->host != nullptr) {
        safe_free(&buf->host);
    }
    geist_backend_free(be, buf);
}

enum geist_status geist_host_buffer_upload(struct geist_buffer *buf,
                                           size_t               n_bytes,
                                           const uint8_t        src[static n_bytes]) {
    if (buf == nullptr || n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    memcpy(buf->host, src, n_bytes);
    return GEIST_OK;
}

enum geist_status geist_host_buffer_download(size_t                     n_bytes,
                                             uint8_t                    dst[static n_bytes],
                                             const struct geist_buffer *buf) {
    if (buf == nullptr || n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    memcpy(dst, buf->host, n_bytes);
    return GEIST_OK;
}

void *geist_host_buffer_map(struct geist_buffer *buf) {
    return buf != nullptr ? buf->host : nullptr;
}

void geist_host_buffer_unmap(struct geist_buffer *buf) {
    /* No-op on CPU; nothing to sync. */
    (void) buf;
}
