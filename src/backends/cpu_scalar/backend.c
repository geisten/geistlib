/*
 * src/backends/cpu_scalar/backend.c — pure-C reference backend.
 *
 * Layer: BACKEND.
 *
 * Reference implementation: every dtype the resolver accepts runs
 * dequant-to-fp32 plus a naive matmul. Correctness oracle for the
 * optimized backends, not a performance path.
 *
 * The geist_buffer for this backend wraps a host pointer plus its size,
 * role, and memory_flags. Allocation routes through the user-provided
 * geist_allocator (default: heap.h via geist_libc_allocator).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include <geist.h>
#include <geist_backend.h>

#include "heap.h"
#include "quant.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---------- Lifecycle ---------- */

[[nodiscard]] static enum geist_status cpu_scalar_create(struct geist_backend            *be,
                                                         const struct geist_backend_opts *opts) {
    (void) opts; /* B-2b will read max_threads, log_cb, etc. */

    struct cpu_scalar_state *st =
            geist_backend_alloc(be, sizeof(*st), alignof(struct cpu_scalar_state));
    if (st == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "cpu_scalar: failed to allocate %zu-byte state", sizeof(*st));
        return GEIST_E_OOM;
    }
    *st       = (struct cpu_scalar_state) {0};
    be->state = st;
    return GEIST_OK;
}

static void cpu_scalar_destroy(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    geist_backend_free(be, be->state);
    be->state = nullptr;
}

/* ---------- Buffer ops ---------- */

[[nodiscard]] static enum geist_status cpu_scalar_buffer_create(struct geist_backend  *be,
                                                                size_t                 bytes,
                                                                enum geist_buffer_role role,
                                                                unsigned int           memory_flags,
                                                                struct geist_buffer  **out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (bytes == 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar: zero-byte buffer requested");
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "cpu_scalar: failed to allocate buffer handle");
        return GEIST_E_OOM;
    }

    /* Use SIMD-friendly alignment for activation/weight data. */
    void *host = heap_alloc_aligned(bytes, OPTIMAL_ALIGNMENT);
    if (host == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(
                be, GEIST_E_OOM, "cpu_scalar: failed to allocate %zu-byte host buffer", bytes);
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

[[nodiscard]] static enum geist_status cpu_scalar_buffer_create_aliased(struct geist_backend *be,
                                                                        void  *host_ptr,
                                                                        size_t n_bytes,
                                                                        enum geist_buffer_role role,
                                                                        struct geist_buffer **out) {

    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (host_ptr == nullptr || n_bytes == 0) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_scalar: aliased buffer needs host_ptr + bytes");
        return GEIST_E_INVALID_ARG;
    }
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "cpu_scalar: buffer handle alloc");
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

static void cpu_scalar_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    /* Aliased buffers (P0.3): see cpu_neon mirror — don't free host_ptr. */
    if ((buf->memory_flags & GEIST_MEMORY_ALIASED) == 0 && buf->host != nullptr) {
        safe_free(&buf->host);
    }
    geist_backend_free(be, buf);
}

[[nodiscard]] static enum geist_status cpu_scalar_buffer_upload(struct geist_buffer *buf,
                                                                size_t               n_bytes,
                                                                const uint8_t src[static n_bytes]) {
    if (buf == nullptr || src == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    memcpy(buf->host, src, n_bytes);
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status cpu_scalar_buffer_download(size_t  n_bytes,
                                                                  uint8_t dst[static n_bytes],
                                                                  const struct geist_buffer *buf) {
    if (buf == nullptr || dst == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    memcpy(dst, buf->host, n_bytes);
    return GEIST_OK;
}

static void *cpu_scalar_buffer_map(struct geist_buffer *buf) {
    return buf != nullptr ? buf->host : nullptr;
}

static void cpu_scalar_buffer_unmap(struct geist_buffer *buf) {
    /* No-op on CPU; nothing to sync. */
    (void) buf;
}

/* ---------- Op dispatcher: linear() -------------------------------------
 * Routes to the format-specific implementation based on (W.dtype,
 * W.layout). cpu_scalar_linear_f32_dense lives in linear.c; future
 * implementations (Q4_K, Q3_K, Q8_0) will join in B-2c. */

/* ---------- Vtable + Descriptor ---------- */

/* ponytail: exported (not static) so cpu_x86 can reuse it as its Phase-0
 * vtbl. Per-op slots get replaced with native VPDPBUSD/VDPBF16PS kernels as
 * Phase 1a/1b/2 land. See docs/LINUX_X86_SPEC.md. */
const struct geist_backend_vtbl cpu_scalar_vtbl = {
        .create                = cpu_scalar_create,
        .destroy               = cpu_scalar_destroy,
        .buffer_create         = cpu_scalar_buffer_create,
        .buffer_destroy        = cpu_scalar_buffer_destroy,
        .buffer_create_aliased = cpu_scalar_buffer_create_aliased,
        .buffer_upload         = cpu_scalar_buffer_upload,
        .buffer_download       = cpu_scalar_buffer_download,
        .buffer_map            = cpu_scalar_buffer_map,
        .buffer_unmap          = cpu_scalar_buffer_unmap,
        .resolve_weight        = cpu_scalar_resolve_weight,
};

/* Elementwise F32 fusions hold for any geometry and any m; nothing else
 * is implemented. */
static bool cpu_scalar_fused_supported(struct geist_backend            *be,
                                       const struct geist_fusion_query *q) {
    (void) be;
    if (q == nullptr) {
        return false;
    }
    return q->op == GEIST_FUSED_GELU_TANH_MUL || q->op == GEIST_FUSED_GELU_TANH_MUL_SCALED;
}

/* Exported like the vtbl: cpu_x86 struct-copies these as its Phase-0
 * baseline. */
const struct geist_backend_primitives cpu_scalar_prims = {
        .rmsnorm          = cpu_scalar_rmsnorm,
        .add              = cpu_scalar_add,
        .mul              = cpu_scalar_mul,
        .gelu_tanh        = cpu_scalar_gelu_tanh,
        .silu             = cpu_scalar_silu,
        .relu_squared     = cpu_scalar_relu_squared,
        .rope_apply       = cpu_scalar_rope_apply,
        .embedding_lookup = cpu_scalar_embedding_lookup,
        .attention        = cpu_scalar_attention,
};

const struct geist_backend_fused cpu_scalar_fused = {
        .supported            = cpu_scalar_fused_supported,
        .gelu_tanh_mul        = cpu_scalar_gelu_tanh_mul,
        .gelu_tanh_mul_scaled = cpu_scalar_gelu_tanh_mul_scaled,
};

const struct geist_backend_descriptor geist_backend_cpu_scalar = {
        .name  = "cpu_scalar",
        .vtbl  = &cpu_scalar_vtbl,
        .prims = &cpu_scalar_prims,
        .fused = &cpu_scalar_fused,
        .caps =
                {
                        .max_m = GEIST_QUANT_M_CAP,
#if defined(__APPLE__)
                        .preferred_kv_mode = GEIST_KV_FP32,
#else
                        .preferred_kv_mode = GEIST_KV_INT8,
#endif
                },
};
