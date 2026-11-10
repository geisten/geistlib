/*
 * src/backends/cpu_scalar/backend.c — pure-C reference backend.
 *
 * Layer: BACKEND.
 *
 * Reference implementation: every dtype the resolver accepts runs
 * dequant-to-fp32 plus a naive matmul. Correctness oracle for the
 * optimized backends, not a performance path.
 *
 * Buffers are the CPU backends' shared host buffers
 * (common/host_buffer.c); handles route through the user-provided
 * geist_allocator (default: heap.h via geist_libc_allocator).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include <geist.h>
#include <geist_backend.h>
#include "backend_tables.h"

#include "hadamard.h"
#include "heap.h"
#include "host_ops.h"
#include "quant.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---------- Lifecycle ---------- */

[[nodiscard]] static enum geist_status cpu_scalar_create(struct geist_backend            *be,
                                                         const struct geist_backend_opts *opts) {
    (void) opts;

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

/* Buffer ops: geist_host_buffer_* (common/host_buffer.c). */

/* ---------- Vtable + Descriptor ---------- */

static const struct geist_backend_vtbl cpu_scalar_vtbl =
        GEIST_BACKEND_VTBL(cpu_scalar_destroy,
                           geist_host_buffer_create,
                           geist_host_buffer_destroy,
                           geist_host_buffer_create_aliased,
                           geist_host_buffer_upload,
                           geist_host_buffer_download,
                           geist_host_buffer_map,
                           geist_host_buffer_unmap,
                           .create         = cpu_scalar_create,
                           .resolve_weight = cpu_scalar_resolve_weight);

/* Elementwise F32 fusions hold for any geometry and any m; the Hadamard
 * rotation for the geometry the shared host transform accepts. Nothing
 * else is implemented. */
static bool cpu_scalar_fused_supported(struct geist_backend            *be,
                                       const struct geist_fusion_query *q) {
    (void) be;
    if (q == nullptr) {
        return false;
    }
    switch (q->op) {
    case GEIST_FUSED_GELU_TANH_MUL:
    case GEIST_FUSED_GELU_TANH_MUL_SCALED:
        return true;
    case GEIST_FUSED_HADAMARD_ROTATE:
        return geist_hadamard_query_ok(q);
    case GEIST_FUSED_SILU_MUL:
    case GEIST_FUSED_FFN_GATE_UP:
    case GEIST_FUSED_FFN_NORM_GATE_UP:
    case GEIST_FUSED_RMSNORM_ADD:
    case GEIST_FUSED_ATTN_QKV_PREP:
    case GEIST_FUSED_PLE_BLOCK:
    case GEIST_FUSED_EMBEDDING_LOOKUP_SCALED:
    case GEIST_FUSED_ARGMAX_F32:
    case GEIST_FUSED_BITNET_ACT_QUANT:
    case GEIST_FUSED_ATTN_KV_INT8:
    case GEIST_FUSED_ATTN_KV_INT4:
    case GEIST_FUSED_DELTANET_MIX:
    case GEIST_FUSED_RMSNORM_ADD_NORM:
    case GEIST_FUSED_LINEAR_GROUPED:
    case GEIST_FUSED_ROWS_WEIGHTED_SUM:
    case GEIST_FUSED_FFN_GATE_UP_SILU:
    case GEIST_FUSED_ATTN_ROPE_APPEND:
    case GEIST_FUSED_ROWS_GATHER:
    case GEIST_FUSED_MOE_DEVICE:
        break;
    }
    return false;
}

static const struct geist_backend_primitives cpu_scalar_prims =
        GEIST_BACKEND_PRIMS(cpu_scalar_rmsnorm,
                            cpu_scalar_add,
                            cpu_scalar_mul,
                            cpu_scalar_gelu_tanh,
                            cpu_scalar_silu,
                            cpu_scalar_rope_apply,
                            cpu_scalar_attention,
                            .relu_squared = cpu_scalar_relu_squared,
                            .add_bias     = geist_host_add_bias,
                            .layernorm    = geist_host_layernorm);

/* Exported: cpu_x86 copies it as the base of its own. */
const struct geist_backend_fused cpu_scalar_fused = {
        .supported            = cpu_scalar_fused_supported,
        .gelu_tanh_mul        = cpu_scalar_gelu_tanh_mul,
        .gelu_tanh_mul_scaled = cpu_scalar_gelu_tanh_mul_scaled,
        .hadamard_rotate      = geist_host_hadamard_rotate,
        .kv_append_f16        = geist_host_kv_append_f16,
};

/* The KV cache a session gets on GEIST_KV_AUTO. A constant, not an #if
 * inside the descriptor's macro arguments, where a directive is undefined. */
#if defined(__APPLE__)
static constexpr enum geist_kv_mode cpu_scalar_kv_mode = GEIST_KV_FP32;
#else
static constexpr enum geist_kv_mode cpu_scalar_kv_mode = GEIST_KV_INT8;
#endif

const struct geist_backend_descriptor geist_backend_cpu_scalar =
        GEIST_BACKEND_DESCRIPTOR("cpu_scalar",
                                 cpu_scalar_vtbl,
                                 cpu_scalar_prims,
                                 .fused = &cpu_scalar_fused,
                                 .caps  = {
                                         .max_m = GEIST_QUANT_M_CAP,
                                         /* host DeltaNet sub-chunks (layer_deltanet.c) */
                                         .dn_subchunk = true,
                                         /* F16 KV on request only (kv_f16_by_default unset):
                                          * a family or session asks for it (#48). */
                                         .kv_f16_attention  = true,
                                         .preferred_kv_mode = cpu_scalar_kv_mode,
                                 });
