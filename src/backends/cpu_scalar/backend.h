/*
 * src/backends/cpu_scalar/backend.h — pure-C reference backend.
 *
 * Layer: BACKEND. Architecture-agnostic primitive op implementations
 * in scalar C (no SIMD intrinsics). Serves three roles:
 *   1. Reference implementation against which optimized backends validate
 *      (the cross-reference *_unit tests).
 *   2. Portability fallback for x86 dev hosts (no NEON).
 *   3. Algorithm documentation — clearer to read than NEON-optimized code.
 *
 * Defined in (src/backends/cpu_scalar/):
 *   backend.c         — descriptor, lifecycle, capability
 *   weight_resolve.c  — load-time kernel binding (row dequant + dot)
 *   elementwise.c     — per-element ops and norms
 *   transformer_ops.c — RoPE, embedding lookup, attention
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_SCALAR_H
#define GEIST_INTERNAL_BACKEND_CPU_SCALAR_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_scalar/backend.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_types.h>

/* Backend vtable — every backend implements this (engine-known shape). */
struct geist_backend_vtbl;

struct geist_backend_descriptor {
    const char                      *name;
    const struct geist_backend_vtbl *vtbl;
    /* Capability matrix — declared per-op, NULL-terminated. */
    const struct geist_op_support_query *caps;
    size_t                               n_caps;
};

extern const struct geist_backend_descriptor geist_backend_cpu_scalar;

#endif /* GEIST_INTERNAL_BACKEND_CPU_SCALAR_H */
