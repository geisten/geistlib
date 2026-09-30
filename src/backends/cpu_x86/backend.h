/*
 * src/backends/cpu_x86/backend.h — x86_64 backend (AVX2 / AVX-512 / +VNNI / +BF16).
 *
 * Layer: BACKEND.
 *
 * The descriptor (gated by GEIST_BACKEND_CPU_X86) starts from cpu_scalar's
 * vtbl, prims and fused tables and overrides what cpu_x86 has native code
 * for: create/destroy, resolve_weight (the AVX2 / AVX-512 linear kernels,
 * per dtype), the per-phase OpenMP regions, attention and the GELU
 * entries. backend.c has the list.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/backend.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_types.h>

struct geist_backend_descriptor;

extern const struct geist_backend_descriptor geist_backend_cpu_x86;

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_H */
