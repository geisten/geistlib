/*
 * src/backends/cpu_neon/backend.h — ARM64 NEON-optimized backend.
 *
 * Layer: BACKEND. ARMv8-A AArch64 NEON intrinsics for hot kernels
 * (quantized linear, fused attention, ...). Ops without a NEON path use
 * scalar code inside this backend, never cpu_scalar's symbols.
 *
 * Defined in (src/backends/cpu_neon/):
 *   backend.c          — descriptor, lifecycle, capability
 *   weight_resolve.c   — load-time kernel binding (the dtype × ISA table)
 *   kernels/           — the linear kernels, one file per weight format
 *   tl1.c              — W1.58 × A8 LUT-GEMV decode kernel
 *   elementwise.c, transformer_ops.c — elementwise ops and norms; RoPE,
 *                        embedding, attention
 *   parallel.h, workspace.c — schedule(dynamic) on geist_par_for, per-thread
 *                        scratch
 *   kernel_catalog.c, calibrate.c — kernel policy and its calibration
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_neon/backend.h is internal to the backend layer."
#endif

#include <geist.h>

struct geist_backend_vtbl;

struct geist_backend_descriptor;

extern const struct geist_backend_descriptor geist_backend_cpu_neon;

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_H */
