/*
 * src/backends/cpu_x86/linear_pq2_0.h — cpu_x86 PQ2_0 linear (W2 x A8).
 *
 * Layer: BACKEND (cpu_x86, internal). See linear_pq2_0.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_PQ2_0_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_PQ2_0_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_pq2_0.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

/* Bind w->linear_m1 / linear_mN to the int8 PQ2_0 kernels: weights
 * straight from the GGUF bytes, each activation row quantized to int8 once
 * per call (a GEMV for M=1, a GEMM for M>1: AVX2, or AMX-INT8 where
 * cpu_x86_linear_pq2_0_amx_usable). w is PQ2_0 and has passed
 * quant_weight_extent_ok, so n_in is a whole number of 128-element blocks.
 * No repack, no aux memory. */
void cpu_x86_linear_pq2_0_bind(struct geist_weight *w);

/* Whether the bind installs the AMX-INT8 GEMM for M>1 on this host: the
 * dispatcher tier (GEIST_FORCE_ISA below avx512_vnni turns it off), cpuid,
 * and Linux's permission for the tile data, which this call requests. */
[[nodiscard]] bool cpu_x86_linear_pq2_0_amx_usable(void);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_PQ2_0_H */
