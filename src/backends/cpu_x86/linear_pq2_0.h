/*
 * src/backends/cpu_x86/linear_pq2_0.h — cpu_x86 PQ2_0 decode GEMV (W2 x A8).
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

/* Bind w->linear_m1 to the int8 PQ2_0 GEMV (weights straight from the GGUF
 * bytes, the activation quantized to int8 once per call). linear_mN is not
 * touched: the caller binds prefill first. Returns false and leaves `w`
 * untouched unless w is PQ2_0 with n_in a whole number of 128-element
 * blocks. No repack, no aux memory. */
[[nodiscard]] bool cpu_x86_linear_pq2_0_bind_m1(struct geist_weight *w);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_PQ2_0_H */
