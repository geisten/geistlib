/*
 * src/backends/cpu_x86/linear_tq2_0.h — cpu_x86 native TQ2_0 linear.
 *
 * Layer: BACKEND (cpu_x86, internal). See linear_tq2_0.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_TQ2_0_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_TQ2_0_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_tq2_0.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

/* Bind w->linear_m1 / linear_mN to the int8 TQ2_0 kernels (ternary codes
 * straight from the GGUF bytes, activations quantized to int8 with one
 * scale per 256 elements, once per call). Returns false and leaves `w`
 * untouched unless w is TQ2_0 with n_in a whole number of 256-element
 * blocks. No repack, no aux memory. */
[[nodiscard]] bool cpu_x86_linear_tq2_0_bind(struct geist_weight *w);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_TQ2_0_H */
