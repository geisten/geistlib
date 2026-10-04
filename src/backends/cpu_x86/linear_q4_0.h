/*
 * src/backends/cpu_x86/linear_q4_0.h — cpu_x86 native Q4_0 / Q4_1 / IQ4_NL /
 * IQ4_XS linear.
 *
 * Layer: BACKEND (cpu_x86, internal). See linear_q4_0.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4_0_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4_0_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_q4_0.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

/* Bind w->linear_m1 / linear_mN to the int8 Q4_0 / Q4_1 kernels (weights
 * straight from the GGUF bytes, activations quantized to Q8_0 blocks once
 * per call). Returns false and leaves `w` untouched unless w is Q4_0 or
 * Q4_1 with n_in a whole number of 32-element blocks. No repack, no aux
 * memory. */
[[nodiscard]] bool cpu_x86_linear_q4_0_bind(struct geist_weight *w);

/* The same for IQ4_NL (n_in a whole number of 32-element blocks) and IQ4_XS
 * (256-element super-blocks). Returns false and leaves `w` untouched for any
 * other dtype or shape. */
[[nodiscard]] bool cpu_x86_linear_iq4_bind(struct geist_weight *w);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4_0_H */
