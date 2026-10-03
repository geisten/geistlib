/*
 * src/backends/cpu_x86/linear_q8_0.h — cpu_x86 native Q8_0 linear.
 *
 * Layer: BACKEND (cpu_x86, internal). See linear_q8_0.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q8_0_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q8_0_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_q8_0.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

/* Bind w->linear_m1 / linear_mN to the int8 Q8_0 kernels (Q8_0 weights
 * straight from the GGUF bytes, activations quantized to Q8_0 blocks once
 * per call). Returns false and leaves `w` untouched unless w is Q8_0 with
 * n_in a whole number of 32-element blocks. No repack, no aux memory.
 * M>1 takes the AVX-512 VNNI tiles where the ISA gate allows them. */
[[nodiscard]] bool cpu_x86_linear_q8_0_bind(struct geist_weight *w);

/* The same for Q4_0 weights (nibble codes unpacked to int8 per block,
 * straight from the GGUF bytes). M>1 runs the AVX2 kernel on every host. */
[[nodiscard]] bool cpu_x86_linear_q4_0_bind(struct geist_weight *w);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q8_0_H */
