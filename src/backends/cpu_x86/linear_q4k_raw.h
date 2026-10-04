/*
 * src/backends/cpu_x86/linear_q4k_raw.h — cpu_x86 native Q4_K / Q5_K linear on the GGUF bytes.
 *
 * Layer: BACKEND (cpu_x86, internal). See linear_q4k_raw.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4K_RAW_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4K_RAW_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_q4k_raw.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

/* Bind w->linear_m1 / linear_mN to the int8 Q4_K kernels that read the
 * GGUF block layout directly (activations quantized to int8 with one scale
 * per 256 elements, once per call). Returns false and leaves `w` untouched
 * unless w is Q4_K with n_in a whole number of 256-element superblocks. No
 * repack, no aux memory. */
[[nodiscard]] bool cpu_x86_linear_q4k_raw_bind(struct geist_weight *w);

/* The same kernels for Q5_K (the fifth bit of each q from qh). Q5_K has no
 * repack on cpu_x86, so this is its kernel on every host. */
[[nodiscard]] bool cpu_x86_linear_q5k_bind(struct geist_weight *w);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4K_RAW_H */
