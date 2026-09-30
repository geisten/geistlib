/*
 * src/backends/cpu_x86/linear_generic.h — cpu_x86's linear for every dtype
 * that has no native x86 kernel.
 *
 * Layer: BACKEND (cpu_x86, internal). See linear_generic.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_GENERIC_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_GENERIC_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_generic.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>

/* Bind w->linear_m1 / linear_mN to the generic kernels when they can serve
 * the weight: a dtype with a whole-row decoder (quant.h's formats, F16,
 * BF16 — not I2_S, whose per-tensor scale sits past the last row and which
 * has native kernels) and n_in a whole number of its blocks. Returns false
 * and leaves `w` untouched otherwise.
 *
 * The kernels split rows across OpenMP threads; each thread dequantizes its
 * rows into a private row of the calling thread's workspace and dots them in
 * fp32. M>1 dequantizes each weight row once for all m activation rows. No
 * heap allocation once that workspace has grown to the shape. */
[[nodiscard]] bool cpu_x86_linear_generic_bind(struct geist_weight *w);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_GENERIC_H */
