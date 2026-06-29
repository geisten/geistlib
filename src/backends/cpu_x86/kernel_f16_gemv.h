/*
 * src/backends/cpu_x86/kernel_f16_gemv.h — F16-weight decode GEMV.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * y[r] = Σ_i f16_to_f32(W[r,i]) * x[i],  W row-major [n_out, n_in].
 *
 * The motivating case is BitNet-2B-4T's tied lm_head (F16, 128 K × 2560 =
 * 657 MB): read once per decode step, so this is bandwidth-bound and the
 * lever is OMP across rows + an in-register F16C convert (no f32
 * materialization of the weight). Compiled at the x86-64-v3 baseline
 * (F16C + FMA are in v3); no AVX-512 needed.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_F16_GEMV_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_F16_GEMV_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_f16_gemv.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

void f16_gemv_m1(size_t        n_out,
                 size_t        n_in,
                 const float  *x,
                 const uint16_t w_f16[],
                 float         y[static n_out]);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_F16_GEMV_H */
