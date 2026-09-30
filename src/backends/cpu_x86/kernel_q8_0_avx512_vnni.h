/*
 * src/backends/cpu_x86/kernel_q8_0_avx512_vnni.h — Q8_0 x Q8_0 prefill tiles,
 * AVX-512 VNNI.
 *
 * Layer: BACKEND (cpu_x86, internal). See kernel_q8_0_avx512_vnni.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q8_0_AVX512_VNNI_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q8_0_AVX512_VNNI_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_q8_0_avx512_vnni.h is internal to the backend layer."
#endif

#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>

/* Output rows per register tile; callers split n_out into groups of this. */
constexpr size_t Q8_0_VNNI_TILE_ROWS = 4;

/* Output rows [j0, j0 + n_rows) of y = W x for all m activation rows:
 *
 *   y[t * n_out + j] = sum_b d_w[j,b] * d_x[t,b] * sum_k q_w[j,b,k] * q_x[t,b,k]
 *
 * W is the Q8_0 weight, row-major, nb blocks per row (n_out rows). q_x / d_x
 * are the activations quantized to Q8_0: row t at q_x + t * nb * 32 and
 * d_x + t * nb. y is [m, n_out] row-major; only the named rows are written.
 * n_rows is Q8_0_VNNI_TILE_ROWS except for a last, shorter group.
 * Extents are products over runtime dimensions, hence plain pointers.
 *
 * Bit-identical to linear_q8_0.c's AVX2 M>1 kernel for finite activations.
 * Only call on a host whose dispatcher tier is AVX512_VNNI: this TU is
 * compiled with -mavx512*. */
void q8_0_gemm_rows_avx512_vnni(size_t                     m,
                                size_t                     nb,
                                size_t                     n_out,
                                size_t                     j0,
                                size_t                     n_rows,
                                const struct block_q8_0_t *w,
                                const int8_t              *q_x,
                                const float               *d_x,
                                float                     *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q8_0_AVX512_VNNI_H */
