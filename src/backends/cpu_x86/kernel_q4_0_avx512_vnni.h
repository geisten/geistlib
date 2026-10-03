/*
 * src/backends/cpu_x86/kernel_q4_0_avx512_vnni.h — Q4_0 / Q4_1 x Q8_0
 * prefill tiles, AVX-512 VNNI.
 *
 * Layer: BACKEND (cpu_x86, internal). See kernel_q4_0_avx512_vnni.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q4_0_AVX512_VNNI_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q4_0_AVX512_VNNI_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_q4_0_avx512_vnni.h is internal to the backend layer."
#endif

#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>

/* Output rows per register tile; callers split n_out into groups of this. */
constexpr size_t Q4_0_VNNI_TILE_ROWS = 4;

/* Output rows [j0, j0 + n_rows) of y = W x for all m activation rows. W is
 * the Q4_0 (or Q4_1) weight, row-major, nb blocks per row. q_x / d_x / s_x
 * are the activations quantized to Q8_0 plus each block's integer sum: row
 * t at q_x + t * nb * 32, d_x + t * nb, s_x + t * nb. y is [m, n_out]
 * row-major; only the named rows are written. n_rows is
 * Q4_0_VNNI_TILE_ROWS except for a last, shorter group. Extents are
 * products over runtime dimensions, hence plain pointers.
 *
 * The int32 block sums are those of linear_q4_0.c's AVX2 kernel; the fp32
 * summation order differs, so the two agree to float rounding. Only call on
 * a host whose dispatcher tier is AVX512_VNNI: this TU is compiled with
 * -mavx512*. */
void q4_0_gemm_rows_avx512_vnni(size_t                     m,
                                size_t                     nb,
                                size_t                     n_out,
                                size_t                     j0,
                                size_t                     n_rows,
                                const struct block_q4_0_t *w,
                                const int8_t              *q_x,
                                const float               *d_x,
                                const int32_t             *s_x,
                                float                     *y);

void q4_1_gemm_rows_avx512_vnni(size_t                     m,
                                size_t                     nb,
                                size_t                     n_out,
                                size_t                     j0,
                                size_t                     n_rows,
                                const struct block_q4_1_t *w,
                                const int8_t              *q_x,
                                const float               *d_x,
                                const int32_t             *s_x,
                                float                     *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q4_0_AVX512_VNNI_H */
