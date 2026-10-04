/*
 * src/backends/cpu_x86/kernel_tq2_0_avx512_vnni.h — TQ2_0 x int8 prefill
 * tiles, AVX-512 VNNI.
 *
 * Layer: BACKEND (cpu_x86, internal). See kernel_tq2_0_avx512_vnni.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_TQ2_0_AVX512_VNNI_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_TQ2_0_AVX512_VNNI_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_tq2_0_avx512_vnni.h is internal to the backend layer."
#endif

#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>

/* Output rows per register tile; callers split n_out into groups of this. */
constexpr size_t TQ2_0_VNNI_TILE_ROWS = 4;

/* Output rows [j0, j0 + n_rows) of y = W x for all m activation rows. W is
 * the TQ2_0 weight, row-major, nb blocks of 256 per row. q_x / d_x / s_x
 * are the activations quantized to int8 with one scale and one integer sum
 * per 256 elements: row t at q_x + t * nb * 256, d_x + t * nb, s_x + t * nb.
 * y is [m, n_out] row-major; only the named rows are written. n_rows is
 * TQ2_0_VNNI_TILE_ROWS except for a last, shorter group. Extents are
 * products over runtime dimensions, hence plain pointers.
 *
 * The int32 block sums are those of linear_tq2_0.c's AVX2 kernel; the fp32
 * summation order differs, so the two agree to float rounding. Only call on
 * a host whose dispatcher tier is AVX512_VNNI: this TU is compiled with
 * -mavx512*. */
void tq2_0_gemm_rows_avx512_vnni(size_t                      m,
                                 size_t                      nb,
                                 size_t                      n_out,
                                 size_t                      j0,
                                 size_t                      n_rows,
                                 const struct block_tq2_0_t *w,
                                 const int8_t               *q_x,
                                 const float                *d_x,
                                 const int32_t              *s_x,
                                 float                      *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_TQ2_0_AVX512_VNNI_H */
