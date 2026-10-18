/*
 * src/backends/cpu_x86/kernel_pq2_0_avx512_vnni.h — PQ2_0 x int8 prefill
 * tiles, AVX-512 VNNI.
 *
 * Layer: BACKEND (cpu_x86, internal). See kernel_pq2_0_avx512_vnni.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_PQ2_0_AVX512_VNNI_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_PQ2_0_AVX512_VNNI_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_pq2_0_avx512_vnni.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

/* Output rows per register tile; callers split n_out into groups of this. */
constexpr size_t PQ2_0_VNNI_TILE_ROWS = 4;

/* Output rows [j0, j0 + n_rows) of y = W x for all m activation rows, in
 * linear_pq2_0.c's M>1 layout: raw is the PQ2_0 weight, row-major, rb bytes
 * per row (nb blocks of 34); block b of token t's int8 activation at
 * xq + (b * m + t) * 128, in the codes' order (shift level l of code byte
 * k is element 4k + l, at 32 * l + k); neg_sum[b * m + t] is minus that
 * block's activation sum, inv[t] the token's inverse scale. y is [m, n_out]
 * row-major; only the named rows are written. n_rows is
 * PQ2_0_VNNI_TILE_ROWS except for a last, shorter group.
 *
 * The int32 block sums equal the AVX2 kernel's; only the fp32 summation
 * order differs. Only call on a host whose dispatcher tier is AVX512_VNNI:
 * this TU is compiled with -mavx512*. */
void pq2_0_gemm_rows_avx512_vnni(size_t         m,
                                 size_t         nb,
                                 size_t         rb,
                                 size_t         n_out,
                                 size_t         j0,
                                 size_t         n_rows,
                                 const uint8_t *raw,
                                 const int8_t  *xq,
                                 const int32_t *neg_sum,
                                 const float   *inv,
                                 float         *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_PQ2_0_AVX512_VNNI_H */
