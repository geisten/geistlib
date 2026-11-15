/*
 * src/backends/cpu_neon/kernels/f32_rows.h — fp32 GEMM on a dequantized
 * weight tile whose rows do not depend on the row count (#528).
 *
 * Layer: BACKEND (cpu_neon, internal).
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_F32_ROWS_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_F32_ROWS_H

#include <stddef.h>

/* y[i * ldy + r] = dot(x row i, w row r) for i < m, r < n_rows: x is
 * [m, n_in] and w is [n_rows, n_in], both row-major and dense, n_in a
 * multiple of 4 (the block formats' are multiples of 32); y has m rows of
 * stride ldy >= n_rows, of which columns [0, n_rows) are written.
 *
 * Every output is one dot product summed in one fixed order (four lane
 * sums over n_in in steps of 4, reduced low + high, then the pair),
 * whatever m and n_rows are and wherever the row sits in the call. A BLAS
 * gives no such promise: Accelerate and OpenBLAS send a one-row product
 * through gemv, which sums in another order than gemm. Single thread; the
 * caller tiles over the weight rows. */
void f32_rows_gemm(
        size_t m, size_t n_rows, size_t n_in, size_t ldy, const float *x, const float *w, float *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_F32_ROWS_H */
