/*
 * src/backends/common/geist_gemm.h — single dense-fp32 GEMM/GEMV facade.
 *
 * All dense fp32 matmul/matvec call sites route through geist_sgemm /
 * geist_sgemv instead of calling cblas directly. The backend is selected at
 * build time in geist_gemm.c:
 *   - CBLAS (Accelerate on macOS / OpenBLAS on Linux) by default, OR
 *   - a dependency-free native implementation when GEIST_GEMM_NATIVE is set.
 *
 * Row-major is implied (every call site uses it). Quantized paths have their
 * own native kernels and do NOT go through here — this is only for dense fp32.
 */
#ifndef GEIST_GEMM_H
#define GEIST_GEMM_H

/* Transpose selector. Values match CBLAS (NoTrans=111, Trans=112) so the cblas
 * backend forwards them verbatim. */
enum { GEIST_OP_N = 111, GEIST_OP_T = 112 };

/* C[M,N] = alpha * op(A) * op(B) + beta * C   (row-major).
 *   op(A) = A    if transA==GEIST_OP_N (A is [M,K]), else A^T (A is [K,M]).
 *   op(B) = B    if transB==GEIST_OP_N (B is [K,N]), else B^T (B is [N,K]).
 *   lda/ldb/ldc are the row strides of A/B/C. beta==0 treats C as write-only. */
void geist_sgemm(int          transA,
                 int          transB,
                 int          M,
                 int          N,
                 int          K,
                 float        alpha,
                 const float *A,
                 int          lda,
                 const float *B,
                 int          ldb,
                 float        beta,
                 float       *C,
                 int          ldc);

/* y = alpha * op(A) * x + beta * y, with A stored row-major as [M,N].
 *   transA==GEIST_OP_N: y has length M, x has length N.
 *   transA==GEIST_OP_T: y has length N, x has length M.
 *   incx/incy are element strides; beta==0 treats y as write-only. */
void geist_sgemv(int          transA,
                 int          M,
                 int          N,
                 float        alpha,
                 const float *A,
                 int          lda,
                 const float *x,
                 int          incx,
                 float        beta,
                 float       *y,
                 int          incy);

#endif /* GEIST_GEMM_H */
