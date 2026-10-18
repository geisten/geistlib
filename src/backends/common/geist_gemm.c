/*
 * src/backends/common/geist_gemm.c — geist_sgemm / geist_sgemv backends.
 *
 * The dense fp32 GEMM provider is chosen at build time by GEMM_PROVIDER
 * (mk/gemm-<provider>.mk). The `native` provider defines GEIST_GEMM_NATIVE and
 * uses the dependency-free path below; `openblas` / `accelerate` leave it
 * undefined and forward to the CBLAS symbols their library exports. This is
 * orthogonal to HAVE_ACCELERATE, which signals Accelerate's vDSP (FFT,
 * elementwise) is available regardless of the GEMM provider.
 */
#include "geist_gemm.h"
#include <geist_types.h>

#include <stddef.h>

/* CBLAS path: used unless the `native` GEMM provider defined GEIST_GEMM_NATIVE.
 * The cblas symbols come from whichever library the provider linked (Accelerate
 * on macOS, OpenBLAS on Linux/Pi). */
#if !defined(GEIST_GEMM_NATIVE)

/* Declare the two CBLAS entry points directly. We deliberately avoid Apple's
 * <Accelerate/Accelerate.h> umbrella (it drags in vImage and slows the build)
 * and OpenBLAS's <cblas.h>; both libraries export these C symbols with plain
 * int enum arguments, and GEIST_OP_N/T already equal CBLAS NoTrans/Trans. */
enum { GEIST_CBLAS_ROW_MAJOR = 101 };
extern void cblas_sgemm(int          order,
                        int          transA,
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
extern void cblas_sgemv(int          order,
                        int          transA,
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

/* Pin OpenBLAS to 1 thread: only small, skinny dense matmuls come through
 * here (e.g. Gemma 4 PLE projections), where its per-call thread spawn/sync
 * dominates (+9 % Gemma 4 prefill). OpenBLAS-only: the symbol does not exist
 * under Accelerate, and a weak undefined symbol is a link error on Mach-O.
 *
 * pthread_once, not a first-use flag: the first cblas call of a process can
 * come from every thread of an omp team at once (the chunked DeltaNet prefill
 * calls geist_sgemm per v-head). A plain flag lets several threads call
 * openblas_set_num_threads concurrently, and lets a thread that sees it set
 * enter cblas before the pin has taken effect — a multithreaded BLAS call
 * inside an omp region. pthread_once runs the pin once and returns to every
 * caller only after it has completed. */
#if defined(GEIST_GEMM_OPENBLAS)
#include <pthread.h>

extern void openblas_set_num_threads(int);

static pthread_once_t g_blas_pin_once = PTHREAD_ONCE_INIT;

static void geist_blas_pin_init(void) {
    openblas_set_num_threads(1);
}

static void geist_blas_pin_single_thread(void) {
    (void) pthread_once(&g_blas_pin_once, geist_blas_pin_init);
}
#else
static void geist_blas_pin_single_thread(void) {
}
#endif

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
                 int          ldc) {
    geist_blas_pin_single_thread();
    cblas_sgemm(
            GEIST_CBLAS_ROW_MAJOR, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

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
                 int          incy) {
    geist_blas_pin_single_thread();
    cblas_sgemv(GEIST_CBLAS_ROW_MAJOR, transA, M, N, alpha, A, lda, x, incx, beta, y, incy);
}

#else /* native, dependency-free provider */

/* NEON-vectorized for the dominant y = x*W^T pattern (transA=N, transB=T,
 * beta=0); a scalar triple loop covers the rest. Dense fp32 is a few percent
 * of inference, so no OpenBLAS-grade blocking. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

/* x86 AVX2/FMA kernels (#12). x86-64-v3, the Linux x86 target's baseline,
 * guarantees both. Before them the native provider ran every x86 sgemm
 * through the scalar triple loop below, and Qwen3.5's chunked DeltaNet
 * prefill (eight sgemm per chunk) ran at half the OpenBLAS build's speed. */

/* Lanes [0, n) of an 8-float vector, for column tails; n <= 0 is empty,
 * n >= 8 full. A masked load of an empty mask reads nothing. */
static inline __m256i avx2_lane_mask(int n) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}

/* C[i0:i0+MR, j0:j0+w] = alpha * op(A) B + beta C for B not transposed,
 * w <= 16: each k broadcasts one op(A) element per row and FMAs it into two
 * 8-float accumulators per row against a contiguous row of B. MR is a
 * constant at every call, so the accumulators live in registers (6x16:
 * 12 accumulators, 2 B vectors, 1 broadcast of the 16 ymm registers). */
#define AVX2_MR 6
static inline __attribute__((always_inline)) void avx2_tile_bn(int          MR,
                                                               int          w,
                                                               int          transA,
                                                               int          K,
                                                               float        alpha,
                                                               const float *A,
                                                               int          lda,
                                                               const float *B,
                                                               int          ldb,
                                                               float        beta,
                                                               float       *C,
                                                               int          ldc) {
    const __m256i m0 = avx2_lane_mask(w), m1 = avx2_lane_mask(w - 8);
    __m256        acc[AVX2_MR][2];
    for (int r = 0; r < MR; r++)
        acc[r][0] = acc[r][1] = _mm256_setzero_ps();
    for (int k = 0; k < K; k++) {
        const float *Bk = B + (size_t) k * ldb;
        __m256       b0, b1;
        if (w == 16) {
            b0 = _mm256_loadu_ps(Bk);
            b1 = _mm256_loadu_ps(Bk + 8);
        } else {
            b0 = _mm256_maskload_ps(Bk, m0);
            b1 = _mm256_maskload_ps(Bk + 8, m1);
        }
        for (int r = 0; r < MR; r++) {
            const float a =
                    transA == GEIST_OP_N ? A[(size_t) r * lda + k] : A[(size_t) k * lda + r];
            const __m256 av = _mm256_broadcast_ss(&a);
            acc[r][0]       = _mm256_fmadd_ps(av, b0, acc[r][0]);
            acc[r][1]       = _mm256_fmadd_ps(av, b1, acc[r][1]);
        }
    }
    const __m256 va = _mm256_set1_ps(alpha), vb = _mm256_set1_ps(beta);
    for (int r = 0; r < MR; r++) {
        float *Cr = C + (size_t) r * ldc;
        for (int h = 0; h < 2; h++) {
            const __m256i m = h == 0 ? m0 : m1;
            __m256        v = _mm256_mul_ps(va, acc[r][h]);
            /* beta == 0: C is write-only and may hold anything, even NaN */
            if (beta != 0.0f)
                v = _mm256_fmadd_ps(vb, _mm256_maskload_ps(Cr + 8 * h, m), v);
            _mm256_maskstore_ps(Cr + 8 * h, m, v);
        }
    }
}

static void avx2_sgemm_bn(int          transA,
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
                          int          ldc) {
    for (int j = 0; j < N; j += 16) {
        const int w = N - j < 16 ? N - j : 16;
        int       i = 0;
        /* op(A) row i starts at A[i*lda] (N) or column i at A[i] (T). */
        for (; i + AVX2_MR <= M; i += AVX2_MR)
            avx2_tile_bn(AVX2_MR,
                         w,
                         transA,
                         K,
                         alpha,
                         transA == GEIST_OP_N ? A + (size_t) i * lda : A + i,
                         lda,
                         B + j,
                         ldb,
                         beta,
                         C + (size_t) i * ldc + j,
                         ldc);
        for (; i < M; i++)
            avx2_tile_bn(1,
                         w,
                         transA,
                         K,
                         alpha,
                         transA == GEIST_OP_N ? A + (size_t) i * lda : A + i,
                         lda,
                         B + j,
                         ldb,
                         beta,
                         C + (size_t) i * ldc + j,
                         ldc);
    }
}

static inline float avx2_hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s        = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s        = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* C[i0:i0+MR, j0:j0+NR] for C = alpha A B^T + beta C (both operands
 * K-contiguous): dot products along K, 8 at a time, a K tail under a mask.
 * 2x4 tiles: 8 accumulators. */
static inline __attribute__((always_inline)) void avx2_tile_nt(int          MR,
                                                               int          NR,
                                                               int          K,
                                                               float        alpha,
                                                               const float *A,
                                                               int          lda,
                                                               const float *B,
                                                               int          ldb,
                                                               float        beta,
                                                               float       *C,
                                                               int          ldc) {
    __m256 acc[2][4];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            acc[r][c] = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 a[2];
        for (int r = 0; r < MR; r++)
            a[r] = _mm256_loadu_ps(A + (size_t) r * lda + k);
        for (int c = 0; c < NR; c++) {
            const __m256 b = _mm256_loadu_ps(B + (size_t) c * ldb + k);
            for (int r = 0; r < MR; r++)
                acc[r][c] = _mm256_fmadd_ps(a[r], b, acc[r][c]);
        }
    }
    if (k < K) {
        const __m256i m = avx2_lane_mask(K - k);
        __m256        a[2];
        for (int r = 0; r < MR; r++)
            a[r] = _mm256_maskload_ps(A + (size_t) r * lda + k, m);
        for (int c = 0; c < NR; c++) {
            const __m256 b = _mm256_maskload_ps(B + (size_t) c * ldb + k, m);
            for (int r = 0; r < MR; r++)
                acc[r][c] = _mm256_fmadd_ps(a[r], b, acc[r][c]);
        }
    }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++) {
            float      *dst = C + (size_t) r * ldc + c;
            const float v   = alpha * avx2_hsum(acc[r][c]);
            *dst            = beta == 0.0f ? v : v + beta * *dst;
        }
}

static void avx2_sgemm_nt(int          M,
                          int          N,
                          int          K,
                          float        alpha,
                          const float *A,
                          int          lda,
                          const float *B,
                          int          ldb,
                          float        beta,
                          float       *C,
                          int          ldc) {
    int i = 0;
    for (; i + 2 <= M; i += 2) {
        int j = 0;
        for (; j + 4 <= N; j += 4)
            avx2_tile_nt(2,
                         4,
                         K,
                         alpha,
                         A + (size_t) i * lda,
                         lda,
                         B + (size_t) j * ldb,
                         ldb,
                         beta,
                         C + (size_t) i * ldc + j,
                         ldc);
        for (; j < N; j++)
            avx2_tile_nt(2,
                         1,
                         K,
                         alpha,
                         A + (size_t) i * lda,
                         lda,
                         B + (size_t) j * ldb,
                         ldb,
                         beta,
                         C + (size_t) i * ldc + j,
                         ldc);
    }
    for (; i < M; i++) {
        int j = 0;
        for (; j + 4 <= N; j += 4)
            avx2_tile_nt(1,
                         4,
                         K,
                         alpha,
                         A + (size_t) i * lda,
                         lda,
                         B + (size_t) j * ldb,
                         ldb,
                         beta,
                         C + (size_t) i * ldc + j,
                         ldc);
        for (; j < N; j++)
            avx2_tile_nt(1,
                         1,
                         K,
                         alpha,
                         A + (size_t) i * lda,
                         lda,
                         B + (size_t) j * ldb,
                         ldb,
                         beta,
                         C + (size_t) i * ldc + j,
                         ldc);
    }
}
#endif

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
                 int          ldc) {
#if defined(__ARM_NEON)
    if (transA == GEIST_OP_N && transB == GEIST_OP_T && beta == 0.0f) {
        /* C[M,N] = alpha * A[M,K] * B[N,K]^T (the y = x*W^T pattern). 4x4
         * register-blocked microkernel: a 4x4 C-tile in 16 accumulators, so
         * each B row is reused across 4 A rows — 4x less weight traffic than
         * a 1-row sweep. Edges below. */
        const int Kv = K & ~3;
        int       i  = 0;
        for (; i + 4 <= M; i += 4) {
            const float *Ap[4] = {A + (size_t) (i + 0) * lda,
                                  A + (size_t) (i + 1) * lda,
                                  A + (size_t) (i + 2) * lda,
                                  A + (size_t) (i + 3) * lda};
            float       *Cp[4] = {C + (size_t) (i + 0) * ldc,
                                  C + (size_t) (i + 1) * ldc,
                                  C + (size_t) (i + 2) * ldc,
                                  C + (size_t) (i + 3) * ldc};
            int          j     = 0;
            for (; j + 4 <= N; j += 4) {
                const float *Bp[4] = {B + (size_t) (j + 0) * ldb,
                                      B + (size_t) (j + 1) * ldb,
                                      B + (size_t) (j + 2) * ldb,
                                      B + (size_t) (j + 3) * ldb};
                float32x4_t  c[4][4];
                for (int ii = 0; ii < 4; ii++)
                    for (int jj = 0; jj < 4; jj++)
                        c[ii][jj] = vdupq_n_f32(0);
                for (int k = 0; k < Kv; k += 4) {
                    float32x4_t av[4], bv[4];
                    for (int ii = 0; ii < 4; ii++)
                        av[ii] = vld1q_f32(Ap[ii] + k);
                    for (int jj = 0; jj < 4; jj++)
                        bv[jj] = vld1q_f32(Bp[jj] + k);
                    for (int ii = 0; ii < 4; ii++)
                        for (int jj = 0; jj < 4; jj++)
                            c[ii][jj] = vfmaq_f32(c[ii][jj], av[ii], bv[jj]);
                }
                for (int ii = 0; ii < 4; ii++)
                    for (int jj = 0; jj < 4; jj++) {
                        float r = vaddvq_f32(c[ii][jj]);
                        for (int k = Kv; k < K; k++)
                            r += Ap[ii][k] * Bp[jj][k];
                        Cp[ii][j + jj] = alpha * r;
                    }
            }
            for (; j < N; j++) { /* N-edge: 4 A-rows x 1 B-row */
                const float *Bj    = B + (size_t) j * ldb;
                float32x4_t  cc[4] = {
                        vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
                for (int k = 0; k < Kv; k += 4) {
                    const float32x4_t bv = vld1q_f32(Bj + k);
                    for (int ii = 0; ii < 4; ii++)
                        cc[ii] = vfmaq_f32(cc[ii], vld1q_f32(Ap[ii] + k), bv);
                }
                for (int ii = 0; ii < 4; ii++) {
                    float r = vaddvq_f32(cc[ii]);
                    for (int k = Kv; k < K; k++)
                        r += Ap[ii][k] * Bj[k];
                    Cp[ii][j] = alpha * r;
                }
            }
        }
        for (; i < M; i++) { /* M-edge: 1 A-row, block N by 4 */
            const float *Ai = A + (size_t) i * lda;
            float       *Ci = C + (size_t) i * ldc;
            int          j  = 0;
            for (; j + 4 <= N; j += 4) {
                const float *Bp[4] = {B + (size_t) (j + 0) * ldb,
                                      B + (size_t) (j + 1) * ldb,
                                      B + (size_t) (j + 2) * ldb,
                                      B + (size_t) (j + 3) * ldb};
                float32x4_t  cc[4] = {
                        vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
                for (int k = 0; k < Kv; k += 4) {
                    const float32x4_t av = vld1q_f32(Ai + k);
                    for (int jj = 0; jj < 4; jj++)
                        cc[jj] = vfmaq_f32(cc[jj], av, vld1q_f32(Bp[jj] + k));
                }
                for (int jj = 0; jj < 4; jj++) {
                    float r = vaddvq_f32(cc[jj]);
                    for (int k = Kv; k < K; k++)
                        r += Ai[k] * Bp[jj][k];
                    Ci[j + jj] = alpha * r;
                }
            }
            for (; j < N; j++) {
                const float *Bj = B + (size_t) j * ldb;
                float32x4_t  a  = vdupq_n_f32(0);
                int          k  = 0;
                for (; k < Kv; k += 4)
                    a = vfmaq_f32(a, vld1q_f32(Ai + k), vld1q_f32(Bj + k));
                float r = vaddvq_f32(a);
                for (; k < K; k++)
                    r += Ai[k] * Bj[k];
                Ci[j] = alpha * r;
            }
        }
        return;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    if (M > 0 && N > 0 && K > 0) {
        if (transB == GEIST_OP_N) {
            avx2_sgemm_bn(transA, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
        if (transA == GEIST_OP_N) {
            avx2_sgemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
    }
#endif
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                const float a =
                        (transA == GEIST_OP_N) ? A[(size_t) i * lda + k] : A[(size_t) k * lda + i];
                const float b =
                        (transB == GEIST_OP_N) ? B[(size_t) k * ldb + j] : B[(size_t) j * ldb + k];
                acc += a * b;
            }
            float *c = &C[(size_t) i * ldc + j];
            *c       = alpha * acc + (beta == 0.0f ? 0.0f : beta * *c);
        }
    }
}

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
                 int          incy) {
    if (transA == GEIST_OP_N) {
#if defined(__ARM_NEON)
        if (incx == 1 && incy == 1 && beta == 0.0f) {
            const int Nv = N & ~3;
            for (int i = 0; i < M; i++) {
                const float *Ai = A + (size_t) i * lda;
                float32x4_t  a  = vdupq_n_f32(0);
                int          j  = 0;
                for (; j < Nv; j += 4)
                    a = vfmaq_f32(a, vld1q_f32(Ai + j), vld1q_f32(x + j));
                float c = vaddvq_f32(a);
                for (; j < N; j++)
                    c += Ai[j] * x[j];
                y[i] = alpha * c;
            }
            return;
        }
#endif
        for (int i = 0; i < M; i++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += A[(size_t) i * lda + j] * x[(size_t) j * incx];
            float *yi = &y[(size_t) i * incy];
            *yi       = alpha * acc + (beta == 0.0f ? 0.0f : beta * *yi);
        }
    } else {
        /* y[N] = alpha * A^T x[M] + beta y : scale y first, then accumulate. */
        for (int j = 0; j < N; j++) {
            float *yj = &y[(size_t) j * incy];
            *yj       = (beta == 0.0f ? 0.0f : beta * *yj);
        }
        for (int i = 0; i < M; i++) {
            const float xi = alpha * x[(size_t) i * incx];
            for (int j = 0; j < N; j++)
                y[(size_t) j * incy] += A[(size_t) i * lda + j] * xi;
        }
    }
}

#endif
