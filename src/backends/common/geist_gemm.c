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

/* ---- optional profiling (GEIST_PROFILE_GEMM=1) ---------------------------
 * Accumulates wall time spent in the dense-fp32 facade across the whole run
 * and prints it at exit. Used to size the dense-fp32 share of inference
 * (ROADMAP.md Step 2) — the gate for going BLAS-free on linux-arm64. Backend-
 * agnostic: wraps whichever backend is compiled in. */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdatomic.h>

static _Atomic unsigned long long g_gemm_ns = 0, g_gemm_calls = 0;
static _Atomic unsigned long long g_gemv_ns = 0, g_gemv_calls = 0;

static void geist_gemm_report(void) {
    fprintf(stderr,
            "[geist_gemm] dense fp32: sgemm %llu calls / %.2f ms, "
            "sgemv %llu calls / %.2f ms, total %.2f ms\n",
            (unsigned long long) g_gemm_calls,
            g_gemm_ns / 1e6,
            (unsigned long long) g_gemv_calls,
            g_gemv_ns / 1e6,
            (g_gemm_ns + g_gemv_ns) / 1e6);
}

static int geist_gemm_prof(void) {
    static _Atomic int enabled = -1;
    int                e       = atomic_load(&enabled);
    if (e < 0) {
        const char *s = getenv("GEIST_PROFILE_GEMM");
        e             = (s != NULL && s[0] == '1') ? 1 : 0;
        int expected  = -1;
        if (atomic_compare_exchange_strong(&enabled, &expected, e) && e)
            atexit(geist_gemm_report);
    }
    return e;
}

static unsigned long long geist_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long) ts.tv_sec * 1000000000ull + (unsigned long long) ts.tv_nsec;
}

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

/* OpenBLAS multithreads each cblas call via its own pool. geist already
 * parallelizes the heavy (quantized) matmuls itself and only routes small,
 * skinny F32/BF16 dense matmuls (e.g. Gemma 4's per-layer PLE projections,
 * called ~70× per prefill chunk) through cblas — there the per-call thread
 * spawn/sync overhead dominates. Pin BLAS to 1 thread once (measured +9% on
 * Gemma 4 prefill). OpenBLAS-only: the symbol doesn't exist under Accelerate
 * (which manages its own threading) and a weak undefined symbol is a hard
 * link error on Mach-O — so gate it on the openblas provider macro. */
#if defined(GEIST_GEMM_OPENBLAS)
extern void openblas_set_num_threads(int);

static void geist_blas_pin_single_thread(void) {
    static int done = 0;
    if (done) {
        return;
    }
    done = 1;
    openblas_set_num_threads(1);
}
#else
static void geist_blas_pin_single_thread(void) {
}
#endif

static void geist_sgemm_impl(int          transA,
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

static void geist_sgemv_impl(int          transA,
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

#else /* ---- native, dependency-free fallback -------------------------------     \
       * NEON-vectorized for the dominant y = x*W^T pattern (transA=N, transB=T,   \
       * beta=0); a scalar triple loop covers the rare cases (N/N, transposed A,   \
       * beta!=0). Used by the BLAS-free build only; the quant W*A8 hot path never \
       * reaches here. Per ROADMAP.md Step 2, dense fp32 is ~2.6% of inference, so \
       * "decent" suffices — this needs no OpenBLAS-grade blocking. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static void geist_sgemm_impl(int          transA,
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
         * register-blocked microkernel: a 4-row x 4-col C-tile is held in 16
         * accumulators, so each B-row K-vector is reused across 4 A-rows (and
         * vice versa) — 4x less weight traffic than a 1-row sweep, which on
         * the big model_proj GEMM (B re-read once per token otherwise) is the
         * difference between ~6x and ~1.5x off OpenBLAS. Edges handled scalar-
         * vectorized. */
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

static void geist_sgemv_impl(int          transA,
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

/* Public entry points: dispatch to the selected backend, optionally timing the
 * call when GEIST_PROFILE_GEMM=1. */
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
    if (!geist_gemm_prof()) {
        geist_sgemm_impl(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    const unsigned long long t0 = geist_now_ns();
    geist_sgemm_impl(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    atomic_fetch_add(&g_gemm_ns, geist_now_ns() - t0);
    atomic_fetch_add(&g_gemm_calls, 1);
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
    if (!geist_gemm_prof()) {
        geist_sgemv_impl(transA, M, N, alpha, A, lda, x, incx, beta, y, incy);
        return;
    }
    const unsigned long long t0 = geist_now_ns();
    geist_sgemv_impl(transA, M, N, alpha, A, lda, x, incx, beta, y, incy);
    atomic_fetch_add(&g_gemv_ns, geist_now_ns() - t0);
    atomic_fetch_add(&g_gemv_calls, 1);
}
