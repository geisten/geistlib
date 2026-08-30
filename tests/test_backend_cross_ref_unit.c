/*
 * test_backend_cross_ref_unit — verifies cpu_scalar ≡ cpu_neon for
 * F32 DENSE linear, plus measures dispatch overhead.
 *
 * Cross-reference test: identical input through both backends; outputs
 * must match within rtol=1e-4 (BLAS sgemm vs naive triple-loop differ
 * in reduction order, so ULP-1 doesn't apply).
 *
 * Bench: same workload via both backends + a direct (no-vtable) cblas
 * sgemm baseline. Reports ms/call and GFLOP/s, plus the dispatch overhead
 * of going through the backend vtable.
 *
 * Skips cleanly if cpu_neon is not compiled in (BACKENDS=cpu_scalar only).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Forward-declare cblas_sgemm for the no-dispatch baseline (same pattern
 * as the cpu_neon backend uses internally). */
typedef enum CBLAS_ORDER { CblasRowMajor = 101 } CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 } CBLAS_TRANSPOSE;
extern void cblas_sgemm(CBLAS_ORDER,
                        CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB,
                        int             M,
                        int             N,
                        int             K,
                        float           alpha,
                        const float    *A,
                        int             lda,
                        const float    *B,
                        int             ldb,
                        float           beta,
                        float          *C,
                        int             ldc);

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

static void fill_random_f32(float *p, size_t n, uint32_t seed) {
    /* Deterministic xorshift32 — same input across runs and backends. */
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        /* Map to [-1, 1) */
        p[i] = (float) ((int32_t) s) * (1.0f / (float) INT32_MAX);
    }
}

/* Run linear() via the given backend and time the matmul portion. */
[[nodiscard]] static enum geist_status run_via_backend(struct geist_backend *be,
                                                       size_t                M,
                                                       size_t                K,
                                                       size_t                N,
                                                       const float          *xdata,
                                                       const float          *wdata,
                                                       float                *yout,
                                                       double               *out_ms) {
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    enum geist_status    s;

    s = be->desc->vtbl->buffer_create(
            be, M * K * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bx);
    if (s != GEIST_OK)
        goto cleanup;
    s = be->desc->vtbl->buffer_create(
            be, N * K * sizeof(float), GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &bw);
    if (s != GEIST_OK)
        goto cleanup;
    s = be->desc->vtbl->buffer_create(
            be, M * N * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    if (s != GEIST_OK)
        goto cleanup;

    s = be->desc->vtbl->buffer_upload(bx, M * K * sizeof(float), (const uint8_t *) xdata);
    if (s != GEIST_OK)
        goto cleanup;
    s = be->desc->vtbl->buffer_upload(bw, N * K * sizeof(float), (const uint8_t *) wdata);
    if (s != GEIST_OK)
        goto cleanup;

    /* P2.e: resolve once, time the resolver-installed kernel. The
     * legacy v->linear() vtable slot is gone after P2.e. */
    void *w_host = be->desc->vtbl->buffer_map(bw);
    if (w_host == nullptr) {
        s = GEIST_E_BACKEND;
        goto cleanup;
    }
    struct geist_weight wkr = {
            .raw        = w_host,
            .raw_nbytes = (size_t) K * N * sizeof(float),
            .n_in       = (int32_t) K,
            .n_out      = (int32_t) N,
            .dtype      = (uint16_t) GEIST_DTYPE_F32,
    };
    s = be->desc->vtbl->resolve_weight(be, &wkr);
    be->desc->vtbl->buffer_unmap(bw);
    if (s != GEIST_OK || wkr.linear_mN == nullptr)
        goto cleanup;

    void *x_host = be->desc->vtbl->buffer_map(bx);
    void *y_host = be->desc->vtbl->buffer_map(by);
    if (x_host == nullptr || y_host == nullptr) {
        s = GEIST_E_BACKEND;
        goto cleanup;
    }

    /* Warm-up call (BLAS often does first-call lazy init). */
    wkr.linear_mN(M, (const float *) x_host, &wkr, be, (float *) y_host);

    /* Timed runs: take min over 5 iterations. */
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = monotonic_ms();
        wkr.linear_mN(M, (const float *) x_host, &wkr, be, (float *) y_host);
        double dt = monotonic_ms() - t0;
        if (dt < best)
            best = dt;
    }

    be->desc->vtbl->buffer_unmap(bx);
    be->desc->vtbl->buffer_unmap(by);
    *out_ms = best;
    s       = be->desc->vtbl->buffer_download(M * N * sizeof(float), (uint8_t *) yout, by);

cleanup:
    if (bx)
        be->desc->vtbl->buffer_destroy(be, bx);
    if (bw)
        be->desc->vtbl->buffer_destroy(be, bw);
    if (by)
        be->desc->vtbl->buffer_destroy(be, by);
    return s;
}

int main(void) {
    /* Workload: representative LM matmul shape (Gemma 4 q_proj-like).
     * M=8 (batched prefill), K=2048 (hidden), N=2048 (out_features).
     * 33 MFLOPs per call, easy to time. */
    const size_t M = 8, K = 2048, N = 2048;
    const double mflops = 2.0 * (double) M * (double) K * (double) N / 1e6;

    float *xdata    = aligned_alloc(64, M * K * sizeof(float));
    float *wdata    = aligned_alloc(64, N * K * sizeof(float));
    float *y_scalar = aligned_alloc(64, M * N * sizeof(float));
    float *y_neon   = aligned_alloc(64, M * N * sizeof(float));
    float *y_direct = aligned_alloc(64, M * N * sizeof(float));

    if (!xdata || !wdata || !y_scalar || !y_neon || !y_direct) {
        fprintf(stderr, "alloc failed\n");
        return GEIST_TEST_ERROR;
    }

    fill_random_f32(xdata, M * K, 0xCAFEBABE);
    fill_random_f32(wdata, N * K, 0xDEADBEEF);

    int               fails = 0;
    enum geist_status s;

    /* ---- Direct cblas_sgemm baseline (no vtable) ---- */
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                (int) M,
                (int) N,
                (int) K,
                1.0f,
                xdata,
                (int) K,
                wdata,
                (int) K,
                0.0f,
                y_direct,
                (int) N); /* warmup */
    double best_direct = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = monotonic_ms();
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    (int) M,
                    (int) N,
                    (int) K,
                    1.0f,
                    xdata,
                    (int) K,
                    wdata,
                    (int) K,
                    0.0f,
                    y_direct,
                    (int) N);
        double dt = monotonic_ms() - t0;
        if (dt < best_direct)
            best_direct = dt;
    }

    /* ---- cpu_scalar via vtable ---- */
    struct geist_backend *be_scalar = nullptr;
    s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be_scalar);
    if (s != GEIST_OK) {
        fprintf(stderr, "cpu_scalar create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    double scalar_ms = 0;
    s                = run_via_backend(be_scalar, M, K, N, xdata, wdata, y_scalar, &scalar_ms);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "scalar run: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be_scalar));
        return GEIST_TEST_FAIL;
    }
    geist_backend_destroy(be_scalar);

    /* ---- cpu_neon via vtable (skip if not linked) ---- */
    struct geist_backend *be_neon = nullptr;
    s                             = geist_backend_create("cpu_neon", nullptr, nullptr, &be_neon);
    if (s == GEIST_E_NOT_FOUND) {
        printf("SKIP partial: cpu_neon not compiled in; only cpu_scalar measured\n");
        printf("  cpu_scalar (M=%zu K=%zu N=%zu): %.3f ms = %.2f GFLOP/s\n",
               M,
               K,
               N,
               scalar_ms,
               mflops / scalar_ms);
        return GEIST_TEST_PASS;
    } else if (s != GEIST_OK) {
        fprintf(stderr, "cpu_neon create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    double neon_ms = 0;
    s              = run_via_backend(be_neon, M, K, N, xdata, wdata, y_neon, &neon_ms);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "neon run: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be_neon));
        return GEIST_TEST_FAIL;
    }
    geist_backend_destroy(be_neon);

    /* ---- Cross-reference: scalar ≈ neon ---- */
    ptrdiff_t bad = geist_fp32_close_array(y_scalar, y_neon, M * N, 1e-4f, 1e-3f);
    if (bad >= 0) {
        fprintf(stderr,
                "FAIL: scalar[%td]=%.6f vs neon[%td]=%.6f\n",
                bad,
                (double) y_scalar[bad],
                bad,
                (double) y_neon[bad]);
        fails++;
    }

    /* Also: neon ≈ direct cblas (should be near-zero diff) */
    bad = geist_fp32_close_array(y_neon, y_direct, M * N, 1e-6f, 1e-6f);
    if (bad >= 0) {
        fprintf(stderr,
                "FAIL: neon[%td]=%.6f vs direct[%td]=%.6f\n",
                bad,
                (double) y_neon[bad],
                bad,
                (double) y_direct[bad]);
        fails++;
    }

    /* ---- Report ---- */
    printf("F32 DENSE linear (M=%zu K=%zu N=%zu, %.1f MFLOPs):\n", M, K, N, mflops);
    printf("  cpu_scalar (vtable) : %8.3f ms = %6.2f GFLOP/s\n", scalar_ms, mflops / scalar_ms);
    printf("  cpu_neon   (vtable) : %8.3f ms = %6.2f GFLOP/s\n", neon_ms, mflops / neon_ms);
    printf("  direct sgemm (none) : %8.3f ms = %6.2f GFLOP/s\n", best_direct, mflops / best_direct);
    printf("  vtable overhead     : %8.3f ms (= %.2f%% of best path)\n",
           neon_ms - best_direct,
           (neon_ms - best_direct) / best_direct * 100.0);
    printf("  scalar speedup      : %.1fx slower than direct\n", scalar_ms / best_direct);

    free(xdata);
    free(wdata);
    free(y_scalar);
    free(y_neon);
    free(y_direct);

    if (fails == 0) {
        printf("PASS: scalar/neon/direct all agree within tolerance\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
