/*
 * test_geist_gemm_unit — geist_sgemm against a double-precision reference
 * (#12): every transpose combination, beta 0 / 1 / 0.5, alpha 1 / -1 / 0.7,
 * the gated-DeltaNet chunk shapes and ragged edges, with row strides wider
 * than the rows. With beta 0, C starts as NaN: it must be write-only.
 *
 * Whatever provider the build links (OpenBLAS, Accelerate, or the native
 * one with its NEON / AVX2 kernels) must agree with the reference within
 * the error bound of an fp32 dot product: |err| <= 4 * K * eps * sum|a*b|
 * (+ beta's term), the accumulation order being the provider's choice.
 * Built with -fno-finite-math-only (mk/common.mk): the NaN sentinels are the
 * point, and the Linux targets' -ffast-math would fold isnan() away.
 */
#include "test_helpers.h"

#include "geist_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;

static float uniform(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (float) ((double) (rng >> 11) / (double) (1ull << 53) * 2.0 - 1.0);
}

/* One case: op(A)[M,K] op(B)[K,N], strides padded by `pad`. */
static int check(int ta, int tb, int M, int N, int K, float alpha, float beta, int pad) {
    const int    lda = (ta == GEIST_OP_N ? K : M) + pad;
    const int    ldb = (tb == GEIST_OP_N ? N : K) + pad;
    const int    ldc = N + pad;
    const size_t na  = (size_t) (ta == GEIST_OP_N ? M : K) * lda;
    const size_t nb  = (size_t) (tb == GEIST_OP_N ? K : N) * ldb;
    const size_t nc  = (size_t) M * ldc;
    float       *A = malloc(na * sizeof *A), *B = malloc(nb * sizeof *B);
    float       *C = malloc(nc * sizeof *C), *C0 = malloc(nc * sizeof *C0);
    if (A == nullptr || B == nullptr || C == nullptr || C0 == nullptr) {
        fprintf(stderr, "FAIL: allocation\n");
        exit(GEIST_TEST_FAIL);
    }
    for (size_t i = 0; i < na; i++)
        A[i] = uniform();
    for (size_t i = 0; i < nb; i++)
        B[i] = uniform();
    for (size_t i = 0; i < nc; i++)
        C[i] = C0[i] = beta == 0.0f ? NAN : uniform();

    geist_sgemm(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    int bad = 0;
    for (int i = 0; i < M && !bad; i++) {
        for (int j = 0; j < N; j++) {
            double ref = 0.0, mag = 0.0;
            for (int k = 0; k < K; k++) {
                const double a =
                        ta == GEIST_OP_N ? A[(size_t) i * lda + k] : A[(size_t) k * lda + i];
                const double b =
                        tb == GEIST_OP_N ? B[(size_t) k * ldb + j] : B[(size_t) j * ldb + k];
                ref += a * b;
                mag += fabs(a * b);
            }
            const double c0 = C0[(size_t) i * ldc + j];
            ref             = alpha * ref + (beta == 0.0f ? 0.0 : beta * c0);
            const double tol =
                    4.0 * (K + 2) * 1.2e-7 *
                            (fabs(alpha) * mag + (beta == 0.0f ? 0.0 : fabs(beta * c0))) +
                    1e-30;
            const double got = C[(size_t) i * ldc + j];
            if (!(fabs(got - ref) <= tol)) {
                fprintf(stderr,
                        "FAIL: op %c%c M=%d N=%d K=%d alpha=%g beta=%g pad=%d: C[%d,%d] = %.9g, "
                        "reference %.9g (tol %.3g)\n",
                        ta == GEIST_OP_N ? 'N' : 'T',
                        tb == GEIST_OP_N ? 'N' : 'T',
                        M,
                        N,
                        K,
                        (double) alpha,
                        (double) beta,
                        pad,
                        i,
                        j,
                        got,
                        ref,
                        tol);
                bad = 1;
                break;
            }
        }
    }
    /* Padding columns of C are not part of the result and must stay, bit
     * for bit (NaN sentinels included, so compare bytes, not floats). */
    for (int i = 0; i < M && !bad; i++)
        if (ldc > N && memcmp(C + (size_t) i * ldc + N,
                              C0 + (size_t) i * ldc + N,
                              (size_t) (ldc - N) * sizeof *C) != 0) {
            fprintf(stderr,
                    "FAIL: write past N into C's row padding (op %c%c M=%d N=%d K=%d alpha=%g "
                    "beta=%g)\n",
                    ta == GEIST_OP_N ? 'N' : 'T',
                    tb == GEIST_OP_N ? 'N' : 'T',
                    M,
                    N,
                    K,
                    (double) alpha,
                    (double) beta);
            bad = 1;
        }
    free(A);
    free(B);
    free(C);
    free(C0);
    return bad;
}

int main(void) {
    const int ops[2] = {GEIST_OP_N, GEIST_OP_T};
    /* The DeltaNet chunk shapes (layer_deltanet.c, chunk C = 64, d = 128),
     * a projection-like shape, and ragged edges around the kernels' 6x16,
     * 2x4 and 8-wide K tiles. */
    const int   shapes[][3] = {{64, 64, 128},
                               {64, 128, 64},
                               {64, 128, 128},
                               {128, 128, 64},
                               {7, 384, 96},
                               {1, 1, 1},
                               {1, 17, 9},
                               {5, 15, 7},
                               {6, 16, 8},
                               {13, 33, 17},
                               {12, 31, 1},
                               {2, 4, 3},
                               {19, 9, 40},
                               {3, 50, 65}};
    const float alphas[3]   = {1.0f, -1.0f, 0.7f};
    const float betas[3]    = {0.0f, 1.0f, 0.5f};
    int         fails = 0, cases = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                for (int al = 0; al < 3; al++)
                    for (int be = 0; be < 3; be++)
                        for (int pad = 0; pad <= 3; pad += 3) {
                            fails += check(ops[a],
                                           ops[b],
                                           shapes[s][0],
                                           shapes[s][1],
                                           shapes[s][2],
                                           alphas[al],
                                           betas[be],
                                           pad);
                            cases++;
                        }
    if (fails) {
        fprintf(stderr, "%d of %d sgemm cases failed\n", fails, cases);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: geist_sgemm matches the double reference in %d cases\n", cases);
    return GEIST_TEST_PASS;
}
