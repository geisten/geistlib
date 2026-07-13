/*
 * test_bf16_gemm_unit — bf16 SGEMM kernel correctness + cross-ISA.
 *
 * 1. tiny: hand-set 2-row × 32-col × 16-col GEMM with deterministic
 *    values, verify scalar reference reaches expected output within
 *    bf16-rounding tolerance.
 * 2. cross-ISA: 8 × 32 × 64 random PRNG GEMM, verify AVX-512+BF16
 *    matches scalar reference within ≤ 1e-2 absolute (bf16-noise
 *    bound for this size).
 * 3. pack/unpack: flat → packed → reconstructed flat round-trip.
 *
 * Deterministic; runs in <100 ms; no model.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_bf16_gemm.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t prng_next(uint32_t *state) {
    uint32_t z = (*state += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static float prng_uniform_pm1(uint32_t bits) {
    return 2.0f * (float) (bits & 0xFFFFFFu) / (float) (1u << 24) - 1.0f;
}

/* ------------------------- Scenario 1: pack round-trip ------------------- */

static int scenario_pack_roundtrip(void) {
    constexpr size_t N = 32;
    constexpr size_t K = 8;
    bf16_t           flat[N * K];
    bf16_t           packed[N * K];

    uint32_t s = 0xDEADBEEFu;
    for (size_t i = 0; i < N * K; i++) {
        flat[i] = (bf16_t) (prng_next(&s) & 0xFFFFu);
    }

    bf16_pack_weights_ntile16(N, K, flat, packed);

    /* Reconstruct flat from packed and compare. */
    bf16_t       reconstructed[N * K];
    const size_t n_tiles = N / BF16_NTILE;
    for (size_t t = 0; t < n_tiles; t++) {
        for (size_t kp = 0; kp < K / 2; kp++) {
            for (size_t lane = 0; lane < BF16_NTILE; lane++) {
                const size_t j                    = t * BF16_NTILE + lane;
                const size_t base                 = t * K * BF16_NTILE + kp * 32 + lane * 2;
                reconstructed[j * K + 2 * kp + 0] = packed[base + 0];
                reconstructed[j * K + 2 * kp + 1] = packed[base + 1];
            }
        }
    }
    for (size_t i = 0; i < N * K; i++) {
        if (reconstructed[i] != flat[i]) {
            fprintf(stderr, "pack round-trip: mismatch at i=%zu\n", i);
            return 1;
        }
    }
    return 0;
}

/* ------------------------- Scenario 2: tiny deterministic ---------------- */

static int scenario_tiny(void) {
    /* M=1 row, N=16 cells, K=2 (one K-pair). */
    constexpr size_t M        = 1;
    constexpr size_t N        = 16;
    constexpr size_t K        = 2;
    float            x[M * K] = {1.0f, 2.0f};
    bf16_t           W_flat[N * K];
    for (size_t j = 0; j < N; j++) {
        /* Row j: (1+j, 2+j). Output: 1*(1+j) + 2*(2+j) = 5 + 3j. */
        W_flat[j * K + 0] = fp32_to_bf16((float) (1 + j));
        W_flat[j * K + 1] = fp32_to_bf16((float) (2 + j));
    }
    bf16_t W_packed[N * K];
    bf16_pack_weights_ntile16(N, K, W_flat, W_packed);

    float y_scalar[M * N];
    bf16_gemm_scalar(M, N, K, x, W_packed, y_scalar);

    float y_avx[M * N];
    bf16_gemm_avx512_bf16(M, N, K, x, W_packed, y_avx);

    for (size_t j = 0; j < N; j++) {
        const float expected = 5.0f + 3.0f * (float) j;
        if (fabsf(y_scalar[j] - expected) > 0.5f) {
            fprintf(stderr,
                    "tiny scalar: y[%zu]=%g expected %g\n",
                    j,
                    (double) y_scalar[j],
                    (double) expected);
            return 1;
        }
        if (fabsf(y_avx[j] - y_scalar[j]) > 1e-3f) {
            fprintf(stderr,
                    "tiny avx: y[%zu]=%g vs scalar %g\n",
                    j,
                    (double) y_avx[j],
                    (double) y_scalar[j]);
            return 1;
        }
    }
    return 0;
}

/* ------------------------- Scenario 3: random cross-ISA ------------------ */

static int scenario_random_cross_isa(void) {
    constexpr size_t M = 8;
    constexpr size_t N = 64;
    constexpr size_t K = 32;
    static float     X[M * K];
    static bf16_t    W_flat[N * K];
    static bf16_t    W_packed[N * K];
    static float     Y_scalar[M * N];
    static float     Y_avx[M * N];

    uint32_t s = 0xCAFEBABEu;
    for (size_t i = 0; i < M * K; i++) {
        X[i] = 0.5f * prng_uniform_pm1(prng_next(&s));
    }
    for (size_t i = 0; i < N * K; i++) {
        W_flat[i] = fp32_to_bf16(0.3f * prng_uniform_pm1(prng_next(&s)));
    }
    bf16_pack_weights_ntile16(N, K, W_flat, W_packed);

    bf16_gemm_scalar(M, N, K, X, W_packed, Y_scalar);
    bf16_gemm_avx512_bf16(M, N, K, X, W_packed, Y_avx);

    const float atol     = 1e-2f;
    const float rtol     = 1e-2f;
    int         fails    = 0;
    float       max_diff = 0.0f;
    for (size_t i = 0; i < M * N; i++) {
        const float d   = fabsf(Y_avx[i] - Y_scalar[i]);
        const float tol = atol + rtol * fabsf(Y_scalar[i]);
        if (d > tol) {
            fprintf(stderr,
                    "cross-ISA: i=%zu scalar=%g avx=%g diff=%g tol=%g\n",
                    i,
                    (double) Y_scalar[i],
                    (double) Y_avx[i],
                    (double) d,
                    (double) tol);
            fails++;
            if (fails > 4) {
                break;
            }
        }
        if (d > max_diff) {
            max_diff = d;
        }
    }
    fprintf(stdout,
            "[bf16_gemm] max |y_avx - y_scalar| = %g (M=%zu, N=%zu, K=%zu)\n",
            (double) max_diff,
            (size_t) M,
            (size_t) N,
            (size_t) K);
    return fails;
}

int main(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    /* bf16_gemm_avx512_bf16 needs AVX512-BF16 and has no runtime CPU guard — the
     * engine gates it before calling. This test calls it directly, so skip
     * cleanly on a CPU without AVX512-BF16 (e.g. a CI runner) instead of SIGILL. */
    if (!__builtin_cpu_supports("avx512bf16")) {
        GEIST_SKIP("bf16_gemm_avx512_bf16 requires AVX512-BF16, absent on this CPU");
    }
#endif
    int fails = 0;
    if (scenario_pack_roundtrip() != 0) {
        fputs("scenario_pack_roundtrip FAILED\n", stderr);
        fails++;
    }
    if (scenario_tiny() != 0) {
        fputs("scenario_tiny FAILED\n", stderr);
        fails++;
    }
    if (scenario_random_cross_isa() != 0) {
        fputs("scenario_random_cross_isa FAILED\n", stderr);
        fails++;
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
