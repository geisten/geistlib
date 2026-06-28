/*
 * test_w8a8_gemm_unit — Q6_K prefill GEMM correctness (cpu_x86).
 *
 * The W8A8 tiled GEMM (cpu_x86_linear_q6k_mN's inner kernel) must match
 * the trusted per-token W8A8 GEMV (the decode m1 path) for the same
 * weights and activations. Synthetic W8A8 weights (uint8 [0,63] + per-16
 * scale/offset, matching the Q6_K predecoder range), random fp32 acts
 * quantized to int8, GEMM vs per-token GEMV.
 *
 * Deterministic; no GGUF needed; runs in <100 ms.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_w8a8.h"
#include "../src/backends/cpu_x86/kernel_w4a8.h" /* w4a8_quantize_acts_row */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t prng_next(uint32_t *s) {
    uint32_t z = (*s += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static int scenario(size_t M, size_t N, size_t K) {
    const size_t NB = K / W8A8_BLOCK_ELEMS;
    uint32_t     s  = 0xB16B00B5u;

    uint8_t *W  = malloc(N * NB * W8A8_BLOCK_ELEMS);
    float   *ws = malloc(N * NB * sizeof(float));
    float   *wo = malloc(N * NB * sizeof(float));
    for (size_t i = 0; i < N * NB * W8A8_BLOCK_ELEMS; i++) W[i] = (uint8_t) (prng_next(&s) & 0x3Fu);
    for (size_t i = 0; i < N * NB; i++) {
        ws[i] = 0.001f + ((prng_next(&s) & 0xFFFFu) / 65536.0f) * 0.01f;
        wo[i] = ((prng_next(&s) & 0xFFFFu) / 65536.0f) * 0.5f;
    }

    float *X = malloc(M * K * sizeof(float));
    for (size_t i = 0; i < M * K; i++) X[i] = 2.0f * ((prng_next(&s) & 0xFFFFu) / 65536.0f) - 1.0f;

    int8_t  *acts = malloc(M * K);
    int32_t *sa   = malloc(M * NB * sizeof(int32_t));
    float   *sx   = malloc(M * sizeof(float));
    int32_t *tmp  = malloc((K / W4A8_BLOCK_ELEMS) * sizeof(int32_t));
    for (size_t j = 0; j < M; j++) {
        sx[j] = w4a8_quantize_acts_row(K, X + j * K, acts + j * K, tmp);
        for (size_t b = 0; b < NB; b++) {
            int32_t t = 0;
            for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++) t += acts[j * K + b * W8A8_BLOCK_ELEMS + i];
            sa[j * NB + b] = t;
        }
    }

    float *Yg   = malloc(M * N * sizeof(float));
    float *Yref = malloc(M * N * sizeof(float));
    w8a8_gemm(M, N, NB, W, ws, wo, acts, sa, sx, Yg);
    for (size_t j = 0; j < M; j++)
        w8a8_gemv(N, NB, W, ws, wo, acts + j * K, sa + j * NB, sx[j], Yref + j * N);

    double maxd = 0.0;
    int    fails = 0;
    for (size_t i = 0; i < M * N; i++) {
        const double d = fabs((double) Yg[i] - (double) Yref[i]);
        if (d > maxd) maxd = d;
        if (d > 1e-3) fails++;
    }
    fprintf(stdout, "[w8a8_gemm  M=%zu N=%zu K=%zu] max |Δ vs gemv| = %g, fails=%d\n",
            M, N, K, maxd, fails);

    /* Lane-parallel W8x8 (VNNI hosts; N % 8 == 0). Repack the row-major
     * weights and compare the lane-parallel GEMM to the same GEMV oracle. */
    if (w8a8_isa_is_vnni() && (N % W8X8_NROWS) == 0) {
        uint8_t *qs  = malloc(N * K);
        float   *qsc = malloc(N * NB * sizeof(float));
        float   *qof = malloc(N * NB * sizeof(float));
        float   *Yx  = malloc(M * N * sizeof(float));
        w8x8_repack(N, K, W, ws, wo, qs, qsc, qof);
        w8x8_gemm(M, N, NB, qs, qsc, qof, acts, sa, sx, Yx);
        double maxdx = 0.0;
        for (size_t i = 0; i < M * N; i++) {
            const double d = fabs((double) Yx[i] - (double) Yref[i]);
            if (d > maxdx) maxdx = d;
            if (d > 1e-3) fails++;
        }
        fprintf(stdout, "[w8x8_gemm  M=%zu N=%zu K=%zu] max |Δ vs gemv| = %g, fails=%d\n",
                M, N, K, maxdx, fails);
        free(qs); free(qsc); free(qof); free(Yx);
    }

    /* Lane-parallel 512-bit W8x16 (N % 16 == 0). */
    if (w8a8_isa_is_vnni() && (N % W8X16_NROWS) == 0) {
        uint8_t *qs  = malloc(N * K);
        float   *qsc = malloc(N * NB * sizeof(float));
        float   *qof = malloc(N * NB * sizeof(float));
        float   *Yx  = malloc(M * N * sizeof(float));
        w8x16_repack(N, K, W, ws, wo, qs, qsc, qof);
        w8x16_gemm(M, N, NB, qs, qsc, qof, acts, sa, sx, Yx);
        double maxdx = 0.0;
        for (size_t i = 0; i < M * N; i++) {
            const double d = fabs((double) Yx[i] - (double) Yref[i]);
            if (d > maxdx) maxdx = d;
            if (d > 1e-3) fails++;
        }
        fprintf(stdout, "[w8x16_gemm M=%zu N=%zu K=%zu] max |Δ vs gemv| = %g, fails=%d\n",
                M, N, K, maxdx, fails);
        free(qs); free(qsc); free(qof); free(Yx);
    }

    free(W); free(ws); free(wo); free(X);
    free(acts); free(sa); free(sx); free(tmp); free(Yg); free(Yref);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += scenario(4, 32, 256);     /* JT-exact, small */
    fails += scenario(7, 48, 512);     /* M not divisible by JT (=4) → tail */
    fails += scenario(64, 1536, 12288); /* real Gemma 4 ffn_down shape */
    if (fails == 0) fprintf(stdout, "OK\n");
    return fails == 0 ? 0 : 1;
}
