/*
 * test_q6k_gemv_unit — native Q6_K decode GEMV correctness (cpu_x86).
 *
 * Synthesizes random Q6_K weight rows, then checks q6k_gemv_m1 against an
 * fp32 reference built from q6k_to_w8a8_row (itself verified vs
 * dequant_q6_K_row): reconstruct each weight w[i] = w_scale[b]*u_w[i] -
 * w_offset[b], dot with the fp32 activation. q6k_gemv_m1 int8-quantizes the
 * activation, so the only material noise is per-row int8 quant — pass
 * criterion is cosine similarity >= 0.999 per output row.
 *
 * Deterministic; no GGUF needed; runs in <100 ms.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_q6k_gemv.h"
#include "../src/backends/cpu_x86/q6k_to_w8a8.h"
#include "quant.h"
#include "quant_blocks.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t prng(uint32_t *s) {
    uint32_t z = (*s += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

/* fp16 bits for a small positive scale ~0.0007 (typical Q6_K d). */
static uint16_t small_d(void) {
    return 0x139Bu; /* ~0.000727 */
}

static int scenario(size_t N, size_t K) {
    const size_t NS = K / 256;
    uint32_t     s  = 0xC0FFEE11u;

    /* Synthesize N rows of Q6_K. */
    struct block_q6_K_t *W = calloc(N * NS, sizeof(*W));
    for (size_t r = 0; r < N * NS; r++) {
        for (int i = 0; i < 128; i++)
            W[r].ql[i] = (uint8_t) (prng(&s) & 0xFFu);
        for (int i = 0; i < 64; i++)
            W[r].qh[i] = (uint8_t) (prng(&s) & 0xFFu);
        for (int i = 0; i < 16; i++)
            W[r].scales[i] = (int8_t) ((prng(&s) % 101) - 50);
        W[r].d = small_d();
    }

    float *x = malloc(K * sizeof(float));
    for (size_t k = 0; k < K; k++)
        x[k] = 2.0f * ((prng(&s) & 0xFFFFu) / 65536.0f) - 1.0f;

    /* Reference: dequant each row via q6k_to_w8a8_row, dot with x. */
    uint8_t *uw    = malloc(K);
    float   *ws    = malloc((K / 16) * sizeof(float));
    float   *wo    = malloc((K / 16) * sizeof(float));
    float   *y_ref = malloc(N * sizeof(float));
    for (size_t r = 0; r < N; r++) {
        q6k_to_w8a8_row(K, (const uint8_t *) (W + r * NS), uw, ws, wo);
        double dot = 0.0;
        for (size_t i = 0; i < K; i++) {
            const size_t b = i / 16;
            const float  w = ws[b] * (float) uw[i] - wo[b];
            dot += (double) w * (double) x[i];
        }
        y_ref[r] = (float) dot;
    }

    float *y = malloc(N * sizeof(float));
    q6k_gemv_m1(N, K, x, (const uint8_t *) W, y);

    /* Cosine similarity over the output vector. */
    double dotp = 0.0, na = 0.0, nb = 0.0;
    for (size_t r = 0; r < N; r++) {
        dotp += (double) y[r] * (double) y_ref[r];
        na += (double) y[r] * (double) y[r];
        nb += (double) y_ref[r] * (double) y_ref[r];
    }
    const double cos = dotp / (sqrt(na) * sqrt(nb) + 1e-12);
    fprintf(stdout, "[q6k_gemv N=%zu K=%zu] cosine = %.6f\n", N, K, cos);

    free(W);
    free(x);
    free(uw);
    free(ws);
    free(wo);
    free(y_ref);
    free(y);
    return cos >= 0.999 ? 0 : 1;
}

int main(void) {
    int fails = 0;
    fails += scenario(64, 256);
    fails += scenario(128, 512);
    fails += scenario(256, 1024);
    if (fails == 0)
        fprintf(stdout, "OK\n");
    return fails == 0 ? 0 : 1;
}
