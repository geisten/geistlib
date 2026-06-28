/*
 * test_f32q_unit — accuracy of the F32 → W8A8 PLE quantization (cpu_x86).
 *
 * Quantizes a random F32 weight matrix via f32_to_w8a8_row, runs the W8A8
 * GEMV (the f32q m1 path), and compares to the exact F32 matmul. Both the
 * weights (per-16-block asymmetric int8) and the activation (per-row int8)
 * are quantized, so the pass criterion is cosine similarity >= 0.999 per
 * output vector — well within what the Gemma 4 PLE projections tolerate
 * (confirmed by coherent generation).
 *
 * Deterministic; no GGUF needed; runs in <100 ms.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/linear_f32q.h"
#include "../src/backends/cpu_x86/kernel_w8a8.h"
#include "../src/backends/cpu_x86/kernel_w4a8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t prng(uint32_t *s) {
    uint32_t z = (*s += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}
static float frand(uint32_t *s) {
    return 2.0f * ((prng(s) & 0xFFFFu) / 65536.0f) - 1.0f;
}

static int scenario(size_t N, size_t K) {
    const size_t nblk = K / W8A8_BLOCK_ELEMS;
    uint32_t     s    = 0xA5A5F00Du;

    float *W = malloc(N * K * sizeof(float));
    float *x = malloc(K * sizeof(float));
    for (size_t i = 0; i < N * K; i++)
        W[i] = frand(&s) * 0.05f; /* PLE-ish magnitudes */
    for (size_t k = 0; k < K; k++)
        x[k] = frand(&s);

    /* Exact F32 reference. */
    float *y_ref = malloc(N * sizeof(float));
    for (size_t r = 0; r < N; r++) {
        double d = 0.0;
        for (size_t k = 0; k < K; k++)
            d += (double) W[r * K + k] * (double) x[k];
        y_ref[r] = (float) d;
    }

    /* Quantize weights → W8A8. */
    uint8_t *uw = malloc(N * K);
    float   *ws = malloc(N * nblk * sizeof(float));
    float   *wo = malloc(N * nblk * sizeof(float));
    for (size_t r = 0; r < N; r++) {
        f32_to_w8a8_row(K, W + r * K, uw + r * K, ws + r * nblk, wo + r * nblk);
    }

    /* Quantize activation → int8 + 16-block sums. */
    int8_t     *acts    = malloc(K);
    int32_t    *sa      = malloc(nblk * sizeof(int32_t));
    int32_t    *tmp     = malloc((K / 32 + 1) * sizeof(int32_t));
    const float scale_x = w4a8_quantize_acts_row(K, x, acts, tmp);
    for (size_t b = 0; b < nblk; b++) {
        int32_t v = 0;
        for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++)
            v += (int32_t) acts[b * W8A8_BLOCK_ELEMS + i];
        sa[b] = v;
    }

    float *y = malloc(N * sizeof(float));
    w8a8_gemv(N, nblk, uw, ws, wo, acts, sa, scale_x, y);

    double dp = 0.0, na = 0.0, nb = 0.0;
    for (size_t r = 0; r < N; r++) {
        dp += (double) y[r] * (double) y_ref[r];
        na += (double) y[r] * (double) y[r];
        nb += (double) y_ref[r] * (double) y_ref[r];
    }
    const double cos = dp / (sqrt(na) * sqrt(nb) + 1e-12);
    fprintf(stdout, "[f32q N=%zu K=%zu] cosine = %.6f\n", N, K, cos);

    free(W);
    free(x);
    free(y_ref);
    free(uw);
    free(ws);
    free(wo);
    free(acts);
    free(sa);
    free(tmp);
    free(y);
    return cos >= 0.999 ? 0 : 1;
}

int main(void) {
    int fails = 0;
    fails += scenario(256, 1536);  /* gemma4 inp_gate: 1536→256 */
    fails += scenario(1536, 256);  /* gemma4 proj:     256→1536 */
    fails += scenario(8960, 1536); /* gemma4 model_proj-ish */
    if (fails == 0)
        fprintf(stdout, "OK\n");
    return fails == 0 ? 0 : 1;
}
