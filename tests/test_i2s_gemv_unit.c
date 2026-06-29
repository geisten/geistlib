/*
 * test_i2s_gemv_unit — BitNet b1.58 I2_S ternary decode GEMV (cpu_x86).
 *
 * Synthesizes a packed I2_S weight blob (random ternary trits) + one f32
 * per-tensor scale at the tail, then checks:
 *   (1) i2s_gemv_m1 (dispatch: VNNI when available) vs i2s_gemv_m1_scalar
 *       — both run the identical int8-quantized integer math, so they must
 *       agree to within fp rounding of the final scale (Δ <= 1e-3 abs).
 *   (2) the scalar oracle vs a pure-f32 reference (trit·x, no act quant)
 *       — only per-row int8 activation quant differs; pass at cosine
 *       similarity >= 0.999 per output row.
 *
 * Deterministic; no GGUF needed; runs in <100 ms.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_i2s.h"

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

static int scenario(size_t N, size_t K) {
    const size_t NB        = K / I2S_BLOCK_ELEMS;
    const size_t row_bytes = K / 4;
    uint32_t     s         = 0xBEEF1234u;

    const float scale = 0.0123f; /* per-tensor scale */

    /* Packed weight blob: N rows * row_bytes, + 4-byte scale tail. */
    uint8_t *W     = calloc(N * row_bytes + sizeof(float), 1);
    int8_t  *trits = malloc(N * K); /* ground-truth trit per (row, elem) */
    for (size_t r = 0; r < N; r++) {
        for (size_t b = 0; b < NB; b++) {
            uint8_t *qs = W + r * row_bytes + b * I2S_BLOCK_BYTES;
            for (size_t h = 0; h < 2; h++) {
                for (size_t bb = 0; bb < 32; bb++) {
                    uint8_t byte = 0;
                    for (size_t g = 0; g < 4; g++) {
                        const uint8_t code = (uint8_t) (prng(&s) % 3); /* 0,1,2 */
                        byte               = (uint8_t) (byte | (code << (6 - 2 * g)));
                        trits[r * K + b * 256 + h * 128 + g * 32 + bb] = (int8_t) ((int) code - 1);
                    }
                    qs[h * 32 + bb] = byte;
                }
            }
        }
    }
    memcpy(W + N * row_bytes, &scale, sizeof scale);

    float *x = malloc(K * sizeof(float));
    for (size_t k = 0; k < K; k++) {
        x[k] = 2.0f * ((prng(&s) & 0xFFFFu) / 65536.0f) - 1.0f;
    }

    /* Pure-f32 reference. */
    float *ref = malloc(N * sizeof(float));
    for (size_t r = 0; r < N; r++) {
        double acc = 0.0;
        for (size_t k = 0; k < K; k++) {
            acc += (double) trits[r * K + k] * (double) x[k];
        }
        ref[r] = (float) (acc * scale);
    }

    float *y_disp = malloc(N * sizeof(float));
    float *y_scal = malloc(N * sizeof(float));
    i2s_gemv_m1(N, K, x, W, scale, y_disp);
    i2s_gemv_m1_scalar(N, K, x, W, scale, y_scal);

    /* (1) dispatch vs scalar: identical integer math. */
    int    fail     = 0;
    double max_dd   = 0.0;
    for (size_t r = 0; r < N; r++) {
        const double dd = fabs((double) y_disp[r] - (double) y_scal[r]);
        if (dd > max_dd) {
            max_dd = dd;
        }
    }
    if (max_dd > 1e-3) {
        fprintf(stderr, "  [N=%zu K=%zu] dispatch vs scalar Δ=%.3e > 1e-3\n", N, K, max_dd);
        fail = 1;
    }

    /* (2) scalar oracle vs f32 ref: cosine similarity. */
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t r = 0; r < N; r++) {
        dot += (double) y_scal[r] * (double) ref[r];
        na += (double) y_scal[r] * (double) y_scal[r];
        nb += (double) ref[r] * (double) ref[r];
    }
    const double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    if (cos < 0.999) {
        fprintf(stderr, "  [N=%zu K=%zu] cosine=%.6f < 0.999\n", N, K, cos);
        fail = 1;
    }
    printf("  [N=%zu K=%zu] vnni=%d  Δ(disp,scal)=%.3e  cos(scal,f32)=%.6f%s\n",
           N, K, i2s_isa_is_vnni(), max_dd, cos, fail ? "  FAIL" : "");

    free(W);
    free(trits);
    free(x);
    free(ref);
    free(y_disp);
    free(y_scal);
    return fail;
}

int main(void) {
    int fail = 0;
    fail |= scenario(64, 2560);  /* BitNet-2B-4T attn_output / q_proj */
    fail |= scenario(640, 2560); /* k/v proj n_out */
    fail |= scenario(32, 6912);  /* ffn_down n_in */
    fail |= scenario(48, 512);   /* small, non-multiple-of-16 row count */
    if (fail) {
        fprintf(stderr, "test_i2s_gemv_unit: FAIL\n");
        return 1;
    }
    printf("test_i2s_gemv_unit: pass\n");
    return 0;
}
