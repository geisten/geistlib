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
    int    fail   = 0;
    double max_dd = 0.0;
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
    /* (3) prefill GEMM (M tokens): each row vs the m1 dispatch result. */
    const size_t M  = 9; /* spans two+ JT=4 tiles incl. a partial tail */
    float       *xm = malloc(M * K * sizeof(float));
    for (size_t i = 0; i < M * K; i++) {
        xm[i] = 2.0f * ((prng(&s) & 0xFFFFu) / 65536.0f) - 1.0f;
    }
    float *y_gemm = malloc(M * N * sizeof(float));
    float *y_row  = malloc(N * sizeof(float));
    i2s_gemm_mN(M, N, K, xm, W, scale, y_gemm);
    double max_gd = 0.0;
    for (size_t i = 0; i < M; i++) {
        i2s_gemv_m1(N, K, xm + i * K, W, scale, y_row);
        for (size_t r = 0; r < N; r++) {
            const double gd = fabs((double) y_gemm[i * N + r] - (double) y_row[r]);
            if (gd > max_gd) {
                max_gd = gd;
            }
        }
    }
    if (max_gd > 1e-3) {
        fprintf(stderr, "  [N=%zu K=%zu] gemm vs m1 Δ=%.3e > 1e-3\n", N, K, max_gd);
        fail = 1;
    }

    /* (4) x4 row-interleaved path vs scalar oracle (VNNI only). */
    double max_x4 = 0.0, max_x4g = 0.0;
    if (i2s_isa_is_vnni() && N % 4 == 0 && K % 64 == 0) {
        uint8_t *x4 = malloc(N * K / 4);
        i2s_to_x4(N, K, W, x4);
        float *y_x4 = malloc(N * sizeof(float));
        i2s_x4_gemv_m1(N, K, x, x4, scale, y_x4);
        for (size_t r = 0; r < N; r++) {
            const double d = fabs((double) y_x4[r] - (double) y_scal[r]);
            if (d > max_x4) {
                max_x4 = d;
            }
        }
        if (max_x4 > 1e-3) {
            fprintf(stderr, "  [N=%zu K=%zu] x4 gemv vs scalar Δ=%.3e\n", N, K, max_x4);
            fail = 1;
        }
        float *y_x4g = malloc(M * N * sizeof(float));
        i2s_x4_gemm_mN(M, N, K, xm, x4, scale, y_x4g);
        for (size_t i = 0; i < M; i++) {
            i2s_x4_gemv_m1(N, K, xm + i * K, x4, scale, y_row);
            for (size_t r = 0; r < N; r++) {
                const double d = fabs((double) y_x4g[i * N + r] - (double) y_row[r]);
                if (d > max_x4g) {
                    max_x4g = d;
                }
            }
        }
        if (max_x4g > 1e-3) {
            fprintf(stderr, "  [N=%zu K=%zu] x4 gemm vs gemv Δ=%.3e\n", N, K, max_x4g);
            fail = 1;
        }

        /* (5) fused pair == two separate m1, BYTE-exact. Two DISTINCT weights
         * with different n_out (like q 2560 + k 640, gate/up 6912): a slot or
         * offset mix-up between the pair's two weights is invisible when both
         * slots get the same buffer — that hole hid the #102 Phase 2 pair
         * regression, so this now builds a second weight (different trits,
         * N2 = N/2 rounded to a multiple of 4) and memcmp's both outputs. */
        size_t N2 = (N / 2) & ~(size_t) 3;
        if (N2 == 0) {
            N2 = N;
        }
        uint8_t *W2 = calloc(N2 * row_bytes + sizeof(float), 1);
        for (size_t i = 0; i < N2 * row_bytes; i++) {
            W2[i] = (uint8_t) (prng(&s) % 255); /* raw 2-bit fields, values 0..2 per field */
        }
        /* Re-encode to valid trit codes (each 2-bit field in {0,1,2}). */
        for (size_t i = 0; i < N2 * row_bytes; i++) {
            uint8_t b = 0;
            for (size_t g = 0; g < 4; g++) {
                b = (uint8_t) (b | ((prng(&s) % 3) << (6 - 2 * g)));
            }
            W2[i] = b;
        }
        uint8_t *x4_2 = malloc(N2 * K / 4);
        i2s_to_x4(N2, K, W2, x4_2);
        float *yp0   = malloc(N * sizeof(float));
        float *yp1   = malloc(N2 * sizeof(float));
        float *yref1 = malloc(N2 * sizeof(float));
        i2s_x4_gemv_pair_m1(K, x, x4, scale, N, yp0, x4_2, scale * 0.5f, N2, yp1);
        i2s_x4_gemv_m1(N2, K, x, x4_2, scale * 0.5f, yref1);
        if (memcmp(yp0, y_x4, N * sizeof(float)) != 0 ||
            memcmp(yp1, yref1, N2 * sizeof(float)) != 0) {
            fprintf(stderr, "  [N=%zu/%zu K=%zu] pair vs m1 NOT byte-identical\n", N, N2, K);
            for (size_t r = 0; r < N; r++) {
                if (yp0[r] != y_x4[r]) {
                    fprintf(stderr,
                            "    slot0 first diff @%zu: pair=%.9g m1=%.9g\n",
                            r,
                            (double) yp0[r],
                            (double) y_x4[r]);
                    break;
                }
            }
            for (size_t r = 0; r < N2; r++) {
                if (yp1[r] != yref1[r]) {
                    fprintf(stderr,
                            "    slot1 first diff @%zu: pair=%.9g m1=%.9g\n",
                            r,
                            (double) yp1[r],
                            (double) yref1[r]);
                    break;
                }
            }
            fail = 1;
        }
        free(W2);
        free(x4_2);
        free(yp0);
        free(yp1);
        free(yref1);
        free(x4);
        free(y_x4);
        free(y_x4g);
    }

    printf("  [N=%zu K=%zu] vnni=%d  Δ(disp,scal)=%.3e  cos=%.6f  Δ(gemm,m1)=%.3e  "
           "Δx4(gemv)=%.3e Δx4(gemm)=%.3e%s\n",
           N,
           K,
           i2s_isa_is_vnni(),
           max_dd,
           cos,
           max_gd,
           max_x4,
           max_x4g,
           fail ? "  FAIL" : "");

    free(xm);
    free(y_gemm);
    free(y_row);
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
    fail |= scenario(64, 2560);   /* BitNet-2B-4T attn_output / q_proj */
    fail |= scenario(640, 2560);  /* k/v proj n_out */
    fail |= scenario(32, 6912);   /* ffn_down n_in */
    fail |= scenario(48, 512);    /* small, non-multiple-of-16 row count */
    fail |= scenario(6912, 2560); /* full-size ffn gate/up (pair in vivo) */
    if (fail) {
        fprintf(stderr, "test_i2s_gemv_unit: FAIL\n");
        return 1;
    }
    printf("test_i2s_gemv_unit: pass\n");
    return 0;
}
