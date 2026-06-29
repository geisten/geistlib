/*
 * test_q8w_gemv_unit — Q8-weight decode GEMV (cpu_x86 lm_head fast path).
 *
 * Synthesizes a random F16 weight + f32 activation, quantizes the weight to
 * per-row int8 (f16_to_q8w) and checks q8w_gemv_m1 against an exact f32
 * reference. Per-row int8 weight quant is the only error source → pass at
 * cosine similarity >= 0.999 (and the matching f16 GEMV agrees to ~1e-3).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_f16_gemv.h"

#include "quant.h" /* fp16_to_fp32 */

#include <immintrin.h> /* _cvtss_sh — F16C f32->f16 (x86-64-v3) */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static inline uint16_t f32_to_f16(float v) {
    return (uint16_t) _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static uint32_t prng(uint32_t *s) {
    uint32_t z = (*s += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static int scenario(size_t N, size_t K) {
    uint32_t  s  = 0x51A0u;
    uint16_t *Wf = malloc(N * K * sizeof(uint16_t));
    float    *Wr = malloc(N * K * sizeof(float));
    for (size_t i = 0; i < N * K; i++) {
        const float v = 2.0f * ((prng(&s) & 0xFFFFu) / 65536.0f) - 1.0f;
        Wf[i]         = f32_to_f16(v);
        Wr[i]         = fp16_to_fp32(Wf[i]); /* the value the kernels actually see */
    }
    float *x = malloc(K * sizeof(float));
    for (size_t k = 0; k < K; k++) {
        x[k] = 2.0f * ((prng(&s) & 0xFFFFu) / 65536.0f) - 1.0f;
    }

    float *ref = malloc(N * sizeof(float));
    for (size_t r = 0; r < N; r++) {
        double acc = 0.0;
        for (size_t k = 0; k < K; k++) {
            acc += (double) Wr[r * K + k] * (double) x[k];
        }
        ref[r] = (float) acc;
    }

    int8_t *wq     = malloc(N * K);
    float  *scales = malloc(N * sizeof(float));
    f16_to_q8w(N, K, Wf, wq, scales);
    float *yq = malloc(N * sizeof(float));
    q8w_gemv_m1(N, K, x, wq, scales, yq);

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t r = 0; r < N; r++) {
        dot += (double) yq[r] * (double) ref[r];
        na += (double) yq[r] * (double) yq[r];
        nb += (double) ref[r] * (double) ref[r];
    }
    const double cos  = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    const int    fail = cos < 0.999;
    printf("  [N=%zu K=%zu] cos(q8w,f32)=%.6f%s\n", N, K, cos, fail ? "  FAIL" : "");

    free(Wf);
    free(Wr);
    free(x);
    free(ref);
    free(wq);
    free(scales);
    free(yq);
    return fail;
}

int main(void) {
    int fail = 0;
    fail |= scenario(256, 2560);  /* lm_head-shaped */
    fail |= scenario(127, 320);   /* odd N, tail K */
    fail |= scenario(64, 100);    /* non-multiple-of-8 K */
    if (fail) {
        fprintf(stderr, "test_q8w_gemv_unit: FAIL\n");
        return 1;
    }
    printf("test_q8w_gemv_unit: pass\n");
    return 0;
}
