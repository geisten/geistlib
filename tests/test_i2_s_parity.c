/*
 * tests/test_i2_s_parity.c — verify the i2_s (BitNet b1.58 official) NEON
 * kernels (cpu_neon_w_i2_s_q8a_m1 / _mN) against a hand-computed reference.
 *
 * Builds a synthetic i2_s weight tensor (reversed-shift 2-bit packing + ONE
 * per-tensor f32 scale at the tail, exactly as bitnet.cpp quantize_i2_s does),
 * runs the kernels through a real cpu_neon backend, and asserts the outputs
 * match Σ trit·xq · scale · inv_act_scale per row. No fp16 roundtrip (i2_s
 * scale is f32), so the match is near-exact.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_neon/internal.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Pack the whole [n_out, n_in] trit tensor as i2_s (flat row-major, 64-byte
 * blocks, byte b holds elements b,32+b,64+b,96+b at shifts 6,4,2,0) followed
 * by ONE f32 per-tensor scale at offset n_in*n_out/4. */
static void pack_i2_s(const int8_t *trits, size_t n_in, size_t n_out, float scale, uint8_t *buf) {
    const size_t total = n_in * n_out;
    memset(buf, 0, total / 4);
    for (size_t r = 0; r < n_out; r++) {
        const int8_t *tr  = trits + r * n_in;
        uint8_t      *row = buf + r * (n_in / 4);
        for (size_t b = 0; b < n_in / 256; b++) {
            uint8_t *qs = row + b * 64;
            for (size_t h = 0; h < 2; h++) {
                for (size_t bb = 0; bb < 32; bb++) {
                    uint8_t byte = 0;
                    for (size_t g = 0; g < 4; g++) {
                        const int     trit = tr[b * 256 + h * 128 + g * 32 + bb];
                        const uint8_t bits = (uint8_t) ((trit + 1) & 3);
                        byte |= (uint8_t) (bits << (6 - 2 * g));
                    }
                    qs[h * 32 + bb] = byte;
                }
            }
        }
    }
    memcpy(buf + total / 4, &scale, sizeof scale);
}

/* Reference: same int8 act-quant as the kernel, single tensor scale. */
static float ref_dot_i2s(const int8_t *trits, float scale, const float *x, size_t n_in) {
    int8_t *xq      = malloc(n_in);
    float   max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = x[i] < 0 ? -x[i] : x[i];
        if (a > max_abs)
            max_abs = a;
    }
    const float act_scale     = 127.0f / max_abs;
    const float inv_act_scale = max_abs / 127.0f;
    for (size_t i = 0; i < n_in; i++) {
        float   q  = x[i] * act_scale;
        int32_t qi = (int32_t) (q < 0 ? q - 0.5f : q + 0.5f);
        if (qi > 127)
            qi = 127;
        if (qi < -128)
            qi = -128;
        xq[i] = (int8_t) qi;
    }
    int64_t dot = 0;
    for (size_t i = 0; i < n_in; i++)
        dot += (int64_t) trits[i] * xq[i];
    free(xq);
    return (float) dot * scale * inv_act_scale;
}

int main(void) {
    const size_t n_in = 2560, n_out = 64;
    const float  scale = 0.0123f;
    srand(4321);
    int8_t *trits = malloc(n_in * n_out);
    for (size_t i = 0; i < n_in * n_out; i++)
        trits[i] = (int8_t) ((rand() % 3) - 1);

    uint8_t *buf = aligned_alloc(64, ((n_in * n_out / 4) + 128 + 63) & ~63ULL);
    pack_i2_s(trits, n_in, n_out, scale, buf);

    struct geist_backend *be = NULL;
    if (geist_backend_create("auto", NULL, NULL, &be) != GEIST_OK || be == NULL) {
        printf("backend_create failed\n");
        return 2;
    }

    struct geist_weight w = {0};
    w.raw                 = buf;
    w.raw_nbytes          = i2_s_scale_offset(n_in * n_out) + sizeof(float);
    w.n_in                = (int32_t) n_in;
    w.n_out               = (int32_t) n_out;
    w.dtype               = GEIST_DTYPE_I2_S;

    int fails = 0;

    /* --- M=1 decode --- */
    {
        float *x = malloc(n_in * sizeof(float));
        for (size_t i = 0; i < n_in; i++)
            x[i] = ((rand() % 2001) - 1000) * 0.01f;
        float *y = malloc(n_out * sizeof(float));
        cpu_neon_w_i2_s_q8a_m1(x, &w, be, y);
        int mm = 0;
        for (size_t r = 0; r < n_out; r++) {
            const float ref = ref_dot_i2s(trits + r * n_in, scale, x, n_in);
            const float rel = fabsf(ref - y[r]) / (fabsf(ref) + 1e-9f);
            if (rel > 1e-4f) {
                if (mm < 5)
                    printf("m1 MISMATCH r=%zu ref=%.6f got=%.6f rel=%.2e\n", r, ref, y[r], rel);
                mm++;
            }
        }
        printf("m1: %d / %zu mismatches\n", mm, n_out);
        fails += mm;
        free(x);
        free(y);
    }

    /* --- M>1 prefill (tiled mN), exercise mt8/mt4/scalar tiers --- */
    for (size_t m = 1; m <= 20; m += (m == 1 ? 7 : 3)) { /* 1,8,11,14,17,20 — mt16/mt8/mt4/scalar */
        float *x = malloc(m * n_in * sizeof(float));
        for (size_t i = 0; i < m * n_in; i++)
            x[i] = ((rand() % 2001) - 1000) * 0.01f;
        float *y = malloc(m * n_out * sizeof(float));
        cpu_neon_w_i2_s_q8a_mN(m, x, &w, be, y);
        int mm = 0;
        for (size_t t = 0; t < m; t++) {
            for (size_t r = 0; r < n_out; r++) {
                const float ref = ref_dot_i2s(trits + r * n_in, scale, x + t * n_in, n_in);
                const float got = y[t * n_out + r];
                const float rel = fabsf(ref - got) / (fabsf(ref) + 1e-9f);
                if (rel > 1e-4f) {
                    if (mm < 5)
                        printf("mN m=%zu MISMATCH t=%zu r=%zu ref=%.6f got=%.6f rel=%.2e\n",
                               m,
                               t,
                               r,
                               ref,
                               got,
                               rel);
                    mm++;
                }
            }
        }
        printf("mN m=%zu: %d / %zu mismatches\n", m, mm, m * n_out);
        fails += mm;
        free(x);
        free(y);
    }

    /* --- Fused F16 lm-head GEMV --- */
    {
        const size_t fn_in = 2560, fn_out = 96;
        uint16_t    *Wf = malloc(fn_in * fn_out * sizeof(uint16_t));
        for (size_t i = 0; i < fn_in * fn_out; i++) {
            const uint16_t sign = (uint16_t) (rand() & 1);
            const uint16_t exp  = (uint16_t) (10 + (rand() % 8)); /* ~2^-5..2^2 */
            const uint16_t mant = (uint16_t) (rand() & 0x3FF);
            Wf[i]               = (uint16_t) ((sign << 15) | (exp << 10) | mant);
        }
        float *x = malloc(fn_in * sizeof(float));
        for (size_t i = 0; i < fn_in; i++)
            x[i] = ((rand() % 2001) - 1000) * 0.01f;
        float              *y  = malloc(fn_out * sizeof(float));
        struct geist_weight wf = {0};
        wf.raw                 = Wf;
        wf.raw_nbytes          = fn_in * fn_out * sizeof(uint16_t);
        wf.n_in                = (int32_t) fn_in;
        wf.n_out               = (int32_t) fn_out;
        wf.dtype               = GEIST_DTYPE_F16;
        cpu_neon_w_f16_m1(x, &wf, be, y);
        int mm = 0;
        for (size_t r = 0; r < fn_out; r++) {
            float ref = 0.0f;
            for (size_t k = 0; k < fn_in; k++)
                ref += fp16_to_fp32(Wf[r * fn_in + k]) * x[k];
            const float rel = fabsf(ref - y[r]) / (fabsf(ref) + 1e-6f);
            /* 2e-3: scalar ref and the NEON gemv sum 2560 terms in different
             * orders — pure fp-association noise, not a correctness issue. */
            if (rel > 2e-3f) {
                if (mm < 5)
                    printf("f16 MISMATCH r=%zu ref=%.6f got=%.6f rel=%.2e\n", r, ref, y[r], rel);
                mm++;
            }
        }
        printf("f16: %d / %zu mismatches\n", mm, fn_out);
        fails += mm;
        free(Wf);
        free(x);
        free(y);
    }

    geist_backend_destroy(be);
    free(trits);
    free(buf);
    printf(fails == 0 ? "I2_S PARITY: PASS\n" : "I2_S PARITY: FAIL (%d)\n", fails);
    return fails == 0 ? 0 : 1;
}
