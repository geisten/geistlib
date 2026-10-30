/*
 * test_q4k_tiled_unit — the tiled int8 Q4_K GEMM (#18) against the scalar
 * Q4_Kx8 GEMM on the same operands.
 *
 * Both take the Q4_Kx8 repack and the Q8_Kx4 activations, so every integer
 * product and subblock sum is the same; only the float accumulation order
 * differs (the tiled kernel converts once per K block, the scalar one per
 * subblock). Shapes cross every edge of the kernel: windows over 256 rows
 * on both sides, N a multiple of 8 but not of 16, several K blocks, M at
 * the 16-row activation band and above.
 *
 * The Q6_K GEMM reads GGUF bytes; it is held against the double-precision
 * product of the dequantized Q6_K weights and the dequantized Q8_Kx4
 * activations, which its exact integer sums must reproduce up to float
 * rounding.
 *
 * Skips on hosts without AVX-512 VNNI (the CI's AVX-512 legs run it).
 */
#define GEIST_INTERNAL_BACKEND_LAYER
#include "../src/backends/cpu_x86/backend_state.h"
#include "../src/backends/cpu_x86/kernel_q4k_tiled.h"
#include "../src/backends/cpu_x86/kernel_q4kx8_gemm.h"
#include "../src/backends/cpu_x86/linear_util.h"
#include "../src/backends/cpu_x86/q4k_to_q4kx8.h"
#include "../src/backends/cpu_x86/q8_kx4.h"
#include "quant.h"
#include "quant_blocks.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t fp32_to_fp16(float f) {
    union {
        float    f;
        uint32_t u;
    } bits            = {.f = f};
    uint32_t x        = bits.u;
    uint32_t sign     = (x >> 31) & 0x1u;
    int32_t  exp32    = (int32_t) ((x >> 23) & 0xFFu) - 127;
    uint32_t mant32   = x & 0x7FFFFFu;
    uint16_t out_sign = (uint16_t) (sign << 15);
    if (exp32 == 128) {
        return (uint16_t) (out_sign | 0x7C00u | (mant32 != 0));
    }
    if (exp32 < -14) {
        return out_sign;
    }
    if (exp32 > 15) {
        return (uint16_t) (out_sign | 0x7C00u);
    }
    uint16_t exp16     = (uint16_t) ((exp32 + 15) & 0x1Fu);
    uint32_t mant16    = mant32 >> 13;
    uint32_t round_bit = (mant32 >> 12) & 1u;
    uint32_t sticky    = mant32 & 0xFFFu;
    if (round_bit != 0 && (sticky != 0 || (mant16 & 1u) != 0)) {
        mant16 += 1;
        if (mant16 == 0x400u) {
            mant16 = 0;
            exp16  = (uint16_t) (exp16 + 1u);
        }
    }
    return (uint16_t) (out_sign | (uint16_t) (exp16 << 10) | (uint16_t) mant16);
}

static uint32_t prng_next(uint32_t *state) {
    uint32_t z = (*state += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static float prng_uniform_pm1(uint32_t bits) {
    return 2.0f * (float) (bits & 0xFFFFFFu) / (float) (1u << 24) - 1.0f;
}

static void set_scale_min_k4(int j, uint8_t scales[12], uint8_t d_in, uint8_t m_in) {
    if (j < 4) {
        scales[j]     = (uint8_t) ((scales[j] & 0xC0u) | (d_in & 0x3Fu));
        scales[j + 4] = (uint8_t) ((scales[j + 4] & 0xC0u) | (m_in & 0x3Fu));
    } else {
        const uint8_t d_lo4 = d_in & 0x0Fu;
        const uint8_t d_hi2 = (d_in >> 4) & 0x03u;
        const uint8_t m_lo4 = m_in & 0x0Fu;
        const uint8_t m_hi2 = (m_in >> 4) & 0x03u;
        scales[j + 4]       = (uint8_t) (d_lo4 | (m_lo4 << 4));
        scales[j - 4]       = (uint8_t) ((scales[j - 4] & 0x3Fu) | (d_hi2 << 6));
        scales[j]           = (uint8_t) ((scales[j] & 0x3Fu) | (m_hi2 << 6));
    }
}

/* One shape: random Q4_K weights (all 6-bit scales and mins in use) and
 * activations, scalar GEMM vs tiled GEMM. */
static int check(size_t M, size_t N, size_t K, struct cpu_x86_state *st) {
    const size_t         n_super = K / 256;
    uint32_t             s       = 0xC0FFEEu ^ (uint32_t) (M * 131 + N * 7 + K);
    struct block_q4_K_t *wq      = calloc(N * n_super, sizeof *wq);
    struct block_q4_Kx8 *wx8     = calloc(N / 8 * n_super, sizeof *wx8);
    float               *x       = malloc(M * K * sizeof *x);
    struct block_q8_Kx4 *xq      = calloc(M / 4 * n_super, sizeof *xq);
    float               *ref     = malloc(M * N * sizeof *ref);
    float               *got     = malloc(M * N * sizeof *got);
    if (!wq || !wx8 || !x || !xq || !ref || !got) {
        fprintf(stderr, "FAIL: allocation\n");
        exit(GEIST_TEST_FAIL);
    }
    for (size_t r = 0; r < N * n_super; r++) {
        wq[r].d    = fp32_to_fp16(0.002f + 0.005f * (float) (prng_next(&s) & 0xFFFF) / 65536.0f);
        wq[r].dmin = fp32_to_fp16(0.001f + 0.003f * (float) (prng_next(&s) & 0xFFFF) / 65536.0f);
        for (int sb = 0; sb < 8; sb++) {
            set_scale_min_k4(sb,
                             wq[r].scales,
                             (uint8_t) (prng_next(&s) & 0x3F),
                             (uint8_t) (prng_next(&s) & 0x3F));
        }
        for (int k = 0; k < 128; k++) {
            wq[r].qs[k] = (uint8_t) prng_next(&s);
        }
    }
    for (size_t o = 0; o < N / 8; o++) {
        q4k_to_q4kx8_octet(n_super, (const uint8_t *) &wq[o * 8 * n_super], &wx8[o * n_super]);
    }
    for (size_t i = 0; i < M * K; i++) {
        x[i] = prng_uniform_pm1(prng_next(&s));
    }
    for (size_t g = 0; g < M / 4; g++) {
        quantize_q8_Kx4(K, x + g * 4 * K, xq + g * n_super);
    }
    q4kx8_gemm_scalar(M, N, K, xq, wx8, ref);
    for (size_t i = 0; i < M * N; i++) {
        got[i] = NAN;
    }
    int fails = 0;
    if (!q4k_tiled_gemm_avx512_vnni(M, N, K, xq, wx8, got, st)) {
        fprintf(stderr,
                "FAIL: M=%zu N=%zu K=%zu: tiled GEMM reported a workspace failure\n",
                M,
                N,
                K);
        fails = 1;
    }
    double rms = 0.0;
    for (size_t i = 0; i < M * N; i++) {
        rms += (double) ref[i] * ref[i];
    }
    rms            = sqrt(rms / (double) (M * N));
    double worst   = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < M * N && !fails; i++) {
        const double d = fabs((double) got[i] - ref[i]);
        if (!(d <= worst)) {
            worst   = isnan(d) ? INFINITY : d;
            worst_i = i;
        }
    }
    /* float rounding of two accumulation orders: well under 1e-4 of the
     * output's scale (llama.cpp reports 1e-4 max on its own shapes). */
    const double tol = 1e-4 * (rms + 1e-6);
    if (!fails && !(worst <= tol)) {
        fprintf(stderr,
                "FAIL: M=%zu N=%zu K=%zu: Y[%zu,%zu] tiled %.9g, scalar %.9g (|d| %.3g > tol "
                "%.3g)\n",
                M,
                N,
                K,
                worst_i / N,
                worst_i % N,
                (double) got[worst_i],
                (double) ref[worst_i],
                worst,
                tol);
        fails = 1;
    }
    printf("  M=%4zu N=%4zu K=%4zu: max |d| %.3g (rms %.3g) %s\n",
           M,
           N,
           K,
           worst,
           rms,
           fails ? "FAIL" : "ok");
    free(wq);
    free(wx8);
    free(x);
    free(xq);
    free(ref);
    free(got);
    return fails;
}

/* Element k of activation row r, dequantized from the Q8_Kx4 layout. */
static float act_at(size_t K, const struct block_q8_Kx4 *xq, size_t r, size_t k) {
    const struct block_q8_Kx4 *b = &xq[(r / 4) * (K / 256) + k / 256];
    const size_t               e = k % 256, lane = r % 4;
    return b->d[lane] *
           (float) (int8_t) b->qs[(e / 64) * 256 + ((e % 64) / 8) * 32 + lane * 8 + e % 8];
}

/* q5k: Q5_K instead of Q6_K. */
static int check_raw(bool q5k, size_t M, size_t N, size_t K, struct cpu_x86_state *st) {
    const size_t         n_super = K / 256;
    uint32_t             s       = 0x5EEDu ^ (uint32_t) (M * 31 + N * 3 + K);
    struct block_q6_K_t *wq      = calloc(N * n_super, sizeof *wq);
    struct block_q5_K_t *w5      = calloc(N * n_super, sizeof *w5);
    float               *wf      = malloc(N * K * sizeof *wf);
    float               *x       = malloc(M * K * sizeof *x);
    struct block_q8_Kx4 *xq      = calloc(M / 4 * n_super, sizeof *xq);
    float               *got     = malloc(M * N * sizeof *got);
    if (!wq || !w5 || !wf || !x || !xq || !got) {
        fprintf(stderr, "FAIL: allocation\n");
        exit(GEIST_TEST_FAIL);
    }
    for (size_t b = 0; b < N * n_super; b++) {
        for (int i = 0; i < 128; i++) {
            wq[b].ql[i] = (uint8_t) prng_next(&s);
        }
        for (int i = 0; i < 64; i++) {
            wq[b].qh[i] = (uint8_t) prng_next(&s);
        }
        for (int i = 0; i < 16; i++) {
            wq[b].scales[i] = (int8_t) (prng_next(&s) & 0xFF); /* full int8 range */
        }
        wq[b].d = fp32_to_fp16(0.0005f + 0.002f * (float) (prng_next(&s) & 0xFFFF) / 65536.0f);
    }
    for (size_t b = 0; b < N * n_super; b++) {
        w5[b].d    = fp32_to_fp16(0.002f + 0.004f * (float) (prng_next(&s) & 0xFFFF) / 65536.0f);
        w5[b].dmin = fp32_to_fp16(0.001f + 0.003f * (float) (prng_next(&s) & 0xFFFF) / 65536.0f);
        for (int sb = 0; sb < 8; sb++) {
            set_scale_min_k4(sb,
                             w5[b].scales,
                             (uint8_t) (prng_next(&s) & 0x3F),
                             (uint8_t) (prng_next(&s) & 0x3F));
        }
        for (int i = 0; i < 32; i++) {
            w5[b].qh[i] = (uint8_t) prng_next(&s);
        }
        for (int i = 0; i < 128; i++) {
            w5[b].qs[i] = (uint8_t) prng_next(&s);
        }
    }
    for (size_t r = 0; r < N; r++) {
        if (q5k) {
            dequant_q5_K_row(K, &w5[r * n_super], wf + r * K);
        } else {
            dequant_q6_K_row(K, &wq[r * n_super], wf + r * K);
        }
    }
    for (size_t i = 0; i < M * K; i++) {
        x[i] = prng_uniform_pm1(prng_next(&s));
    }
    for (size_t g = 0; g < M / 4; g++) {
        quantize_q8_Kx4(K, x + g * 4 * K, xq + g * n_super);
    }
    for (size_t i = 0; i < M * N; i++) {
        got[i] = NAN;
    }
    int fails = 0;
    if (!(q5k ? q5k_tiled_gemm_avx512_vnni(M, N, K, xq, w5, got, st)
              : q6k_tiled_gemm_avx512_vnni(M, N, K, xq, wq, got, st))) {
        fprintf(stderr,
                "FAIL: %s M=%zu N=%zu K=%zu: workspace failure\n",
                q5k ? "Q5_K" : "Q6_K",
                M,
                N,
                K);
        fails = 1;
    }
    double worst = 0.0, rms = 0.0;
    size_t wi = 0;
    for (size_t i = 0; i < M && !fails; i++) {
        for (size_t j = 0; j < N; j++) {
            double ref = 0.0;
            for (size_t k = 0; k < K; k++) {
                ref += (double) act_at(K, xq, i, k) * wf[j * K + k];
            }
            rms += ref * ref;
            const double d = fabs((double) got[i * N + j] - ref);
            if (!(d <= worst)) {
                worst = isnan(d) ? INFINITY : d;
                wi    = i * N + j;
            }
        }
    }
    rms = sqrt(rms / (double) (M * N));
    if (!fails && !(worst <= 1e-4 * (rms + 1e-6))) {
        fprintf(stderr,
                "FAIL: %s M=%zu N=%zu K=%zu: Y[%zu,%zu] = %.9g, |d| %.3g > 1e-4 * rms %.3g\n",
                q5k ? "Q5_K" : "Q6_K",
                M,
                N,
                K,
                wi / N,
                wi % N,
                (double) got[wi],
                worst,
                rms);
        fails = 1;
    }
    printf("  %s M=%4zu N=%4zu K=%4zu: max |d| %.3g (rms %.3g) %s\n",
           q5k ? "Q5_K" : "Q6_K",
           M,
           N,
           K,
           worst,
           rms,
           fails ? "FAIL" : "ok");
    free(wq);
    free(w5);
    free(wf);
    free(x);
    free(xq);
    free(got);
    return fails;
}

/* Q4_0 (q4_1: Q4_1) x per-row Q8_0, against the double product of the
 * dequantized weights and activations. */
static int check_q4_0(bool q4_1, size_t M, size_t N, size_t K, struct cpu_x86_state *st) {
    const size_t nb  = K / 32;
    const size_t bsz = q4_1 ? sizeof(struct block_q4_1_t) : sizeof(struct block_q4_0_t);
    uint32_t     s   = 0xA11CEu ^ (uint32_t) (M * 7 + N * 13 + K + (q4_1 ? 1 : 0));
    uint8_t     *w   = calloc(N * nb, bsz);
    float       *wf  = malloc(N * K * sizeof *wf);
    int8_t      *qx  = malloc(M * K);
    float       *dx  = malloc(M * nb * sizeof *dx);
    int32_t     *sx  = malloc(M * nb * sizeof *sx);
    float       *got = malloc(M * N * sizeof *got);
    if (!w || !wf || !qx || !dx || !sx || !got) {
        fprintf(stderr, "FAIL: allocation\n");
        exit(GEIST_TEST_FAIL);
    }
    for (size_t b = 0; b < N * nb; b++) {
        uint8_t       *blk = w + b * bsz;
        const uint16_t d =
                fp32_to_fp16(0.004f + 0.01f * (float) (prng_next(&s) & 0xFFFF) / 65536.0f);
        memcpy(blk, &d, 2);
        if (q4_1) {
            const uint16_t m = fp32_to_fp16(-0.05f * (float) (prng_next(&s) & 0xFFFF) / 65536.0f);
            memcpy(blk + 2, &m, 2);
        }
        for (int i = 0; i < 16; i++) {
            blk[(q4_1 ? 4 : 2) + i] = (uint8_t) prng_next(&s);
        }
    }
    for (size_t r = 0; r < N; r++) {
        if (q4_1) {
            dequant_q4_1_row(K, w + r * nb * bsz, wf + r * K);
        } else {
            dequant_q4_0_row(K, w + r * nb * bsz, wf + r * K);
        }
    }
    for (size_t r = 0; r < M; r++) {
        for (size_t b = 0; b < nb; b++) {
            float x[32], amax = 0.0f;
            for (int i = 0; i < 32; i++) {
                x[i] = prng_uniform_pm1(prng_next(&s));
                amax = fmaxf(amax, fabsf(x[i]));
            }
            const float d   = amax / 127.0f;
            int32_t     sum = 0;
            for (int i = 0; i < 32; i++) {
                const int8_t q         = (int8_t) lrintf(d != 0.0f ? x[i] / d : 0.0f);
                qx[r * K + b * 32 + i] = q;
                sum += q;
            }
            dx[r * nb + b] = d;
            sx[r * nb + b] = sum;
        }
    }
    for (size_t i = 0; i < M * N; i++) {
        got[i] = NAN;
    }
    int fails = 0;
    if (!q4_0_tiled_gemm_avx512_vnni(M, N, K, qx, dx, sx, w, q4_1, got, st)) {
        fprintf(stderr, "FAIL: %s: workspace failure\n", q4_1 ? "Q4_1" : "Q4_0");
        fails = 1;
    }
    double worst = 0.0, rms = 0.0;
    size_t wi = 0;
    for (size_t i = 0; i < M && !fails; i++) {
        for (size_t j = 0; j < N; j++) {
            double ref = 0.0;
            for (size_t k = 0; k < K; k++) {
                ref += (double) dx[i * nb + k / 32] * qx[i * K + k] * wf[j * K + k];
            }
            rms += ref * ref;
            const double d = fabs((double) got[i * N + j] - ref);
            if (!(d <= worst)) {
                worst = isnan(d) ? INFINITY : d;
                wi    = i * N + j;
            }
        }
    }
    rms = sqrt(rms / (double) (M * N));
    if (!fails && !(worst <= 1e-4 * (rms + 1e-6))) {
        fprintf(stderr,
                "FAIL: %s M=%zu N=%zu K=%zu: Y[%zu,%zu] = %.9g, |d| %.3g > 1e-4 * rms %.3g\n",
                q4_1 ? "Q4_1" : "Q4_0",
                M,
                N,
                K,
                wi / N,
                wi % N,
                (double) got[wi],
                worst,
                rms);
        fails = 1;
    }
    printf("  %s M=%4zu N=%4zu K=%4zu: max |d| %.3g (rms %.3g) %s\n",
           q4_1 ? "Q4_1" : "Q4_0",
           M,
           N,
           K,
           worst,
           rms,
           fails ? "FAIL" : "ok");
    free(w);
    free(wf);
    free(qx);
    free(dx);
    free(sx);
    free(got);
    return fails;
}

int main(void) {
    if (!vnni_tiles_usable()) {
        GEIST_SKIP("no AVX-512 VNNI on this host");
    }
    struct cpu_x86_state st         = {0};
    st.ws_generation                = geist_thread_ws_next_generation();
    static const size_t shapes[][3] = {
            {16, 16, 256},   /* one band, one microtile */
            {20, 24, 256},   /* ragged band, N % 16 == 8 */
            {64, 128, 512},  /* a prefill chunk, two K blocks */
            {260, 264, 768}, /* windows past 256 on both sides */
            {512, 136, 1024},
            {520, 24, 512}, /* past one 512-row block of prepared activations */
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
        fails += check(shapes[i][0], shapes[i][1], shapes[i][2], &st);
    }
    static const size_t q6k_shapes[][3] = {
            {16, 16, 256},
            {20, 40, 256},
            {64, 136, 512},
            {260, 257, 768},
            {520, 40, 256},
    };
    for (size_t i = 0; i < sizeof q6k_shapes / sizeof q6k_shapes[0]; i++) {
        fails += check_raw(false, q6k_shapes[i][0], q6k_shapes[i][1], q6k_shapes[i][2], &st);
        fails += check_raw(true, q6k_shapes[i][0], q6k_shapes[i][1], q6k_shapes[i][2], &st);
    }
    static const size_t q40_shapes[][3] = {
            {16, 16, 256},
            {17, 40, 256},
            {64, 136, 512},
            {261, 257, 768},
            {517, 40, 256},
    };
    for (size_t i = 0; i < sizeof q40_shapes / sizeof q40_shapes[0]; i++) {
        fails += check_q4_0(false, q40_shapes[i][0], q40_shapes[i][1], q40_shapes[i][2], &st);
        fails += check_q4_0(true, q40_shapes[i][0], q40_shapes[i][1], q40_shapes[i][2], &st);
    }
    cpu_x86_ws_destroy_all(&st);
    printf("test_q4k_tiled_unit: %s\n", fails ? "FAIL" : "PASS");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
