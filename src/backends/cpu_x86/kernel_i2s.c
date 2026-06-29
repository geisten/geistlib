/*
 * src/backends/cpu_x86/kernel_i2s.c — I2_S ternary GEMV dispatcher + scalar.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Compiled at baseline -march=x86-64-v3; the AVX-512+VNNI variant lives in
 * kernel_i2s_avx512_vnni.c with -mavx512vnni. See kernel_i2s.h for the
 * ternary algebra.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_i2s.h"

#include "kernel_w4a8.h" /* w4a8_dispatcher_init — shared ISA selection */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* VNNI path (kernel_i2s_avx512_vnni.c). */
void i2s_gemv_m1_avx512_vnni(size_t        n_out,
                             size_t        n_in,
                             const int8_t *xq,
                             int32_t       sum_a,
                             const uint8_t w_raw[],
                             float         scale,
                             float         y[static n_out]);

/* --- Activation quant: per-row symmetric int8, scale = 127/max|x|. ------- */
static float quantize_act_row(size_t n_in, const float *x, int8_t *xq, int32_t *sum_a_out) {
    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = fabsf(x[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }
    const float act_scale = 127.0f / max_abs;
    int32_t     sum_a     = 0;
    for (size_t i = 0; i < n_in; i++) {
        const float q  = x[i] * act_scale;
        int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
        if (qi > 127) {
            qi = 127;
        }
        if (qi < -128) {
            qi = -128;
        }
        xq[i] = (int8_t) qi;
        sum_a += qi;
    }
    *sum_a_out = sum_a;
    return max_abs / 127.0f; /* inv_act_scale */
}

/* --- Scalar reference (oracle) ------------------------------------------- */
static int32_t i2s_row_dot_scalar(size_t n_blocks, const uint8_t *Wr, const int8_t *xq) {
    int32_t acc = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *qs = Wr + b * I2S_BLOCK_BYTES;
        const int8_t  *xb = xq + b * I2S_BLOCK_ELEMS;
        for (size_t h = 0; h < 2; h++) {
            for (size_t bb = 0; bb < 32; bb++) {
                const uint8_t byte = qs[h * 32 + bb];
                for (size_t g = 0; g < 4; g++) {
                    const int trit = (int) ((byte >> (6 - 2 * g)) & 3) - 1;
                    acc += trit * (int) xb[h * 128 + g * 32 + bb];
                }
            }
        }
    }
    return acc;
}

void i2s_gemv_m1_scalar(size_t        n_out,
                        size_t        n_in,
                        const float  *x,
                        const uint8_t w_raw[],
                        float         tensor_scale,
                        float         y[static n_out]) {
    const size_t n_blocks  = n_in / I2S_BLOCK_ELEMS;
    const size_t row_bytes = n_in / 4;
    int8_t      *xq        = (int8_t *) __builtin_alloca(n_in);
    int32_t      sum_a;
    const float  scale = tensor_scale * quantize_act_row(n_in, x, xq, &sum_a);
    (void) sum_a;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_out; r++) {
        const int32_t dot = i2s_row_dot_scalar(n_blocks, w_raw + r * row_bytes, xq);
        y[r]              = (float) dot * scale;
    }
}

/* --- Dispatch ------------------------------------------------------------ */
static int g_i2s_vnni   = -1;

[[nodiscard]] int i2s_isa_is_vnni(void) {
    if (g_i2s_vnni < 0) {
        const enum w4a8_isa tier = w4a8_dispatcher_init();
        g_i2s_vnni = (tier == W4A8_ISA_AVX512_VNNI || tier == W4A8_ISA_AVX512_BF16) ? 1 : 0;
    }
    return g_i2s_vnni;
}

void i2s_gemv_m1(size_t        n_out,
                 size_t        n_in,
                 const float  *x,
                 const uint8_t w_raw[],
                 float         tensor_scale,
                 float         y[static n_out]) {
    if (n_in % I2S_BLOCK_ELEMS != 0) {
        i2s_gemv_m1_scalar(n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    if (!i2s_isa_is_vnni()) {
        i2s_gemv_m1_scalar(n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    int8_t     *xq = (int8_t *) __builtin_alloca(n_in);
    int32_t     sum_a;
    const float scale = tensor_scale * quantize_act_row(n_in, x, xq, &sum_a);
    i2s_gemv_m1_avx512_vnni(n_out, n_in, xq, sum_a, w_raw, scale, y);
}
