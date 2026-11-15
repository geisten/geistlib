/*
 * src/backends/cpu_neon/kernels/f32_rows.c — fp32 GEMM on a dequantized
 * weight tile whose rows do not depend on the row count (#528).
 *
 * Pure compute. An MoE expert's rows per call are how many of a prefill
 * chunk's tokens chose it, so a kernel that picks its path by row count
 * gives a token other bits per chunking (#527). Here each output element
 * has one vector accumulator, stepped through n_in in one order; the
 * register tiles below only decide how many of those independent chains
 * run side by side (16 for four activation rows, 8 for one), never the
 * order inside one.
 */
#include "f32_rows.h"

#include <arm_neon.h>

/* The four lane sums of one output to its scalar: the one reduction every
 * tile shape goes through. Not inlined, so it is compiled once and
 * -ffast-math cannot pair the lanes differently per call site; one call per
 * output against n_in / 4 multiply-adds. */
[[gnu::noinline]] static float f32_rows_reduce(float32x4_t acc) {
    const float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    return vget_lane_f32(vpadd_f32(s2, s2), 0);
}

/* A activation rows against W weight rows: A * W accumulators, each the
 * sum over k of x[a][k .. k + 4) * w[c][k .. k + 4) by fused multiply-add.
 * A and W are constants at every call, so the loops unroll. */
[[gnu::always_inline]] static inline void f32_rows_tile(
        size_t A, size_t W, size_t n_in, size_t ldy, const float *x, const float *w, float *y) {
    float32x4_t acc[4][8];
    for (size_t a = 0; a < A; a++) {
        for (size_t c = 0; c < W; c++) {
            acc[a][c] = vdupq_n_f32(0.0f);
        }
    }
    for (size_t k = 0; k < n_in; k += 4) {
        float32x4_t wv[8];
        for (size_t c = 0; c < W; c++) {
            wv[c] = vld1q_f32(w + c * n_in + k);
        }
        for (size_t a = 0; a < A; a++) {
            const float32x4_t xv = vld1q_f32(x + a * n_in + k);
            for (size_t c = 0; c < W; c++) {
                acc[a][c] = vfmaq_f32(acc[a][c], xv, wv[c]);
            }
        }
    }
    for (size_t a = 0; a < A; a++) {
        for (size_t c = 0; c < W; c++) {
            y[a * ldy + c] = f32_rows_reduce(acc[a][c]);
        }
    }
}

void f32_rows_gemm(size_t       m,
                   size_t       n_rows,
                   size_t       n_in,
                   size_t       ldy,
                   const float *x,
                   const float *w,
                   float       *y) {
    const size_t m4 = m / 4 * 4, r4 = n_rows / 4 * 4, r8 = n_rows / 8 * 8;
    for (size_t i = 0; i < m4; i += 4) {
        const float *xi = x + i * n_in;
        float       *yi = y + i * ldy;
        for (size_t r = 0; r < r4; r += 4) {
            f32_rows_tile(4, 4, n_in, ldy, xi, w + r * n_in, yi + r);
        }
        for (size_t r = r4; r < n_rows; r++) {
            f32_rows_tile(4, 1, n_in, ldy, xi, w + r * n_in, yi + r);
        }
    }
    for (size_t i = m4; i < m; i++) {
        const float *xi = x + i * n_in;
        float       *yi = y + i * ldy;
        for (size_t r = 0; r < r8; r += 8) {
            f32_rows_tile(1, 8, n_in, ldy, xi, w + r * n_in, yi + r);
        }
        for (size_t r = r8; r < n_rows; r++) {
            f32_rows_tile(1, 1, n_in, ldy, xi, w + r * n_in, yi + r);
        }
    }
}
