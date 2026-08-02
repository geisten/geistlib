/*
 * src/backends/cpu_neon/kernels/q8_0.c — Q8_0 W8A8 NEON kernels.
 *
 * Pure compute. Block layout from src/quant/quant_blocks.h.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void linear_q8_0_decode_w8a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_q8, size_t n_in, size_t n_out, float *y) {
    const struct block_q8_0_t *w          = (const struct block_q8_0_t *) w_q8;
    const size_t               nb_per_row = n_in / Q8_0_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q8_0_t *row = w + n * nb_per_row;
        float                      acc = 0.0f;
        for (size_t b = 0; b < nb_per_row; b++) {
            const struct block_q8_0_t *blk = &row[b];
            const float                d   = fp16_to_fp32(blk->d);
            const int8_t              *xb  = x_q8 + b * Q8_0_BLOCK_ELEMS;

            int32_t int_dot = 0;
#if defined(__ARM_NEON)
            /* 32 int8 elements = two 16-byte chunks. */
            int_dot += dot16_i8(xb, vld1q_s8(blk->qs));
            int_dot += dot16_i8(xb + 16, vld1q_s8(blk->qs + 16));
#else
            for (int j = 0; j < Q8_0_BLOCK_ELEMS; j++) {
                int_dot += (int32_t) xb[j] * (int32_t) blk->qs[j];
            }
#endif
            acc += d * scale_x * (float) int_dot;
        }
        y[n] = acc;
    }
}

void linear_q8_0_decode_w8a8(
        const float *x, const void *w_q8, size_t n_in, size_t n_out, float *y) {
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    float   scale_x = quantize_x_int8_sym(x, n_in, x_q8);
    linear_q8_0_decode_w8a8_pre(x_q8, scale_x, w_q8, n_in, n_out, y);
    safe_free((void **) &x_q8);
}

void linear_q8_0_w8a8_prefill_pre(const int8_t *x_q8,
                                  const float  *scale_x,
                                  size_t        m,
                                  const void   *w_q8,
                                  size_t        n_in,
                                  size_t        n_out,
                                  float        *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    const struct block_q8_0_t *w          = (const struct block_q8_0_t *) w_q8;
    const size_t               nb_per_row = n_in / Q8_0_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q8_0_t *row = w + n * nb_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + nb_per_row, 0, 0);

        float accs[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        for (size_t i = 0; i < m; i++)
            accs[i] = 0.0f;

        for (size_t b = 0; b < nb_per_row; b++) {
            const struct block_q8_0_t *blk = &row[b];
            if (b + 2 < nb_per_row)
                __builtin_prefetch(&row[b + 2], 0, 0);
            const float d = fp16_to_fp32(blk->d);
            /* Read the 32 int8 weight bytes ONCE per super-block. */
            int8x16_t qv0 = vld1q_s8(blk->qs);
            int8x16_t qv1 = vld1q_s8(blk->qs + 16);
            for (size_t i = 0; i < m; i++) {
                const int8_t *xb      = x_q8 + i * n_in + b * Q8_0_BLOCK_ELEMS;
                int32_t       int_dot = dot16_i8(xb, qv0) + dot16_i8(xb + 16, qv1);
                accs[i] += d * scale_x[i] * (float) int_dot;
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] = accs[i];
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) m;
    (void) w_q8;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q8_0_w8a8_prefill_pre: NEON required\n");
#endif
}

void linear_q8_0_w8a8_prefill(
        const float *x, const void *w_q8, size_t m, size_t n_in, size_t n_out, float *y) {
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, m *n_in);
    float  *scale_x = heap_alloc_array_aligned(float, m);
    for (size_t i = 0; i < m; i++) {
        scale_x[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
    }
    linear_q8_0_w8a8_prefill_pre(x_q8, scale_x, m, w_q8, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &scale_x);
}
