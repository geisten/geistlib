/*
 * src/backends/cpu_neon/kernels/q8_0.c — Q8_0 W8A8 NEON kernels.
 *
 * Pure compute. Block layout from src/quant/quant_blocks.h.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "par.h"
#include "quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* One Q8_0 W8A8 call, for the row bodies below. */
struct q8_0_rows_job {
    const struct block_q8_0_t *w;
    const int8_t              *x_q8;
    const float               *x_scales;
    size_t                     m, n_in, n_out, nb_per_row;
    float                     *y;
};

/* Output rows [n0, n1) of linear_q8_0_decode_w8a8_pre. */
static void q8_0_decode_rows(void *ctx, size_t n0, size_t n1) {
    const struct q8_0_rows_job *job        = ctx;
    const struct block_q8_0_t  *w          = job->w;
    const int8_t               *x_q8       = job->x_q8;
    const float                *x_scales   = job->x_scales;
    const size_t                nb_per_row = job->nb_per_row;
    float                      *y          = job->y;
    for (size_t n = n0; n < n1; n++) {
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
            for (size_t j = 0; j < Q8_0_BLOCK_ELEMS; j++) {
                int_dot += (int32_t) xb[j] * (int32_t) blk->qs[j];
            }
#endif
            acc += d * x_scales[b] * (float) int_dot;
        }
        y[n] = acc;
    }
}

void linear_q8_0_decode_w8a8_pre(size_t       n_in,
                                 size_t       n_out,
                                 const float *x_scales,
                                 const int8_t x_q8[static n_in],
                                 const void  *w_q8,
                                 float        y[static n_out]) {
    struct q8_0_rows_job job = {.w          = (const struct block_q8_0_t *) w_q8,
                                .x_q8       = x_q8,
                                .x_scales   = x_scales,
                                .n_in       = n_in,
                                .n_out      = n_out,
                                .nb_per_row = n_in / Q8_0_BLOCK_ELEMS,
                                .y          = y};
    geist_par_for(n_out, q8_0_decode_rows, &job);
}

void linear_q8_0_decode_w8a8(size_t      n_in,
                             size_t      n_out,
                             const float x[static n_in],
                             const void *w_q8,
                             float       y[static n_out]) {
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    float *x_scales = heap_alloc_array_aligned(float, geist_act_groups(n_in, GEIST_ACT_Q8_0_ELEMS));
    if (x_q8 == nullptr || x_scales == nullptr) {
        safe_free((void **) &x_q8);
        safe_free((void **) &x_scales);
        return;
    }
    quantize_x_q8_groups(n_in, GEIST_ACT_Q8_0_ELEMS, x, x_q8, x_scales, nullptr);
    linear_q8_0_decode_w8a8_pre(n_in, n_out, x_scales, x_q8, w_q8, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &x_scales);
}

#if defined(__ARM_NEON)
/* Output rows [n0, n1) of linear_q8_0_w8a8_prefill_pre. */
static void q8_0_prefill_rows(void *ctx, size_t n0, size_t n1) {
    const struct q8_0_rows_job *j          = ctx;
    const struct block_q8_0_t  *w          = j->w;
    const int8_t               *x_q8       = j->x_q8;
    const float                *x_scales   = j->x_scales;
    const size_t                m          = j->m;
    const size_t                n_in       = j->n_in;
    const size_t                n_out      = j->n_out;
    const size_t                nb_per_row = j->nb_per_row;
    float                      *y          = j->y;
    for (size_t n = n0; n < n1; n++) {
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
            /* Load the block's 32 weight bytes once; reuse for all m rows. */
            int8x16_t qv0 = vld1q_s8(blk->qs);
            int8x16_t qv1 = vld1q_s8(blk->qs + 16);
            for (size_t i = 0; i < m; i++) {
                const int8_t *xb      = x_q8 + i * n_in + b * Q8_0_BLOCK_ELEMS;
                int32_t       int_dot = dot16_i8(xb, qv0) + dot16_i8(xb + 16, qv1);
                accs[i] += d * x_scales[i * nb_per_row + b] * (float) int_dot;
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] = accs[i];
    }
}
#endif

void linear_q8_0_w8a8_prefill_pre(size_t        m,
                                  size_t        n_in,
                                  size_t        n_out,
                                  const int8_t *x_q8,
                                  const float  *x_scales,
                                  const void   *w_q8,
                                  float        *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    struct q8_0_rows_job job = {.w          = (const struct block_q8_0_t *) w_q8,
                                .x_q8       = x_q8,
                                .x_scales   = x_scales,
                                .m          = m,
                                .n_in       = n_in,
                                .n_out      = n_out,
                                .nb_per_row = n_in / Q8_0_BLOCK_ELEMS,
                                .y          = y};
    geist_par_for(n_out, q8_0_prefill_rows, &job);
#else
    (void) x_q8;
    (void) x_scales;
    (void) m;
    (void) w_q8;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q8_0_w8a8_prefill_pre: NEON required\n");
#endif
}

void linear_q8_0_w8a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w_q8, float *y) {
    const size_t n_groups = geist_act_groups(n_in, GEIST_ACT_Q8_0_ELEMS);
    int8_t      *x_q8     = heap_alloc_array_aligned(int8_t, m *n_in);
    float       *x_scales = heap_alloc_array_aligned(float, m *n_groups);
    if (x_q8 == nullptr || x_scales == nullptr) {
        safe_free((void **) &x_q8);
        safe_free((void **) &x_scales);
        return;
    }
    for (size_t i = 0; i < m; i++) {
        quantize_x_q8_groups(n_in,
                             GEIST_ACT_Q8_0_ELEMS,
                             x + i * n_in,
                             x_q8 + i * n_in,
                             x_scales + i * n_groups,
                             nullptr);
    }
    linear_q8_0_w8a8_prefill_pre(m, n_in, n_out, x_q8, x_scales, w_q8, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &x_scales);
}
