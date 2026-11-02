/*
 * src/backends/cpu_neon/kernels/mxfp4.c — MXFP4 W4A8 NEON GEMV and
 * prefill (GPT-OSS's experts, #243).
 *
 * kvalues_mxfp4 holds twice the E2M1 values, all exact int8, so a block's
 * 32 weights are one vqtbl1q_s8 per 16 nibbles and two SDOTs against the
 * activation's int8 block (one scale per 32, as Q8_0). The block's float
 * factor is mxfp4_e8m0_half(e) * x_scale, the halving undoing the doubled
 * table. Layout in src/quant/quant_blocks.h (17-byte blocks, unaligned).
 *
 * Per (row, token) the prefill does the m=1 kernel's operations in its
 * order, so a token's output does not depend on how many share the call.
 */
#include "par.h"
#include "quant.h"
#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

struct mxfp4_rows_job {
    const struct block_mxfp4_t *w;
    const int8_t               *x_q8;
    const float                *x_scales;
    size_t                      m, n_in, n_out, nb;
    float                      *y;
};

#if defined(__ARM_NEON)
/* The 32 weights of block blk as int8: lo = elements 0..15, hi = 16..31. */
static inline __attribute__((always_inline)) void
mxfp4_block_s8(const struct block_mxfp4_t *blk, int8x16_t kv, int8x16_t *lo, int8x16_t *hi) {
    const uint8x16_t q = vld1q_u8(blk->qs);
    *lo                = vqtbl1q_s8(kv, vandq_u8(q, vdupq_n_u8(0x0f)));
    *hi                = vqtbl1q_s8(kv, vshrq_n_u8(q, 4));
}

/* One token's dot with one row: float lanes accumulate d_w * d_x * idot
 * per block, summed across lanes at the end. */
static inline __attribute__((always_inline)) float
mxfp4_row_dot(size_t nb, const struct block_mxfp4_t *row, const int8_t *xq, const float *xs) {
    const int8x16_t kv  = vld1q_s8(kvalues_mxfp4);
    float32x4_t     acc = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        int8x16_t lo, hi;
        mxfp4_block_s8(&row[b], kv, &lo, &hi);
        int32x4_t dot = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xq + 32 * b), lo);
        dot           = vdotq_s32(dot, vld1q_s8(xq + 32 * b + 16), hi);
        acc           = vmlaq_n_f32(acc, vcvtq_f32_s32(dot), mxfp4_e8m0_half(row[b].e) * xs[b]);
    }
    return vaddvq_f32(acc);
}

static void mxfp4_decode_rows(void *ctx, size_t n0, size_t n1) {
    const struct mxfp4_rows_job *job = ctx;
    for (size_t n = n0; n < n1; n++) {
        const struct block_mxfp4_t *row = job->w + n * job->nb;
        if (n + 1 < n1)
            __builtin_prefetch(row + job->nb, 0, 0);
        job->y[n] = mxfp4_row_dot(job->nb, row, job->x_q8, job->x_scales);
    }
}

/* Output rows [n0, n1) for all m tokens, 4 tokens per decoded block. */
static void mxfp4_prefill_rows(void *ctx, size_t n0, size_t n1) {
    const struct mxfp4_rows_job *job = ctx;
    const size_t                 nb = job->nb, n_in = job->n_in, n_out = job->n_out, m = job->m;
    const int8x16_t              kv = vld1q_s8(kvalues_mxfp4);
    for (size_t n = n0; n < n1; n++) {
        const struct block_mxfp4_t *row = job->w + n * nb;
        size_t                      t0  = 0;
        for (; t0 + 4 <= m; t0 += 4) {
            float32x4_t   a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
            const int8_t *x0 = job->x_q8 + t0 * n_in;
            const float  *s0 = job->x_scales + t0 * nb;
            for (size_t b = 0; b < nb; b++) {
                int8x16_t lo, hi;
                mxfp4_block_s8(&row[b], kv, &lo, &hi);
                const float   d  = mxfp4_e8m0_half(row[b].e);
                const int8_t *x  = x0 + 32 * b;
                int32x4_t     q0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(x), lo);
                int32x4_t     q1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(x + n_in), lo);
                int32x4_t     q2 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(x + 2 * n_in), lo);
                int32x4_t     q3 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(x + 3 * n_in), lo);
                q0               = vdotq_s32(q0, vld1q_s8(x + 16), hi);
                q1               = vdotq_s32(q1, vld1q_s8(x + n_in + 16), hi);
                q2               = vdotq_s32(q2, vld1q_s8(x + 2 * n_in + 16), hi);
                q3               = vdotq_s32(q3, vld1q_s8(x + 3 * n_in + 16), hi);
                a0               = vmlaq_n_f32(a0, vcvtq_f32_s32(q0), d * s0[b]);
                a1               = vmlaq_n_f32(a1, vcvtq_f32_s32(q1), d * s0[nb + b]);
                a2               = vmlaq_n_f32(a2, vcvtq_f32_s32(q2), d * s0[2 * nb + b]);
                a3               = vmlaq_n_f32(a3, vcvtq_f32_s32(q3), d * s0[3 * nb + b]);
            }
            job->y[t0 * n_out + n]       = vaddvq_f32(a0);
            job->y[(t0 + 1) * n_out + n] = vaddvq_f32(a1);
            job->y[(t0 + 2) * n_out + n] = vaddvq_f32(a2);
            job->y[(t0 + 3) * n_out + n] = vaddvq_f32(a3);
        }
        for (; t0 < m; t0++)
            job->y[t0 * n_out + n] =
                    mxfp4_row_dot(nb, row, job->x_q8 + t0 * n_in, job->x_scales + t0 * nb);
    }
}
#endif

void linear_mxfp4_decode_w4a8_pre(size_t       n_in,
                                  size_t       n_out,
                                  const float *x_scales,
                                  const int8_t x_q8[static n_in],
                                  const void  *w_mxfp4,
                                  float        y[static n_out]) {
#if defined(__ARM_NEON)
    struct mxfp4_rows_job job = {.w        = (const struct block_mxfp4_t *) w_mxfp4,
                                 .x_q8     = x_q8,
                                 .x_scales = x_scales,
                                 .m        = 1,
                                 .n_in     = n_in,
                                 .n_out    = n_out,
                                 .nb       = n_in / MXFP4_BLOCK_ELEMS,
                                 .y        = y};
    geist_par_for(n_out, mxfp4_decode_rows, &job);
#else
    (void) x_scales;
    (void) x_q8;
    (void) w_mxfp4;
    (void) y;
    fprintf(stderr, "linear_mxfp4_decode_w4a8_pre: NEON required\n");
#endif
}

void linear_mxfp4_w4a8_prefill_pre(size_t        m,
                                   size_t        n_in,
                                   size_t        n_out,
                                   const int8_t *x_q8,
                                   const float  *x_scales,
                                   const void   *w_mxfp4,
                                   float        *y) {
#if defined(__ARM_NEON)
    if (m == 0)
        return;
    struct mxfp4_rows_job job = {.w        = (const struct block_mxfp4_t *) w_mxfp4,
                                 .x_q8     = x_q8,
                                 .x_scales = x_scales,
                                 .m        = m,
                                 .n_in     = n_in,
                                 .n_out    = n_out,
                                 .nb       = n_in / MXFP4_BLOCK_ELEMS,
                                 .y        = y};
    geist_par_for(n_out, mxfp4_prefill_rows, &job);
#else
    (void) m;
    (void) n_in;
    (void) n_out;
    (void) x_q8;
    (void) x_scales;
    (void) w_mxfp4;
    (void) y;
    fprintf(stderr, "linear_mxfp4_w4a8_prefill_pre: NEON required\n");
#endif
}
