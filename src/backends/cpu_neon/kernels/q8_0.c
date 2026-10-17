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
#include <stdatomic.h>
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

#if defined(__ARM_NEON)
/* 4 output rows x 8 tokens, SDOT register-tiled (#7, the Q4_K kernel's
 * structure from #16): the 4 rows of a 32-weight block are transposed into
 * 8 k-groups of [row0..3][4 bytes], so one vdotq_laneq_s32 adds a token's
 * 4-byte k-group to all four rows; 8 tokens are 8 independent chains. The
 * activations stay row-major (the lane picks the k-group). Each block's
 * int32 sum is exact, as in q8_0_prefill_rows; it is scaled by d_w * d_x
 * in one vector FMA. */
static void q8_0_sdot_tile4x8(const struct q8_0_rows_job *j, size_t n0, size_t mt) {
    const size_t               nb = j->nb_per_row, n_in = j->n_in;
    const struct block_q8_0_t *r0 = j->w + n0 * nb;
    float32x4_t                yacc[8];
    for (int t = 0; t < 8; t++) {
        yacc[t] = vdupq_n_f32(0.0f);
    }
    for (size_t b = 0; b < nb; b++) {
        const struct block_q8_0_t *bl[4] = {r0 + b, r0 + nb + b, r0 + 2 * nb + b, r0 + 3 * nb + b};
        int8x16_t                  w[8];
        for (int c = 0; c < 2; c++) {
            const int32x4_t q0 = vreinterpretq_s32_s8(vld1q_s8(bl[0]->qs + 16 * c));
            const int32x4_t q1 = vreinterpretq_s32_s8(vld1q_s8(bl[1]->qs + 16 * c));
            const int32x4_t q2 = vreinterpretq_s32_s8(vld1q_s8(bl[2]->qs + 16 * c));
            const int32x4_t q3 = vreinterpretq_s32_s8(vld1q_s8(bl[3]->qs + 16 * c));
            const int64x2_t t0 = vreinterpretq_s64_s32(vtrn1q_s32(q0, q1));
            const int64x2_t t1 = vreinterpretq_s64_s32(vtrn2q_s32(q0, q1));
            const int64x2_t t2 = vreinterpretq_s64_s32(vtrn1q_s32(q2, q3));
            const int64x2_t t3 = vreinterpretq_s64_s32(vtrn2q_s32(q2, q3));
            w[4 * c + 0]       = vreinterpretq_s8_s64(vtrn1q_s64(t0, t2));
            w[4 * c + 1]       = vreinterpretq_s8_s64(vtrn1q_s64(t1, t3));
            w[4 * c + 2]       = vreinterpretq_s8_s64(vtrn2q_s64(t0, t2));
            w[4 * c + 3]       = vreinterpretq_s8_s64(vtrn2q_s64(t1, t3));
        }
        const float       d_a[4] = {fp16_to_fp32(bl[0]->d),
                                    fp16_to_fp32(bl[1]->d),
                                    fp16_to_fp32(bl[2]->d),
                                    fp16_to_fp32(bl[3]->d)};
        const float32x4_t dv     = vld1q_f32(d_a);
        int32x4_t         acc[8];
        int8x16_t         a[8];
        const int8_t     *xb = j->x_q8 + mt * n_in + b * Q8_0_BLOCK_ELEMS;
        for (int t = 0; t < 8; t++) {
            acc[t] = vdupq_n_s32(0);
            a[t]   = vld1q_s8(xb + (size_t) t * n_in);
        }
        for (int t = 0; t < 8; t++) {
            acc[t] = vdotq_laneq_s32(acc[t], w[0], a[t], 0);
        }
        for (int t = 0; t < 8; t++) {
            acc[t] = vdotq_laneq_s32(acc[t], w[1], a[t], 1);
        }
        for (int t = 0; t < 8; t++) {
            acc[t] = vdotq_laneq_s32(acc[t], w[2], a[t], 2);
        }
        for (int t = 0; t < 8; t++) {
            acc[t] = vdotq_laneq_s32(acc[t], w[3], a[t], 3);
        }
        for (int t = 0; t < 8; t++) {
            a[t] = vld1q_s8(xb + (size_t) t * n_in + 16);
        }
        for (int t = 0; t < 8; t++) {
            acc[t] = vdotq_laneq_s32(acc[t], w[4], a[t], 0);
        }
        for (int t = 0; t < 8; t++) {
            acc[t] = vdotq_laneq_s32(acc[t], w[5], a[t], 1);
        }
        for (int t = 0; t < 8; t++) {
            acc[t] = vdotq_laneq_s32(acc[t], w[6], a[t], 2);
        }
        for (int t = 0; t < 8; t++) {
            acc[t] = vdotq_laneq_s32(acc[t], w[7], a[t], 3);
        }
        for (int t = 0; t < 8; t++) {
            const float xs = j->x_scales[(mt + (size_t) t) * nb + b];
            yacc[t]        = vfmaq_f32(yacc[t], vcvtq_f32_s32(acc[t]), vmulq_n_f32(dv, xs));
        }
    }
    for (int t = 0; t < 8; t++) {
        vst1q_f32(j->y + (mt + (size_t) t) * j->n_out + n0, yacc[t]);
    }
}

/* 4-row tiles [nt0, nt1): full tiles and full groups of 8 tokens on the
 * SDOT tile, the rest (a ragged last tile, m % 8 tokens) on
 * q8_0_prefill_rows. */
static void q8_0_prefill_sdot_tiles(void *ctx, size_t nt0, size_t nt1) {
    const struct q8_0_rows_job *j  = ctx;
    const size_t                m8 = j->m / 8 * 8;
    for (size_t nt = nt0; nt < nt1; nt++) {
        const size_t n0 = nt * 4;
        if (n0 + 4 > j->n_out || m8 == 0) {
            q8_0_prefill_rows(ctx, n0, n0 + 4 < j->n_out ? n0 + 4 : j->n_out);
            continue;
        }
        for (size_t mt = 0; mt < m8; mt += 8) {
            q8_0_sdot_tile4x8(j, n0, mt);
        }
        if (m8 < j->m) {
            struct q8_0_rows_job tail = *j;
            tail.m                    = j->m - m8;
            tail.x_q8                 = j->x_q8 + m8 * j->n_in;
            tail.x_scales             = j->x_scales + m8 * j->nb_per_row;
            tail.y                    = j->y + m8 * j->n_out;
            q8_0_prefill_rows(&tail, n0, n0 + 4);
        }
    }
}

static bool q8_0_sdot_on(void) {
    static _Atomic int on = -1;
    int                v  = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("GEIST_Q8_0_SDOT_PREFILL");
        v             = e == nullptr || e[0] != '0';
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v != 0;
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
    if (q8_0_sdot_on() && m >= 8) {
        geist_par_for((n_out + 3) / 4, q8_0_prefill_sdot_tiles, &job);
    } else {
        geist_par_for(n_out, q8_0_prefill_rows, &job);
    }
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
