/*
 * src/backends/cpu_neon/kernels/q5_K.c — Q5_K W5A8 NEON kernels.
 *
 * Pure compute. Block layout from src/quant/quant_blocks.h; the
 * dequantizer dequant_q5_K_row lives in src/formats/gguf/q5_K.c.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "par.h"
#include "quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* One Q5_K W5A8 call, for the row bodies below. */
struct q5k_rows_job {
    const struct block_q5_K_t *w;
    const int8_t              *x_q8;
    const float               *x_scales;
    const int32_t             *sum32;
    size_t                     m, n_in, n_out, n_blocks_per_row, n_chunks;
    float                     *y;
};

/* Output rows [n0, n1) of linear_q5k_decode_w5a8_pre. */
static void q5k_decode_rows(void *ctx, size_t n0, size_t n1) {
    const struct q5k_rows_job *job              = ctx;
    const struct block_q5_K_t *w                = job->w;
    const int8_t              *x_q8             = job->x_q8;
    const float               *x_scales         = job->x_scales;
    const int32_t             *sum32            = job->sum32;
    const size_t               n_out            = job->n_out;
    const size_t               n_blocks_per_row = job->n_blocks_per_row;
    float                     *y                = job->y;
    for (size_t n = n0; n < n1; n++) {
        const struct block_q5_K_t *row = w + n * n_blocks_per_row;
        float                      acc = 0.0f;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q5_K_t *blk = &row[b];
            if (b + 2 < n_blocks_per_row)
                __builtin_prefetch(&row[b + 2], 0, 0);
            const float    d_blk    = fp16_to_fp32(blk->d);
            const float    dmin_blk = fp16_to_fp32(blk->dmin);
            const uint8_t *ql       = blk->qs;
            const uint8_t *qh       = blk->qh; /* 32 bytes, shared across all sub-pairs */
            const int8_t  *xb       = x_q8 + b * Q5_K_BLOCK_ELEMS;
            const int32_t *sump     = sum32 + (b * Q5_K_BLOCK_ELEMS) / 32;
            uint8_t        u1 = 1, u2 = 2, sc, mn;

            for (int is = 0; is < 8; is += 2, ql += 32, xb += 64, sump += 2) {
                get_scale_min_k4(is + 0, blk->scales, &sc, &mn);
                const float d1  = d_blk * (float) sc;
                const float m1f = dmin_blk * (float) mn;
                get_scale_min_k4(is + 1, blk->scales, &sc, &mn);
                const float d2  = d_blk * (float) sc;
                const float m2f = dmin_blk * (float) mn;

                int32_t acc1 = 0, acc2 = 0;
#if defined(__ARM_NEON)
                for (int half = 0; half < 32; half += 16) {
                    uint8x16_t qv   = vld1q_u8(ql + half);
                    uint8x16_t qhv  = vld1q_u8(qh + half);
                    uint8x16_t lo4  = vandq_u8(qv, vdupq_n_u8(0x0F));
                    uint8x16_t hi4  = vshrq_n_u8(qv, 4);
                    uint8x16_t m_lo = vtstq_u8(qhv, vdupq_n_u8(u1));
                    uint8x16_t m_hi = vtstq_u8(qhv, vdupq_n_u8(u2));
                    int8x16_t  q_lo =
                            vreinterpretq_s8_u8(vaddq_u8(lo4, vandq_u8(m_lo, vdupq_n_u8(16))));
                    int8x16_t q_hi =
                            vreinterpretq_s8_u8(vaddq_u8(hi4, vandq_u8(m_hi, vdupq_n_u8(16))));
                    acc1 += dot16_i8(xb + half, q_lo);
                    acc2 += dot16_i8(xb + 32 + half, q_hi);
                }
#else
                for (int l = 0; l < 32; l++) {
                    int q1 = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                    int q2 = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                    acc1 += (int32_t) q1 * (int32_t) xb[l];
                    acc2 += (int32_t) q2 * (int32_t) xb[32 + l];
                }
#endif
                acc += x_scales[b] * (d1 * (float) acc1 - m1f * (float) sump[0]);
                acc += x_scales[b] * (d2 * (float) acc2 - m2f * (float) sump[1]);
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        y[n] = acc;
    }
}

void linear_q5k_decode_w5a8_pre(size_t         n_in,
                                size_t         n_out,
                                const float   *x_scales,
                                const int8_t   x_q8[static n_in],
                                const int32_t *sum32,
                                const void    *w_q5k,
                                float          y[static n_out]) {
    const struct block_q5_K_t *w                = (const struct block_q5_K_t *) w_q5k;
    const size_t               n_blocks_per_row = n_in / Q5_K_BLOCK_ELEMS;

    struct q5k_rows_job job = {.w                = w,
                               .x_q8             = x_q8,
                               .x_scales         = x_scales,
                               .sum32            = sum32,
                               .n_out            = n_out,
                               .n_blocks_per_row = n_blocks_per_row,
                               .y                = y};
    geist_par_for(n_out, q5k_decode_rows, &job);
}

void linear_q5k_decode_w5a8(size_t      n_in,
                            size_t      n_out,
                            const float x[static n_in],
                            const void *w_q5k,
                            float       y[static n_out]) {
    int8_t  *x_q8   = heap_alloc_array_aligned(int8_t, n_in);
    int32_t *sum32  = heap_alloc_array_aligned(int32_t, (n_in / 32));
    float *x_scales = heap_alloc_array_aligned(float, geist_act_groups(n_in, GEIST_ACT_Q8K_ELEMS));
    if (x_q8 == nullptr || sum32 == nullptr || x_scales == nullptr) {
        safe_free((void **) &x_q8);
        safe_free((void **) &sum32);
        safe_free((void **) &x_scales);
        return;
    }
    quantize_x_q8_groups(n_in, GEIST_ACT_Q8K_ELEMS, x, x_q8, x_scales, sum32);
    linear_q5k_decode_w5a8_pre(n_in, n_out, x_scales, x_q8, sum32, w_q5k, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &sum32);
    safe_free((void **) &x_scales);
}

#if defined(__ARM_NEON)
/* Output rows [n0, n1) of linear_q5k_w5a8_prefill_pre. */
static void q5k_prefill_rows(void *ctx, size_t n0, size_t n1) {
    const struct q5k_rows_job *job              = ctx;
    const struct block_q5_K_t *w                = job->w;
    const int8_t              *x_q8             = job->x_q8;
    const float               *x_scales         = job->x_scales;
    const int32_t             *sum32            = job->sum32;
    const size_t               m                = job->m;
    const size_t               n_in             = job->n_in;
    const size_t               n_out            = job->n_out;
    const size_t               n_blocks_per_row = job->n_blocks_per_row;
    const size_t               n_chunks         = job->n_chunks;
    float                     *y                = job->y;
    for (size_t n = n0; n < n1; n++) {
        const struct block_q5_K_t *row = w + n * n_blocks_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        float   accs[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        int32_t acc1[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        int32_t acc2[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        for (size_t i = 0; i < m; i++)
            accs[i] = 0.0f;

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q5_K_t *blk = &row[b];
            if (b + 2 < n_blocks_per_row)
                __builtin_prefetch(&row[b + 2], 0, 0);
            const float    d_blk    = fp16_to_fp32(blk->d);
            const float    dmin_blk = fp16_to_fp32(blk->dmin);
            const uint8_t *ql       = blk->qs;
            const uint8_t *qh       = blk->qh;
            uint8_t        u1 = 1, u2 = 2, sc, mn;

            for (int is = 0; is < 8; is += 2, ql += 32) {
                get_scale_min_k4(is + 0, blk->scales, &sc, &mn);
                const float d1  = d_blk * (float) sc;
                const float m1f = dmin_blk * (float) mn;
                get_scale_min_k4(is + 1, blk->scales, &sc, &mn);
                const float d2  = d_blk * (float) sc;
                const float m2f = dmin_blk * (float) mn;

                for (size_t i = 0; i < m; i++) {
                    acc1[i] = 0;
                    acc2[i] = 0;
                }

                for (int half = 0; half < 32; half += 16) {
                    uint8x16_t qv   = vld1q_u8(ql + half);
                    uint8x16_t qhv  = vld1q_u8(qh + half);
                    uint8x16_t lo4  = vandq_u8(qv, vdupq_n_u8(0x0F));
                    uint8x16_t hi4  = vshrq_n_u8(qv, 4);
                    uint8x16_t m_lo = vtstq_u8(qhv, vdupq_n_u8(u1));
                    uint8x16_t m_hi = vtstq_u8(qhv, vdupq_n_u8(u2));
                    int8x16_t  q_lo =
                            vreinterpretq_s8_u8(vaddq_u8(lo4, vandq_u8(m_lo, vdupq_n_u8(16))));
                    int8x16_t q_hi =
                            vreinterpretq_s8_u8(vaddq_u8(hi4, vandq_u8(m_hi, vdupq_n_u8(16))));
                    const size_t xb_off =
                            b * Q5_K_BLOCK_ELEMS + (size_t) (is / 2) * 64 + (size_t) half;
                    for (size_t i = 0; i < m; i++) {
                        const int8_t *xb_lo = x_q8 + i * n_in + xb_off;
                        const int8_t *xb_hi = xb_lo + 32;
                        acc1[i] += dot16_i8(xb_lo, q_lo);
                        acc2[i] += dot16_i8(xb_hi, q_hi);
                    }
                }

                const size_t sump_lo_idx = (b * Q5_K_BLOCK_ELEMS + (size_t) (is / 2) * 64) / 32;
                const size_t sump_hi_idx = sump_lo_idx + 1;
                for (size_t i = 0; i < m; i++) {
                    const int32_t s_lo = sum32[i * n_chunks + sump_lo_idx];
                    const int32_t s_hi = sum32[i * n_chunks + sump_hi_idx];
                    const float   sx   = x_scales[i * n_blocks_per_row + b];
                    accs[i] += sx * (d1 * (float) acc1[i] - m1f * (float) s_lo);
                    accs[i] += sx * (d2 * (float) acc2[i] - m2f * (float) s_hi);
                }
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] = accs[i];
    }
}
#endif

void linear_q5k_w5a8_prefill_pre(size_t         m,
                                 size_t         n_in,
                                 size_t         n_out,
                                 const int8_t  *x_q8,
                                 const float   *x_scales,
                                 const int32_t *sum32,
                                 const void    *w_q5k,
                                 float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    const struct block_q5_K_t *w                = (const struct block_q5_K_t *) w_q5k;
    const size_t               n_blocks_per_row = n_in / Q5_K_BLOCK_ELEMS;
    const size_t               n_chunks         = n_in / 32;

    struct q5k_rows_job job = {.w                = w,
                               .x_q8             = x_q8,
                               .x_scales         = x_scales,
                               .sum32            = sum32,
                               .m                = m,
                               .n_in             = n_in,
                               .n_out            = n_out,
                               .n_blocks_per_row = n_blocks_per_row,
                               .n_chunks         = n_chunks,
                               .y                = y};
    geist_par_for(n_out, q5k_prefill_rows, &job);
#else
    (void) x_q8;
    (void) x_scales;
    (void) sum32;
    (void) m;
    (void) w_q5k;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q5k_w5a8_prefill_pre: NEON required\n");
#endif
}

void linear_q5k_w5a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w_q5k, float *y) {
    const size_t n_groups = geist_act_groups(n_in, GEIST_ACT_Q8K_ELEMS);
    int8_t      *x_q8     = heap_alloc_array_aligned(int8_t, m *n_in);
    int32_t     *sum32    = heap_alloc_array_aligned(int32_t, m *(n_in / 32));
    float       *x_scales = heap_alloc_array_aligned(float, m *n_groups);
    if (x_q8 == nullptr || sum32 == nullptr || x_scales == nullptr) {
        safe_free((void **) &x_q8);
        safe_free((void **) &sum32);
        safe_free((void **) &x_scales);
        return;
    }
    for (size_t i = 0; i < m; i++) {
        quantize_x_q8_groups(n_in,
                             GEIST_ACT_Q8K_ELEMS,
                             x + i * n_in,
                             x_q8 + i * n_in,
                             x_scales + i * n_groups,
                             sum32 + i * (n_in / 32));
    }
    linear_q5k_w5a8_prefill_pre(m, n_in, n_out, x_q8, x_scales, sum32, w_q5k, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &sum32);
    safe_free((void **) &x_scales);
}
