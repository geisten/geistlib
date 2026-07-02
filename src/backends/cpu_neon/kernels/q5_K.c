/*
 * src/backends/cpu_neon/kernels/q5_K.c — Q5_K W5A8 NEON kernels.
 *
 * Pure compute. The on-disk Q5_K block layout (struct block_q5_K_t)
 * comes from src/quant/quant_blocks.h; the file-format decoder
 * dequant_q5_K_row stays in src/formats/gguf/q5_K.c.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void linear_q5k_decode_w5a8_pre(const int8_t  *x_q8,
                                float          scale_x,
                                const int32_t *sum32,
                                const void    *w_q5k,
                                size_t         n_in,
                                size_t         n_out,
                                float         *y) {
    const struct block_q5_K_t *w                = (const struct block_q5_K_t *) w_q5k;
    const size_t               n_blocks_per_row = n_in / Q5_K_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
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
                acc += scale_x * (d1 * (float) acc1 - m1f * (float) sump[0]);
                acc += scale_x * (d2 * (float) acc2 - m2f * (float) sump[1]);
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        y[n] = acc;
    }
}

void linear_q5k_decode_w5a8(
        const float *x, const void *w_q5k, size_t n_in, size_t n_out, float *y) {
    int8_t  *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    int32_t *sum32   = heap_alloc_array_aligned(int32_t, (n_in / 32));
    float    scale_x = quantize_x_for_q4k(x, n_in, x_q8, sum32);
    linear_q5k_decode_w5a8_pre(x_q8, scale_x, sum32, w_q5k, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &sum32);
}

void linear_q5k_w5a8_prefill_pre(const int8_t  *x_q8,
                                 const float   *scale_x,
                                 const int32_t *sum32,
                                 size_t         m,
                                 const void    *w_q5k,
                                 size_t         n_in,
                                 size_t         n_out,
                                 float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    const struct block_q5_K_t *w                = (const struct block_q5_K_t *) w_q5k;
    const size_t               n_blocks_per_row = n_in / Q5_K_BLOCK_ELEMS;
    const size_t               n_chunks         = n_in / 32;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
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
                    accs[i] += scale_x[i] * (d1 * (float) acc1[i] - m1f * (float) s_lo);
                    accs[i] += scale_x[i] * (d2 * (float) acc2[i] - m2f * (float) s_hi);
                }
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] = accs[i];
    }
#else
    (void) x_q8;
    (void) scale_x;
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
        const float *x, const void *w_q5k, size_t m, size_t n_in, size_t n_out, float *y) {
    int8_t  *x_q8    = heap_alloc_array_aligned(int8_t, m *n_in);
    int32_t *sum32   = heap_alloc_array_aligned(int32_t, m *(n_in / 32));
    float   *scale_x = heap_alloc_array_aligned(float, m);
    for (size_t i = 0; i < m; i++) {
        scale_x[i] =
                quantize_x_for_q4k(x + i * n_in, n_in, x_q8 + i * n_in, sum32 + i * (n_in / 32));
    }
    linear_q5k_w5a8_prefill_pre(x_q8, scale_x, sum32, m, w_q5k, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &sum32);
    safe_free((void **) &scale_x);
}
