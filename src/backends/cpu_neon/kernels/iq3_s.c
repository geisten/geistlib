/*
 * src/backends/cpu_neon/kernels/iq3_s.c — IQ3_S W3A8 NEON kernels.
 *
 * Block layout from src/quant/quant_blocks.h. iq3s_subblock_to_int8
 * is duplicated as static inline (same in formats/gguf/iq3_s.c) — the
 * helper is shared between dequant + W3A8 inner loop.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "quant.h"
#include "iq_grids.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Shared with formats/gguf/iq3_s.c — IQ3_S sub-block grid lookup +
 * sign expansion into a 32-int8 working tile. */
static inline void iq3s_subblock_to_int8(int8_t         out[32],
                                         const uint8_t *qs_lo,  /* 8 bytes */
                                         const uint8_t *sign_b, /* 4 bytes */
                                         uint8_t        qh) {
#if defined(__ARM_NEON)
    const uint8x8_t kmask_v = vld1_u8(kmask_iq2xs);
    for (int l = 0; l < 4; l++) {
        const uint32_t idx1 = (uint32_t) qs_lo[2 * l + 0] | ((uint32_t) (qh << (8 - 2 * l)) & 256);
        const uint32_t idx2 = (uint32_t) qs_lo[2 * l + 1] | ((uint32_t) (qh << (7 - 2 * l)) & 256);
        /* Pack grid1[0..3] || grid2[0..3] into one 8-byte NEON vector. */
        uint32x2_t pair          = vdup_n_u32(0);
        pair                     = vld1_lane_u32((const uint32_t *) (iq3s_grid + idx1), pair, 0);
        pair                     = vld1_lane_u32((const uint32_t *) (iq3s_grid + idx2), pair, 1);
        const int8x8_t  grid_v   = vreinterpret_s8_u32(pair);
        const uint8x8_t sig_dup  = vdup_n_u8(sign_b[l]);
        const uint8x8_t neg_msk  = vtst_u8(sig_dup, kmask_v);
        const int8x8_t  signed_v = vbsl_s8(neg_msk, vneg_s8(grid_v), grid_v);
        vst1_s8(out + l * 8, signed_v);
    }
#else
    for (int l = 0; l < 4; l++) {
        const uint32_t idx1  = (uint32_t) qs_lo[2 * l + 0] | ((uint32_t) (qh << (8 - 2 * l)) & 256);
        const uint32_t idx2  = (uint32_t) qs_lo[2 * l + 1] | ((uint32_t) (qh << (7 - 2 * l)) & 256);
        const uint8_t *grid1 = (const uint8_t *) (iq3s_grid + idx1);
        const uint8_t *grid2 = (const uint8_t *) (iq3s_grid + idx2);
        const uint8_t  sig   = sign_b[l];
        for (int j = 0; j < 4; j++) {
            const int8_t v1    = (int8_t) grid1[j];
            const int8_t v2    = (int8_t) grid2[j];
            out[l * 8 + j + 0] = (sig & kmask_iq2xs[j + 0]) ? (int8_t) -v1 : v1;
            out[l * 8 + j + 4] = (sig & kmask_iq2xs[j + 4]) ? (int8_t) -v2 : v2;
        }
    }
#endif
}

void linear_iq3s_decode_w3a8_pre(const int8_t *x_q8,
                                 float         scale_x,
                                 const void   *w_iq3s,
                                 size_t        n_in,
                                 size_t        n_out,
                                 float        *y) {
#if defined(__ARM_NEON)
    const struct block_iq3_s_t *w                = (const struct block_iq3_s_t *) w_iq3s;
    const size_t                n_blocks_per_row = n_in / IQ3_S_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_iq3_s_t *row = w + n * n_blocks_per_row;
        float                       acc = 0.0f;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_iq3_s_t *blk = &row[b];
            if (b + 2 < n_blocks_per_row)
                __builtin_prefetch(&row[b + 2], 0, 0);

            const float   d  = fp16_to_fp32(blk->d);
            const int8_t *xb = x_q8 + b * IQ3_S_BLOCK_ELEMS;

            int32x4_t int_acc = vdupq_n_s32(0);
            int8_t    w_i8[32] __attribute__((aligned(16)));

            /* 4 outer iterations; each covers two 32-element sub-blocks
             * sharing one scales byte (low nibble + high nibble). */
            for (int ib = 0; ib < 4; ib++) {
                const int32_t s_a = 1 + 2 * (blk->scales[ib] & 0xf);
                const int32_t s_b = 1 + 2 * (blk->scales[ib] >> 4);

                /* First 32-element half. */
                iq3s_subblock_to_int8(
                        w_i8, &blk->qs[ib * 16 + 0], &blk->signs[ib * 8 + 0], blk->qh[ib * 2 + 0]);
                {
                    const int8x16_t x_lo = vld1q_s8(xb + 0);
                    const int8x16_t w_lo = vld1q_s8(w_i8 + 0);
                    const int8x16_t x_hi = vld1q_s8(xb + 16);
                    const int8x16_t w_hi = vld1q_s8(w_i8 + 16);
                    int32x4_t       d_lo = vdotq_s32(vdupq_n_s32(0), x_lo, w_lo);
                    d_lo                 = vdotq_s32(d_lo, x_hi, w_hi);
                    int_acc              = vmlaq_n_s32(int_acc, d_lo, s_a);
                }
                xb += 32;

                /* Second 32-element half. */
                iq3s_subblock_to_int8(
                        w_i8, &blk->qs[ib * 16 + 8], &blk->signs[ib * 8 + 4], blk->qh[ib * 2 + 1]);
                {
                    const int8x16_t x_lo = vld1q_s8(xb + 0);
                    const int8x16_t w_lo = vld1q_s8(w_i8 + 0);
                    const int8x16_t x_hi = vld1q_s8(xb + 16);
                    const int8x16_t w_hi = vld1q_s8(w_i8 + 16);
                    int32x4_t       d_lo = vdotq_s32(vdupq_n_s32(0), x_lo, w_lo);
                    d_lo                 = vdotq_s32(d_lo, x_hi, w_hi);
                    int_acc              = vmlaq_n_s32(int_acc, d_lo, s_b);
                }
                xb += 32;
            }
            const int32_t block_acc = vaddvq_s32(int_acc);
            acc += d * scale_x * (float) block_acc;
        }
        y[n] = acc;
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) w_iq3s;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_iq3s_decode_w3a8_pre: NEON required\n");
#endif
}

void linear_iq3s_decode_w3a8(
        const float *x, const void *w_iq3s, size_t n_in, size_t n_out, float *y) {
    int8_t     *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    const float scale_x = quantize_x_int8_sym(x, n_in, x_q8);
    linear_iq3s_decode_w3a8_pre(x_q8, scale_x, w_iq3s, n_in, n_out, y);
    safe_free((void **) &x_q8);
}

void linear_iq3s_w3a8_prefill_pre(const int8_t *x_q8,
                                  const float  *scale_x,
                                  size_t        m,
                                  const void   *w_iq3s,
                                  size_t        n_in,
                                  size_t        n_out,
                                  float        *y) {
#if defined(__ARM_NEON)
    /* m is caller-bounded by the engine's m_max (= GEIST_QUANT_M_CAP); the
     * runtime guard catches contract violations without crashing. */
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    const struct block_iq3_s_t *w                = (const struct block_iq3_s_t *) w_iq3s;
    const size_t                n_blocks_per_row = n_in / IQ3_S_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_iq3_s_t *row = w + n * n_blocks_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        float     accs[GEIST_QUANT_M_CAP] = {0};
        int32x4_t row_acc[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_iq3_s_t *blk = &row[b];
            if (b + 2 < n_blocks_per_row)
                __builtin_prefetch(&row[b + 2], 0, 0);
            const float d = fp16_to_fp32(blk->d);

            for (size_t i = 0; i < m; i++)
                row_acc[i] = vdupq_n_s32(0);

            int8_t w_i8[32] __attribute__((aligned(16)));
            /* 4 outer iterations × 2 halves = 8 sub-blocks of 32 elements. */
            for (int ib = 0; ib < 4; ib++) {
                const int32_t s_a = 1 + 2 * (blk->scales[ib] & 0xf);
                const int32_t s_b = 1 + 2 * (blk->scales[ib] >> 4);

                /* First 32-element half (scale s_a). */
                iq3s_subblock_to_int8(
                        w_i8, &blk->qs[ib * 16 + 0], &blk->signs[ib * 8 + 0], blk->qh[ib * 2 + 0]);
                {
                    const int8x16_t w_lo   = vld1q_s8(w_i8 + 0);
                    const int8x16_t w_hi   = vld1q_s8(w_i8 + 16);
                    const size_t    xb_off = b * IQ3_S_BLOCK_ELEMS + (size_t) ib * 64;
                    for (size_t i = 0; i < m; i++) {
                        const int8_t *xb    = x_q8 + i * n_in + xb_off;
                        int32x4_t     d_acc = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xb + 0), w_lo);
                        d_acc               = vdotq_s32(d_acc, vld1q_s8(xb + 16), w_hi);
                        row_acc[i]          = vmlaq_n_s32(row_acc[i], d_acc, s_a);
                    }
                }

                /* Second 32-element half (scale s_b). */
                iq3s_subblock_to_int8(
                        w_i8, &blk->qs[ib * 16 + 8], &blk->signs[ib * 8 + 4], blk->qh[ib * 2 + 1]);
                {
                    const int8x16_t w_lo   = vld1q_s8(w_i8 + 0);
                    const int8x16_t w_hi   = vld1q_s8(w_i8 + 16);
                    const size_t    xb_off = b * IQ3_S_BLOCK_ELEMS + (size_t) ib * 64 + 32;
                    for (size_t i = 0; i < m; i++) {
                        const int8_t *xb    = x_q8 + i * n_in + xb_off;
                        int32x4_t     d_acc = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xb + 0), w_lo);
                        d_acc               = vdotq_s32(d_acc, vld1q_s8(xb + 16), w_hi);
                        row_acc[i]          = vmlaq_n_s32(row_acc[i], d_acc, s_b);
                    }
                }
            }
            for (size_t i = 0; i < m; i++) {
                accs[i] += d * scale_x[i] * (float) vaddvq_s32(row_acc[i]);
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] = accs[i];
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) m;
    (void) w_iq3s;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_iq3s_w3a8_prefill_pre: NEON required\n");
#endif
}

void linear_iq3s_w3a8_prefill(
        const float *x, const void *w_iq3s, size_t m, size_t n_in, size_t n_out, float *y) {
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, m *n_in);
    float  *scale_x = heap_alloc_array_aligned(float, m);
    for (size_t i = 0; i < m; i++) {
        scale_x[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
    }
    linear_iq3s_w3a8_prefill_pre(x_q8, scale_x, m, w_iq3s, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &scale_x);
}
