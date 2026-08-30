/*
 * src/backends/cpu_neon/kernels/iq4_xs.c — IQ4_XS / IQ4_NL W4A8 NEON
 * decode GEMVs.
 *
 * Both formats store 4-bit indices into the fixed 16-value kvalues_iq4nl
 * table — exactly one vqtbl1q_s8 per 16 nibbles, no reconstruction
 * scratch. Block layouts from src/quant/quant_blocks.h. Replaces the
 * dequant-and-sgemv trampoline for M=1 (Pi 5 baseline: 4.4 tok/s decode
 * on a 0.8B IQ4_XS vs 13.2 on the twice-as-large Q8_0) and — for
 * IQ4_XS on non-Accelerate hosts — for M>1 via the #321 tile kernel
 * below. IQ4_NL prefill stays on the trampoline.
 */
#include "heap.h"
#include "quant.h"
#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static const int8_t kvalues_iq4nl_k[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

void linear_iq4xs_decode_w4a8_pre(const int8_t *x_q8,
                                  float         scale_x,
                                  const void   *w_iq4xs,
                                  size_t        n_in,
                                  size_t        n_out,
                                  float        *y) {
#if defined(__ARM_NEON)
    const struct block_iq4_xs_t *w                = (const struct block_iq4_xs_t *) w_iq4xs;
    const size_t                 n_blocks_per_row = n_in / IQ4_XS_BLOCK_ELEMS;
    const int8x16_t              kv               = vld1q_s8(kvalues_iq4nl_k);
    const uint8x16_t             low4             = vdupq_n_u8(0x0f);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_iq4_xs_t *row = w + n * n_blocks_per_row;
        float                        acc = 0.0f;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_iq4_xs_t *blk = &row[b];
            if (b + 1 < n_blocks_per_row)
                __builtin_prefetch(&row[b + 1], 0, 0);

            const float   d  = fp16_to_fp32(blk->d);
            const int8_t *xb = x_q8 + b * IQ4_XS_BLOCK_ELEMS;

            int32x4_t int_acc = vdupq_n_s32(0);
            for (int ib = 0; ib < 8; ib++) {
                const int32_t    ls = (int32_t) ((blk->scales_l[ib / 2] >> (4 * (ib & 1))) & 0xf) |
                                      (int32_t) (((blk->scales_h >> (2 * ib)) & 3) << 4);
                const uint8x16_t q  = vld1q_u8(blk->qs + 16 * ib);
                const int8x16_t  w_lo = vqtbl1q_s8(kv, vandq_u8(q, low4));
                const int8x16_t  w_hi = vqtbl1q_s8(kv, vshrq_n_u8(q, 4));
                int32x4_t        dot  = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xb + 0), w_lo);
                dot                   = vdotq_s32(dot, vld1q_s8(xb + 16), w_hi);
                int_acc               = vmlaq_n_s32(int_acc, dot, ls - 32);
                xb += 32;
            }
            acc += d * (float) vaddvq_s32(int_acc);
        }
        y[n] = acc * scale_x;
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) w_iq4xs;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_iq4xs_decode_w4a8_pre: NEON required\n");
#endif
}

void linear_iq4xs_decode_w4a8(
        const float *x, const void *w_iq4xs, size_t n_in, size_t n_out, float *y) {
    int8_t     *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    const float scale_x = quantize_x_int8_sym(x, n_in, x_q8);
    linear_iq4xs_decode_w4a8_pre(x_q8, scale_x, w_iq4xs, n_in, n_out, y);
    safe_free((void **) &x_q8);
}

/* ---- IQ4_XS int8 mN prefill (#321) ------------------------------------ *
 *
 * Replaces the dequant+SGEMM trampoline for M>1. Row-major sweep with a
 * 4-token register tile: each 136-byte block is LUT-decoded ONCE (16
 * vqtbl1q) and dotted against 4 activation rows, and the row's blocks
 * stay L1-resident across all token groups, so both decode work and
 * weight traffic amortize. Per (row, token) the op order matches the m1
 * kernel exactly — the output is bit-identical to calling it m times
 * (pinned by test_iq4_dequant_unit). No repack, no extra RSS; the
 * 8-row interleaved lane-SDOT variant stays a documented follow-up if
 * this plateau is not enough. */
void linear_iq4xs_w4a8_prefill_pre(const int8_t *x_q8,
                                   const float  *scale_x,
                                   size_t        m,
                                   const void   *w_iq4xs,
                                   size_t        n_in,
                                   size_t        n_out,
                                   float        *y) {
#if defined(__ARM_NEON)
    if (m == 0)
        return;
    const struct block_iq4_xs_t *w                = (const struct block_iq4_xs_t *) w_iq4xs;
    const size_t                 n_blocks_per_row = n_in / IQ4_XS_BLOCK_ELEMS;
    const int8x16_t              kv               = vld1q_s8(kvalues_iq4nl_k);
    const uint8x16_t             low4             = vdupq_n_u8(0x0f);
    const float                 *scales           = scale_x;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_iq4_xs_t *row = w + n * n_blocks_per_row;
        for (size_t t0 = 0; t0 < m; t0 += 4) {
            const size_t tcnt    = (m - t0 < 4) ? (m - t0) : 4;
            float        accf[4] = {0};
            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct block_iq4_xs_t *blk = &row[b];
                __builtin_prefetch(blk + 1, 0, 0);
                const float d = fp16_to_fp32(blk->d);
                int8x16_t   w_lo[8], w_hi[8];
                int32_t     lsv[8];
                for (int ib = 0; ib < 8; ib++) {
                    lsv[ib] = ((int32_t) ((blk->scales_l[ib / 2] >> (4 * (ib & 1))) & 0xf) |
                               (int32_t) (((blk->scales_h >> (2 * ib)) & 3) << 4)) -
                              32;
                    const uint8x16_t q = vld1q_u8(blk->qs + 16 * ib);
                    w_lo[ib]           = vqtbl1q_s8(kv, vandq_u8(q, low4));
                    w_hi[ib]           = vqtbl1q_s8(kv, vshrq_n_u8(q, 4));
                }
                for (size_t t = 0; t < tcnt; t++) {
                    const int8_t *xb      = x_q8 + (t0 + t) * n_in + b * IQ4_XS_BLOCK_ELEMS;
                    int32x4_t     int_acc = vdupq_n_s32(0);
                    for (int ib = 0; ib < 8; ib++) {
                        int32x4_t dot = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xb), w_lo[ib]);
                        dot           = vdotq_s32(dot, vld1q_s8(xb + 16), w_hi[ib]);
                        int_acc       = vmlaq_n_s32(int_acc, dot, lsv[ib]);
                        xb += 32;
                    }
                    accf[t] += d * (float) vaddvq_s32(int_acc);
                }
            }
            for (size_t t = 0; t < tcnt; t++)
                y[(t0 + t) * n_out + n] = accf[t] * scales[t0 + t];
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) m;
    (void) w_iq4xs;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_iq4xs_w4a8_prefill_pre: NEON required\n");
#endif
}

void linear_iq4xs_w4a8_prefill(
        const float *x, size_t m, const void *w_iq4xs, size_t n_in, size_t n_out, float *y) {
    int8_t *x_q8   = heap_alloc_array_aligned(int8_t, m *n_in);
    float  *scales = heap_alloc_array_aligned(float, m);
    if (x_q8 == nullptr || scales == nullptr) {
        safe_free((void **) &x_q8);
        safe_free((void **) &scales);
        return;
    }
    for (size_t i = 0; i < m; i++)
        scales[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
    linear_iq4xs_w4a8_prefill_pre(x_q8, scales, m, w_iq4xs, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &scales);
}

void linear_iq4nl_decode_w4a8_pre(const int8_t *x_q8,
                                  float         scale_x,
                                  const void   *w_iq4nl,
                                  size_t        n_in,
                                  size_t        n_out,
                                  float        *y) {
#if defined(__ARM_NEON)
    const struct block_iq4_nl_t *w                = (const struct block_iq4_nl_t *) w_iq4nl;
    const size_t                 n_blocks_per_row = n_in / IQ4_NL_BLOCK_ELEMS;
    const int8x16_t              kv               = vld1q_s8(kvalues_iq4nl_k);
    const uint8x16_t             low4             = vdupq_n_u8(0x0f);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_iq4_nl_t *row = w + n * n_blocks_per_row;
        float                        acc = 0.0f;
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_iq4_nl_t *blk  = &row[b];
            const int8_t                *xb   = x_q8 + b * IQ4_NL_BLOCK_ELEMS;
            const uint8x16_t             q    = vld1q_u8(blk->qs);
            const int8x16_t              w_lo = vqtbl1q_s8(kv, vandq_u8(q, low4));
            const int8x16_t              w_hi = vqtbl1q_s8(kv, vshrq_n_u8(q, 4));
            int32x4_t                    dot  = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xb + 0), w_lo);
            dot                               = vdotq_s32(dot, vld1q_s8(xb + 16), w_hi);
            acc += fp16_to_fp32(blk->d) * (float) vaddvq_s32(dot);
        }
        y[n] = acc * scale_x;
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) w_iq4nl;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_iq4nl_decode_w4a8_pre: NEON required\n");
#endif
}

void linear_iq4nl_decode_w4a8(
        const float *x, const void *w_iq4nl, size_t n_in, size_t n_out, float *y) {
    int8_t     *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    const float scale_x = quantize_x_int8_sym(x, n_in, x_q8);
    linear_iq4nl_decode_w4a8_pre(x_q8, scale_x, w_iq4nl, n_in, n_out, y);
    safe_free((void **) &x_q8);
}
