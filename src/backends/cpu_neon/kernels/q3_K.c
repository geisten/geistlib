/*
 * src/backends/cpu_neon/kernels/q3_K.c — Q3_K W3A8 NEON kernels.
 *
 * Block layout from src/quant/quant_blocks.h. Two helpers are
 * shared with the format-side dequant in src/formats/gguf/q3_K.c
 * (unpack_q3k_scales, q3k_reconstruct_q32); both are `static inline`
 * and duplicated here rather than promoted to a cross-layer header.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "quant.h"

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

/* Shared with formats/gguf/q3_K.c (pure scalar, format-decode helper). */
static inline void unpack_q3k_scales(const uint8_t *sc_packed, int8_t *sc_out) {
    for (int j = 0; j < 16; j++) {
        const uint8_t low4  = (uint8_t) ((sc_packed[j % 8] >> ((j / 8) * 4u)) & 0x0Fu);
        const uint8_t high2 = (uint8_t) ((sc_packed[8 + (j % 4)] >> ((j / 4) * 2u)) & 0x03u);
        sc_out[j]           = (int8_t) ((int) (low4 | (high2 << 4u)) - 32);
    }
}

#if defined(__ARM_NEON)
/* Shared with formats/gguf/q3_K.c (NEON-only dequant helper). */
static inline void q3k_reconstruct_q32(uint8x16_t qs_shifted_lo,
                                       uint8x16_t qs_shifted_hi,
                                       uint8x16_t hm_shifted_lo,
                                       uint8x16_t hm_shifted_hi,
                                       int8x16_t *out_lo,
                                       int8x16_t *out_hi) {
    const uint8x16_t mask3  = vdupq_n_u8(0x03);
    const uint8x16_t mask1  = vdupq_n_u8(0x01);
    const int8x16_t  bias4  = vdupq_n_s8(4);
    uint8x16_t       low2_l = vandq_u8(qs_shifted_lo, mask3);
    uint8x16_t       low2_h = vandq_u8(qs_shifted_hi, mask3);
    uint8x16_t       hi1_l  = vandq_u8(hm_shifted_lo, mask1);
    uint8x16_t       hi1_h  = vandq_u8(hm_shifted_hi, mask1);
    *out_lo = vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(low2_l, vshlq_n_u8(hi1_l, 2))), bias4);
    *out_hi = vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(low2_h, vshlq_n_u8(hi1_h, 2))), bias4);
}

/* Kernel-only: accumulator variant for the W3A8 inner loop. */
static inline int32x4_t q3k_w3a8_acc32(int32x4_t     acc,
                                       uint8x16_t    qs_shifted_lo,
                                       uint8x16_t    qs_shifted_hi,
                                       uint8x16_t    hm_shifted_lo,
                                       uint8x16_t    hm_shifted_hi,
                                       const int8_t *xb_g32,
                                       int8_t        s_first16,
                                       int8_t        s_second16) {
    const uint8x16_t mask3  = vdupq_n_u8(0x03);
    const uint8x16_t mask1  = vdupq_n_u8(0x01);
    const int8x16_t  bias4  = vdupq_n_s8(4);
    uint8x16_t       low2_l = vandq_u8(qs_shifted_lo, mask3);
    uint8x16_t       low2_h = vandq_u8(qs_shifted_hi, mask3);
    uint8x16_t       hi1_l  = vandq_u8(hm_shifted_lo, mask1);
    uint8x16_t       hi1_h  = vandq_u8(hm_shifted_hi, mask1);
    int8x16_t q_l   = vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(low2_l, vshlq_n_u8(hi1_l, 2))), bias4);
    int8x16_t q_h   = vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(low2_h, vshlq_n_u8(hi1_h, 2))), bias4);
    int32x4_t dot_l = vdotq_s32(vdupq_n_s32(0), q_l, vld1q_s8(xb_g32 + 0));
    int32x4_t dot_h = vdotq_s32(vdupq_n_s32(0), q_h, vld1q_s8(xb_g32 + 16));
    acc             = vmlaq_n_s32(acc, dot_l, (int32_t) s_first16);
    acc             = vmlaq_n_s32(acc, dot_h, (int32_t) s_second16);
    return acc;
}
#endif

void linear_q3k_decode_w3a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_q3k, size_t n_in, size_t n_out, float *y) {
#if defined(__ARM_NEON)
    const struct block_q3_K_t *w                = (const struct block_q3_K_t *) w_q3k;
    const size_t               n_blocks_per_row = n_in / Q3_K_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q3_K_t *row = w + n * n_blocks_per_row;
        float                      acc = 0.0f;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q3_K_t *blk = &row[b];
            if (b + 2 < n_blocks_per_row)
                __builtin_prefetch(&row[b + 2], 0, 0);

            const float   d  = fp16_to_fp32(blk->d);
            const int8_t *xb = x_q8 + b * Q3_K_BLOCK_ELEMS;

            int8_t sc[16];
            unpack_q3k_scales(blk->scales, sc);

            const uint8x16_t hm_lo  = vld1q_u8(blk->hmask + 0);
            const uint8x16_t hm_hi  = vld1q_u8(blk->hmask + 16);
            const uint8x16_t qsA_lo = vld1q_u8(blk->qs + 0);
            const uint8x16_t qsA_hi = vld1q_u8(blk->qs + 16);
            const uint8x16_t qsB_lo = vld1q_u8(blk->qs + 32);
            const uint8x16_t qsB_hi = vld1q_u8(blk->qs + 48);

            /* Accumulator-style: scaled int dots fold into int32x4_t via
             * vmlaq_n_s32, reduce ONCE at end of super-block (1 vaddvq vs 16). */
            int32x4_t int_acc = vdupq_n_s32(0);
            int_acc = q3k_w3a8_acc32(int_acc, qsA_lo, qsA_hi, hm_lo, hm_hi, xb + 0, sc[0], sc[1]);
            int_acc = q3k_w3a8_acc32(int_acc,
                                     vshrq_n_u8(qsA_lo, 2),
                                     vshrq_n_u8(qsA_hi, 2),
                                     vshrq_n_u8(hm_lo, 1),
                                     vshrq_n_u8(hm_hi, 1),
                                     xb + 32,
                                     sc[2],
                                     sc[3]);
            int_acc = q3k_w3a8_acc32(int_acc,
                                     vshrq_n_u8(qsA_lo, 4),
                                     vshrq_n_u8(qsA_hi, 4),
                                     vshrq_n_u8(hm_lo, 2),
                                     vshrq_n_u8(hm_hi, 2),
                                     xb + 64,
                                     sc[4],
                                     sc[5]);
            int_acc = q3k_w3a8_acc32(int_acc,
                                     vshrq_n_u8(qsA_lo, 6),
                                     vshrq_n_u8(qsA_hi, 6),
                                     vshrq_n_u8(hm_lo, 3),
                                     vshrq_n_u8(hm_hi, 3),
                                     xb + 96,
                                     sc[6],
                                     sc[7]);
            int_acc = q3k_w3a8_acc32(int_acc,
                                     qsB_lo,
                                     qsB_hi,
                                     vshrq_n_u8(hm_lo, 4),
                                     vshrq_n_u8(hm_hi, 4),
                                     xb + 128,
                                     sc[8],
                                     sc[9]);
            int_acc = q3k_w3a8_acc32(int_acc,
                                     vshrq_n_u8(qsB_lo, 2),
                                     vshrq_n_u8(qsB_hi, 2),
                                     vshrq_n_u8(hm_lo, 5),
                                     vshrq_n_u8(hm_hi, 5),
                                     xb + 160,
                                     sc[10],
                                     sc[11]);
            int_acc = q3k_w3a8_acc32(int_acc,
                                     vshrq_n_u8(qsB_lo, 4),
                                     vshrq_n_u8(qsB_hi, 4),
                                     vshrq_n_u8(hm_lo, 6),
                                     vshrq_n_u8(hm_hi, 6),
                                     xb + 192,
                                     sc[12],
                                     sc[13]);
            int_acc = q3k_w3a8_acc32(int_acc,
                                     vshrq_n_u8(qsB_lo, 6),
                                     vshrq_n_u8(qsB_hi, 6),
                                     vshrq_n_u8(hm_lo, 7),
                                     vshrq_n_u8(hm_hi, 7),
                                     xb + 224,
                                     sc[14],
                                     sc[15]);
            const int32_t block_acc = vaddvq_s32(int_acc);

            acc += d * scale_x * (float) block_acc;
        }
        y[n] = acc;
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) w_q3k;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q3k_decode_w3a8_pre: NEON required\n");
#endif
}

void linear_q3k_decode_w3a8(
        const float *x, const void *w_q3k, size_t n_in, size_t n_out, float *y) {
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    float   scale_x = quantize_x_int8_sym(x, n_in, x_q8);
    linear_q3k_decode_w3a8_pre(x_q8, scale_x, w_q3k, n_in, n_out, y);
    safe_free((void **) &x_q8);
}

void linear_q3k_w3a8_prefill_pre(const int8_t *x_q8,
                                 const float  *scale_x,
                                 size_t        m,
                                 const void   *w_q3k,
                                 size_t        n_in,
                                 size_t        n_out,
                                 float        *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    const struct block_q3_K_t *w                = (const struct block_q3_K_t *) w_q3k;
    const size_t               n_blocks_per_row = n_in / Q3_K_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q3_K_t *row = w + n * n_blocks_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        float   my_accs[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        int32_t block_int_accs[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        for (size_t i = 0; i < m; i++)
            my_accs[i] = 0.0f;

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q3_K_t *blk = &row[b];
            if (b + 2 < n_blocks_per_row)
                __builtin_prefetch(&row[b + 2], 0, 0);

            const float d = fp16_to_fp32(blk->d);
            int8_t      sc[16];
            unpack_q3k_scales(blk->scales, sc);

            const uint8x16_t hm_lo  = vld1q_u8(blk->hmask + 0);
            const uint8x16_t hm_hi  = vld1q_u8(blk->hmask + 16);
            const uint8x16_t qsA_lo = vld1q_u8(blk->qs + 0);
            const uint8x16_t qsA_hi = vld1q_u8(blk->qs + 16);
            const uint8x16_t qsB_lo = vld1q_u8(blk->qs + 32);
            const uint8x16_t qsB_hi = vld1q_u8(blk->qs + 48);

            for (size_t i = 0; i < m; i++)
                block_int_accs[i] = 0;

#define PROCESS_GROUP_M(G, QSL, QSH, HML, HMH, XOFF)                               \
    do {                                                                           \
        int8x16_t q_l, q_h;                                                        \
        q3k_reconstruct_q32(QSL, QSH, HML, HMH, &q_l, &q_h);                       \
        const int32_t s_first  = (int32_t) sc[2 * (G)];                            \
        const int32_t s_second = (int32_t) sc[2 * (G) + 1];                        \
        for (size_t i = 0; i < m; i++) {                                           \
            const int8_t *xb    = x_q8 + i * n_in + b * Q3_K_BLOCK_ELEMS + (XOFF); \
            int32_t       dot_l = dot16_i8(xb + 0, q_l);                           \
            int32_t       dot_h = dot16_i8(xb + 16, q_h);                          \
            block_int_accs[i] += s_first * dot_l + s_second * dot_h;               \
        }                                                                          \
    } while (0)

            PROCESS_GROUP_M(0, qsA_lo, qsA_hi, hm_lo, hm_hi, 0);
            PROCESS_GROUP_M(1,
                            vshrq_n_u8(qsA_lo, 2),
                            vshrq_n_u8(qsA_hi, 2),
                            vshrq_n_u8(hm_lo, 1),
                            vshrq_n_u8(hm_hi, 1),
                            32);
            PROCESS_GROUP_M(2,
                            vshrq_n_u8(qsA_lo, 4),
                            vshrq_n_u8(qsA_hi, 4),
                            vshrq_n_u8(hm_lo, 2),
                            vshrq_n_u8(hm_hi, 2),
                            64);
            PROCESS_GROUP_M(3,
                            vshrq_n_u8(qsA_lo, 6),
                            vshrq_n_u8(qsA_hi, 6),
                            vshrq_n_u8(hm_lo, 3),
                            vshrq_n_u8(hm_hi, 3),
                            96);
            PROCESS_GROUP_M(4, qsB_lo, qsB_hi, vshrq_n_u8(hm_lo, 4), vshrq_n_u8(hm_hi, 4), 128);
            PROCESS_GROUP_M(5,
                            vshrq_n_u8(qsB_lo, 2),
                            vshrq_n_u8(qsB_hi, 2),
                            vshrq_n_u8(hm_lo, 5),
                            vshrq_n_u8(hm_hi, 5),
                            160);
            PROCESS_GROUP_M(6,
                            vshrq_n_u8(qsB_lo, 4),
                            vshrq_n_u8(qsB_hi, 4),
                            vshrq_n_u8(hm_lo, 6),
                            vshrq_n_u8(hm_hi, 6),
                            192);
            PROCESS_GROUP_M(7,
                            vshrq_n_u8(qsB_lo, 6),
                            vshrq_n_u8(qsB_hi, 6),
                            vshrq_n_u8(hm_lo, 7),
                            vshrq_n_u8(hm_hi, 7),
                            224);
#undef PROCESS_GROUP_M

            for (size_t i = 0; i < m; i++) {
                my_accs[i] += d * scale_x[i] * (float) block_int_accs[i];
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] = my_accs[i];
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) m;
    (void) w_q3k;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q3k_w3a8_prefill_pre: NEON required\n");
#endif
}

void linear_q3k_w3a8_prefill(
        const float *x, const void *w_q3k, size_t m, size_t n_in, size_t n_out, float *y) {
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, m *n_in);
    float  *scale_x = heap_alloc_array_aligned(float, m);
    for (size_t i = 0; i < m; i++) {
        scale_x[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
    }
    linear_q3k_w3a8_prefill_pre(x_q8, scale_x, m, w_q3k, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &scale_x);
}
