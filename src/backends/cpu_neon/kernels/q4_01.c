/*
 * src/backends/cpu_neon/kernels/q4_01.c — Q4_0 / Q4_1 W8A8 NEON kernels.
 *
 * Pure compute. Replaces the generic dequant trampoline for the two
 * 32-element traditional quants (#281 perf phase: the qwen35 Q4_0
 * exports spend 65% of decode in trampolined FFN linears).
 *
 * Q4_0 block (18 B): fp16 d + 16 nibble bytes. Element j in [0,16) is
 * lo(qs[j]) - 8, element j+16 is hi(qs[j]) - 8 — so the int8 dot pairs
 * x[0..15] with the low nibbles and x[16..31] with the high nibbles.
 *
 * Q4_1 block (20 B): fp16 d, fp16 m + 16 nibble bytes, value = d*q + m
 * with unsigned q. With symmetric int8 activations (x = x_q8 * sx):
 *   sum_j x_j * (d*q_j + m) = sx * (d * dot(x_q8, q) + m * bsum)
 * where bsum = sum of x_q8 over the block — computed once per x, not
 * per row.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "quant.h"

#include <stdint.h>
#include <stdlib.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

struct block_q4_0k_t {
    uint16_t d;
    uint8_t  qs[16];
} __attribute__((packed));
_Static_assert(sizeof(struct block_q4_0k_t) == 18, "q4_0 block size");

struct block_q4_1k_t {
    uint16_t d;
    uint16_t m;
    uint8_t  qs[16];
} __attribute__((packed));
_Static_assert(sizeof(struct block_q4_1k_t) == 20, "q4_1 block size");

/* dot of 32 activations against one nibble block, weights biased by -8
 * (Q4_0) or unbiased (Q4_1: bias handled via bsum outside). */
static inline int32_t q4_nibble_dot(const int8_t *xb, const uint8_t *qs, int biased) {
#if defined(__ARM_NEON)
    const uint8x16_t q  = vld1q_u8(qs);
    int8x16_t        lo = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
    int8x16_t        hi = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
    if (biased) {
        lo = vsubq_s8(lo, vdupq_n_s8(8));
        hi = vsubq_s8(hi, vdupq_n_s8(8));
    }
    return dot16_i8(xb, lo) + dot16_i8(xb + 16, hi);
#else
    int32_t acc = 0;
    for (int j = 0; j < 16; j++) {
        const int lo = (int) (qs[j] & 0x0F) - (biased ? 8 : 0);
        const int hi = (int) (qs[j] >> 4) - (biased ? 8 : 0);
        acc += (int32_t) xb[j] * lo + (int32_t) xb[j + 16] * hi;
    }
    return acc;
#endif
}

void linear_q4_0_decode_w4a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_q4, size_t n_in, size_t n_out, float *y) {
    const struct block_q4_0k_t *w          = (const struct block_q4_0k_t *) w_q4;
    const size_t                nb_per_row = n_in / Q4_0_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q4_0k_t *row = w + n * nb_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + nb_per_row, 0, 0);
        float acc = 0.0f;
        for (size_t b = 0; b < nb_per_row; b++) {
            const int32_t dot = q4_nibble_dot(x_q8 + b * Q4_0_BLOCK_ELEMS, row[b].qs, 1);
            acc += fp16_to_fp32(row[b].d) * (float) dot;
        }
        y[n] = acc * scale_x;
    }
}

void linear_q4_0_decode_w4a8(
        const float *x, const void *w_q4, size_t n_in, size_t n_out, float *y) {
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    float   scale_x = quantize_x_int8_sym(x, n_in, x_q8);
    linear_q4_0_decode_w4a8_pre(x_q8, scale_x, w_q4, n_in, n_out, y);
    safe_free((void **) &x_q8);
}

void linear_q4_1_decode_w4a8_pre(const int8_t  *x_q8,
                                 float          scale_x,
                                 const int32_t *bsum,
                                 const void    *w_q4,
                                 size_t         n_in,
                                 size_t         n_out,
                                 float         *y) {
    const struct block_q4_1k_t *w          = (const struct block_q4_1k_t *) w_q4;
    const size_t                nb_per_row = n_in / Q4_1_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q4_1k_t *row = w + n * nb_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + nb_per_row, 0, 0);
        float acc = 0.0f;
        for (size_t b = 0; b < nb_per_row; b++) {
            const int32_t dot = q4_nibble_dot(x_q8 + b * Q4_1_BLOCK_ELEMS, row[b].qs, 0);
            acc += fp16_to_fp32(row[b].d) * (float) dot + fp16_to_fp32(row[b].m) * (float) bsum[b];
        }
        y[n] = acc * scale_x;
    }
}

void linear_q4_1_decode_w4a8(
        const float *x, const void *w_q4, size_t n_in, size_t n_out, float *y) {
    const size_t nb      = n_in / Q4_1_BLOCK_ELEMS;
    int8_t      *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    int32_t     *bsum    = heap_alloc_array_aligned(int32_t, nb);
    float        scale_x = quantize_x_int8_sym(x, n_in, x_q8);
    for (size_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (size_t j = 0; j < Q4_1_BLOCK_ELEMS; j++)
            s += x_q8[b * Q4_1_BLOCK_ELEMS + j];
        bsum[b] = s;
    }
    linear_q4_1_decode_w4a8_pre(x_q8, scale_x, bsum, w_q4, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &bsum);
}

/* ---- M>1 prefill: quantize each row once, tile over output rows. ---- */

void linear_q4_0_w4a8_prefill(
        const float *x, size_t m, const void *w_q4, size_t n_in, size_t n_out, float *y) {
    int8_t *x_q8   = heap_alloc_array_aligned(int8_t, m *n_in);
    float  *scales = heap_alloc_array_aligned(float, m);
    for (size_t i = 0; i < m; i++)
        scales[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);

    const struct block_q4_0k_t *w          = (const struct block_q4_0k_t *) w_q4;
    const size_t                nb_per_row = n_in / Q4_0_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q4_0k_t *row = w + n * nb_per_row;
        for (size_t i = 0; i < m; i++) {
            const int8_t *xr  = x_q8 + i * n_in;
            float         acc = 0.0f;
            for (size_t b = 0; b < nb_per_row; b++) {
                const int32_t dot = q4_nibble_dot(xr + b * Q4_0_BLOCK_ELEMS, row[b].qs, 1);
                acc += fp16_to_fp32(row[b].d) * (float) dot;
            }
            y[i * n_out + n] = acc * scales[i];
        }
    }
    safe_free((void **) &x_q8);
    safe_free((void **) &scales);
}

void linear_q4_1_w4a8_prefill(
        const float *x, size_t m, const void *w_q4, size_t n_in, size_t n_out, float *y) {
    const size_t nb     = n_in / Q4_1_BLOCK_ELEMS;
    int8_t      *x_q8   = heap_alloc_array_aligned(int8_t, m *n_in);
    float       *scales = heap_alloc_array_aligned(float, m);
    int32_t     *bsums  = heap_alloc_array_aligned(int32_t, m *nb);
    for (size_t i = 0; i < m; i++) {
        scales[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
        for (size_t b = 0; b < nb; b++) {
            int32_t s = 0;
            for (size_t j = 0; j < Q4_1_BLOCK_ELEMS; j++)
                s += x_q8[i * n_in + b * Q4_1_BLOCK_ELEMS + j];
            bsums[i * nb + b] = s;
        }
    }

    const struct block_q4_1k_t *w          = (const struct block_q4_1k_t *) w_q4;
    const size_t                nb_per_row = n_in / Q4_1_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q4_1k_t *row = w + n * nb_per_row;
        for (size_t i = 0; i < m; i++) {
            const int8_t  *xr  = x_q8 + i * n_in;
            const int32_t *bs  = bsums + i * nb;
            float          acc = 0.0f;
            for (size_t b = 0; b < nb_per_row; b++) {
                const int32_t dot = q4_nibble_dot(xr + b * Q4_1_BLOCK_ELEMS, row[b].qs, 0);
                acc += fp16_to_fp32(row[b].d) * (float) dot +
                       fp16_to_fp32(row[b].m) * (float) bs[b];
            }
            y[i * n_out + n] = acc * scales[i];
        }
    }
    safe_free((void **) &x_q8);
    safe_free((void **) &scales);
    safe_free((void **) &bsums);
}
