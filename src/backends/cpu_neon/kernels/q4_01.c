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

/* ---- Q4_0 x8 interleaved GEMV (decode lever, #281) -------------------- *
 *
 * Same idea as the Q6_K x8 layout: pack 8 consecutive output rows so one
 * sequential streaming pass serves all 8 — activation loads amortize 8x
 * and the walk is pure 144-byte chunks. The -8 weight bias is deferred
 * through per-block activation sums (dot_raw - 8*bsum), saving the two
 * vsubq_s8 per row-block of the row-major kernel. */

struct q4_0_x8_header {
    uint32_t magic;
    uint32_t n_in;
    uint32_t n_out;
    uint32_t block_bytes;
};

struct q4_0_x8_block {
    uint16_t d[8];
    uint8_t  qs[16 * 8]; /* row r nibbles at qs + r*16 */
};
_Static_assert(sizeof(struct q4_0_x8_block) == 8 * 18, "q4_0 x8 block size");

static const uint32_t Q4_0_X8_GEMV_MAGIC = 0x38583430u; /* "04X8" */

size_t q4_0_x8_gemv_size_bytes(size_t n_in, size_t n_out) {
    if (n_in == 0 || n_out == 0 || n_in % Q4_0_BLOCK_ELEMS != 0 || n_out % 8 != 0)
        return 0;
    if (n_in > UINT32_MAX || n_out > UINT32_MAX)
        return 0;
    const size_t n_blocks = (n_out / 8) * (n_in / Q4_0_BLOCK_ELEMS);
    if (n_blocks > (SIZE_MAX - sizeof(struct q4_0_x8_header)) / sizeof(struct q4_0_x8_block))
        return 0;
    return sizeof(struct q4_0_x8_header) + n_blocks * sizeof(struct q4_0_x8_block);
}

int q4_0_x8_gemv_pack(const void *w_q4, size_t n_in, size_t n_out, void *dst) {
    if (q4_0_x8_gemv_size_bytes(n_in, n_out) == 0 || w_q4 == NULL || dst == NULL)
        return -1;
    const struct block_q4_0k_t *src        = (const struct block_q4_0k_t *) w_q4;
    const size_t                nb_per_row = n_in / Q4_0_BLOCK_ELEMS;
    struct q4_0_x8_header      *h          = (struct q4_0_x8_header *) dst;
    h->magic                               = Q4_0_X8_GEMV_MAGIC;
    h->n_in                                = (uint32_t) n_in;
    h->n_out                               = (uint32_t) n_out;
    h->block_bytes                         = (uint32_t) sizeof(struct q4_0_x8_block);
    struct q4_0_x8_block *out              = (struct q4_0_x8_block *) (h + 1);
    for (size_t tile = 0; tile < n_out / 8; tile++) {
        for (size_t b = 0; b < nb_per_row; b++) {
            struct q4_0_x8_block *ob = out + tile * nb_per_row + b;
            for (size_t r = 0; r < 8; r++) {
                const struct block_q4_0k_t *sb = src + (tile * 8 + r) * nb_per_row + b;
                ob->d[r]                       = sb->d;
                for (int j = 0; j < 16; j++)
                    ob->qs[r * 16 + j] = sb->qs[j];
            }
        }
    }
    return 0;
}

static int q4_0_x8_valid(const void *packed, size_t n_in, size_t n_out) {
    const struct q4_0_x8_header *h = (const struct q4_0_x8_header *) packed;
    return packed != NULL && h->magic == Q4_0_X8_GEMV_MAGIC && h->n_in == n_in &&
           h->n_out == n_out && h->block_bytes == sizeof(struct q4_0_x8_block);
}

void linear_q4_0_decode_w4a8_x8_pre(const int8_t  *x_q8,
                                    float          scale_x,
                                    const int32_t *bsum,
                                    const void    *packed,
                                    size_t         n_in,
                                    size_t         n_out,
                                    float         *y) {
    if (!q4_0_x8_valid(packed, n_in, n_out))
        return;
    const size_t                nb_per_row = n_in / Q4_0_BLOCK_ELEMS;
    const struct q4_0_x8_block *w = (const struct q4_0_x8_block *) ((const uint8_t *) packed +
                                                                    sizeof(struct q4_0_x8_header));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t tile = 0; tile < n_out / 8; tile++) {
        const struct q4_0_x8_block *row     = w + tile * nb_per_row;
        float                       accf[8] = {0};
        for (size_t b = 0; b < nb_per_row; b++) {
            const struct q4_0_x8_block *blk = row + b;
            __builtin_prefetch(blk + 2, 0, 0);
            const int8_t *xb    = x_q8 + b * Q4_0_BLOCK_ELEMS;
            const float   bias8 = 8.0f * (float) bsum[b];
#if defined(__ARM_NEON)
            const int8x16_t xa = vld1q_s8(xb);
            const int8x16_t xz = vld1q_s8(xb + 16);
            for (int r = 0; r < 8; r++) {
                const uint8x16_t q   = vld1q_u8(blk->qs + r * 16);
                const int8x16_t  lo  = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
                const int8x16_t  hi  = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
                int32x4_t        d32 = vdotq_s32(vdupq_n_s32(0), lo, xa);
                d32                  = vdotq_s32(d32, hi, xz);
                accf[r] += fp16_to_fp32(blk->d[r]) * ((float) vaddvq_s32(d32) - bias8);
            }
#else
            for (int r = 0; r < 8; r++) {
                int32_t acc = 0;
                for (int j = 0; j < 16; j++) {
                    acc += (int32_t) xb[j] * (int32_t) (blk->qs[r * 16 + j] & 0x0F);
                    acc += (int32_t) xb[j + 16] * (int32_t) (blk->qs[r * 16 + j] >> 4);
                }
                accf[r] += fp16_to_fp32(blk->d[r]) * ((float) acc - bias8);
            }
#endif
        }
        for (int r = 0; r < 8; r++)
            y[tile * 8 + r] = accf[r] * scale_x;
    }
}

void linear_q4_0_decode_w4a8_x8(
        const float *x, const void *packed, size_t n_in, size_t n_out, float *y) {
    const size_t nb   = n_in / Q4_0_BLOCK_ELEMS;
    int8_t      *x_q8 = heap_alloc_array_aligned(int8_t, n_in);
    int32_t     *bsum = heap_alloc_array_aligned(int32_t, nb);
    if (x_q8 == NULL || bsum == NULL) {
        safe_free((void **) &x_q8);
        safe_free((void **) &bsum);
        return;
    }
    const float scale_x = quantize_x_int8_sym(x, n_in, x_q8);
    for (size_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (size_t j = 0; j < Q4_0_BLOCK_ELEMS; j++)
            s += x_q8[b * Q4_0_BLOCK_ELEMS + j];
        bsum[b] = s;
    }
    linear_q4_0_decode_w4a8_x8_pre(x_q8, scale_x, bsum, packed, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &bsum);
}

/* ---- Q4_0 x8 int8 mN GEMM (#295 prefill lever) ------------------------ *
 *
 * The Mac prefill path dequantizes weight tiles to f32 and calls
 * Accelerate SGEMM — two memory passes and 8x the weight traffic.
 * This kernel runs int8 GEMM directly on the #291 x8 layout: each
 * 144-byte block (8 rows x 32 elems) is loaded ONCE and dotted
 * against T activation rows, so weight traffic amortizes T-fold and
 * the arithmetic is SDOT. Register tiling: 8 rows x 4 tokens = 16
 * int32x4 accumulators alive per block column pass.
 */

void linear_q4_0_w4a8_prefill_x8(
        const float *x, size_t m, const void *packed, size_t n_in, size_t n_out, float *y) {
    if (!q4_0_x8_valid(packed, n_in, n_out) || m == 0)
        return;
    const size_t nb     = n_in / Q4_0_BLOCK_ELEMS;
    int8_t      *x_q8   = heap_alloc_array_aligned(int8_t, m *n_in);
    float       *scales = heap_alloc_array_aligned(float, m);
    int32_t     *bsums  = heap_alloc_array_aligned(int32_t, m *nb);
    if (x_q8 == NULL || scales == NULL || bsums == NULL) {
        safe_free((void **) &x_q8);
        safe_free((void **) &scales);
        safe_free((void **) &bsums);
        return;
    }
    for (size_t i = 0; i < m; i++) {
        scales[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
        for (size_t b = 0; b < nb; b++) {
            int32_t acc = 0;
            for (size_t j = 0; j < Q4_0_BLOCK_ELEMS; j++)
                acc += x_q8[i * n_in + b * Q4_0_BLOCK_ELEMS + j];
            bsums[i * nb + b] = acc;
        }
    }
    const struct q4_0_x8_block *w = (const struct q4_0_x8_block *) ((const uint8_t *) packed +
                                                                    sizeof(struct q4_0_x8_header));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t tile = 0; tile < n_out / 8; tile++) {
        const struct q4_0_x8_block *row = w + tile * nb;
        for (size_t t0 = 0; t0 < m; t0 += 4) {
            const size_t tcnt = (m - t0 < 4) ? (m - t0) : 4;
#if defined(__ARM_NEON)
            /* f32 accumulators per token (d varies per block) */
            float32x4_t facc[4][2];
            for (size_t t = 0; t < 4; t++) {
                facc[t][0] = vdupq_n_f32(0.0f);
                facc[t][1] = vdupq_n_f32(0.0f);
            }
            for (size_t b = 0; b < nb; b++) {
                const struct q4_0_x8_block *blk = row + b;
                __builtin_prefetch(blk + 2, 0, 0);
                float d_arr[8];
                for (int r = 0; r < 8; r++)
                    d_arr[r] = fp16_to_fp32(blk->d[r]);
                const float32x4_t d0 = vld1q_f32(d_arr + 0);
                const float32x4_t d1 = vld1q_f32(d_arr + 4);
                int8x16_t         lo[8], hi[8];
                for (int r = 0; r < 8; r++) {
                    const uint8x16_t q = vld1q_u8(blk->qs + r * 16);
                    lo[r]              = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
                    hi[r]              = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
                }
                for (size_t t = 0; t < tcnt; t++) {
                    const int8_t   *xb = x_q8 + (t0 + t) * n_in + b * Q4_0_BLOCK_ELEMS;
                    const int8x16_t xa = vld1q_s8(xb);
                    const int8x16_t xz = vld1q_s8(xb + 16);
                    int32_t         d[8];
                    for (int r = 0; r < 8; r++) {
                        int32x4_t dd = vdotq_s32(vdupq_n_s32(0), lo[r], xa);
                        dd           = vdotq_s32(dd, hi[r], xz);
                        d[r]         = vaddvq_s32(dd);
                    }
                    const float bias8 = 8.0f * (float) bsums[(t0 + t) * nb + b];
                    facc[t][0]        = vfmaq_f32(
                            facc[t][0],
                            d0,
                            vsubq_f32(vcvtq_f32_s32(vld1q_s32(d + 0)), vdupq_n_f32(bias8)));
                    facc[t][1] = vfmaq_f32(
                            facc[t][1],
                            d1,
                            vsubq_f32(vcvtq_f32_s32(vld1q_s32(d + 4)), vdupq_n_f32(bias8)));
                }
            }
            for (size_t t = 0; t < tcnt; t++) {
                float *yt = y + (t0 + t) * n_out + tile * 8;
                vst1q_f32(yt + 0, vmulq_n_f32(facc[t][0], scales[t0 + t]));
                vst1q_f32(yt + 4, vmulq_n_f32(facc[t][1], scales[t0 + t]));
            }
#else
            for (size_t t = 0; t < tcnt; t++) {
                float *yt = y + (t0 + t) * n_out + tile * 8;
                for (int r = 0; r < 8; r++) {
                    float acc_f = 0.0f;
                    for (size_t b = 0; b < nb; b++) {
                        const struct q4_0_x8_block *blk = row + b;
                        const int8_t *xb = x_q8 + (t0 + t) * n_in + b * Q4_0_BLOCK_ELEMS;
                        int32_t       dt = 0;
                        for (int j = 0; j < 16; j++) {
                            dt += (int32_t) xb[j] * (int32_t) (blk->qs[r * 16 + j] & 0x0F);
                            dt += (int32_t) xb[j + 16] * (int32_t) (blk->qs[r * 16 + j] >> 4);
                        }
                        acc_f += fp16_to_fp32(blk->d[r]) *
                                 ((float) dt - 8.0f * (float) bsums[(t0 + t) * nb + b]);
                    }
                    yt[r] = acc_f * scales[t0 + t];
                }
            }
#endif
        }
    }
    safe_free((void **) &x_q8);
    safe_free((void **) &scales);
    safe_free((void **) &bsums);
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
