/*
 * src/backends/cpu_neon/kernels/iq4_xs.c — IQ4_XS / IQ4_NL W4A8 NEON
 * decode GEMVs.
 *
 * Both formats store 4-bit indices into the fixed 16-value kvalues_iq4nl
 * table — one vqtbl1q_s8 per 16 nibbles, no reconstruction scratch.
 * Block layouts from src/quant/quant_blocks.h. IQ4_XS also has an M>1
 * tile kernel (used on non-Accelerate hosts); IQ4_NL prefill uses the
 * dequant+SGEMM path.
 */
#include "heap.h"
#include "par.h"
#include "quant.h"
#include "quant_blocks.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>

static const int8_t kvalues_iq4nl_k[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
#endif

/* One call's operands, for the range bodies below. */
struct iq4_rows_job {
    const void   *w;
    const int8_t *x_q8;
    const float  *x_scales;
    size_t        m, n_in, n_out, n_blocks_per_row;
    float        *y;
};

#if defined(__ARM_NEON)
/* Output rows [n0, n1) of linear_iq4xs_decode_w4a8_pre. */
static void iq4xs_decode_rows(void *ctx, size_t n0, size_t n1) {
    const struct iq4_rows_job   *job              = ctx;
    const struct block_iq4_xs_t *w                = job->w;
    const int8_t                *x_q8             = job->x_q8;
    const float                 *x_scales         = job->x_scales;
    const size_t                 n_out            = job->n_out;
    const size_t                 n_blocks_per_row = job->n_blocks_per_row;
    float                       *y                = job->y;
    const int8x16_t              kv               = vld1q_s8(kvalues_iq4nl_k);
    const uint8x16_t             low4             = vdupq_n_u8(0x0f);
    for (size_t n = n0; n < n1; n++) {
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
            acc = fmaf(d * x_scales[b], (float) vaddvq_s32(int_acc), acc);
        }
        y[n] = acc;
    }
}
#endif

void linear_iq4xs_decode_w4a8_pre(size_t       n_in,
                                  size_t       n_out,
                                  const float *x_scales,
                                  const int8_t x_q8[static n_in],
                                  const void  *w_iq4xs,
                                  float        y[static n_out]) {
#if defined(__ARM_NEON)
    const struct block_iq4_xs_t *w                = (const struct block_iq4_xs_t *) w_iq4xs;
    const size_t                 n_blocks_per_row = n_in / IQ4_XS_BLOCK_ELEMS;

    struct iq4_rows_job job = {.w                = w,
                               .x_q8             = x_q8,
                               .x_scales         = x_scales,
                               .n_out            = n_out,
                               .n_blocks_per_row = n_blocks_per_row,
                               .y                = y};
    geist_par_for(n_out, iq4xs_decode_rows, &job);
#else
    (void) x_q8;
    (void) x_scales;
    (void) w_iq4xs;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_iq4xs_decode_w4a8_pre: NEON required\n");
#endif
}

void linear_iq4xs_decode_w4a8(size_t      n_in,
                              size_t      n_out,
                              const float x[static n_in],
                              const void *w_iq4xs,
                              float       y[static n_out]) {
    int8_t *x_q8     = heap_alloc_array_aligned(int8_t, n_in);
    float  *x_scales = heap_alloc_array_aligned(float, geist_act_groups(n_in, GEIST_ACT_Q8K_ELEMS));
    if (x_q8 == nullptr || x_scales == nullptr) {
        safe_free((void **) &x_q8);
        safe_free((void **) &x_scales);
        return;
    }
    quantize_x_q8_groups(n_in, GEIST_ACT_Q8K_ELEMS, x, x_q8, x_scales, nullptr);
    linear_iq4xs_decode_w4a8_pre(n_in, n_out, x_scales, x_q8, w_iq4xs, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &x_scales);
}

#if defined(__ARM_NEON)
/* Output rows [n0, n1) of linear_iq4xs_w4a8_prefill_pre. */
static void iq4xs_prefill_rows(void *ctx, size_t n0, size_t n1) {
    const struct iq4_rows_job   *job              = ctx;
    const struct block_iq4_xs_t *w                = job->w;
    const int8_t                *x_q8             = job->x_q8;
    const float                 *x_scales         = job->x_scales;
    const size_t                 m                = job->m;
    const size_t                 n_in             = job->n_in;
    const size_t                 n_out            = job->n_out;
    const size_t                 n_blocks_per_row = job->n_blocks_per_row;
    float                       *y                = job->y;
    const int8x16_t              kv               = vld1q_s8(kvalues_iq4nl_k);
    const uint8x16_t             low4             = vdupq_n_u8(0x0f);
    for (size_t n = n0; n < n1; n++) {
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
                    accf[t] = fmaf(d * x_scales[(t0 + t) * n_blocks_per_row + b],
                                   (float) vaddvq_s32(int_acc),
                                   accf[t]);
                }
            }
            for (size_t t = 0; t < tcnt; t++)
                y[(t0 + t) * n_out + n] = accf[t];
        }
    }
}
#endif

/* IQ4_XS int8 mN prefill (see #321). Row-major sweep with a 4-token
 * register tile: each 136-byte block is LUT-decoded once and dotted
 * against 4 activation rows; the row's blocks stay L1-resident across
 * token groups. Per (row, token) the op order matches the m1 kernel, so
 * the output is bit-identical to m decode calls (test_iq4_dequant_unit). */
void linear_iq4xs_w4a8_prefill_pre(size_t        m,
                                   size_t        n_in,
                                   size_t        n_out,
                                   const int8_t *x_q8,
                                   const float  *x_scales,
                                   const void   *w_iq4xs,
                                   float        *y) {
#if defined(__ARM_NEON)
    if (m == 0)
        return;
    const struct block_iq4_xs_t *w                = (const struct block_iq4_xs_t *) w_iq4xs;
    const size_t                 n_blocks_per_row = n_in / IQ4_XS_BLOCK_ELEMS;

    struct iq4_rows_job job = {.w                = w,
                               .x_q8             = x_q8,
                               .x_scales         = x_scales,
                               .m                = m,
                               .n_in             = n_in,
                               .n_out            = n_out,
                               .n_blocks_per_row = n_blocks_per_row,
                               .y                = y};
    geist_par_for(n_out, iq4xs_prefill_rows, &job);
#else
    (void) x_q8;
    (void) x_scales;
    (void) m;
    (void) w_iq4xs;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_iq4xs_w4a8_prefill_pre: NEON required\n");
#endif
}

void linear_iq4xs_w4a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w_iq4xs, float *y) {
    const size_t n_groups = geist_act_groups(n_in, GEIST_ACT_Q8K_ELEMS);
    int8_t      *x_q8     = heap_alloc_array_aligned(int8_t, m *n_in);
    float       *x_scales = heap_alloc_array_aligned(float, m *n_groups);
    if (x_q8 == nullptr || x_scales == nullptr) {
        safe_free((void **) &x_q8);
        safe_free((void **) &x_scales);
        return;
    }
    for (size_t i = 0; i < m; i++) {
        quantize_x_q8_groups(n_in,
                             GEIST_ACT_Q8K_ELEMS,
                             x + i * n_in,
                             x_q8 + i * n_in,
                             x_scales + i * n_groups,
                             nullptr);
    }
    linear_iq4xs_w4a8_prefill_pre(m, n_in, n_out, x_q8, x_scales, w_iq4xs, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &x_scales);
}

#if defined(__ARM_NEON)
/* Output rows [n0, n1) of linear_iq4nl_decode_w4a8_pre. */
static void iq4nl_decode_rows(void *ctx, size_t n0, size_t n1) {
    const struct iq4_rows_job   *job              = ctx;
    const struct block_iq4_nl_t *w                = job->w;
    const int8_t                *x_q8             = job->x_q8;
    const float                 *x_scales         = job->x_scales;
    const size_t                 n_blocks_per_row = job->n_blocks_per_row;
    float                       *y                = job->y;
    const int8x16_t              kv               = vld1q_s8(kvalues_iq4nl_k);
    const uint8x16_t             low4             = vdupq_n_u8(0x0f);
    for (size_t n = n0; n < n1; n++) {
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
            acc += fp16_to_fp32(blk->d) * x_scales[b] * (float) vaddvq_s32(dot);
        }
        y[n] = acc;
    }
}
#endif

void linear_iq4nl_decode_w4a8_pre(size_t       n_in,
                                  size_t       n_out,
                                  const float *x_scales,
                                  const int8_t x_q8[static n_in],
                                  const void  *w_iq4nl,
                                  float        y[static n_out]) {
#if defined(__ARM_NEON)
    const struct block_iq4_nl_t *w                = (const struct block_iq4_nl_t *) w_iq4nl;
    const size_t                 n_blocks_per_row = n_in / IQ4_NL_BLOCK_ELEMS;

    struct iq4_rows_job job = {.w                = w,
                               .x_q8             = x_q8,
                               .x_scales         = x_scales,
                               .n_blocks_per_row = n_blocks_per_row,
                               .y                = y};
    geist_par_for(n_out, iq4nl_decode_rows, &job);
#else
    (void) x_q8;
    (void) x_scales;
    (void) w_iq4nl;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_iq4nl_decode_w4a8_pre: NEON required\n");
#endif
}

void linear_iq4nl_decode_w4a8(size_t      n_in,
                              size_t      n_out,
                              const float x[static n_in],
                              const void *w_iq4nl,
                              float       y[static n_out]) {
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, n_in);
    float *x_scales = heap_alloc_array_aligned(float, geist_act_groups(n_in, GEIST_ACT_Q8_0_ELEMS));
    if (x_q8 == nullptr || x_scales == nullptr) {
        safe_free((void **) &x_q8);
        safe_free((void **) &x_scales);
        return;
    }
    quantize_x_q8_groups(n_in, GEIST_ACT_Q8_0_ELEMS, x, x_q8, x_scales, nullptr);
    linear_iq4nl_decode_w4a8_pre(n_in, n_out, x_scales, x_q8, w_iq4nl, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &x_scales);
}
