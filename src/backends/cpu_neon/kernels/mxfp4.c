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
#include <math.h>
#include <stdint.h>

#include <arm_neon.h>

struct mxfp4_rows_job {
    const struct block_mxfp4_t *w;
    const int8_t               *x_q8;
    const float                *x_scales;
    size_t                      m, n_in, n_out, nb;
    float                      *y;
};

/* The 32 weights of block blk as int8: lo = elements 0..15, hi = 16..31. */
[[gnu::always_inline]] static inline void
mxfp4_block_s8(const struct block_mxfp4_t *blk, int8x16_t kv, int8x16_t *lo, int8x16_t *hi) {
    const uint8x16_t q = vld1q_u8(blk->qs);
    *lo                = vqtbl1q_s8(kv, vandq_u8(q, vdupq_n_u8(0x0f)));
    *hi                = vqtbl1q_s8(kv, vshrq_n_u8(q, 4));
}

/* A block's integer dot with 32 int8 activations, as 4 partial lanes. */
[[gnu::always_inline]] static inline int32x4_t
mxfp4_idot(int8x16_t lo, int8x16_t hi, const int8_t *x) {
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(x), lo), vld1q_s8(x + 16), hi);
}

/* One block group's float factors: 2^(e - 128) of each of the 4 blocks
 * times its activation scale. */
[[gnu::always_inline]] static inline float32x4_t mxfp4_scales4(const struct block_mxfp4_t *blk,
                                                               const float                *xs) {
    const float sc[4] = {mxfp4_e8m0_half(blk[0].e) * xs[0],
                         mxfp4_e8m0_half(blk[1].e) * xs[1],
                         mxfp4_e8m0_half(blk[2].e) * xs[2],
                         mxfp4_e8m0_half(blk[3].e) * xs[3]};
    return vld1q_f32(sc);
}

/* The four blocks' integer sums [s0, s1, s2, s3] from their partials. */
[[gnu::always_inline]] static inline int32x4_t
mxfp4_sum4(int32x4_t d0, int32x4_t d1, int32x4_t d2, int32x4_t d3) {
    return vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
}

/* One token's dot with one row. Blocks go in groups of 4: their integer
 * sums gathered into one vector, converted once and scaled by a vector of
 * their factors; the nb % 4 tail blocks into a scalar. The prefill below
 * repeats exactly these operations per token. */
[[gnu::always_inline]] static inline float
mxfp4_row_dot(size_t nb, const struct block_mxfp4_t *row, const int8_t *xq, const float *xs) {
    const int8x16_t kv  = vld1q_s8(kvalues_mxfp4);
    float32x4_t     acc = vdupq_n_f32(0.0f);
    size_t          b   = 0;
    for (; b + 4 <= nb; b += 4) {
        int8x16_t l0, h0, l1, h1, l2, h2, l3, h3;
        mxfp4_block_s8(&row[b], kv, &l0, &h0);
        mxfp4_block_s8(&row[b + 1], kv, &l1, &h1);
        mxfp4_block_s8(&row[b + 2], kv, &l2, &h2);
        mxfp4_block_s8(&row[b + 3], kv, &l3, &h3);
        const int8_t   *x = xq + 32 * b;
        const int32x4_t s = mxfp4_sum4(mxfp4_idot(l0, h0, x),
                                       mxfp4_idot(l1, h1, x + 32),
                                       mxfp4_idot(l2, h2, x + 64),
                                       mxfp4_idot(l3, h3, x + 96));
        acc               = vfmaq_f32(acc, vcvtq_f32_s32(s), mxfp4_scales4(&row[b], xs + b));
    }
    float tail = 0.0f;
    for (; b < nb; b++) {
        int8x16_t lo, hi;
        mxfp4_block_s8(&row[b], kv, &lo, &hi);
        tail = fmaf(mxfp4_e8m0_half(row[b].e) * xs[b],
                    (float) vaddvq_s32(mxfp4_idot(lo, hi, xq + 32 * b)),
                    tail);
    }
    return vaddvq_f32(acc) + tail;
}

/* Two rows against one token: the activation blocks loaded once for both,
 * per row mxfp4_row_dot's operations in its order (the same bits). */
[[gnu::always_inline]] static inline void mxfp4_row2_dot(size_t                      nb,
                                                         const struct block_mxfp4_t *r0,
                                                         const struct block_mxfp4_t *r1,
                                                         const int8_t               *xq,
                                                         const float                *xs,
                                                         float                      *y0,
                                                         float                      *y1) {
    const int8x16_t kv = vld1q_s8(kvalues_mxfp4);
    float32x4_t     a0 = vdupq_n_f32(0.0f), a1 = a0;
    size_t          b = 0;
    for (; b + 4 <= nb; b += 4) {
        const int8_t   *x   = xq + 32 * b;
        const int8x16_t xa0 = vld1q_s8(x), xb0 = vld1q_s8(x + 16), xa1 = vld1q_s8(x + 32),
                        xb1 = vld1q_s8(x + 48), xa2 = vld1q_s8(x + 64), xb2 = vld1q_s8(x + 80),
                        xa3 = vld1q_s8(x + 96), xb3 = vld1q_s8(x + 112);
        int32x4_t       d[2][4];
        for (size_t r = 0; r < 2; r++) {
            const struct block_mxfp4_t *row = r == 0 ? r0 : r1;
            int8x16_t                   lo, hi;
            mxfp4_block_s8(&row[b], kv, &lo, &hi);
            d[r][0] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), xa0, lo), xb0, hi);
            mxfp4_block_s8(&row[b + 1], kv, &lo, &hi);
            d[r][1] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), xa1, lo), xb1, hi);
            mxfp4_block_s8(&row[b + 2], kv, &lo, &hi);
            d[r][2] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), xa2, lo), xb2, hi);
            mxfp4_block_s8(&row[b + 3], kv, &lo, &hi);
            d[r][3] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), xa3, lo), xb3, hi);
        }
        a0 = vfmaq_f32(a0,
                       vcvtq_f32_s32(mxfp4_sum4(d[0][0], d[0][1], d[0][2], d[0][3])),
                       mxfp4_scales4(&r0[b], xs + b));
        a1 = vfmaq_f32(a1,
                       vcvtq_f32_s32(mxfp4_sum4(d[1][0], d[1][1], d[1][2], d[1][3])),
                       mxfp4_scales4(&r1[b], xs + b));
    }
    float t0 = 0.0f, t1 = 0.0f;
    for (; b < nb; b++) {
        int8x16_t lo, hi;
        mxfp4_block_s8(&r0[b], kv, &lo, &hi);
        t0 = fmaf(mxfp4_e8m0_half(r0[b].e) * xs[b],
                  (float) vaddvq_s32(mxfp4_idot(lo, hi, xq + 32 * b)),
                  t0);
        mxfp4_block_s8(&r1[b], kv, &lo, &hi);
        t1 = fmaf(mxfp4_e8m0_half(r1[b].e) * xs[b],
                  (float) vaddvq_s32(mxfp4_idot(lo, hi, xq + 32 * b)),
                  t1);
    }
    *y0 = vaddvq_f32(a0) + t0;
    *y1 = vaddvq_f32(a1) + t1;
}

static void mxfp4_decode_rows(void *ctx, size_t n0, size_t n1) {
    const struct mxfp4_rows_job *job = ctx;
    const size_t                 nb  = job->nb;
    size_t                       n   = n0;
    for (; n + 2 <= n1; n += 2) {
        const struct block_mxfp4_t *row = job->w + n * nb;
        __builtin_prefetch(row + 2 * nb, 0, 0);
        mxfp4_row2_dot(nb, row, row + nb, job->x_q8, job->x_scales, &job->y[n], &job->y[n + 1]);
    }
    if (n < n1)
        job->y[n] = mxfp4_row_dot(nb, job->w + n * nb, job->x_q8, job->x_scales);
}

/* Output rows [n0, n1) for all m tokens: each group of 4 blocks decoded
 * once for 4 tokens, each token through mxfp4_row_dot's operations. */
static void mxfp4_prefill_rows(void *ctx, size_t n0, size_t n1) {
    const struct mxfp4_rows_job *job = ctx;
    const size_t                 nb = job->nb, n_in = job->n_in, n_out = job->n_out, m = job->m;
    const int8x16_t              kv = vld1q_s8(kvalues_mxfp4);
    for (size_t n = n0; n < n1; n++) {
        const struct block_mxfp4_t *row = job->w + n * nb;
        size_t                      t0  = 0;
        for (; t0 + 4 <= m; t0 += 4) {
            float32x4_t a[4] = {
                    vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            size_t b = 0;
            for (; b + 4 <= nb; b += 4) {
                int8x16_t l0, h0, l1, h1, l2, h2, l3, h3;
                mxfp4_block_s8(&row[b], kv, &l0, &h0);
                mxfp4_block_s8(&row[b + 1], kv, &l1, &h1);
                mxfp4_block_s8(&row[b + 2], kv, &l2, &h2);
                mxfp4_block_s8(&row[b + 3], kv, &l3, &h3);
                for (size_t t = 0; t < 4; t++) {
                    const int8_t   *x = job->x_q8 + (t0 + t) * n_in + 32 * b;
                    const int32x4_t s = mxfp4_sum4(mxfp4_idot(l0, h0, x),
                                                   mxfp4_idot(l1, h1, x + 32),
                                                   mxfp4_idot(l2, h2, x + 64),
                                                   mxfp4_idot(l3, h3, x + 96));
                    a[t] = vfmaq_f32(a[t],
                                     vcvtq_f32_s32(s),
                                     mxfp4_scales4(&row[b], job->x_scales + (t0 + t) * nb + b));
                }
            }
            float tail[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (; b < nb; b++) {
                int8x16_t lo, hi;
                mxfp4_block_s8(&row[b], kv, &lo, &hi);
                const float d = mxfp4_e8m0_half(row[b].e);
                for (size_t t = 0; t < 4; t++) {
                    const int8_t *x = job->x_q8 + (t0 + t) * n_in + 32 * b;
                    tail[t]         = fmaf(d * job->x_scales[(t0 + t) * nb + b],
                                           (float) vaddvq_s32(mxfp4_idot(lo, hi, x)),
                                           tail[t]);
                }
            }
            for (size_t t = 0; t < 4; t++)
                job->y[(t0 + t) * n_out + n] = vaddvq_f32(a[t]) + tail[t];
        }
        for (; t0 < m; t0++)
            job->y[t0 * n_out + n] =
                    mxfp4_row_dot(nb, row, job->x_q8 + t0 * n_in, job->x_scales + t0 * nb);
    }
}

void linear_mxfp4_decode_w4a8_pre(size_t       n_in,
                                  size_t       n_out,
                                  const float *x_scales,
                                  const int8_t x_q8[static n_in],
                                  const void  *w_mxfp4,
                                  float        y[static n_out]) {
    struct mxfp4_rows_job job = {.w        = (const struct block_mxfp4_t *) w_mxfp4,
                                 .x_q8     = x_q8,
                                 .x_scales = x_scales,
                                 .m        = 1,
                                 .n_in     = n_in,
                                 .n_out    = n_out,
                                 .nb       = n_in / MXFP4_BLOCK_ELEMS,
                                 .y        = y};
    geist_par_for(n_out, mxfp4_decode_rows, &job);
}

void linear_mxfp4_w4a8_prefill_pre(size_t        m,
                                   size_t        n_in,
                                   size_t        n_out,
                                   const int8_t *x_q8,
                                   const float  *x_scales,
                                   const void   *w_mxfp4,
                                   float        *y) {
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
}
