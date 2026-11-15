/*
 * src/backends/cpu_neon/kernels/q5_0.c — Q5_0 W5A8 NEON kernels (#528).
 *
 * Pure compute. Block layout from src/quant/quant_blocks.h: fp16 d, 4 bytes
 * of fifth bits, 16 nibble bytes; value = d * (code - 16), code in [0, 32).
 * A block unpacks to 32 int8 in [-16, 15] in registers (nothing is
 * repacked or kept), and from there the kernels are the Q8_0 ones
 * (q8_0.c): activations quantized to int8 with one scale per 32 elements
 * (GEIST_ACT_Q8_0_ELEMS, llama.cpp's Q8_0 for this weight), an exact int32
 * dot per block, one fused multiply-add per block into the output.
 *
 * Every output is summed in block order by the same float step on every
 * path (kq_epilogue.h), so a row has the same bits whatever the row count
 * of the call: an MoE expert's is how many of a prefill chunk's tokens
 * chose it (#527).
 */
#include "quant_blocks.h"
#include "par.h"
#include "quant.h"

#include "kq_epilogue.h"

#include <stdint.h>

#include <arm_neon.h>

/* The 32 codes of one block minus 16: elements [0, 16) in *lo, [16, 32) in
 * *hi. Element j's fifth bit is bit j of qh read as a little-endian
 * uint32, so qh[0..1] belong to the low nibbles and qh[2..3] to the high. */
[[gnu::always_inline]] static inline void
q5_0_unpack(const struct block_q5_0_t *b, int8x16_t *lo, int8x16_t *hi) {
    static const uint8_t BITS[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t     bits     = vld1q_u8(BITS);
    const uint8x16_t     fifth    = vdupq_n_u8(0x10);
    const uint8x16_t     q        = vld1q_u8(b->qs);
    const uint8x16_t     h_lo     = vcombine_u8(vdup_n_u8(b->qh[0]), vdup_n_u8(b->qh[1]));
    const uint8x16_t     h_hi     = vcombine_u8(vdup_n_u8(b->qh[2]), vdup_n_u8(b->qh[3]));
    const uint8x16_t     c_lo =
            vorrq_u8(vandq_u8(q, vdupq_n_u8(0x0F)), vandq_u8(vtstq_u8(h_lo, bits), fifth));
    const uint8x16_t c_hi = vorrq_u8(vshrq_n_u8(q, 4), vandq_u8(vtstq_u8(h_hi, bits), fifth));
    *lo                   = vsubq_s8(vreinterpretq_s8_u8(c_lo), vdupq_n_s8(16));
    *hi                   = vsubq_s8(vreinterpretq_s8_u8(c_hi), vdupq_n_s8(16));
}

/* One Q5_0 W5A8 call, for the range bodies below. */
struct q5_0_rows_job {
    const struct block_q5_0_t *w;
    const int8_t              *x_q8;
    const float               *x_scales;
    size_t                     m, n_in, n_out, nb_per_row;
    float                     *y;
};

/* The exact dot of one block (lo, hi) with 32 int8 activations. */
[[gnu::always_inline]] static inline int32_t
q5_0_block_dot(const int8_t *xb, int8x16_t lo, int8x16_t hi) {
    int32x4_t acc = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xb), lo);
    acc           = vdotq_s32(acc, vld1q_s8(xb + 16), hi);
    return vaddvq_s32(acc);
}

/* Output rows [n0, n1) of linear_q5_0_decode_w5a8_pre. */
static void q5_0_decode_rows(void *ctx, size_t n0, size_t n1) {
    const struct q5_0_rows_job *j  = ctx;
    const size_t                nb = j->nb_per_row;
    for (size_t n = n0; n < n1; n++) {
        const struct block_q5_0_t *row = j->w + n * nb;
        if (n + 1 < j->n_out)
            __builtin_prefetch(row + nb, 0, 0);
        float acc = 0.0f;
        for (size_t b = 0; b < nb; b++) {
            int8x16_t lo, hi;
            q5_0_unpack(&row[b], &lo, &hi);
            acc = kq_epi_dsx(acc,
                             fp16_to_fp32(row[b].d),
                             j->x_scales[b],
                             q5_0_block_dot(j->x_q8 + b * Q5_0_BLOCK_ELEMS, lo, hi));
        }
        j->y[n] = acc;
    }
}

void linear_q5_0_decode_w5a8_pre(size_t       n_in,
                                 size_t       n_out,
                                 const float *x_scales,
                                 const int8_t x_q8[static n_in],
                                 const void  *w_q5,
                                 float        y[static n_out]) {
    struct q5_0_rows_job job = {.w          = (const struct block_q5_0_t *) w_q5,
                                .x_q8       = x_q8,
                                .x_scales   = x_scales,
                                .n_in       = n_in,
                                .n_out      = n_out,
                                .nb_per_row = n_in / Q5_0_BLOCK_ELEMS,
                                .y          = y};
    geist_par_for(n_out, q5_0_decode_rows, &job);
}

/* Output rows [n0, n1) for all m tokens: each block unpacked once and
 * dotted with every token. */
static void q5_0_prefill_rows(void *ctx, size_t n0, size_t n1) {
    const struct q5_0_rows_job *j  = ctx;
    const size_t                nb = j->nb_per_row, m = j->m, n_in = j->n_in;
    for (size_t n = n0; n < n1; n++) {
        const struct block_q5_0_t *row = j->w + n * nb;
        if (n + 1 < j->n_out)
            __builtin_prefetch(row + nb, 0, 0);

        alignas(16) float accs[GEIST_QUANT_M_CAP];
        for (size_t i = 0; i < m; i++)
            accs[i] = 0.0f;

        for (size_t b = 0; b < nb; b++) {
            const float d = fp16_to_fp32(row[b].d);
            int8x16_t   lo, hi;
            q5_0_unpack(&row[b], &lo, &hi);
            for (size_t i = 0; i < m; i++) {
                const int8_t *xb = j->x_q8 + i * n_in + b * Q5_0_BLOCK_ELEMS;
                accs[i] =
                        kq_epi_dsx(accs[i], d, j->x_scales[i * nb + b], q5_0_block_dot(xb, lo, hi));
            }
        }
        for (size_t i = 0; i < m; i++)
            j->y[i * j->n_out + n] = accs[i];
    }
}

/* 4 output rows x 8 tokens, the Q8_0 SDOT register tile (q8_0_sdot_tile4x8)
 * behind the unpack: the 4 rows of a block are transposed into 8 k-groups
 * of [row0..3][4 codes], so one vdotq_laneq_s32 adds a token's 4-element
 * k-group to all four rows; 8 tokens are 8 independent chains. Each block's
 * int32 sum is exact and is scaled by d_w * d_x in one vector FMA, which
 * is kq_epi_dsx per lane: a token gets the bits q5_0_prefill_rows gives it. */
static void q5_0_sdot_tile4x8(const struct q5_0_rows_job *j, size_t n0, size_t mt) {
    const size_t               nb = j->nb_per_row, n_in = j->n_in;
    const struct block_q5_0_t *r0 = j->w + n0 * nb;
    float32x4_t                yacc[8];
    for (int t = 0; t < 8; t++) {
        yacc[t] = vdupq_n_f32(0.0f);
    }
    for (size_t b = 0; b < nb; b++) {
        const struct block_q5_0_t *bl[4] = {r0 + b, r0 + nb + b, r0 + 2 * nb + b, r0 + 3 * nb + b};
        int8x16_t                  half[2][4];
        for (int r = 0; r < 4; r++) {
            q5_0_unpack(bl[r], &half[0][r], &half[1][r]);
        }
        int8x16_t w[8];
        for (int c = 0; c < 2; c++) {
            const int32x4_t q0 = vreinterpretq_s32_s8(half[c][0]);
            const int32x4_t q1 = vreinterpretq_s32_s8(half[c][1]);
            const int32x4_t q2 = vreinterpretq_s32_s8(half[c][2]);
            const int32x4_t q3 = vreinterpretq_s32_s8(half[c][3]);
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
        const int8_t     *xb = j->x_q8 + mt * n_in + b * Q5_0_BLOCK_ELEMS;
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
 * q5_0_prefill_rows. */
static void q5_0_prefill_tiles(void *ctx, size_t nt0, size_t nt1) {
    const struct q5_0_rows_job *j  = ctx;
    const size_t                m8 = j->m / 8 * 8;
    for (size_t nt = nt0; nt < nt1; nt++) {
        const size_t n0 = nt * 4;
        if (n0 + 4 > j->n_out || m8 == 0) {
            q5_0_prefill_rows(ctx, n0, n0 + 4 < j->n_out ? n0 + 4 : j->n_out);
            continue;
        }
        for (size_t mt = 0; mt < m8; mt += 8) {
            q5_0_sdot_tile4x8(j, n0, mt);
        }
        if (m8 < j->m) {
            struct q5_0_rows_job tail = *j;
            tail.m                    = j->m - m8;
            tail.x_q8                 = j->x_q8 + m8 * j->n_in;
            tail.x_scales             = j->x_scales + m8 * j->nb_per_row;
            tail.y                    = j->y + m8 * j->n_out;
            q5_0_prefill_rows(&tail, n0, n0 + 4);
        }
    }
}

void linear_q5_0_w5a8_prefill_pre(size_t        m,
                                  size_t        n_in,
                                  size_t        n_out,
                                  const int8_t *x_q8,
                                  const float  *x_scales,
                                  const void   *w_q5,
                                  float        *y) {
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    struct q5_0_rows_job job = {.w          = (const struct block_q5_0_t *) w_q5,
                                .x_q8       = x_q8,
                                .x_scales   = x_scales,
                                .m          = m,
                                .n_in       = n_in,
                                .n_out      = n_out,
                                .nb_per_row = n_in / Q5_0_BLOCK_ELEMS,
                                .y          = y};
    geist_par_for((n_out + 3) / 4, q5_0_prefill_tiles, &job);
}
