/*
 * src/backends/cpu_x86/kernel_q4kx8_gemm_scalar.c — scalar reference.
 *
 * Walks both interleaved layouts (Q4_Kx8 weights, Q8_Kx4 acts) and
 * produces fp32 outputs. Used as correctness oracle for the AVX-512
 * variant. Per super-block:
 *
 *   y[m, n] += d_a[m] * d_w[n] * sum_b (sc[b][n] * sum_q[b][m, n])
 *             - d_a[m] * dmin_w[n] * sum_g (mn[g][n] * bsum[g][m])
 *
 * where d_a is the per-row acts scale, d_w / dmin_w are the per-row
 * weight scale / min, sc and mn are the 6-bit sub-scales/sub-mins
 * (decoded from the 96-byte scales[] field), q is the unsigned 4-bit
 * weight nibble, and bsum is the precomputed sum of 16 acts per group.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_q4kx8_gemm.h"

#include "quant.h"
#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Decode the 96-byte scales[] field of one block_q4_Kx8 back into 8 rows
 * of 16 sub-blocks worth of (scale, min) pairs (each 6-bit unsigned). */
static void
decode_q4kx8_scales(const struct block_q4_Kx8 *b, uint8_t sc[8][16], uint8_t mn[8][16]) {
    /* The scales[] format is the inverse of make_block_q4_kx8's pack.
     * Per llama.cpp's GEMM, the layout per 12-byte block is:
     *   [0..3]: low-6-bit packed s0..s3 (with high 2 bits of s4..s7)
     *   [4..7]: low-6-bit packed m0..m3 (with high 2 bits of m4..m7)
     *   [8..11]: low 4 bits = low nibble of s4..s7, high 4 bits = m4..m7
     * For i in 0..3, scales[i*12 + .] holds sub-block i across 8 rows.
     * For i in 0..3, scales[48 + i*12 + .] holds sub-block i+4 across 8 rows.
     */
    for (int half = 0; half < 2; half++) {
        const int      sb_base = half * 4;
        const uint8_t *base    = b->scales + half * 48;
        for (int i = 0; i < 4; i++) {
            const uint8_t *p = base + i * 12;
            uint8_t        s_lo4[4], m_lo4[4], s_hi2[4], m_hi2[4];
            /* p[0..3]: low6 of s[0..3] + high2 of s[4..7] (×4). */
            for (int j = 0; j < 4; j++) {
                /* low 6 bits */
                const uint8_t low6_s   = p[j] & 63;
                const uint8_t high2_s4 = (uint8_t) ((p[j] >> 6) & 3);
                /* p[4..7] holds low6 of m[0..3] + high2 of m[4..7] */
                const uint8_t low6_m   = p[j + 4] & 63;
                const uint8_t high2_m4 = (uint8_t) ((p[j + 4] >> 6) & 3);
                /* p[8..11]: low4 = low4 of s[4..7], high4 = m[4..7] low4 */
                const uint8_t s_lo4_j = p[j + 8] & 15;
                const uint8_t m_lo4_j = (uint8_t) ((p[j + 8] >> 4) & 15);

                sc[j][sb_base + i] = low6_s;
                mn[j][sb_base + i] = low6_m;
                s_lo4[j]           = s_lo4_j;
                m_lo4[j]           = m_lo4_j;
                s_hi2[j]           = high2_s4;
                m_hi2[j]           = high2_m4;
            }
            for (int j = 0; j < 4; j++) {
                /* Reconstruct s/m for source rows 4..7. */
                sc[4 + j][sb_base + i] = (uint8_t) (s_lo4[j] | (s_hi2[j] << 4));
                mn[4 + j][sb_base + i] = (uint8_t) (m_lo4[j] | (m_hi2[j] << 4));
            }
        }
    }
}

/* Extract the unsigned 4-bit weight q[n_row][k] for source row n_row ∈ [0,8)
 * and K-position k ∈ [0, 256) from one Q4_Kx8 super-block.
 *
 * qs layout: 128 stripes of 8 bytes each. Stripe i comes from
 * source row (i % 8), source byte offset (i / 8) * 8 + [0..7].
 * Each source byte holds 2 nibbles (low + high). Q4_K's nibble layout
 * per source row: byte at offset b holds (q[b], q[b + 32]) for the first
 * 32-byte chunk, etc. — same as dequant_q4_K_row's q[l] & 0xF / q[l] >> 4.
 *
 * For element at K position k of source row n_row:
 *   The source row's qs[] is 128 bytes. The K → (byte_index, nibble_lo_hi)
 *   mapping follows Q4_K's dequant pattern:
 *     For l in 0..31: y[l + 0] = q[l] & 0xF
 *                     y[l + 32] = q[l] >> 4
 *     For l in 0..31: y[l + 64] = q[l+32] & 0xF
 *                     y[l + 96] = q[l+32] >> 4
 *     For l in 0..31: y[l + 128] = q[l+64] & 0xF
 *                     ...
 *   I.e., per 64-element half: bytes [l_offset .. l_offset+31] hold pairs.
 *
 * For the interleaved Q4_Kx8 layout, the byte at source offset b of source
 * row n_row lives at qs[(b / 8) * 64 + n_row * 8 + (b % 8)].
 */
static uint8_t extract_q4kx8_nibble(const struct block_q4_Kx8 *b, int n_row, int k) {
    /* Determine source byte offset and lo/hi within Q4_K layout. */
    const int half        = k / 64;        /* 0..3 */
    const int half_pos    = k % 64;        /* 0..63 */
    const int sub_in_half = half_pos / 32; /* 0 or 1 */
    const int pos_in_sub  = half_pos % 32; /* 0..31 */
    /* Source byte index within source row's 128-byte qs[]. */
    const int src_byte_off = half * 32 + pos_in_sub;
    /* Interleaved layout: stripe (src_byte_off / 8) at position
     * (n_row * 8 + (src_byte_off % 8)). */
    const int     stripe_idx = src_byte_off / 8;
    const int     in_stripe  = n_row * 8 + (src_byte_off % 8);
    const uint8_t byte       = b->qs[stripe_idx * 64 + in_stripe];
    return (sub_in_half == 0) ? (uint8_t) (byte & 0x0F) : (uint8_t) ((byte >> 4) & 0x0F);
}

/* Extract the int8 activation a[m_row][k] for m-row m_row ∈ [0,4) and
 * K-position k ∈ [0, 256) from one block_q8_Kx4 super-block.
 *
 * qs layout per super-block: 4 sub-blocks × 256 bytes each. Per sub-block:
 *   stripe s (0..7) at byte offset s*32; within stripe, bytes [r*8 .. r*8+7]
 *   = row r's elements [s*8 .. s*8+7] within the sub-block.
 */
static int8_t extract_q8kx4_act(const struct block_q8_Kx4 *b, int m_row, int k) {
    const int sb        = k / 64;
    const int pos_in_sb = k % 64;
    const int stripe    = pos_in_sb / 8;
    const int in_stripe = pos_in_sb % 8;
    return b->qs[sb * 256 + stripe * 32 + m_row * 8 + in_stripe];
}

void q4kx8_gemm_scalar(size_t                     M,
                       size_t                     N,
                       size_t                     K,
                       const struct block_q8_Kx4 *X,
                       const struct block_q4_Kx8 *W,
                       float                      Y[static M * N]) {
    const size_t n_super_k = K / 256;
    const size_t M_tiles   = M / 4;
    const size_t N_tiles   = N / 8;

    for (size_t mt = 0; mt < M_tiles; mt++) {
        for (size_t nt = 0; nt < N_tiles; nt++) {
            /* Per (m-row, n-cell) accumulators. */
            float acc[4][8];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 8; j++) {
                    acc[i][j] = 0.0f;
                }
            }

            for (size_t s = 0; s < n_super_k; s++) {
                const struct block_q8_Kx4 *Xb = &X[mt * n_super_k + s];
                const struct block_q4_Kx8 *Wb = &W[nt * n_super_k + s];

                uint8_t sc[8][16], mn[8][16];
                decode_q4kx8_scales(Wb, sc, mn);

                /* d/dmin per source row of weights, as fp32. */
                float dw[8], dmin_w[8];
                for (int r = 0; r < 8; r++) {
                    dw[r]     = fp16_to_fp32(Wb->d[r]);
                    dmin_w[r] = fp16_to_fp32(Wb->dmin[r]);
                }

                /* Q4_K has 8 sub-blocks of 32 elements; Q8_K bsums have
                 * 16 groups of 16. Per weight sub-block wsb (32 elems),
                 * the matching bsum range covers two groups: 2*wsb and
                 * 2*wsb+1.
                 *
                 * Per (i, j) per super-block contribution:
                 *   d_a[i] * dw[j] * sum_wsb sc[j][wsb] * sum_q_a(wsb)
                 *   - d_a[i] * dmin_w[j] * sum_wsb mn[j][wsb] *
                 *       (bsum[i][2*wsb] + bsum[i][2*wsb+1])
                 */
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        float pos_acc = 0.0f;
                        float neg_acc = 0.0f;
                        for (int wsb = 0; wsb < 8; wsb++) {
                            int32_t sum_q_a = 0;
                            for (int kk = 0; kk < 32; kk++) {
                                const int     k_global = wsb * 32 + kk;
                                const uint8_t q        = extract_q4kx8_nibble(Wb, j, k_global);
                                const int8_t  a        = extract_q8kx4_act(Xb, i, k_global);
                                sum_q_a += (int32_t) q * (int32_t) a;
                            }
                            pos_acc += (float) sc[j][wsb] * (float) sum_q_a;
                            const int g0 = 2 * wsb;
                            const int g1 = 2 * wsb + 1;
                            neg_acc += (float) mn[j][wsb] *
                                       (float) (Xb->bsums[i * 16 + g0] + Xb->bsums[i * 16 + g1]);
                        }
                        const float d_a = Xb->d[i];
                        acc[i][j] += d_a * (dw[j] * pos_acc - dmin_w[j] * neg_acc);
                    }
                }
            }

            /* Write tile. */
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 8; j++) {
                    Y[(mt * 4 + i) * N + (nt * 8 + j)] = acc[i][j];
                }
            }
        }
    }
}
