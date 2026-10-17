/*
 * quant_blocks — packed memory layouts of every quantized block type, plus
 * the small inline helpers shared by the dequant codecs (formats/gguf) and
 * the W*A8 compute kernels (backends/cpu_neon).
 *
 * These `__attribute__((packed))` structs ARE the block-layout contract:
 * each is _Static_assert'd against its byte size. They carry no file-format
 * types, so formats/ and backends/ can both include this header.
 *
 * Helpers used by one TU only (q4k_subpair_dots, iq2s/iq3s subblock
 * decoders, etc.) live inside that .c file as `static inline`; the ones a
 * codec and a kernel both need (get_scale_min_k4, unpack_q3k_scales,
 * q3k_reconstruct_q32) live here, once.
 */
#pragma once

#include "quant.h"

#include <stddef.h>
#include <stdint.h>

/* Exact on-disk byte layouts of the GGUF quant blocks. `packed` rules out
 * compiler padding; block constants live in quant.h. */

struct block_q3_K_t {
    uint8_t  hmask[32];
    uint8_t  qs[64];
    uint8_t  scales[12];
    uint16_t d;
} __attribute__((packed));
_Static_assert(sizeof(struct block_q3_K_t) == Q3_K_BLOCK_BYTES, "struct block_q3_K_t size");

struct block_q4_K_t {
    uint16_t d;    /* fp16 */
    uint16_t dmin; /* fp16 */
    uint8_t  scales[12];
    uint8_t  qs[128];
} __attribute__((packed));
_Static_assert(sizeof(struct block_q4_K_t) == 144, "struct block_q4_K_t size");

struct block_q5_K_t {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
} __attribute__((packed));
_Static_assert(sizeof(struct block_q5_K_t) == Q5_K_BLOCK_BYTES, "struct block_q5_K_t size");

struct block_q6_K_t {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
} __attribute__((packed));
_Static_assert(sizeof(struct block_q6_K_t) == 210, "struct block_q6_K_t size");

struct block_q4_0_t {
    uint16_t d;      /* scale (fp16) */
    uint8_t  qs[16]; /* element j in the low nibble of qs[j], j + 16 in the high */
} __attribute__((packed));
_Static_assert(sizeof(struct block_q4_0_t) == Q4_0_BLOCK_BYTES, "struct block_q4_0_t size");

struct block_q4_1_t {
    uint16_t d; /* scale (fp16) */
    uint16_t m; /* min   (fp16) */
    uint8_t  qs[16];
} __attribute__((packed));
_Static_assert(sizeof(struct block_q4_1_t) == Q4_1_BLOCK_BYTES, "struct block_q4_1_t size");

/* Element j's 5-bit code is the low nibble of qs[j] plus bit j of qh (read
 * as a little-endian uint32) as bit 4; element j + 16 is the high nibble of
 * qs[j] plus bit j + 16. Value = d * (code - 16). */
struct block_q5_0_t {
    uint16_t d;     /* scale (fp16) */
    uint8_t  qh[4]; /* fifth bit of each element */
    uint8_t  qs[16];
} __attribute__((packed));
_Static_assert(sizeof(struct block_q5_0_t) == Q5_0_BLOCK_BYTES, "struct block_q5_0_t size");

/* Element (h * 128 + l * 32 + k) is bits 2l..2l+1 of qs[h * 32 + k], h < 2,
 * l < 4, k < 32; the trit is that value minus 1. */
struct block_tq2_0_t {
    uint8_t qs[64];
    uint8_t d[2]; /* fp16, little-endian */
};
static_assert(sizeof(struct block_tq2_0_t) == TQ2_0_BLOCK_BYTES, "struct block_tq2_0_t size");

struct block_q8_0_t {
    uint16_t d;
    int8_t   qs[32];
} __attribute__((packed));
_Static_assert(sizeof(struct block_q8_0_t) == 34, "struct block_q8_0_t size");

struct block_iq2_s_t {
    uint16_t d;
    uint8_t  qs[64];    /* low 8 bits of 32 grid indices + 32 sign bytes */
    uint8_t  qh[8];     /* high 2 bits of 32 grid indices, packed */
    uint8_t  scales[8]; /* 4-bit pair per 32-elem sub-block */
} __attribute__((packed));
_Static_assert(sizeof(struct block_iq2_s_t) == 82, "struct block_iq2_s_t size");

struct block_iq3_s_t {
    uint16_t d;
    uint8_t  qs[64];
    uint8_t  qh[8];
    uint8_t  signs[32];
    uint8_t  scales[4];
} __attribute__((packed));
_Static_assert(sizeof(struct block_iq3_s_t) == 110, "struct block_iq3_s_t size");

struct block_iq4_nl_t {
    uint16_t d;
    uint8_t  qs[16]; /* 32 4-bit indices into kvalues_iq4nl */
} __attribute__((packed));
_Static_assert(sizeof(struct block_iq4_nl_t) == 18, "struct block_iq4_nl_t size");

struct block_iq4_xs_t {
    uint16_t d;
    uint16_t scales_h;    /* 2 high bits per 32-elem sub-block */
    uint8_t  scales_l[4]; /* 4 low bits per sub-block, packed pairs */
    uint8_t  qs[128];     /* 256 4-bit indices into kvalues_iq4nl */
} __attribute__((packed));
_Static_assert(sizeof(struct block_iq4_xs_t) == 136, "struct block_iq4_xs_t size");

/* Shared Q4_K / Q5_K scale-min unpacker.
 *
 * Both formats encode 8 sub-blocks of 32 elements with a 6-bit scale and
 * 6-bit min per sub-block, packed across 12 bytes in a Q4_K/Q5_K super-
 * block. `j` selects the sub-block index in [0, 8). */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d_out, uint8_t *m_out) {
    if (j < 4) {
        *d_out = q[j] & 63;
        *m_out = q[j + 4] & 63;
    } else {
        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m_out = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* Q3_K: unpack the 12 packed scale bytes into 16 signed 6-bit scales. */
static inline void unpack_q3k_scales(const uint8_t *sc_packed, int8_t *sc_out) {
    for (int j = 0; j < 16; j++) {
        const uint8_t low4  = (uint8_t) ((sc_packed[j % 8] >> ((j / 8) * 4u)) & 0x0Fu);
        const uint8_t high2 = (uint8_t) ((sc_packed[8 + (j % 4)] >> ((j / 4) * 2u)) & 0x03u);
        sc_out[j]           = (int8_t) ((int) (low4 | (high2 << 4u)) - 32);
    }
}

/* dot(x_q8[0..15], q_signed_int8[0..15]) using vdotq_s32 when NEON
 * is available. Shared across Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 W*A8
 * kernels — a 16-byte int8 SDOT is the universal inner accumulator.
 *
 * The signed-int8 view of `q` is the caller's responsibility: Q4_K /
 * Q5_K pass unsigned nibbles (0..15 / 0..31), the higher-bit formats
 * pass signed Q-values already in -32..31 range. The function does
 * not interpret the sign of `q` — it just multiplies and sums. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline int32_t dot16_i8(const int8_t *x_q8, int8x16_t q) {
    int8x16_t xv  = vld1q_s8(x_q8);
    int32x4_t acc = vdupq_n_s32(0);
    acc           = vdotq_s32(acc, xv, q);
    return vaddvq_s32(acc);
}

/* Q3_K: rebuild 32 signed 3-bit weights in [-4, 3] from 2 low bits (qs)
 * and 1 high bit (hmask). The caller passes the qs/hmask vectors already
 * shifted (vshrq_n_u8 needs an immediate; shift 0 passes them through). */
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
#endif
