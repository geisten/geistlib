/*
 * quant_blocks — packed memory layouts of every quantized block type, plus
 * the small inline helpers shared by the dequant codecs (formats/gguf) and
 * the W*A8 compute kernels (backends/cpu_neon).
 *
 * These `__attribute__((packed))` structs ARE the block-layout contract:
 * each is _Static_assert'd against its byte size from quant.h. They carry no
 * file-format types, so this header sits in the neutral quant/ module rather
 * than inside formats/gguf, where the NEON kernels used to reach it across a
 * layer boundary.
 *
 * Helpers here:
 *   - get_scale_min_k4: 6-bit scale/min unpack shared by Q4_K and Q5_K
 *   - quantize_x_int8_sym: per-vector symmetric INT8 quant, used by all
 *     W*A8 kernels (also declared in quant.h).
 *
 * Per-format-only helpers (unpack_q3k_scales, q4k_subpair_dots,
 * iq2s/iq3s subblock decoders, q3k NEON inlines, etc.) live inside the
 * format-specific .c file as `static inline`.
 */
#pragma once

#include "quant.h"

#include <stddef.h>
#include <stdint.h>

/* ---- Block layouts ----------------------------------------------------
 *
 * These structs describe the exact on-disk byte layouts of GGUF quant
 * blocks. They are part of the format contract (every backend that
 * decodes GGUF needs to read these), so they live in this format-
 * internal header — kernels in src/backends/<arch>/kernels/ include
 * it to access the fields without duplicating the definition.
 *
 * `__attribute__((packed))` keeps the struct laid out exactly as the
 * file format dictates (no compiler-inserted padding). _Static_assert
 * on each definition verifies the size against the published constant.
 *
 * Block constants live in quant.h (the neutral quant contract). */

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

/* quantize_x_int8_sym (symmetric INT8 activation quant) is declared in
 * quant.h — the W*A8 kernels here use it via that contract. */

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
#endif
