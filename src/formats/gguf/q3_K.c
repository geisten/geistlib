/*
 * src/formats/gguf/q3_K.c — Q3_K block dequantization.
 *
 * Pure file-format decoder. The W3A8 NEON kernels live in
 * src/backends/cpu_neon/kernels/q3_K.c. struct block_q3_K_t lives in
 * internal.h so both files share one definition.
 *
 * Local NEON helpers (q3k_reconstruct_q32, q3k_store16_scaled) are
 * kept here because they are dequant-side. The kernel side duplicates
 * unpack_q3k_scales + q3k_reconstruct_q32 as static inline for the same
 * reason — DRY would mean a third header neither file fully owns.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline void unpack_q3k_scales(const uint8_t *sc_packed, int8_t *sc_out) {
    for (int j = 0; j < 16; j++) {
        const uint8_t low4  = (uint8_t) ((sc_packed[j % 8] >> ((j / 8) * 4u)) & 0x0Fu);
        const uint8_t high2 = (uint8_t) ((sc_packed[8 + (j % 4)] >> ((j / 4) * 2u)) & 0x03u);
        sc_out[j]           = (int8_t) ((int) (low4 | (high2 << 4u)) - 32);
    }
}

#if defined(__ARM_NEON)
/* Reconstruct 32 signed-3-bit q values for a Q3_K quant-group. Caller passes
 * pre-shifted qs/hmask vectors (vshrq_n_u8 needs an immediate; shift=0 means
 * caller passes the source vector through unshifted). */
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

/* Multiply 16 int8 q values by a single fp32 scalar, store to y[0..15]. */
static inline void q3k_store16_scaled(int8x16_t q, float scale, float *y) {
    const float32x4_t v_scale = vdupq_n_f32(scale);
    int16x8_t         qa      = vmovl_s8(vget_low_s8(q));
    int16x8_t         qb      = vmovl_s8(vget_high_s8(q));
    float32x4_t       f0      = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qa)));
    float32x4_t       f1      = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qa)));
    float32x4_t       f2      = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qb)));
    float32x4_t       f3      = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qb)));
    vst1q_f32(y + 0, vmulq_f32(f0, v_scale));
    vst1q_f32(y + 4, vmulq_f32(f1, v_scale));
    vst1q_f32(y + 8, vmulq_f32(f2, v_scale));
    vst1q_f32(y + 12, vmulq_f32(f3, v_scale));
}
#endif

void dequant_q3_K_row(const void *blocks, float *out, size_t n_elems) {
    const struct block_q3_K_t *b  = (const struct block_q3_K_t *) blocks;
    const size_t               nb = n_elems / Q3_K_BLOCK_ELEMS;

    for (size_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(b[i].d);
        float      *y = out + i * Q3_K_BLOCK_ELEMS;

        int8_t scale[16];
        unpack_q3k_scales(b[i].scales, scale);

#if defined(__ARM_NEON)
        const uint8x16_t hm_lo  = vld1q_u8(b[i].hmask + 0);
        const uint8x16_t hm_hi  = vld1q_u8(b[i].hmask + 16);
        const uint8x16_t qsA_lo = vld1q_u8(b[i].qs + 0);
        const uint8x16_t qsA_hi = vld1q_u8(b[i].qs + 16);
        const uint8x16_t qsB_lo = vld1q_u8(b[i].qs + 32);
        const uint8x16_t qsB_hi = vld1q_u8(b[i].qs + 48);

        /* 8 quant-groups of 32 weights. qs shifts: 0,2,4,6 within each 32-byte
         * qs half. hmask shift = group index 0..7 across both halves. */
        int8x16_t q_l, q_h;
#define DEQUANT_GROUP(G, QSL, QSH, HML, HMH, XOFF)                                \
    do {                                                                          \
        q3k_reconstruct_q32(QSL, QSH, HML, HMH, &q_l, &q_h);                      \
        q3k_store16_scaled(q_l, d * (float) scale[2 * (G)], y + (XOFF) + 0);      \
        q3k_store16_scaled(q_h, d * (float) scale[2 * (G) + 1], y + (XOFF) + 16); \
    } while (0)

        DEQUANT_GROUP(0, qsA_lo, qsA_hi, hm_lo, hm_hi, 0);
        DEQUANT_GROUP(1,
                      vshrq_n_u8(qsA_lo, 2),
                      vshrq_n_u8(qsA_hi, 2),
                      vshrq_n_u8(hm_lo, 1),
                      vshrq_n_u8(hm_hi, 1),
                      32);
        DEQUANT_GROUP(2,
                      vshrq_n_u8(qsA_lo, 4),
                      vshrq_n_u8(qsA_hi, 4),
                      vshrq_n_u8(hm_lo, 2),
                      vshrq_n_u8(hm_hi, 2),
                      64);
        DEQUANT_GROUP(3,
                      vshrq_n_u8(qsA_lo, 6),
                      vshrq_n_u8(qsA_hi, 6),
                      vshrq_n_u8(hm_lo, 3),
                      vshrq_n_u8(hm_hi, 3),
                      96);
        DEQUANT_GROUP(4, qsB_lo, qsB_hi, vshrq_n_u8(hm_lo, 4), vshrq_n_u8(hm_hi, 4), 128);
        DEQUANT_GROUP(5,
                      vshrq_n_u8(qsB_lo, 2),
                      vshrq_n_u8(qsB_hi, 2),
                      vshrq_n_u8(hm_lo, 5),
                      vshrq_n_u8(hm_hi, 5),
                      160);
        DEQUANT_GROUP(6,
                      vshrq_n_u8(qsB_lo, 4),
                      vshrq_n_u8(qsB_hi, 4),
                      vshrq_n_u8(hm_lo, 6),
                      vshrq_n_u8(hm_hi, 6),
                      192);
        DEQUANT_GROUP(7,
                      vshrq_n_u8(qsB_lo, 6),
                      vshrq_n_u8(qsB_hi, 6),
                      vshrq_n_u8(hm_lo, 7),
                      vshrq_n_u8(hm_hi, 7),
                      224);
#undef DEQUANT_GROUP
#else
        const uint8_t *hmask = b[i].hmask;
        const uint8_t *qs    = b[i].qs;
        for (size_t w = 0; w < Q3_K_BLOCK_ELEMS; w++) {
            const size_t scale_idx   = w >> 4;
            const size_t byte_idx_lo = ((w >> 7) << 5) | (w & 31);
            const int    shift_lo    = (int) ((w >> 5) & 3) << 1;
            const int    low2        = (qs[byte_idx_lo] >> shift_lo) & 0x03;
            const size_t byte_idx_hi = w & 31;
            const int    shift_hi    = (int) (w >> 5);
            const int    hi1         = (hmask[byte_idx_hi] >> shift_hi) & 0x01;
            const int    q_signed    = low2 - ((hi1 ^ 1) << 2);
            y[w]                     = d * (float) scale[scale_idx] * (float) q_signed;
        }
#endif
    }
}
