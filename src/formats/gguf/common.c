/*
 * src/formats/gguf/common.c — generic dispatchers + shared INT8 vector quant.
 *
 * Layer: BACKEND.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "quant.h"
#include "gguf_dequant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if !defined(GEIST_FP16_TO_FP32_INLINE)
/* IEEE-754 fp16 → fp32. Bit-exact decode (subnormals + inf + nan handled).
 * Only built without hardware fp16; otherwise quant.h has the inline
 * single-instruction version. */
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t) (h >> 15) & 0x1;
    uint32_t exp  = (uint32_t) (h >> 10) & 0x1F;
    uint32_t frac = (uint32_t) (h) & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign << 31;
        } else {
            /* subnormal */
            int e = -1;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                e--;
            }
            frac &= 0x3FF;
            /* Subnormal fp16 with leading 1 at bit b (after shift) has FP32
             * unbiased exponent E = b - 24. Since shift_count = 10 - b and
             * e = -1 - shift_count, we get b = e + 11 and E = e - 13. */
            uint32_t exp32 = (uint32_t) (127 - 13 + e);
            out            = (sign << 31) | (exp32 << 23) | (frac << 13);
        }
    } else if (exp == 0x1F) {
        out = (sign << 31) | (0xFF << 23) | (frac << 13);
    } else {
        out = (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}
#endif

/* I2_S (BitNet b1.58, bitnet.cpp): 256-element / 64-byte ternary blocks,
 * four 2-bit fields per byte in REVERSE order (element 32*g + bb of each
 * 128-element half at shift 6 - 2g), one f32 per-TENSOR scale. */
static void dequant_i2_s_blocks(size_t n, const uint8_t *blocks, float scale, float out[static n]) {
    for (size_t b = 0; b < n / I2_S_BLOCK_ELEMS; b++) {
        const uint8_t *qs = blocks + b * I2_S_BLOCK_BYTES;
        float         *o  = out + b * I2_S_BLOCK_ELEMS;
        for (size_t h = 0; h < 2; h++) {
            for (size_t bb = 0; bb < 32; bb++) {
                const uint8_t byte = qs[h * 32 + bb];
                for (size_t g = 0; g < 4; g++) {
                    const int trit           = (int) ((byte >> (6 - 2 * g)) & 3) - 1;
                    o[h * 128 + g * 32 + bb] = (float) trit * scale;
                }
            }
        }
    }
}

bool quant_dequant_row(enum geist_dtype dt,
                       size_t           n_total,
                       size_t           e0,
                       size_t           n,
                       const void      *raw,
                       float            out[static n]) {
    size_t blk_elems = 0;
    size_t blk_bytes = 0;
    size_t tail      = 0;
    if (!quant_block_layout(dt, &blk_elems, &blk_bytes, &tail) || e0 % blk_elems != 0u ||
        n % blk_elems != 0u || n > n_total || e0 > n_total - n) {
        memset(out, 0, n * sizeof *out);
        return false;
    }
    /* In bounds of the caller's quant_raw_bytes(dt, n_total) extent. */
    const uint8_t *src = (const uint8_t *) raw + e0 / blk_elems * blk_bytes;
    switch (dt) {
    case GEIST_DTYPE_F32:
        memcpy(out, src, n * sizeof(float));
        return true;
    case GEIST_DTYPE_F16:
        /* Little-endian storage, assembled bytewise: `src` need not be
         * 2-byte aligned. */
        for (size_t i = 0; i < n; i++) {
            const uint8_t *h = src + i * 2;
            out[i]           = fp16_to_fp32((uint16_t) ((uint16_t) h[0] | ((uint16_t) h[1] << 8)));
        }
        return true;
    case GEIST_DTYPE_BF16:
        /* BF16 = top 16 bits of FP32; left-shift restores fp32 layout. */
        for (size_t i = 0; i < n; i++) {
            const uint8_t *h = src + i * 2;
            const uint32_t b = (uint32_t) ((uint16_t) h[0] | ((uint16_t) h[1] << 8)) << 16;
            memcpy(&out[i], &b, sizeof b);
        }
        return true;
    case GEIST_DTYPE_Q4_0:
        dequant_q4_0_row(n, src, out);
        return true;
    case GEIST_DTYPE_Q4_1:
        dequant_q4_1_row(n, src, out);
        return true;
    case GEIST_DTYPE_Q5_0:
        dequant_q5_0_row(n, src, out);
        return true;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(n, src, out);
        return true;
    case GEIST_DTYPE_Q3_K:
        dequant_q3_K_row(n, src, out);
        return true;
    case GEIST_DTYPE_Q4_K:
        dequant_q4_K_row(n, src, out);
        return true;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(n, src, out);
        return true;
    case GEIST_DTYPE_Q6_K:
        dequant_q6_K_row(n, src, out);
        return true;
    case GEIST_DTYPE_IQ2_S:
        dequant_iq2_s_row(n, src, out);
        return true;
    case GEIST_DTYPE_IQ3_S:
        dequant_iq3_s_row(n, src, out);
        return true;
    case GEIST_DTYPE_IQ4_NL:
        dequant_iq4_nl_row(n, src, out);
        return true;
    case GEIST_DTYPE_IQ4_XS:
        dequant_iq4_xs_row(n, src, out);
        return true;
    case GEIST_DTYPE_TQ2_0:
        dequant_tq2_0_row(n, src, out);
        return true;
    case GEIST_DTYPE_PQ2_0:
        dequant_pq2_0_row(n, src, out);
        return true;
    case GEIST_DTYPE_I2_S: {
        float scale;
        memcpy(&scale, (const uint8_t *) raw + i2_s_scale_offset(n_total), sizeof scale);
        dequant_i2_s_blocks(n, src, scale, out);
        return true;
    }
    default:
        /* I8 / U8: a raw layout but no f32 decode here. */
        memset(out, 0, n * sizeof *out);
        return false;
    }
}

/* The geist dtype a GGUF tensor's rows decode as; GEIST_DTYPE_CUSTOM (no
 * decoder) for a GGUF type quant_dequant_row does not cover. */
static enum geist_dtype gguf_row_dtype(gguf_dtype_t gd) {
    switch (gd) {
    case GGUF_TYPE_F32:
        return GEIST_DTYPE_F32;
    case GGUF_TYPE_F16:
        return GEIST_DTYPE_F16;
    case GGUF_TYPE_BF16:
        return GEIST_DTYPE_BF16;
    case GGUF_TYPE_Q4_0:
        return GEIST_DTYPE_Q4_0;
    case GGUF_TYPE_Q4_1:
        return GEIST_DTYPE_Q4_1;
    case GGUF_TYPE_Q5_0:
        return GEIST_DTYPE_Q5_0;
    case GGUF_TYPE_Q8_0:
        return GEIST_DTYPE_Q8_0;
    case GGUF_TYPE_Q3_K:
        return GEIST_DTYPE_Q3_K;
    case GGUF_TYPE_Q4_K:
        return GEIST_DTYPE_Q4_K;
    case GGUF_TYPE_Q5_K:
        return GEIST_DTYPE_Q5_K;
    case GGUF_TYPE_Q6_K:
        return GEIST_DTYPE_Q6_K;
    case GGUF_TYPE_IQ2_S:
        return GEIST_DTYPE_IQ2_S;
    case GGUF_TYPE_IQ3_S:
        return GEIST_DTYPE_IQ3_S;
    case GGUF_TYPE_IQ4_NL:
        return GEIST_DTYPE_IQ4_NL;
    case GGUF_TYPE_IQ4_XS:
        return GEIST_DTYPE_IQ4_XS;
    case GGUF_TYPE_TQ2_0:
        return GEIST_DTYPE_TQ2_0;
    case GGUF_TYPE_PQ2_0:
        return GEIST_DTYPE_PQ2_0;
    case GGUF_TYPE_I2_S:
        return GEIST_DTYPE_I2_S;
    default:
        return GEIST_DTYPE_CUSTOM;
    }
}

bool gguf_dequant_row_to_fp32(const struct gguf_tensor_t *t,
                              size_t                      row_idx,
                              size_t                      row_elems,
                              float                      *out) {
    size_t e0 = 0;
    if (!t || ckd_mul(&e0, row_idx, row_elems)) {
        memset(out, 0, row_elems * sizeof *out);
        return false;
    }
    return quant_dequant_row(
            gguf_row_dtype(t->dtype), gguf_tensor_elem_count(t), e0, row_elems, t->data, out);
}

float *gguf_dequant_to_fp32(const struct gguf_tensor_t *t) {
    if (!t)
        return nullptr;
    size_t elems = gguf_tensor_elem_count(t);
    float *out   = heap_alloc_array_aligned(float, elems);
    if (!out)
        return nullptr;
    if (!quant_dequant_row(gguf_row_dtype(t->dtype), elems, 0, elems, t->data, out)) {
        safe_free((void **) &out);
        return nullptr;
    }
    return out;
}

float quantize_x_int8_sym(size_t n, const float x[static n], int8_t x_q8[static n]) {
#if defined(__ARM_NEON)
    float32x4_t amax_v = vdupq_n_f32(0.0f);
    size_t      i      = 0;
    for (; i + 4 <= n; i += 4) {
        amax_v = vmaxq_f32(amax_v, vabsq_f32(vld1q_f32(x + i)));
    }
    float amax = vmaxvq_f32(amax_v);
    for (; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax)
            amax = a;
    }
#else
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax)
            amax = a;
    }
#endif
    float scale = amax / 127.0f;
    if (scale == 0.0f)
        scale = 1.0f;
    float inv = 1.0f / scale;
#if defined(__ARM_NEON)
    /* Pack 16 floats per iteration: round-to-nearest, saturating narrow s32→s16→s8. */
    const float32x4_t invv = vdupq_n_f32(inv);
    size_t            j    = 0;
    for (; j + 16 <= n; j += 16) {
        int32x4_t q0  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + j + 0), invv));
        int32x4_t q1  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + j + 4), invv));
        int32x4_t q2  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + j + 8), invv));
        int32x4_t q3  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + j + 12), invv));
        int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(x_q8 + j, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
    }
    for (; j < n; j++)
        x_q8[j] = (int8_t) lrintf(x[j] * inv);
#else
    /* |x[i]·inv| ≤ 127 by construction (inv = 127/amax), so cast is in-range. */
    for (size_t i = 0; i < n; i++)
        x_q8[i] = (int8_t) lrintf(x[i] * inv);
#endif
    return scale;
}
