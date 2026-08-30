/*
 * src/formats/gguf/common.c — generic dispatchers + shared INT8 vector quant.
 *
 * Layer: BACKEND. Extracted from gguf_quant.c during the per-quant
 * format split.
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

#if !(defined(__ARM_FP) && (__ARM_FP & 2))
/* IEEE-754 fp16 → fp32. Bit-exact decode (subnormals + inf + nan handled).
 *
 * Used only on builds without hardware fp16 support. On ARM64 with native
 * __fp16 (the typical Pi 5 / Apple Silicon case), the header has a
 * `static inline` version that compiles to a single `fcvt` instruction
 * and inlines everywhere, eliminating the call-overhead measured at ~5%
 * of Pi 5 Gemma 4 decode time. */
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

bool gguf_dequant_row_to_fp32(const struct gguf_tensor_t *t,
                              size_t                      row_idx,
                              size_t                      row_elems,
                              float                      *out) {
    if (!t)
        return false;
    switch (t->dtype) {
    case GGUF_TYPE_F32:
        memcpy(out, (const float *) t->data + row_idx * row_elems, row_elems * sizeof(float));
        return true;
    case GGUF_TYPE_F16: {
        const uint16_t *h = (const uint16_t *) t->data + row_idx * row_elems;
        for (size_t i = 0; i < row_elems; i++)
            out[i] = fp16_to_fp32(h[i]);
        return true;
    }
    case GGUF_TYPE_BF16: {
        /* BF16 = top 16 bits of FP32; left-shift restores fp32 layout. */
        const uint16_t *h = (const uint16_t *) t->data + row_idx * row_elems;
        uint32_t       *o = (uint32_t *) out;
        for (size_t i = 0; i < row_elems; i++)
            o[i] = ((uint32_t) h[i]) << 16;
        return true;
    }
    case GGUF_TYPE_Q8_0: {
        const size_t   blocks_per_row = row_elems / Q8_0_BLOCK_ELEMS;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * Q8_0_BLOCK_BYTES;
        dequant_q8_0_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_Q3_K: {
        const size_t   blocks_per_row = row_elems / Q3_K_BLOCK_ELEMS;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * Q3_K_BLOCK_BYTES;
        dequant_q3_K_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_Q4_K: {
        const size_t   blocks_per_row = row_elems / Q4_K_BLOCK_ELEMS;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * Q4_K_BLOCK_BYTES;
        dequant_q4_K_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_Q5_K: {
        const size_t   blocks_per_row = row_elems / Q5_K_BLOCK_ELEMS;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * Q5_K_BLOCK_BYTES;
        dequant_q5_K_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_Q6_K: {
        const size_t   blocks_per_row = row_elems / Q6_K_BLOCK_ELEMS;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * Q6_K_BLOCK_BYTES;
        dequant_q6_K_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_IQ2_S: {
        const size_t   blocks_per_row = row_elems / 256;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * IQ2_S_BLOCK_BYTES;
        dequant_iq2_s_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_IQ3_S: {
        const size_t   blocks_per_row = row_elems / 256;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * IQ3_S_BLOCK_BYTES;
        dequant_iq3_s_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_IQ4_NL: {
        const size_t   blocks_per_row = row_elems / IQ4_NL_BLOCK_ELEMS;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * IQ4_NL_BLOCK_BYTES;
        dequant_iq4_nl_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_IQ4_XS: {
        const size_t   blocks_per_row = row_elems / IQ4_XS_BLOCK_ELEMS;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * IQ4_XS_BLOCK_BYTES;
        dequant_iq4_xs_row(row_elems, base, out);
        return true;
    }
    case GGUF_TYPE_TQ2_0: {
        const size_t   blocks_per_row = row_elems / TQ2_0_BLOCK_ELEMS;
        const uint8_t *base =
                (const uint8_t *) t->data + row_idx * blocks_per_row * TQ2_0_BLOCK_BYTES;
        dequant_tq2_0_row(row_elems, base, out);
        return true;
    }
    default:
        return false;
    }
}

float *gguf_dequant_to_fp32(const struct gguf_tensor_t *t) {
    if (!t)
        return nullptr;
    size_t elems = gguf_tensor_elem_count(t);
    float *out   = heap_alloc_array_aligned(float, elems);
    if (!out)
        return nullptr;

    switch (t->dtype) {
    case GGUF_TYPE_F32:
        memcpy(out, t->data, elems * sizeof(float));
        break;
    case GGUF_TYPE_F16: {
        const uint16_t *h = (const uint16_t *) t->data;
        for (size_t i = 0; i < elems; i++)
            out[i] = fp16_to_fp32(h[i]);
        break;
    }
    case GGUF_TYPE_BF16: {
        /* BF16 = top 16 bits of FP32; left-shift restores fp32 layout. */
        const uint16_t *h = (const uint16_t *) t->data;
        uint32_t       *o = (uint32_t *) out;
        for (size_t i = 0; i < elems; i++)
            o[i] = ((uint32_t) h[i]) << 16;
        break;
    }
    case GGUF_TYPE_Q8_0:
        dequant_q8_0_row(elems, t->data, out);
        break;
    case GGUF_TYPE_Q3_K:
        dequant_q3_K_row(elems, t->data, out);
        break;
    case GGUF_TYPE_Q4_K:
        dequant_q4_K_row(elems, t->data, out);
        break;
    case GGUF_TYPE_Q5_K:
        dequant_q5_K_row(elems, t->data, out);
        break;
    case GGUF_TYPE_Q6_K:
        dequant_q6_K_row(elems, t->data, out);
        break;
    case GGUF_TYPE_IQ2_S:
        dequant_iq2_s_row(elems, t->data, out);
        break;
    case GGUF_TYPE_IQ3_S:
        dequant_iq3_s_row(elems, t->data, out);
        break;
    case GGUF_TYPE_IQ4_NL:
        dequant_iq4_nl_row(elems, t->data, out);
        break;
    case GGUF_TYPE_IQ4_XS:
        dequant_iq4_xs_row(elems, t->data, out);
        break;
    case GGUF_TYPE_TQ2_0:
        dequant_tq2_0_row(elems, t->data, out);
        break;
    default:
        safe_free((void **) &out);
        return nullptr;
    }
    return out;
}

float quantize_x_int8_sym(const float *x, size_t n, int8_t *x_q8) {
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
