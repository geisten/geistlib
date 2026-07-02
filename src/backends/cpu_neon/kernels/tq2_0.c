/*
 * src/backends/cpu_neon/kernels/tq2_0.c — TQ2_0 W1.58×A8 NEON kernels.
 *
 * Layer: BACKEND (cpu_neon). Extracted from weight_resolve.c to isolate
 * the ternary (BitNet b1.58 / TQ2_0) compute path from the shared weight
 * resolver and from the Q4_K/Q6_K (Gemma) kernels — so optimizing one
 * model's kernels cannot disturb the other.
 *
 * Owns the M=1 decode (q8a + fp32 fallback) and M>1 prefill paths for
 * TQ2_0 weights, plus the block-dot helpers and the per-row activation
 * absmax-quant. The pure file-format decoder (dequant_tq2_0_row) and the
 * block struct stay in src/formats/gguf/. The TL1 LUT path lives in tl1.c.
 *
 * Entry points (declared in internal.h, referenced by the resolver table):
 *   cpu_neon_w_tq2_0_q8a_m1  — M=1 decode, int8 SDOT (dotprod hosts)
 *   cpu_neon_w_tq2_0_q8a_mN  — M>1 prefill, int8 SDOT (dotprod hosts)
 *   cpu_neon_w_tq2_0_m1      — M=1 fp32 fallback (no-dotprod hosts)
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../internal.h"
#include "../parallel.h"
#include "heap.h"

#include "quant.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* P3.10: direct TQ2_0 W1.58 A32 NEON kernel — fused dequant+dot. The
 * dequant trampoline materializes 32-row tiles into fp32 then cblas_sgemv
 * — two memory passes per row + one extra MB of L2 churn per tile. This
 * fused path unpacks trits from the packed 2-bit format on-the-fly,
 * converts to fp32 inside NEON registers, and FMAs directly against the
 * activation, never writing the dequanted weights to memory.
 *
 * Block layout (per TQ2_0_BLOCK_ELEMS=256 elements, 66 bytes):
 *   qs[ 0..31]  — packed trits for first 128 elements. Byte (j+m), bit
 *                 pair l*2 holds the trit for element 32*l + m, l in 0..3.
 *   qs[32..63]  — packed trits for second 128 elements (offset by 128).
 *   d[2]        — fp16 row scale.
 * trit ∈ {0,1,2} representing {-1, 0, +1}; final value = (q-1) * d.
 *
 * Each NEON inner stride handles 32 elements: load 16 packed bytes, then
 * per-l (compile-time unrolled) extract the 32 trits across the two
 * halves of the byte block, expand to int8 → int16 → int32 → fp32, and
 * fma against 32 contiguous activations.
 *
 * Theoretical: 8 fp32 FMAs per element across the 256-element block.
 * Versus the trampoline: same final FMA count, but no intermediate fp32
 * store/reload. Pi 5 A76 wins from L2 bandwidth savings on the
 * weight-row stream. */
/* Used by cpu_neon_w_tq2_0_m1 fp32 fallback. Always compiled so the
 * fallback can be selected at runtime on dotprod-built binaries that
 * end up on non-dotprod hosts. */
#if defined(__ARM_NEON)
static inline float tq2_0_block_dot_neon(const uint8_t *qs, const float *xb) {
    const uint8x16_t three  = vdupq_n_u8(3);
    const int8x16_t  one_s8 = vdupq_n_s8(1);
    float32x4_t      acc    = vdupq_n_f32(0.0f);

/* DO_PAIR(P, SHIFT, X_OFF): for the 16 packed bytes in `P`, extract
 * one trit per byte (the SHIFT-th 2-bit slot), convert to fp32, and
 * fma against `xb + X_OFF` (16 activations). Repeats for the 16
 * higher elements via a second invocation. */
/* SHIFT==0: skip vshrq (the intrinsic requires shift ≥ 1); just mask. */
#define DO_PAIR(P, SHIFT, X_OFF)                                                                    \
    do {                                                                                            \
        uint8x16_t  _t  = ((SHIFT) == 0) ? vandq_u8((P), three)                                     \
                                         : vandq_u8(vshrq_n_u8((P), (SHIFT) ? (SHIFT) : 1), three); \
        int8x16_t   _s  = vsubq_s8(vreinterpretq_s8_u8(_t), one_s8);                                \
        int16x8_t   _lo = vmovl_s8(vget_low_s8(_s));                                                \
        int16x8_t   _hi = vmovl_s8(vget_high_s8(_s));                                               \
        float32x4_t _f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_lo)));                              \
        float32x4_t _f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_lo)));                             \
        float32x4_t _f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_hi)));                              \
        float32x4_t _f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_hi)));                             \
        float32x4_t _a0 = vld1q_f32(xb + (X_OFF) + 0);                                              \
        float32x4_t _a1 = vld1q_f32(xb + (X_OFF) + 4);                                              \
        float32x4_t _a2 = vld1q_f32(xb + (X_OFF) + 8);                                              \
        float32x4_t _a3 = vld1q_f32(xb + (X_OFF) + 12);                                             \
        acc             = vfmaq_f32(acc, _f0, _a0);                                                 \
        acc             = vfmaq_f32(acc, _f1, _a1);                                                 \
        acc             = vfmaq_f32(acc, _f2, _a2);                                                 \
        acc             = vfmaq_f32(acc, _f3, _a3);                                                 \
    } while (0)

    /* First 128 elements: bytes 0..31, two halves of 16 bytes each. */
    uint8x16_t pa = vld1q_u8(qs + 0);
    uint8x16_t pb = vld1q_u8(qs + 16);
    DO_PAIR(pa, 0, 0);
    DO_PAIR(pb, 0, 16); /* elements 0..31 */
    DO_PAIR(pa, 2, 32);
    DO_PAIR(pb, 2, 48); /* elements 32..63 */
    DO_PAIR(pa, 4, 64);
    DO_PAIR(pb, 4, 80); /* elements 64..95 */
    DO_PAIR(pa, 6, 96);
    DO_PAIR(pb, 6, 112); /* elements 96..127 */

    /* Second 128 elements: bytes 32..63. */
    pa = vld1q_u8(qs + 32);
    pb = vld1q_u8(qs + 48);
    DO_PAIR(pa, 0, 128);
    DO_PAIR(pb, 0, 144);
    DO_PAIR(pa, 2, 160);
    DO_PAIR(pb, 2, 176);
    DO_PAIR(pa, 4, 192);
    DO_PAIR(pb, 4, 208);
    DO_PAIR(pa, 6, 224);
    DO_PAIR(pb, 6, 240);
#undef DO_PAIR

    return vaddvq_f32(acc);
}
#endif

/* P3.11: W1.58 × A8 — int8 activation quant + ternary×int8 NEON dot.
 *
 * Where P3.10's W1.58 × A32 kernel did fp32 FMAs against fp32 trits,
 * this kernel quantizes the activation to int8 once per call and then
 * uses ARMv8.2 vdotq_s32 (4-way int8×int8 → int32) for the per-block
 * dot. Each vdotq replaces 4 vfmaq calls — fewer instructions and
 * higher throughput on Pi 5 A76 (ARM dotprod extension).
 *
 * Equivalence with HF's BitLinear at inference:
 *   y = round(x · s) · w_ternary · weight_d / s
 *     = (int8_dot(x_q8, w_ternary)) · weight_d / act_scale
 * where s = act_scale = 127/max|x|, x_q8 = round(x · s) clamped [-128,127],
 * and the per-block weight_d comes from TQ2_0's fp16 scale.
 *
 * Geist's forward.c already calls apply_bitnet_input_quant_inplace before
 * each BitLinear (round-trip fake-quant — x values are already
 * representable as int8/s). Re-quanting here is a no-op modulo rounding
 * (the second quant lands on the same grid). The cost is one O(n_in)
 * pass per kernel call (~2560 mul-round-cmp per matmul on 2B-4T;
 * negligible vs the ~6M dot ops). */
#if defined(__ARM_NEON)
/* Block dot product with deferred -1 bias.
 *
 * TQ2_0 trits are stored as {0, 1, 2} representing {-1, 0, +1}. The
 * "obvious" path subtracts 1 from each unpacked nibble before vdotq:
 *
 *   dot = Σ (trit_i - 1) × xq_i
 *
 * which costs one vsubq_s8 per nibble level (16 subs per 256-elem
 * block). The unbiased variant skips the subtract — vdotq sees raw
 * 0/1/2 weights — and the caller applies the bias correction once
 * per block:
 *
 *   dot_raw = Σ trit_i × xq_i        (this function)
 *   dot     = dot_raw - Σ xq_i        (caller, using precomputed bsum)
 *
 * Identical math, ~25% fewer NEON ops in the inner kernel. Pattern
 * mirrors llama.cpp's ggml_vec_dot_tq2_0_q8_K. */
static inline int32_t tq2_0_block_dot_q8a_neon_unbiased(const uint8_t *qs, const int8_t *xb) {
    const uint8x16_t three = vdupq_n_u8(3);
    /* Two int32 accumulators so the Cortex-A76 / Apple-Silicon dual NEON
     * pipes can issue independent vdotq_s32 instructions in parallel.
     * Single-acc serializes the dependency chain. */
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);

#define DO_DOT_PAIR(P, SHIFT, X_OFF_A, X_OFF_B)                                                   \
    do {                                                                                          \
        uint8x16_t _t = ((SHIFT) == 0) ? vandq_u8((P), three)                                     \
                                       : vandq_u8(vshrq_n_u8((P), (SHIFT) ? (SHIFT) : 1), three); \
        const int8x16_t _s  = vreinterpretq_s8_u8(_t);                                            \
        const int8x16_t _a0 = vld1q_s8(xb + (X_OFF_A));                                           \
        const int8x16_t _a1 = vld1q_s8(xb + (X_OFF_B));                                           \
        acc0                = vdotq_s32(acc0, _s, _a0);                                           \
        acc1                = vdotq_s32(acc1, _s, _a1);                                           \
    } while (0)

    /* First half (qs[0..31], xb[0..127]): pa for low 16 trits, pb for high. */
    {
        const uint8x16_t pa = vld1q_u8(qs + 0);
        const uint8x16_t pb = vld1q_u8(qs + 16);
        /* Interleave pa/pb work so the two pipes alternate. */
        const int8x16_t s_a0 = vreinterpretq_s8_u8(vandq_u8(pa, three));
        const int8x16_t s_b0 = vreinterpretq_s8_u8(vandq_u8(pb, three));
        const int8x16_t s_a2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 2), three));
        const int8x16_t s_b2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 2), three));
        const int8x16_t s_a4 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 4), three));
        const int8x16_t s_b4 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 4), three));
        const int8x16_t s_a6 = vreinterpretq_s8_u8(vshrq_n_u8(pa, 6));
        const int8x16_t s_b6 = vreinterpretq_s8_u8(vshrq_n_u8(pb, 6));
        acc0                 = vdotq_s32(acc0, s_a0, vld1q_s8(xb + 0));
        acc1                 = vdotq_s32(acc1, s_b0, vld1q_s8(xb + 16));
        acc0                 = vdotq_s32(acc0, s_a2, vld1q_s8(xb + 32));
        acc1                 = vdotq_s32(acc1, s_b2, vld1q_s8(xb + 48));
        acc0                 = vdotq_s32(acc0, s_a4, vld1q_s8(xb + 64));
        acc1                 = vdotq_s32(acc1, s_b4, vld1q_s8(xb + 80));
        acc0                 = vdotq_s32(acc0, s_a6, vld1q_s8(xb + 96));
        acc1                 = vdotq_s32(acc1, s_b6, vld1q_s8(xb + 112));
    }
    /* Second half (qs[32..63], xb[128..255]). */
    {
        const uint8x16_t pa   = vld1q_u8(qs + 32);
        const uint8x16_t pb   = vld1q_u8(qs + 48);
        const int8x16_t  s_a0 = vreinterpretq_s8_u8(vandq_u8(pa, three));
        const int8x16_t  s_b0 = vreinterpretq_s8_u8(vandq_u8(pb, three));
        const int8x16_t  s_a2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 2), three));
        const int8x16_t  s_b2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 2), three));
        const int8x16_t  s_a4 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 4), three));
        const int8x16_t  s_b4 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 4), three));
        const int8x16_t  s_a6 = vreinterpretq_s8_u8(vshrq_n_u8(pa, 6));
        const int8x16_t  s_b6 = vreinterpretq_s8_u8(vshrq_n_u8(pb, 6));
        acc0                  = vdotq_s32(acc0, s_a0, vld1q_s8(xb + 128));
        acc1                  = vdotq_s32(acc1, s_b0, vld1q_s8(xb + 144));
        acc0                  = vdotq_s32(acc0, s_a2, vld1q_s8(xb + 160));
        acc1                  = vdotq_s32(acc1, s_b2, vld1q_s8(xb + 176));
        acc0                  = vdotq_s32(acc0, s_a4, vld1q_s8(xb + 192));
        acc1                  = vdotq_s32(acc1, s_b4, vld1q_s8(xb + 208));
        acc0                  = vdotq_s32(acc0, s_a6, vld1q_s8(xb + 224));
        acc1                  = vdotq_s32(acc1, s_b6, vld1q_s8(xb + 240));
    }
#undef DO_DOT_PAIR

    return vaddvq_s32(vaddq_s32(acc0, acc1));
}

/* MT=4 token-tile of the unbiased block dot. Unpacks the 16 trit
 * sub-vectors of one weight block ONCE and dots each against the SAME
 * position in FOUR tokens' activations (xb0..xb3), amortizing the 2-bit
 * unpack 4x — exactly bitnet.cpp's i2_s 1x4 register-blocking. The four
 * per-token int32 accumulators give natural dual-NEON-pipe ILP (no need
 * for the single-token acc0/acc1 split). Bit-identical to calling the
 * single-token dot four times (same vdotq ops, same order per token). */
static inline void tq2_0_block_dot_q8a_neon_unbiased_mt4(const uint8_t *qs,
                                                         const int8_t  *xb0,
                                                         const int8_t  *xb1,
                                                         const int8_t  *xb2,
                                                         const int8_t  *xb3,
                                                         int32_t        out[4]) {
    const uint8x16_t three = vdupq_n_u8(3);
    int32x4_t        a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t        a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
#define MT4_POS(S, XOFF)                                                \
    do {                                                                \
        const int8x16_t _w = (S);                                       \
        a0                 = vdotq_s32(a0, _w, vld1q_s8(xb0 + (XOFF))); \
        a1                 = vdotq_s32(a1, _w, vld1q_s8(xb1 + (XOFF))); \
        a2                 = vdotq_s32(a2, _w, vld1q_s8(xb2 + (XOFF))); \
        a3                 = vdotq_s32(a3, _w, vld1q_s8(xb3 + (XOFF))); \
    } while (0)
    {
        const uint8x16_t pa = vld1q_u8(qs + 0);
        const uint8x16_t pb = vld1q_u8(qs + 16);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(pa, three)), 0);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(pb, three)), 16);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 2), three)), 32);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 2), three)), 48);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 4), three)), 64);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 4), three)), 80);
        MT4_POS(vreinterpretq_s8_u8(vshrq_n_u8(pa, 6)), 96);
        MT4_POS(vreinterpretq_s8_u8(vshrq_n_u8(pb, 6)), 112);
    }
    {
        const uint8x16_t pa = vld1q_u8(qs + 32);
        const uint8x16_t pb = vld1q_u8(qs + 48);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(pa, three)), 128);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(pb, three)), 144);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 2), three)), 160);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 2), three)), 176);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 4), three)), 192);
        MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 4), three)), 208);
        MT4_POS(vreinterpretq_s8_u8(vshrq_n_u8(pa, 6)), 224);
        MT4_POS(vreinterpretq_s8_u8(vshrq_n_u8(pb, 6)), 240);
    }
#undef MT4_POS
    out[0] = vaddvq_s32(a0);
    out[1] = vaddvq_s32(a1);
    out[2] = vaddvq_s32(a2);
    out[3] = vaddvq_s32(a3);
}

/* MT=8 variant: unpack one weight block once, dot against 8 tokens —
 * amortizes the 2-bit unpack over 8 tokens (vs 4). Eight per-token int32
 * accumulators stay in registers; the transient weight sub-vector + the
 * 8 activation loads pressure but fit the 32 NEON registers. Bit-identical
 * to 8x the single-token dot. */
static inline void tq2_0_block_dot_q8a_neon_unbiased_mt8(const uint8_t      *qs,
                                                         const int8_t *const xb[8],
                                                         int32_t             out[8]) {
    const uint8x16_t three = vdupq_n_u8(3);
    int32x4_t        a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t        a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32x4_t        a4 = vdupq_n_s32(0), a5 = vdupq_n_s32(0);
    int32x4_t        a6 = vdupq_n_s32(0), a7 = vdupq_n_s32(0);
#define MT8_POS(S, XOFF)                                                  \
    do {                                                                  \
        const int8x16_t _w = (S);                                         \
        a0                 = vdotq_s32(a0, _w, vld1q_s8(xb[0] + (XOFF))); \
        a1                 = vdotq_s32(a1, _w, vld1q_s8(xb[1] + (XOFF))); \
        a2                 = vdotq_s32(a2, _w, vld1q_s8(xb[2] + (XOFF))); \
        a3                 = vdotq_s32(a3, _w, vld1q_s8(xb[3] + (XOFF))); \
        a4                 = vdotq_s32(a4, _w, vld1q_s8(xb[4] + (XOFF))); \
        a5                 = vdotq_s32(a5, _w, vld1q_s8(xb[5] + (XOFF))); \
        a6                 = vdotq_s32(a6, _w, vld1q_s8(xb[6] + (XOFF))); \
        a7                 = vdotq_s32(a7, _w, vld1q_s8(xb[7] + (XOFF))); \
    } while (0)
    for (int h = 0; h < 2; h++) {
        const uint8x16_t pa = vld1q_u8(qs + h * 32 + 0);
        const uint8x16_t pb = vld1q_u8(qs + h * 32 + 16);
        const size_t     xo = (size_t) h * 128;
        MT8_POS(vreinterpretq_s8_u8(vandq_u8(pa, three)), xo + 0);
        MT8_POS(vreinterpretq_s8_u8(vandq_u8(pb, three)), xo + 16);
        MT8_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 2), three)), xo + 32);
        MT8_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 2), three)), xo + 48);
        MT8_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 4), three)), xo + 64);
        MT8_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 4), three)), xo + 80);
        MT8_POS(vreinterpretq_s8_u8(vshrq_n_u8(pa, 6)), xo + 96);
        MT8_POS(vreinterpretq_s8_u8(vshrq_n_u8(pb, 6)), xo + 112);
    }
#undef MT8_POS
    out[0] = vaddvq_s32(a0);
    out[1] = vaddvq_s32(a1);
    out[2] = vaddvq_s32(a2);
    out[3] = vaddvq_s32(a3);
    out[4] = vaddvq_s32(a4);
    out[5] = vaddvq_s32(a5);
    out[6] = vaddvq_s32(a6);
    out[7] = vaddvq_s32(a7);
}

/* Sum of 256 int8 activations as int32. Used to apply the
 * deferred -1 bias on the unbiased dot. */
static inline int32_t q8a_block_bsum(const int8_t *xb) {
    int32x4_t sum = vdupq_n_s32(0);
    for (size_t i = 0; i < 256; i += 16) {
        const int8x16_t v = vld1q_s8(xb + i);
        sum               = vpadalq_s16(sum, vpaddlq_s8(v));
    }
    return vaddvq_s32(sum);
}
#endif

/* Per-row work body for cpu_neon_w_tq2_0_q8a_m1. File-scope so both
 * the custom thread pool (which takes a function pointer) and the OMP
 * fallback can invoke it. */
struct q8a_m1_ctx {
    const uint8_t *W;
    const int8_t  *xq;
#if defined(__ARM_NEON)
    const int32_t *bsum_cache;
#endif
    float *y;
    float  inv_act_scale;
    size_t row_bytes;
    size_t blocks_per_row;
};

static void q8a_m1_row_body(size_t r, void *vctx) {
    const struct q8a_m1_ctx *c       = (const struct q8a_m1_ctx *) vctx;
    const uint8_t           *Wr      = c->W + r * c->row_bytes;
    float                    row_sum = 0.0f;
    for (size_t b = 0; b < c->blocks_per_row; b++) {
        const uint8_t *qs     = Wr + b * 66;
        const uint16_t d_bits = (uint16_t) qs[64] | ((uint16_t) qs[65] << 8);
        const float    d      = fp16_to_fp32(d_bits);
#if defined(__ARM_NEON)
        const int32_t dot_raw = tq2_0_block_dot_q8a_neon_unbiased(qs, c->xq + b * 256);
        const int32_t dot     = dot_raw - c->bsum_cache[b];
        row_sum += (float) dot * d;
#else
        int32_t dot = 0;
        for (size_t j = 0; j < 64; j += 32) {
            const size_t elem_base = (j == 0) ? 0 : 128;
            for (size_t l = 0; l < 4; l++) {
                const int    shift = (int) (l * 2);
                const size_t off   = elem_base + 32 * l;
                for (size_t m = 0; m < 32; m++) {
                    const int trit = (int) ((qs[j + m] >> shift) & 3) - 1;
                    dot += trit * (int) c->xq[b * 256 + off + m];
                }
            }
        }
        row_sum += (float) dot * d;
#endif
    }
    c->y[r] = row_sum * c->inv_act_scale;
}

/* GEIST_PP=1 opts into the custom spin-pool over OMP. Parsed once. */
static int tq2_pp_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("GEIST_PP");
        cached        = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

void cpu_neon_w_tq2_0_q8a_m1(const float               *x,
                             const struct geist_weight *w,
                             struct geist_backend      *be,
                             float                     *y) {
    struct cpu_neon_workspace *ws             = &((struct cpu_neon_state *) be->state)->workspace;
    const size_t               n_in           = (size_t) w->n_in;
    const size_t               n_out          = (size_t) w->n_out;
    const size_t               blocks_per_row = n_in / 256;
    const size_t               row_bytes      = blocks_per_row * 66;
    const uint8_t             *W              = (const uint8_t *) w->raw;

    /* Per-call activation quant: scan x for absmax, scale to [-127, 127],
     * write to int8 scratch. Reused across all n_out output-row dots. */
    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > max_abs)
            max_abs = a;
    }
    const float act_scale     = 127.0f / max_abs;
    const float inv_act_scale = max_abs / 127.0f;

    /* Thread-local int8 activation scratch — avoids reallocation per
     * ~210 kernel calls per decode token. Grows on demand; released
     * either at thread exit or via cpu_neon_release_thread_caches() at
     * backend destroy time (review #4 / V12). The cache pointers live
     * at file scope so the release helper can find them. */
    if (ws->m1_xq_cap < n_in) {
        safe_free((void **) &ws->m1_xq);
        ws->m1_xq = heap_alloc_array_aligned(int8_t, n_in);
        if (ws->m1_xq == nullptr) {
            ws->m1_xq_cap = 0;
            memset(y, 0, n_out * sizeof *y);
            return;
        }
        ws->m1_xq_cap = n_in;
    }
    int8_t *xq = ws->m1_xq;
    for (size_t i = 0; i < n_in; i++) {
        const float q  = x[i] * act_scale;
        int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
        if (qi > 127)
            qi = 127;
        if (qi < -128)
            qi = -128;
        xq[i] = (int8_t) qi;
    }

#if defined(__ARM_NEON)
    /* Precompute per-block sum-of-activations for the deferred -1 bias.
     * One int32 per 256-element block. Reused across all n_out rows.
     * File-scope TLS so backend destroy can release on demand. */
    if (ws->m1_bsum_cap < blocks_per_row) {
        safe_free((void **) &ws->m1_bsum);
        ws->m1_bsum = heap_alloc_array_aligned(int32_t, blocks_per_row);
        if (ws->m1_bsum == nullptr) {
            ws->m1_bsum_cap = 0;
            memset(y, 0, n_out * sizeof *y);
            return;
        }
        ws->m1_bsum_cap = blocks_per_row;
    }
    for (size_t b = 0; b < blocks_per_row; b++) {
        ws->m1_bsum[b] = q8a_block_bsum(xq + b * 256);
    }
    /* Snapshot thread-local pointers so worker threads (whose own
     * _Thread_local storage is NULL) see the main thread's buffers. */
    const int32_t *const bsum_cache = ws->m1_bsum;
#endif

    struct q8a_m1_ctx ctx = {
            .W  = W,
            .xq = xq,
            .y  = y,
#if defined(__ARM_NEON)
            .bsum_cache = bsum_cache,
#endif
            .inv_act_scale  = inv_act_scale,
            .row_bytes      = row_bytes,
            .blocks_per_row = blocks_per_row,
    };

    /* Dispatch:
     *   - GEIST_PP=1            → custom spin-pool (geist_pp_parallel_for)
     *   - else, in OMP team     → omp for (work-share, no team spawn)
     *   - else, standalone OMP  → omp parallel for (spawn a team)
     *   - else                  → serial */
    const int pp_enabled = tq2_pp_enabled();
    if (pp_enabled) {
        geist_pp_parallel_for(n_out, q8a_m1_row_body, &ctx);
    }
#ifdef _OPENMP
    else if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t r = 0; r < n_out; r++)
            q8a_m1_row_body(r, &ctx);
    }
#endif
    else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t r = 0; r < n_out; r++)
            q8a_m1_row_body(r, &ctx);
    }
    /* xq, bsum point to thread-local cache; nothing to free. */
}

/* P3.13: W1.58 × A8 prefill kernel (M>1). Per-input-row int8 quant
 * (each of the m activation rows has its own absmax/scale), then for
 * each (output_row, input_row) cell the dot reuses the same NEON
 * block kernel as M=1.
 *
 * Loop order: outer parallel-over-output-rows; inner serial-over-
 * input-rows. Each output row r streams its weight row through L1
 * once and dots against all m input rows in turn — good cache reuse
 * on the weight bytes (the dominant memory traffic). Each thread
 * holds its own (xq, inv_scales) scratch.
 *
 * y layout: [m, n_out] row-major (same convention as cblas_sgemm). */
struct q8a_mN_ctx {
    const uint8_t *W;
    const int8_t  *xq;
#if defined(__ARM_NEON)
    const int32_t *bsum_cache;
#endif
    const float *inv_scales;
    float       *y;
    size_t       m;
    size_t       n_in;
    size_t       n_out;
    size_t       blocks_per_row;
    size_t       row_bytes;
};

static void q8a_mN_row_body(size_t r, void *vctx) {
    const struct q8a_mN_ctx *c   = (const struct q8a_mN_ctx *) vctx;
    const uint8_t           *Wr  = c->W + r * c->row_bytes;
    const size_t             bpr = c->blocks_per_row;
    size_t                   i   = 0;
#if defined(__ARM_NEON)
    /* MT=4 token tiles: unpack each weight block once, dot against 4 tokens
     * (amortizes the 2-bit unpack 4x). Bit-identical to the per-token path. */
    for (; i + 4 <= c->m; i += 4) {
        const int8_t  *x0  = c->xq + (i + 0) * c->n_in;
        const int8_t  *x1  = c->xq + (i + 1) * c->n_in;
        const int8_t  *x2  = c->xq + (i + 2) * c->n_in;
        const int8_t  *x3  = c->xq + (i + 3) * c->n_in;
        const int32_t *bs0 = c->bsum_cache + (i + 0) * bpr;
        const int32_t *bs1 = c->bsum_cache + (i + 1) * bpr;
        const int32_t *bs2 = c->bsum_cache + (i + 2) * bpr;
        const int32_t *bs3 = c->bsum_cache + (i + 3) * bpr;
        float          rs0 = 0.0f, rs1 = 0.0f, rs2 = 0.0f, rs3 = 0.0f;
        for (size_t b = 0; b < bpr; b++) {
            const uint8_t *qs     = Wr + b * 66;
            const uint16_t d_bits = (uint16_t) qs[64] | ((uint16_t) qs[65] << 8);
            const float    d      = fp16_to_fp32(d_bits);
            int32_t        dots[4];
            tq2_0_block_dot_q8a_neon_unbiased_mt4(
                    qs, x0 + b * 256, x1 + b * 256, x2 + b * 256, x3 + b * 256, dots);
            rs0 += (float) (dots[0] - bs0[b]) * d;
            rs1 += (float) (dots[1] - bs1[b]) * d;
            rs2 += (float) (dots[2] - bs2[b]) * d;
            rs3 += (float) (dots[3] - bs3[b]) * d;
        }
        c->y[(i + 0) * c->n_out + r] = rs0 * c->inv_scales[i + 0];
        c->y[(i + 1) * c->n_out + r] = rs1 * c->inv_scales[i + 1];
        c->y[(i + 2) * c->n_out + r] = rs2 * c->inv_scales[i + 2];
        c->y[(i + 3) * c->n_out + r] = rs3 * c->inv_scales[i + 3];
    }
#endif
    for (; i < c->m; i++) {
        const int8_t *xqi     = c->xq + i * c->n_in;
        float         row_sum = 0.0f;
        for (size_t b = 0; b < bpr; b++) {
            const uint8_t *qs     = Wr + b * 66;
            const uint16_t d_bits = (uint16_t) qs[64] | ((uint16_t) qs[65] << 8);
            const float    d      = fp16_to_fp32(d_bits);
#if defined(__ARM_NEON)
            const int32_t dot_raw = tq2_0_block_dot_q8a_neon_unbiased(qs, xqi + b * 256);
            const int32_t dot     = dot_raw - c->bsum_cache[i * bpr + b];
#else
            int32_t dot = 0;
            for (size_t j = 0; j < 64; j += 32) {
                const size_t elem_base = (j == 0) ? 0 : 128;
                for (size_t l = 0; l < 4; l++) {
                    const int    shift = (int) (l * 2);
                    const size_t off   = elem_base + 32 * l;
                    for (size_t mm = 0; mm < 32; mm++) {
                        const int trit = (int) ((qs[j + mm] >> shift) & 3) - 1;
                        dot += trit * (int) xqi[b * 256 + off + mm];
                    }
                }
            }
#endif
            row_sum += (float) dot * d;
        }
        c->y[i * c->n_out + r] = row_sum * c->inv_scales[i];
    }
}

void cpu_neon_w_tq2_0_q8a_mN(const float               *x,
                             const struct geist_weight *w,
                             size_t                     m,
                             struct geist_backend      *be,
                             float                     *y) {
    struct cpu_neon_workspace *ws             = &((struct cpu_neon_state *) be->state)->workspace;
    const size_t               n_in           = (size_t) w->n_in;
    const size_t               n_out          = (size_t) w->n_out;
    const size_t               blocks_per_row = n_in / 256;
    const size_t               row_bytes      = blocks_per_row * 66;
    const uint8_t             *W              = (const uint8_t *) w->raw;
    /* Match the m-cap guard every sibling mN/prefill kernel enforces: the OMP
     * panel below indexes a fixed-size ytile[GEIST_QUANT_M_CAP * TQ2_NC]. */
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;

    /* Per-row int8 quant of x. Each row has its own absmax. xq is
     * [m, n_in] int8; inv_scales[i] = max_abs_row_i / 127. File-scope
     * TLS so backend destroy can release on demand (review #4 / V12). */
    const size_t xq_need = m * n_in;
    if (ws->mN_xq_cap < xq_need) {
        safe_free((void **) &ws->mN_xq);
        ws->mN_xq = heap_alloc_array_aligned(int8_t, xq_need);
        if (ws->mN_xq == nullptr) {
            ws->mN_xq_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        ws->mN_xq_cap = xq_need;
    }
    if (ws->mN_sc_cap < m) {
        safe_free((void **) &ws->mN_sc);
        ws->mN_sc = heap_alloc_array_aligned(float, m);
        if (ws->mN_sc == nullptr) {
            ws->mN_sc_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        ws->mN_sc_cap = m;
    }
    int8_t *xq         = ws->mN_xq;
    float  *inv_scales = ws->mN_sc;

    /* Per-(activation-row, K-block) bsum for the deferred -1 bias.
     * Layout: bsum[i * blocks_per_row + b]. Built once before fanning
     * out output rows to OMP workers. */
#if defined(__ARM_NEON)
    const size_t bsum_need_mN = m * blocks_per_row;
    if (ws->mN_bsum_cap < bsum_need_mN) {
        safe_free((void **) &ws->mN_bsum);
        ws->mN_bsum = heap_alloc_array_aligned(int32_t, bsum_need_mN);
        if (ws->mN_bsum == nullptr) {
            ws->mN_bsum_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        ws->mN_bsum_cap = bsum_need_mN;
    }
    int32_t *const bsum_cache_mN = ws->mN_bsum;
#endif

    for (size_t i = 0; i < m; i++) {
        const float *xi      = x + i * n_in;
        float        max_abs = 1e-5f;
        for (size_t k = 0; k < n_in; k++) {
            const float a = xi[k] < 0.0f ? -xi[k] : xi[k];
            if (a > max_abs)
                max_abs = a;
        }
        const float act_scale = 127.0f / max_abs;
        inv_scales[i]         = max_abs / 127.0f;
        int8_t *xqi           = xq + i * n_in;
        for (size_t k = 0; k < n_in; k++) {
            const float q  = xi[k] * act_scale;
            int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
            if (qi > 127)
                qi = 127;
            if (qi < -128)
                qi = -128;
            xqi[k] = (int8_t) qi;
        }
#if defined(__ARM_NEON)
        for (size_t b = 0; b < blocks_per_row; b++) {
            bsum_cache_mN[i * blocks_per_row + b] = q8a_block_bsum(xqi + b * 256);
        }
#endif
    }

    const struct q8a_mN_ctx ctx = {
            .W  = W,
            .xq = xq,
#if defined(__ARM_NEON)
            .bsum_cache = bsum_cache_mN,
#endif
            .inv_scales     = inv_scales,
            .y              = y,
            .m              = m,
            .n_in           = n_in,
            .n_out          = n_out,
            .blocks_per_row = blocks_per_row,
            .row_bytes      = row_bytes,
    };

    const int pp_enabled = tq2_pp_enabled();
#if defined(__ARM_NEON) && defined(_OPENMP)
    if (!pp_enabled) {
/* Loop-reordered NC-row panels (q4_K cache-blocking applied to ternary):
 * per panel, loop K-blocks OUTER and output rows INNER so each block's
 * m-token activation (m*256 B) is loaded once and reused L1-resident
 * across all NC rows — eliminating the per-row activation re-stream.
 * Bit-identical: same MT=4 dots, same per-token block-order float
 * accumulation; only the row/block loop nest is reordered. */
#define TQ2_NC 32
        const size_t n_panels = (n_out + (size_t) TQ2_NC - 1) / (size_t) TQ2_NC;
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t p = 0; p < n_panels; p++) {
            const size_t nc0 = p * (size_t) TQ2_NC;
            const size_t nc  = (n_out - nc0 < (size_t) TQ2_NC) ? (n_out - nc0) : (size_t) TQ2_NC;
            float        ytile[GEIST_QUANT_M_CAP * TQ2_NC];
            for (size_t t = 0; t < m * (size_t) TQ2_NC; t++)
                ytile[t] = 0.0f;
            for (size_t b = 0; b < blocks_per_row; b++) {
                for (size_t rl = 0; rl < nc; rl++) {
                    const uint8_t *qs     = W + (nc0 + rl) * row_bytes + b * 66;
                    const uint16_t d_bits = (uint16_t) qs[64] | ((uint16_t) qs[65] << 8);
                    const float    d      = fp16_to_fp32(d_bits);
                    size_t         i      = 0;
                    for (; i + 8 <= m; i += 8) {
                        const int8_t *xb[8] = {
                                xq + (i + 0) * n_in + b * 256,
                                xq + (i + 1) * n_in + b * 256,
                                xq + (i + 2) * n_in + b * 256,
                                xq + (i + 3) * n_in + b * 256,
                                xq + (i + 4) * n_in + b * 256,
                                xq + (i + 5) * n_in + b * 256,
                                xq + (i + 6) * n_in + b * 256,
                                xq + (i + 7) * n_in + b * 256,
                        };
                        int32_t dots[8];
                        tq2_0_block_dot_q8a_neon_unbiased_mt8(qs, xb, dots);
                        for (int k = 0; k < 8; k++)
                            ytile[(i + (size_t) k) * (size_t) TQ2_NC + rl] +=
                                    (float) (dots[k] -
                                             bsum_cache_mN[(i + (size_t) k) * blocks_per_row + b]) *
                                    d;
                    }
                    for (; i + 4 <= m; i += 4) {
                        int32_t dots[4];
                        tq2_0_block_dot_q8a_neon_unbiased_mt4(qs,
                                                              xq + (i + 0) * n_in + b * 256,
                                                              xq + (i + 1) * n_in + b * 256,
                                                              xq + (i + 2) * n_in + b * 256,
                                                              xq + (i + 3) * n_in + b * 256,
                                                              dots);
                        ytile[(i + 0) * (size_t) TQ2_NC + rl] +=
                                (float) (dots[0] - bsum_cache_mN[(i + 0) * blocks_per_row + b]) * d;
                        ytile[(i + 1) * (size_t) TQ2_NC + rl] +=
                                (float) (dots[1] - bsum_cache_mN[(i + 1) * blocks_per_row + b]) * d;
                        ytile[(i + 2) * (size_t) TQ2_NC + rl] +=
                                (float) (dots[2] - bsum_cache_mN[(i + 2) * blocks_per_row + b]) * d;
                        ytile[(i + 3) * (size_t) TQ2_NC + rl] +=
                                (float) (dots[3] - bsum_cache_mN[(i + 3) * blocks_per_row + b]) * d;
                    }
                    for (; i < m; i++) {
                        const int32_t dot =
                                tq2_0_block_dot_q8a_neon_unbiased(qs, xq + i * n_in + b * 256) -
                                bsum_cache_mN[i * blocks_per_row + b];
                        ytile[i * (size_t) TQ2_NC + rl] += (float) dot * d;
                    }
                }
            }
            for (size_t i = 0; i < m; i++)
                for (size_t rl = 0; rl < nc; rl++)
                    y[i * n_out + (nc0 + rl)] = ytile[i * (size_t) TQ2_NC + rl] * inv_scales[i];
        }
#undef TQ2_NC
        return;
    }
#endif
    if (pp_enabled) {
        geist_pp_parallel_for_grain(n_out, 4, q8a_mN_row_body, (void *) &ctx);
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_out; r++) {
        q8a_mN_row_body(r, (void *) &ctx);
    }
}

/* P3.10 fp32 fallback for pre-ARMv8.2 (no dotprod) hosts. Always compiled
 * so the resolver can select it at load time on dotprod-built binaries
 * that end up running on a non-dotprod CPU (cross-build / emulator). */
void cpu_neon_w_tq2_0_m1(const float               *x,
                         const struct geist_weight *w,
                         struct geist_backend      *be,
                         float                     *y) {
    (void) be;
    const size_t   n_in           = (size_t) w->n_in;
    const size_t   n_out          = (size_t) w->n_out;
    const size_t   blocks_per_row = n_in / 256;
    const size_t   row_bytes      = blocks_per_row * 66;
    const uint8_t *W              = (const uint8_t *) w->raw;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_out; r++) {
        const uint8_t *Wr      = W + r * row_bytes;
        float          row_sum = 0.0f;
        for (size_t b = 0; b < blocks_per_row; b++) {
            const uint8_t *qs     = Wr + b * 66;
            const uint16_t d_bits = (uint16_t) qs[64] | ((uint16_t) qs[65] << 8);
            const float    d      = fp16_to_fp32(d_bits);
            const float   *xb     = x + b * 256;
#if defined(__ARM_NEON)
            row_sum += d * tq2_0_block_dot_neon(qs, xb);
#else
            float block_acc = 0.0f;
            for (size_t j = 0; j < 64; j += 32) {
                const size_t elem_base = (j == 0) ? 0 : 128;
                for (size_t l = 0; l < 4; l++) {
                    const int    shift = (int) (l * 2);
                    const size_t off   = elem_base + 32 * l;
                    for (size_t m = 0; m < 32; m++) {
                        const int trit = (int) ((qs[j + m] >> shift) & 3) - 1;
                        block_acc += (float) trit * xb[off + m];
                    }
                }
            }
            row_sum += d * block_acc;
#endif
        }
        y[r] = row_sum;
    }
}

/* ======================= I2_S (BitNet b1.58 official) ======================
 * Microsoft distributes BitNet-2B-4T only as i2_s. It is ternary like TQ2_0
 * (same {0,1,2}={-1,0,+1} trit codes, same 256-elem/64-byte blocks, same int8
 * SDOT compute) and differs in exactly two ways:
 *   (1) the four 2-bit fields within each byte are in REVERSE order — i2_s
 *       stores element 32*g+b at shift (6-2g); TQ2_0 stores it at shift 2g; and
 *   (2) there is NO per-block scale — ONE f32 per-TENSOR scale lives at the tail
 *       of the tensor data (offset n_in*n_out/4) and is applied once per row.
 * So we reuse the whole W1.58×A8 machinery (activation quant, per-block bsum
 * for the −1 bias, the dispatch) and only swap the shift↔activation pairing and
 * fold a single scale at the end. Source: BitNet/src/ggml-bitnet-mad.cpp
 * quantize_i2_s (one scale_ptr[0] per tensor).
 *
 * NOTE: reads the scale at w->raw + n_in*n_out/4, i.e. just past the packed
 * bytes — valid under the default mmap-alias load (the bytes are in the mmap);
 * the β/arena copy path would need to also copy the 4-byte scale tail. */
#if defined(__ARM_NEON)
static inline int32_t i2_s_block_dot_q8a_neon_unbiased(const uint8_t *qs, const int8_t *xb) {
    const uint8x16_t three = vdupq_n_u8(3);
    int32x4_t        acc0  = vdupq_n_s32(0);
    int32x4_t        acc1  = vdupq_n_s32(0);
    for (int h = 0; h < 2; h++) {
        const uint8x16_t pa   = vld1q_u8(qs + h * 32 + 0);
        const uint8x16_t pb   = vld1q_u8(qs + h * 32 + 16);
        const size_t     xo   = (size_t) h * 128;
        const int8x16_t  s_a0 = vreinterpretq_s8_u8(vandq_u8(pa, three));
        const int8x16_t  s_b0 = vreinterpretq_s8_u8(vandq_u8(pb, three));
        const int8x16_t  s_a2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 2), three));
        const int8x16_t  s_b2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 2), three));
        const int8x16_t  s_a4 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 4), three));
        const int8x16_t  s_b4 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 4), three));
        const int8x16_t  s_a6 = vreinterpretq_s8_u8(vshrq_n_u8(pa, 6));
        const int8x16_t  s_b6 = vreinterpretq_s8_u8(vshrq_n_u8(pb, 6));
        /* Reversed shift↔offset pairing vs TQ2_0: shift6→elems 0..31,
         * shift4→32..63, shift2→64..95, shift0→96..127. */
        acc0 = vdotq_s32(acc0, s_a6, vld1q_s8(xb + xo + 0));
        acc1 = vdotq_s32(acc1, s_b6, vld1q_s8(xb + xo + 16));
        acc0 = vdotq_s32(acc0, s_a4, vld1q_s8(xb + xo + 32));
        acc1 = vdotq_s32(acc1, s_b4, vld1q_s8(xb + xo + 48));
        acc0 = vdotq_s32(acc0, s_a2, vld1q_s8(xb + xo + 64));
        acc1 = vdotq_s32(acc1, s_b2, vld1q_s8(xb + xo + 80));
        acc0 = vdotq_s32(acc0, s_a0, vld1q_s8(xb + xo + 96));
        acc1 = vdotq_s32(acc1, s_b0, vld1q_s8(xb + xo + 112));
    }
    return vaddvq_s32(vaddq_s32(acc0, acc1));
}
#endif

static inline float i2_s_tensor_scale(const struct geist_weight *w) {
    const size_t packed = (size_t) w->n_in * (size_t) w->n_out / 4;
    float        s;
    memcpy(&s, (const uint8_t *) w->raw + packed, sizeof s);
    return s;
}

struct i2s_m1_ctx {
    const uint8_t *W;
    const int8_t  *xq;
#if defined(__ARM_NEON)
    const int32_t *bsum_cache;
#endif
    float *y;
    float  scale;     /* tensor_scale * inv_act_scale */
    size_t row_bytes; /* blocks_per_row * 64 (no per-block scale) */
    size_t blocks_per_row;
};

static void i2s_m1_row_body(size_t r, void *vctx) {
    const struct i2s_m1_ctx *c   = (const struct i2s_m1_ctx *) vctx;
    const uint8_t           *Wr  = c->W + r * c->row_bytes;
    int64_t                  acc = 0;
    for (size_t b = 0; b < c->blocks_per_row; b++) {
        const uint8_t *qs = Wr + b * 64;
#if defined(__ARM_NEON)
        const int32_t dot_raw = i2_s_block_dot_q8a_neon_unbiased(qs, c->xq + b * 256);
        acc += (int64_t) (dot_raw - c->bsum_cache[b]);
#else
        const int8_t *xb = c->xq + b * 256;
        for (size_t h = 0; h < 2; h++) {
            for (size_t bb = 0; bb < 32; bb++) {
                const uint8_t byte = qs[h * 32 + bb];
                for (size_t g = 0; g < 4; g++) {
                    const int trit = (int) ((byte >> (6 - 2 * g)) & 3) - 1;
                    acc += (int64_t) trit * (int64_t) xb[h * 128 + g * 32 + bb];
                }
            }
        }
#endif
    }
    c->y[r] = (float) acc * c->scale;
}

void cpu_neon_w_i2_s_q8a_m1(const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y) {
    struct cpu_neon_workspace *ws             = &((struct cpu_neon_state *) be->state)->workspace;
    const size_t               n_in           = (size_t) w->n_in;
    const size_t               n_out          = (size_t) w->n_out;
    const size_t               blocks_per_row = n_in / 256;
    const size_t               row_bytes      = blocks_per_row * 64;
    const uint8_t             *W              = (const uint8_t *) w->raw;

    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > max_abs)
            max_abs = a;
    }
    const float act_scale     = 127.0f / max_abs;
    const float inv_act_scale = max_abs / 127.0f;

    if (ws->m1_xq_cap < n_in) {
        safe_free((void **) &ws->m1_xq);
        ws->m1_xq = heap_alloc_array_aligned(int8_t, n_in);
        if (ws->m1_xq == nullptr) {
            ws->m1_xq_cap = 0;
            memset(y, 0, n_out * sizeof *y);
            return;
        }
        ws->m1_xq_cap = n_in;
    }
    int8_t *xq = ws->m1_xq;
    for (size_t i = 0; i < n_in; i++) {
        const float q  = x[i] * act_scale;
        int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
        if (qi > 127)
            qi = 127;
        if (qi < -128)
            qi = -128;
        xq[i] = (int8_t) qi;
    }

#if defined(__ARM_NEON)
    if (ws->m1_bsum_cap < blocks_per_row) {
        safe_free((void **) &ws->m1_bsum);
        ws->m1_bsum = heap_alloc_array_aligned(int32_t, blocks_per_row);
        if (ws->m1_bsum == nullptr) {
            ws->m1_bsum_cap = 0;
            memset(y, 0, n_out * sizeof *y);
            return;
        }
        ws->m1_bsum_cap = blocks_per_row;
    }
    for (size_t b = 0; b < blocks_per_row; b++) {
        ws->m1_bsum[b] = q8a_block_bsum(xq + b * 256);
    }
    const int32_t *const bsum_cache = ws->m1_bsum;
#endif

    struct i2s_m1_ctx ctx = {
            .W  = W,
            .xq = xq,
            .y  = y,
#if defined(__ARM_NEON)
            .bsum_cache = bsum_cache,
#endif
            .scale          = i2_s_tensor_scale(w) * inv_act_scale,
            .row_bytes      = row_bytes,
            .blocks_per_row = blocks_per_row,
    };

    const int pp_enabled = tq2_pp_enabled();
    if (pp_enabled) {
        geist_pp_parallel_for(n_out, i2s_m1_row_body, &ctx);
    }
#ifdef _OPENMP
    else if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t r = 0; r < n_out; r++)
            i2s_m1_row_body(r, &ctx);
    }
#endif
    else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t r = 0; r < n_out; r++)
            i2s_m1_row_body(r, &ctx);
    }
}

/* MT=4 i2_s block dot — like the tq2_0 mt4 but with the reversed shift↔offset
 * pairing. Unpacks one weight block once and dots it against 4 tokens. */
#if defined(__ARM_NEON)
static inline void i2_s_block_dot_q8a_neon_unbiased_mt4(const uint8_t *qs,
                                                        const int8_t  *xb0,
                                                        const int8_t  *xb1,
                                                        const int8_t  *xb2,
                                                        const int8_t  *xb3,
                                                        int32_t        out[4]) {
    const uint8x16_t three = vdupq_n_u8(3);
    int32x4_t        a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t        a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
#define I2S_MT4_POS(S, XOFF)                                            \
    do {                                                                \
        const int8x16_t _w = (S);                                       \
        a0                 = vdotq_s32(a0, _w, vld1q_s8(xb0 + (XOFF))); \
        a1                 = vdotq_s32(a1, _w, vld1q_s8(xb1 + (XOFF))); \
        a2                 = vdotq_s32(a2, _w, vld1q_s8(xb2 + (XOFF))); \
        a3                 = vdotq_s32(a3, _w, vld1q_s8(xb3 + (XOFF))); \
    } while (0)
    for (int h = 0; h < 2; h++) {
        const uint8x16_t pa = vld1q_u8(qs + h * 32 + 0);
        const uint8x16_t pb = vld1q_u8(qs + h * 32 + 16);
        const size_t     xo = (size_t) h * 128;
        I2S_MT4_POS(vreinterpretq_s8_u8(vshrq_n_u8(pa, 6)), xo + 0);
        I2S_MT4_POS(vreinterpretq_s8_u8(vshrq_n_u8(pb, 6)), xo + 16);
        I2S_MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 4), three)), xo + 32);
        I2S_MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 4), three)), xo + 48);
        I2S_MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 2), three)), xo + 64);
        I2S_MT4_POS(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 2), three)), xo + 80);
        I2S_MT4_POS(vreinterpretq_s8_u8(vandq_u8(pa, three)), xo + 96);
        I2S_MT4_POS(vreinterpretq_s8_u8(vandq_u8(pb, three)), xo + 112);
    }
#undef I2S_MT4_POS
    out[0] = vaddvq_s32(a0);
    out[1] = vaddvq_s32(a1);
    out[2] = vaddvq_s32(a2);
    out[3] = vaddvq_s32(a3);
}
#endif

/* MT=8 i2_s block dot — 8 tokens per weight-block unpack (reversed shifts).
 * ACCUMULATES into the caller's int32x4 lanes ACROSS blocks (no per-block
 * horizontal reduce — the caller reduces once per row), keeping the vdotq
 * pipeline full. This is the structural prefill win. */
#if defined(__ARM_NEON)
static inline void
i2_s_block_dot_q8a_mt8_accum(const uint8_t *qs, const int8_t *const xb[8], int32x4_t acc[8]) {
    const uint8x16_t three = vdupq_n_u8(3);
#define I2S_MT8A(S, XOFF)                                                     \
    do {                                                                      \
        const int8x16_t _w = (S);                                             \
        acc[0]             = vdotq_s32(acc[0], _w, vld1q_s8(xb[0] + (XOFF))); \
        acc[1]             = vdotq_s32(acc[1], _w, vld1q_s8(xb[1] + (XOFF))); \
        acc[2]             = vdotq_s32(acc[2], _w, vld1q_s8(xb[2] + (XOFF))); \
        acc[3]             = vdotq_s32(acc[3], _w, vld1q_s8(xb[3] + (XOFF))); \
        acc[4]             = vdotq_s32(acc[4], _w, vld1q_s8(xb[4] + (XOFF))); \
        acc[5]             = vdotq_s32(acc[5], _w, vld1q_s8(xb[5] + (XOFF))); \
        acc[6]             = vdotq_s32(acc[6], _w, vld1q_s8(xb[6] + (XOFF))); \
        acc[7]             = vdotq_s32(acc[7], _w, vld1q_s8(xb[7] + (XOFF))); \
    } while (0)
    for (int h = 0; h < 2; h++) {
        const uint8x16_t pa = vld1q_u8(qs + h * 32 + 0);
        const uint8x16_t pb = vld1q_u8(qs + h * 32 + 16);
        const size_t     xo = (size_t) h * 128;
        I2S_MT8A(vreinterpretq_s8_u8(vshrq_n_u8(pa, 6)), xo + 0);
        I2S_MT8A(vreinterpretq_s8_u8(vshrq_n_u8(pb, 6)), xo + 16);
        I2S_MT8A(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 4), three)), xo + 32);
        I2S_MT8A(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 4), three)), xo + 48);
        I2S_MT8A(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pa, 2), three)), xo + 64);
        I2S_MT8A(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pb, 2), three)), xo + 80);
        I2S_MT8A(vreinterpretq_s8_u8(vandq_u8(pa, three)), xo + 96);
        I2S_MT8A(vreinterpretq_s8_u8(vandq_u8(pb, three)), xo + 112);
    }
#undef I2S_MT8A
}
#endif

struct i2s_mN_ctx {
    const uint8_t *W;
    const int8_t  *xq;
    const int32_t *bsum_cache;
    const float   *inv_scales;
    float         *y;
    float          scale; /* per-tensor i2_s scale */
    size_t         m, n_in, n_out, blocks_per_row, row_bytes;
};

static void i2s_mN_row_body(size_t r, void *vctx) {
    const struct i2s_mN_ctx *c   = (const struct i2s_mN_ctx *) vctx;
    const uint8_t           *Wr  = c->W + r * c->row_bytes;
    const size_t             bpr = c->blocks_per_row;
    const float              sc  = c->scale;
    size_t                   i   = 0;
#if defined(__ARM_NEON)
    /* mt8 is the sweet spot: mt16 (16 accumulators) spills NEON registers on
     * the A76 and regresses (measured 31 vs 48 tps). The unpack overhead is
     * instruction-count-bound (~2 IPC) but can't be amortized wider here. */
    for (; i + 8 <= c->m; i += 8) {
        const int8_t  *xb[8];
        const int32_t *bs[8];
        int32_t        bsum_tot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int j = 0; j < 8; j++) {
            xb[j] = c->xq + (i + (size_t) j) * c->n_in;
            bs[j] = c->bsum_cache + (i + (size_t) j) * bpr;
        }
        int32x4_t acc[8];
        for (int j = 0; j < 8; j++)
            acc[j] = vdupq_n_s32(0);
        for (size_t b = 0; b < bpr; b++) {
            const uint8_t *qs = Wr + b * 64;
            const int8_t  *xbb[8];
            for (int j = 0; j < 8; j++) {
                xbb[j] = xb[j] + b * 256;
                bsum_tot[j] += bs[j][b];
            }
            i2_s_block_dot_q8a_mt8_accum(qs, xbb, acc);
        }
        for (int j = 0; j < 8; j++)
            c->y[(i + (size_t) j) * c->n_out + r] =
                    (float) (vaddvq_s32(acc[j]) - bsum_tot[j]) * sc * c->inv_scales[i + (size_t) j];
    }
    for (; i + 4 <= c->m; i += 4) {
        const int8_t  *x0 = c->xq + (i + 0) * c->n_in, *x1 = c->xq + (i + 1) * c->n_in;
        const int8_t  *x2 = c->xq + (i + 2) * c->n_in, *x3 = c->xq + (i + 3) * c->n_in;
        const int32_t *bs0  = c->bsum_cache + (i + 0) * bpr;
        const int32_t *bs1  = c->bsum_cache + (i + 1) * bpr;
        const int32_t *bs2  = c->bsum_cache + (i + 2) * bpr;
        const int32_t *bs3  = c->bsum_cache + (i + 3) * bpr;
        int32_t        acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        for (size_t b = 0; b < bpr; b++) {
            const uint8_t *qs = Wr + b * 64;
            int32_t        dots[4];
            i2_s_block_dot_q8a_neon_unbiased_mt4(
                    qs, x0 + b * 256, x1 + b * 256, x2 + b * 256, x3 + b * 256, dots);
            acc0 += dots[0] - bs0[b];
            acc1 += dots[1] - bs1[b];
            acc2 += dots[2] - bs2[b];
            acc3 += dots[3] - bs3[b];
        }
        c->y[(i + 0) * c->n_out + r] = (float) acc0 * sc * c->inv_scales[i + 0];
        c->y[(i + 1) * c->n_out + r] = (float) acc1 * sc * c->inv_scales[i + 1];
        c->y[(i + 2) * c->n_out + r] = (float) acc2 * sc * c->inv_scales[i + 2];
        c->y[(i + 3) * c->n_out + r] = (float) acc3 * sc * c->inv_scales[i + 3];
    }
#endif
    for (; i < c->m; i++) {
        const int8_t *xqi = c->xq + i * c->n_in;
        int64_t       acc = 0;
        for (size_t b = 0; b < bpr; b++) {
            const uint8_t *qs = Wr + b * 64;
#if defined(__ARM_NEON)
            acc += i2_s_block_dot_q8a_neon_unbiased(qs, xqi + b * 256) - c->bsum_cache[i * bpr + b];
#else
            for (size_t h = 0; h < 2; h++)
                for (size_t bb = 0; bb < 32; bb++) {
                    const uint8_t byte = qs[h * 32 + bb];
                    for (size_t g = 0; g < 4; g++) {
                        const int trit = (int) ((byte >> (6 - 2 * g)) & 3) - 1;
                        acc += (int64_t) trit * xqi[b * 256 + h * 128 + g * 32 + bb];
                    }
                }
#endif
        }
        c->y[i * c->n_out + r] = (float) acc * sc * c->inv_scales[i];
    }
}

/* M>1 prefill: per-row int8 activation quant (shared across output rows), then
 * mt4 token-tiled dots reusing each weight row once. Mirrors tq2_0/q8a_mN. */
void cpu_neon_w_i2_s_q8a_mN(const float               *x,
                            const struct geist_weight *w,
                            size_t                     m,
                            struct geist_backend      *be,
                            float                     *y) {
    struct cpu_neon_workspace *ws             = &((struct cpu_neon_state *) be->state)->workspace;
    const size_t               n_in           = (size_t) w->n_in;
    const size_t               n_out          = (size_t) w->n_out;
    const size_t               blocks_per_row = n_in / 256;
    const size_t               row_bytes      = blocks_per_row * 64;
    const uint8_t             *W              = (const uint8_t *) w->raw;
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;

    const size_t xq_need = m * n_in;
    if (ws->mN_xq_cap < xq_need) {
        safe_free((void **) &ws->mN_xq);
        ws->mN_xq = heap_alloc_array_aligned(int8_t, xq_need);
        if (ws->mN_xq == nullptr) {
            ws->mN_xq_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        ws->mN_xq_cap = xq_need;
    }
    if (ws->mN_sc_cap < m) {
        safe_free((void **) &ws->mN_sc);
        ws->mN_sc = heap_alloc_array_aligned(float, m);
        if (ws->mN_sc == nullptr) {
            ws->mN_sc_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        ws->mN_sc_cap = m;
    }
    int8_t *xq         = ws->mN_xq;
    float  *inv_scales = ws->mN_sc;

#if defined(__ARM_NEON)
    const size_t bsum_need = m * blocks_per_row;
    if (ws->mN_bsum_cap < bsum_need) {
        safe_free((void **) &ws->mN_bsum);
        ws->mN_bsum = heap_alloc_array_aligned(int32_t, bsum_need);
        if (ws->mN_bsum == nullptr) {
            ws->mN_bsum_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        ws->mN_bsum_cap = bsum_need;
    }
    int32_t *const bsum_cache = ws->mN_bsum;
#endif

    for (size_t i = 0; i < m; i++) {
        const float *xi      = x + i * n_in;
        float        max_abs = 1e-5f;
        for (size_t k = 0; k < n_in; k++) {
            const float a = xi[k] < 0.0f ? -xi[k] : xi[k];
            if (a > max_abs)
                max_abs = a;
        }
        const float act_scale = 127.0f / max_abs;
        inv_scales[i]         = max_abs / 127.0f;
        int8_t *xqi           = xq + i * n_in;
        for (size_t k = 0; k < n_in; k++) {
            const float q  = xi[k] * act_scale;
            int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
            if (qi > 127)
                qi = 127;
            if (qi < -128)
                qi = -128;
            xqi[k] = (int8_t) qi;
        }
#if defined(__ARM_NEON)
        for (size_t b = 0; b < blocks_per_row; b++) {
            bsum_cache[i * blocks_per_row + b] = q8a_block_bsum(xqi + b * 256);
        }
#endif
    }

    const struct i2s_mN_ctx ctx = {
            .W  = W,
            .xq = xq,
#if defined(__ARM_NEON)
            .bsum_cache = bsum_cache,
#endif
            .inv_scales     = inv_scales,
            .y              = y,
            .scale          = i2_s_tensor_scale(w),
            .m              = m,
            .n_in           = n_in,
            .n_out          = n_out,
            .blocks_per_row = blocks_per_row,
            .row_bytes      = row_bytes,
    };

    const int pp_enabled = tq2_pp_enabled();
    if (pp_enabled) {
        geist_pp_parallel_for(n_out, i2s_mN_row_body, (void *) &ctx);
    }
#ifdef _OPENMP
    else if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t r = 0; r < n_out; r++)
            i2s_mN_row_body(r, (void *) &ctx);
    }
#endif
    else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t r = 0; r < n_out; r++)
            i2s_mN_row_body(r, (void *) &ctx);
    }
}
