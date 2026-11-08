/* Shared body of the 32-element-block matvec kernels (Q4_0, Q4_1, Q8_0,
 * Q5_0, MXFP4). Included by matvec_q4_0.comp / matvec_q4_1.comp /
 * matvec_q8_0.comp / matvec_q5_0.comp / matvec_mxfp4.comp after they define
 * exactly one of DT_Q4_0 / DT_Q4_1 / DT_Q8_0 / DT_Q5_0 / DT_MXFP4, and by
 * matvec_legacy_id_body.glsl (MV_ID: the grouped and _sel variants, which
 * declare the bindings and the push block and define MV_BEGIN, W_LOAD,
 * X_BASE and Y_BASE as matvec_q4k_core.glsl's includers do).
 *
 * Weight layouts (all little-endian words):
 *   Q4_0  struct-of-arrays, built at resolve time: every block's 16 quant
 *         bytes back to back (4 words/block), then one f16 scale per block
 *         (two per word). 18 bytes/block — no padding, and each lane's word
 *         load is coalesced.
 *   Q4_1  native 20-byte block: word0 = d | m << 16, words 1..4 = quants.
 *   Q8_0  struct-of-arrays like Q4_0: 32 quant bytes/block (8 words), then
 *         the f16 scales.
 *   Q5_0  struct-of-arrays (#499): per block the 32 high bits (one word, bit
 *         j of element j) and the 16 quant bytes (4 words), then the f16
 *         scales; element value d * (q - 16).
 *   MXFP4 struct-of-arrays (#499): 16 quant bytes per block (4 words), then
 *         one E8M0 exponent byte per block (four per word); element value
 *         2^(e - 128) * the doubled E2M1 code (mxfp4_e8m0_half, quant.h).
 * 4-bit quants: byte i holds element i (low nibble) and element i+16 (high).
 *
 * Mapping: a 32-lane "warp" (derived from the thread id, not the hardware
 * subgroup, so any subgroup size works) covers 32 / LPB blocks per step, LPB
 * lanes per block; each lane owns one quant word. 4 rows per warp reuse the
 * x loads; the workgroup is 2 warps = 8 rows. Dispatch: gx = ceil(n_out / 8).
 * Reduction is a shared-memory tree.
 *
 * MV_BATCH (matvec_q8_0_mt.comp) runs one batch row per gl_WorkGroupID.y
 * (x and y advanced by x_stride / y_stride; x_stride % 4 == 0): the GEMMs
 * whose n_out is too narrow for any tensor-core tile (Qwen3.5's 16- and
 * 32-row DeltaNet alpha / beta projections, #467). The weight rows are
 * re-read per batch row, from L2. Dispatch: gx = ceil(n_out / 8), gy = rows. */

#include "vk_limits.h"

layout(local_size_x = 64) in;

#ifndef MV_ID
layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
layout(set = 0, binding = 1) readonly buffer W { uint w[]; };
layout(set = 0, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform Push {
    uint n_in;
    uint n_out;
    uint blocks_per_row;
    uint rows;
    uint x_offset;
    uint w_offset;
    uint y_offset;
    uint x_stride;
    uint y_stride;
} pc;

#define MV_BEGIN
#define W_LOAD(i) w[i]
#ifdef MV_BATCH
#define X_BASE (pc.x_offset + gl_WorkGroupID.y * pc.x_stride)
#define Y_BASE (pc.y_offset + gl_WorkGroupID.y * pc.y_stride)
#else
#define X_BASE pc.x_offset
#define Y_BASE pc.y_offset
#endif
#endif

#if defined(DT_Q8_0)
const uint LPB = 8u; /* lanes per block: one quant word (4 int8) each */
#else
const uint LPB = 4u; /* one quant word (8 nibbles) each */
#endif
const uint BPI = 32u / LPB; /* blocks per step per warp */
const uint NUM_ROWS = 4u;
shared float tmpsh[2][NUM_ROWS][32];

/* The f16 scale of block bi in the scale words from sc0 (a function where
 * the weight is a binding, as before #499: the same SPIR-V). */
#ifdef MV_ID
#define BLOCK_SCALE(bi, sc0) unpackHalf2x16(W_LOAD((sc0) + ((bi) >> 1u)) >> (((bi) & 1u) * 16u)).x
#else
float block_scale(uint bi, uint sc0) {
    uint word = w[sc0 + (bi >> 1u)];
    return unpackHalf2x16(word >> ((bi & 1u) * 16u)).x;
}
#define BLOCK_SCALE(bi, sc0) block_scale(bi, sc0)
#endif

#if defined(DT_MXFP4)
/* The doubled E2M1 value of a 4-bit code (kvalues_mxfp4): the magnitudes
 * 0 1 2 3 4 6 8 12 as nibbles of one word, bit 3 the sign. */
float mx4v(uint c) {
    float m = float((0xC8643210u >> ((c & 7u) * 4u)) & 15u);
    return (c & 8u) != 0u ? -m : m;
}
/* The codes in the low nibbles of q's four bytes. */
vec4 mx4(uint q) {
    return vec4(mx4v(q & 15u), mx4v((q >> 8u) & 15u), mx4v((q >> 16u) & 15u), mx4v((q >> 24u) & 15u));
}
/* 2^(e - 128) for the exponent byte e, the denormals at e < 2 included. */
float e8m0_half(uint e) {
    return uintBitsToFloat(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23u));
}
#endif

void main() {
    MV_BEGIN
    uint tid = gl_LocalInvocationID.x;
    uint wrp = tid >> 5u;
    uint lane = tid & 31u;
    uint first_row = gl_WorkGroupID.x * VK_MV_ROWS_PER_WG + wrp * NUM_ROWS;
    uint ib = lane / LPB;
    uint wq = lane % LPB;
    uint nb = pc.blocks_per_row;
    uint num_rows = first_row < pc.n_out ? min(NUM_ROWS, pc.n_out - first_row) : 0u;

    float temp[NUM_ROWS];
    for (uint n = 0u; n < NUM_ROWS; ++n) {
        temp[n] = 0.0;
    }
    uint xbase = (X_BASE) >> 2u;
#if defined(DT_Q4_0) || defined(DT_MXFP4)
    uint sc0 = pc.n_out * nb * 4u;
#elif defined(DT_Q8_0)
    uint sc0 = pc.n_out * nb * 8u;
#elif defined(DT_Q5_0)
    uint sc0 = pc.n_out * nb * 5u;
#endif

    for (uint b = ib; b < nb; b += BPI) {
#if defined(DT_Q8_0)
        vec4 xv = x4[xbase + b * 8u + wq];
#else
        vec4 xlo = x4[xbase + b * 8u + wq];
        vec4 xhi = x4[xbase + b * 8u + 4u + wq];
        float xs = (xlo.x + xlo.y + xlo.z + xlo.w) + (xhi.x + xhi.y + xhi.z + xhi.w);
#endif
        for (uint n = 0u; n < NUM_ROWS; ++n) {
            /* clamp keeps tail-row weight reads in bounds */
            uint bi = min(first_row + n, pc.n_out - 1u) * nb + b;
#if defined(DT_Q4_0)
            uint qw = W_LOAD(bi * 4u + wq);
            float d = BLOCK_SCALE(bi, sc0);
            vec4 lo = vec4(unpack8(qw & 0x0F0F0F0Fu));
            vec4 hi = vec4(unpack8((qw >> 4u) & 0x0F0F0F0Fu));
            temp[n] = fma(d, dot(lo, xlo) + dot(hi, xhi) - 8.0 * xs, temp[n]);
#elif defined(DT_Q4_1)
            vec2 dm = unpackHalf2x16(W_LOAD(bi * 5u));
            uint qw = W_LOAD(bi * 5u + 1u + wq);
            vec4 lo = vec4(unpack8(qw & 0x0F0F0F0Fu));
            vec4 hi = vec4(unpack8((qw >> 4u) & 0x0F0F0F0Fu));
            temp[n] = fma(dm.x, dot(lo, xlo) + dot(hi, xhi), fma(dm.y, xs, temp[n]));
#elif defined(DT_Q5_0)
            uint qh = W_LOAD(bi * 5u);
            uint qw = W_LOAD(bi * 5u + 1u + wq);
            float d = BLOCK_SCALE(bi, sc0);
            /* the high bits of elements 4wq..4wq+3 and 16+4wq.., one per byte */
            uint hl = (((qh >> (wq * 4u)) & 15u) * 0x00204081u) & 0x01010101u;
            uint hh = (((qh >> (wq * 4u + 16u)) & 15u) * 0x00204081u) & 0x01010101u;
            vec4 lo = vec4(unpack8((qw & 0x0F0F0F0Fu) | (hl << 4u)));
            vec4 hi = vec4(unpack8(((qw >> 4u) & 0x0F0F0F0Fu) | (hh << 4u)));
            temp[n] = fma(d, dot(lo, xlo) + dot(hi, xhi) - 16.0 * xs, temp[n]);
#elif defined(DT_MXFP4)
            uint qw = W_LOAD(bi * 4u + wq);
            float d = e8m0_half((W_LOAD(sc0 + (bi >> 2u)) >> ((bi & 3u) * 8u)) & 0xffu);
            temp[n] = fma(d, dot(mx4(qw), xlo) + dot(mx4(qw >> 4u), xhi), temp[n]);
#else
            int qw = int(W_LOAD(bi * 8u + wq));
            float d = BLOCK_SCALE(bi, sc0);
            vec4 q = vec4(float(bitfieldExtract(qw, 0, 8)), float(bitfieldExtract(qw, 8, 8)),
                          float(bitfieldExtract(qw, 16, 8)), float(bitfieldExtract(qw, 24, 8)));
            temp[n] = fma(d, dot(q, xv), temp[n]);
#endif
        }
    }

    for (uint n = 0u; n < NUM_ROWS; ++n) {
        tmpsh[wrp][n][lane] = temp[n];
    }
    barrier();
    for (uint s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            for (uint n = 0u; n < NUM_ROWS; ++n) {
                tmpsh[wrp][n][lane] += tmpsh[wrp][n][lane + s];
            }
        }
        barrier();
    }
    if (lane < num_rows) {
        y[(Y_BASE) + first_row + lane] = tmpsh[wrp][lane][0];
    }
}
