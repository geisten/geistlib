/* Shared body of the 32-element-block matvec kernels (Q4_0, Q4_1, Q8_0).
 * Included by matvec_q4_0.comp / matvec_q4_1.comp / matvec_q8_0.comp after
 * they define exactly one of DT_Q4_0 / DT_Q4_1 / DT_Q8_0.
 *
 * Weight layouts (all little-endian words):
 *   Q4_0  struct-of-arrays, built at resolve time: every block's 16 quant
 *         bytes back to back (4 words/block), then one f16 scale per block
 *         (two per word). 18 bytes/block — no padding, and each lane's word
 *         load is coalesced.
 *   Q4_1  native 20-byte block: word0 = d | m << 16, words 1..4 = quants.
 *   Q8_0  struct-of-arrays like Q4_0: 32 quant bytes/block (8 words), then
 *         the f16 scales.
 * 4-bit quants: byte i holds element i (low nibble) and element i+16 (high).
 *
 * Mapping: a 32-lane "warp" (derived from the thread id, not the hardware
 * subgroup, so any subgroup size works) covers 32 / LPB blocks per step, LPB
 * lanes per block; each lane owns one quant word. 4 rows per warp reuse the
 * x loads; the workgroup is 2 warps = 8 rows. Dispatch: gx = ceil(n_out / 8).
 * Reduction is a shared-memory tree. */

layout(local_size_x = 64) in;

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

#if defined(DT_Q8_0)
const uint LPB = 8u; /* lanes per block: one quant word (4 int8) each */
#else
const uint LPB = 4u; /* one quant word (8 nibbles) each */
#endif
const uint BPI = 32u / LPB; /* blocks per step per warp */
const uint NUM_ROWS = 4u;
shared float tmpsh[2][NUM_ROWS][32];

#if defined(DT_Q4_0) || defined(DT_Q8_0)
float block_scale(uint bi, uint sc0) {
    uint word = w[sc0 + (bi >> 1u)];
    return unpackHalf2x16(word >> ((bi & 1u) * 16u)).x;
}
#endif

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint wrp = tid >> 5u;
    uint lane = tid & 31u;
    uint first_row = gl_WorkGroupID.x * 8u + wrp * NUM_ROWS;
    uint ib = lane / LPB;
    uint wq = lane % LPB;
    uint nb = pc.blocks_per_row;
    uint num_rows = first_row < pc.n_out ? min(NUM_ROWS, pc.n_out - first_row) : 0u;

    float temp[NUM_ROWS];
    for (uint n = 0u; n < NUM_ROWS; ++n) {
        temp[n] = 0.0;
    }
    uint xbase = pc.x_offset >> 2u;
#if defined(DT_Q4_0)
    uint sc0 = pc.n_out * nb * 4u;
#elif defined(DT_Q8_0)
    uint sc0 = pc.n_out * nb * 8u;
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
            uint qw = w[bi * 4u + wq];
            float d = block_scale(bi, sc0);
            vec4 lo = vec4(unpack8(qw & 0x0F0F0F0Fu));
            vec4 hi = vec4(unpack8((qw >> 4u) & 0x0F0F0F0Fu));
            temp[n] = fma(d, dot(lo, xlo) + dot(hi, xhi) - 8.0 * xs, temp[n]);
#elif defined(DT_Q4_1)
            vec2 dm = unpackHalf2x16(w[bi * 5u]);
            uint qw = w[bi * 5u + 1u + wq];
            vec4 lo = vec4(unpack8(qw & 0x0F0F0F0Fu));
            vec4 hi = vec4(unpack8((qw >> 4u) & 0x0F0F0F0Fu));
            temp[n] = fma(dm.x, dot(lo, xlo) + dot(hi, xhi), fma(dm.y, xs, temp[n]));
#else
            int qw = int(w[bi * 8u + wq]);
            float d = block_scale(bi, sc0);
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
        y[pc.y_offset + first_row + lane] = tmpsh[wrp][lane][0];
    }
}
