/* Shared frame of the register-tiled GEMM kernels: one output row per
 * hardware subgroup; each lane dequantizes its weights ONCE per block into
 * registers and reuses them for every batch row of the tile, and the tile
 * ends in one subgroupAdd per batch row. Assumes subgroup size 32 (the host
 * loops the size-agnostic matvec on other sizes).
 * Included after exactly one of:
 *   DT_Q4_0 / DT_Q4_1 / DT_Q8_0  32-element blocks; layouts as in
 *                                mv_legacy.glsl
 *   DT_TQ2_0  layout and lane mapping as in matvec_tq2_0.comp
 *   DT_PQ2_0  layout and lane mapping as in matvec_pq2_0.comp (16 lanes per
 *             128-element block, two blocks per step)
 *   DT_Q4K    native 144-byte superblocks (d, dmin, 12 scale bytes, 128 qs);
 *             lane 0 loads the header and broadcasts it
 *   DT_Q5K    native 176-byte superblocks (d, dmin, 12 scale bytes, 32 qh,
 *             128 qs); lane mapping and value formula as in matvec_q5k.comp
 *   DT_Q6K    native 210-byte superblocks padded to 216 bytes; not 4-aligned
 *             in the file, so ql/qh stay byte-assembled
 *   DT_F32    plain f32 rows (vec4 loads); requires n_in % 8 == 0
 * Geometry: 8 rows x 32 batch rows per 256-thread workgroup, dispatch
 * gx = ceil(n_out / 8), gy = ceil(rows / 32); DT_Q6K and DT_F32 use 4 rows x
 * 16 batch rows per 128 threads, gx = ceil(n_out / 4), gy = ceil(rows / 16). */
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#if defined(DT_Q4K)
#extension GL_KHR_shader_subgroup_ballot : enable
#endif
#include "vk_limits.h"

#if defined(DT_Q6K) || defined(DT_F32)
layout(local_size_x = VK_MM_SMALL_ROWS_PER_WG * 32u) in; /* one row per 32-lane subgroup */

const uint TM = VK_MM_SMALL_TOKENS_PER_WG; /* batch rows per workgroup */
#else
layout(local_size_x = VK_MM_ROWS_PER_WG * 32u) in; /* one row per 32-lane subgroup */

const uint TM = VK_MM_TOKENS_PER_WG; /* batch rows per workgroup */
#endif

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
#if defined(DT_F32)
layout(set = 0, binding = 1) readonly buffer W { vec4 w4[]; };
#else
layout(set = 0, binding = 1) readonly buffer W { uint w[]; };
#endif
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
const uint LPB = 8u;
#elif defined(DT_Q4_0) || defined(DT_Q4_1)
const uint LPB = 4u;
#endif
#if defined(DT_Q4_0) || defined(DT_Q4_1) || defined(DT_Q8_0)
const uint BPI = 32u / LPB;
#endif

#if defined(DT_Q4_0) || defined(DT_Q8_0)
float block_scale(uint bi, uint sc0) {
    uint word = w[sc0 + (bi >> 1u)];
    return unpackHalf2x16(word >> ((bi & 1u) * 16u)).x;
}
#elif defined(DT_TQ2_0)
vec4 trits(uint q, uint sh) {
    return vec4(float((q >> sh) & 3u) - 1.0,
                float((q >> (sh + 8u)) & 3u) - 1.0,
                float((q >> (sh + 16u)) & 3u) - 1.0,
                float((q >> (sh + 24u)) & 3u) - 1.0);
}
#elif defined(DT_PQ2_0)
vec4 trits(uint bits, uint sh) {
    return vec4(float((bits >> sh) & 3u) - 1.0,
                float((bits >> (sh + 2u)) & 3u) - 1.0,
                float((bits >> (sh + 4u)) & 3u) - 1.0,
                float((bits >> (sh + 6u)) & 3u) - 1.0);
}
#elif defined(DT_Q6K)
uint load_u8(uint byte_offset) {
    return (w[byte_offset >> 2] >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

float load_f16(uint byte_offset) {
    uint lo = load_u8(byte_offset);
    uint hi = load_u8(byte_offset + 1u);
    return unpackHalf2x16(lo | (hi << 8u)).x;
}

int load_i8(uint byte_offset) {
    return int(load_u8(byte_offset) ^ 0x80u) - 0x80;
}
#endif

void main() {
    uint row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint t0 = gl_WorkGroupID.y * TM;
    if (t0 >= pc.rows) {
        return;
    }
    uint tm = min(TM, pc.rows - t0);
#if defined(DT_TQ2_0)
    uint g = lane >> 4u;
    uint l = (lane >> 2u) & 3u;
    uint m0 = (lane & 3u) * 8u;
    uint sh = 2u * l;
    uint nb = pc.blocks_per_row;
    uint sc0 = pc.n_out * nb * 16u;
#elif defined(DT_PQ2_0)
    uint ib = lane >> 4u;
    uint wq = lane & 15u;
    uint nb = pc.blocks_per_row;
    uint sc0 = pc.n_out * nb * 8u;
#elif defined(DT_Q4_0) || defined(DT_Q4_1) || defined(DT_Q8_0)
    uint nb = pc.blocks_per_row;
    uint ib = lane / LPB;
    uint wq = lane % LPB;
#endif

    float s[TM];
    for (uint t = 0u; t < TM; ++t) {
        s[t] = 0.0;
    }
#if defined(DT_Q4_0)
    uint sc0 = pc.n_out * nb * 4u;
#elif defined(DT_Q8_0)
    uint sc0 = pc.n_out * nb * 8u;
#endif

    if (row < pc.n_out) {
#if defined(DT_TQ2_0)
        for (uint b = 0u; b < nb; ++b) {
            uint bi = row * nb + b;
            uint qw = bi * 16u + g * 8u + (m0 >> 2u);
            float d = unpackHalf2x16(w[sc0 + (bi >> 1u)] >> ((bi & 1u) * 16u)).x;
            vec4 wv0 = d * trits(w[qw], sh);
            vec4 wv1 = d * trits(w[qw + 1u], sh);
            uint e_base = pc.x_offset + b * 256u + g * 128u + l * 32u + m0;
            for (uint t = 0u; t < TM; ++t) {
                if (t >= tm) {
                    break;
                }
                uint xv = (e_base + (t0 + t) * pc.x_stride) >> 2u;
                s[t] += dot(wv0, x4[xv]) + dot(wv1, x4[xv + 1u]);
            }
        }
#elif defined(DT_PQ2_0)
        for (uint b0 = 0u; b0 < nb; b0 += 2u) {
            uint b = b0 + ib;
            if (b < nb) {
                uint bi = row * nb + b;
                uint bits = (w[bi * 8u + (wq >> 1u)] >> ((wq & 1u) * 16u)) & 0xffffu;
                float d = unpackHalf2x16(w[sc0 + (bi >> 1u)] >> ((bi & 1u) * 16u)).x;
                vec4 wv0 = d * trits(bits, 0u);
                vec4 wv1 = d * trits(bits, 8u);
                uint e_base = pc.x_offset + b * 128u + wq * 8u;
                for (uint t = 0u; t < TM; ++t) {
                    if (t >= tm) {
                        break;
                    }
                    uint xv = (e_base + (t0 + t) * pc.x_stride) >> 2u;
                    s[t] += dot(wv0, x4[xv]) + dot(wv1, x4[xv + 1u]);
                }
            }
        }
#elif defined(DT_Q4K)
        uint sub = lane >> 2; /* int shift literals as in the pre-fold kernel: same SPIR-V */
        uint idx8 = (lane & 3u) * 8u;
        uint shift = (sub & 1u) * 4u;
        uint row_word = (pc.w_offset + row * pc.blocks_per_row * 144u) >> 2u;

        for (uint b = 0u; b < pc.blocks_per_row; ++b) {
            uint blk = row_word + b * 36u;

            uint h0 = 0u;
            uint h1 = 0u;
            uint h2 = 0u;
            uint h3 = 0u;
            if (lane == 0u) {
                h0 = w[blk];
                h1 = w[blk + 1u];
                h2 = w[blk + 2u];
                h3 = w[blk + 3u];
            }
            h0 = subgroupBroadcast(h0, 0);
            h1 = subgroupBroadcast(h1, 0);
            h2 = subgroupBroadcast(h2, 0);
            h3 = subgroupBroadcast(h3, 0);

            vec2 dd = unpackHalf2x16(h0);
            uint sc_u;
            uint mn_u;
            if (sub < 4u) {
                sc_u = (h1 >> (8u * sub)) & 63u;
                mn_u = (h2 >> (8u * sub)) & 63u;
            } else {
                uint j = sub - 4u;
                uint q_j4 = (h3 >> (8u * j)) & 0xffu;
                sc_u = (q_j4 & 15u) | ((((h1 >> (8u * j)) & 0xffu) >> 6u) << 4u);
                mn_u = (q_j4 >> 4u) | ((((h2 >> (8u * j)) & 0xffu) >> 6u) << 4u);
            }

            uint qs = blk + 4u + (sub >> 1u) * 8u + (idx8 >> 2u);
            uint q0 = w[qs];
            uint q1 = w[qs + 1u];

            /* All x runs are 8-float aligned (pool 64 B, idx8 multiple of
             * 8), so the inner product is two vec4 dots. */
            float dsc = dd.x * float(sc_u);
            float dmn = dd.y * float(mn_u);
            vec4 wv0 = dsc * vec4(float((q0 >> shift) & 15u),
                                  float((q0 >> (8u + shift)) & 15u),
                                  float((q0 >> (16u + shift)) & 15u),
                                  float((q0 >> (24u + shift)) & 15u)) -
                       vec4(dmn);
            vec4 wv1 = dsc * vec4(float((q1 >> shift) & 15u),
                                  float((q1 >> (8u + shift)) & 15u),
                                  float((q1 >> (16u + shift)) & 15u),
                                  float((q1 >> (24u + shift)) & 15u)) -
                       vec4(dmn);

            uint e_base = pc.x_offset + b * 256u + sub * 32u + idx8;
            for (uint t = 0u; t < TM; ++t) {
                if (t >= tm) {
                    break;
                }
                uint xv = (e_base + (t0 + t) * pc.x_stride) >> 2u;
                s[t] += dot(wv0, x4[xv]) + dot(wv1, x4[xv + 1u]);
            }
        }
#elif defined(DT_Q6K)
        uint e0 = lane * 8u;
        uint half_idx = e0 >> 7;
        uint in_half = e0 & 127u;
        uint stream = in_half >> 5;
        uint l0 = in_half & 31u;
        uint ql_shift = (stream >= 2u) ? 4u : 0u;
        uint qh_shift = stream * 2u;
        uint sc_idx = stream * 2u + (l0 >> 4);
        uint row_byte = pc.w_offset + row * pc.blocks_per_row * 216u;

        for (uint b = 0u; b < pc.blocks_per_row; ++b) {
            uint block_byte = row_byte + b * 216u;
            uint ql_byte = block_byte + half_idx * 64u + (stream & 1u) * 32u + l0;
            uint qh_byte = block_byte + 128u + half_idx * 32u + l0;
            float d = load_f16(block_byte + 208u);
            int sc = load_i8(block_byte + 192u + half_idx * 8u + sc_idx);
            float dsc = d * float(sc);

            vec4 q_lo;
            vec4 q_hi;
            for (uint j = 0u; j < 4u; ++j) {
                q_lo[j] = float(((load_u8(ql_byte + j) >> ql_shift) & 15u) |
                                (((load_u8(qh_byte + j) >> qh_shift) & 3u) << 4u)) -
                          32.0;
                q_hi[j] = float(((load_u8(ql_byte + 4u + j) >> ql_shift) & 15u) |
                                (((load_u8(qh_byte + 4u + j) >> qh_shift) & 3u) << 4u)) -
                          32.0;
            }
            vec4 wv0 = dsc * q_lo;
            vec4 wv1 = dsc * q_hi;

            uint e_base = pc.x_offset + b * 256u + e0;
            for (uint t = 0u; t < TM; ++t) {
                if (t >= tm) {
                    break;
                }
                uint xv = (e_base + (t0 + t) * pc.x_stride) >> 2u;
                s[t] += dot(wv0, x4[xv]) + dot(wv1, x4[xv + 1u]);
            }
        }
#elif defined(DT_F32)
        uint w_row = (pc.w_offset + row * pc.n_in) >> 2u;
        /* lane covers elements [lane*8, lane*8+8) each 256-element stripe */
        for (uint e = lane * 8u; e < pc.n_in; e += 256u) {
            vec4 wv0 = w4[w_row + (e >> 2u)];
            vec4 wv1 = w4[w_row + (e >> 2u) + 1u];
            uint e_base = pc.x_offset + e;
            for (uint t = 0u; t < TM; ++t) {
                if (t >= tm) {
                    break;
                }
                uint xv = (e_base + (t0 + t) * pc.x_stride) >> 2u;
                s[t] += dot(wv0, x4[xv]) + dot(wv1, x4[xv + 1u]);
            }
        }
#elif defined(DT_Q5K)
        uint sub = lane >> 2u;
        uint idx8 = (lane & 3u) * 8u;
        uint shift = (sub & 1u) * 4u;
        uint row_word = (pc.w_offset >> 2u) + row * pc.blocks_per_row * 44u;

        for (uint b = 0u; b < pc.blocks_per_row; ++b) {
            uint blk = row_word + b * 44u;
            uint h0 = w[blk];
            uint h1 = w[blk + 1u];
            uint h2 = w[blk + 2u];
            uint h3 = w[blk + 3u];
            vec2 dd = unpackHalf2x16(h0);
            uint sc_u;
            uint mn_u;
            if (sub < 4u) {
                sc_u = (h1 >> (8u * sub)) & 63u;
                mn_u = (h2 >> (8u * sub)) & 63u;
            } else {
                uint j = sub - 4u;
                uint q_j4 = (h3 >> (8u * j)) & 0xffu;
                sc_u = (q_j4 & 15u) | ((((h1 >> (8u * j)) & 0xffu) >> 6u) << 4u);
                mn_u = (q_j4 >> 4u) | ((((h2 >> (8u * j)) & 0xffu) >> 6u) << 4u);
            }
            uint qh0 = w[blk + 4u + (idx8 >> 2u)];
            uint qh1 = w[blk + 5u + (idx8 >> 2u)];
            uint qs = blk + 12u + (sub >> 1u) * 8u + (idx8 >> 2u);
            uint q0 = w[qs];
            uint q1 = w[qs + 1u];

            float dsc = dd.x * float(sc_u);
            float dmn = dd.y * float(mn_u);
            vec4 wv0 = dsc * vec4(float(((q0 >> shift) & 15u) | (((qh0 >> sub) & 1u) << 4u)),
                                  float(((q0 >> (8u + shift)) & 15u) | (((qh0 >> (8u + sub)) & 1u) << 4u)),
                                  float(((q0 >> (16u + shift)) & 15u) | (((qh0 >> (16u + sub)) & 1u) << 4u)),
                                  float(((q0 >> (24u + shift)) & 15u) | (((qh0 >> (24u + sub)) & 1u) << 4u))) -
                       vec4(dmn);
            vec4 wv1 = dsc * vec4(float(((q1 >> shift) & 15u) | (((qh1 >> sub) & 1u) << 4u)),
                                  float(((q1 >> (8u + shift)) & 15u) | (((qh1 >> (8u + sub)) & 1u) << 4u)),
                                  float(((q1 >> (16u + shift)) & 15u) | (((qh1 >> (16u + sub)) & 1u) << 4u)),
                                  float(((q1 >> (24u + shift)) & 15u) | (((qh1 >> (24u + sub)) & 1u) << 4u))) -
                       vec4(dmn);

            uint e_base = pc.x_offset + b * 256u + sub * 32u + idx8;
            for (uint t = 0u; t < TM; ++t) {
                if (t >= tm) {
                    break;
                }
                uint xv = (e_base + (t0 + t) * pc.x_stride) >> 2u;
                s[t] += dot(wv0, x4[xv]) + dot(wv1, x4[xv + 1u]);
            }
        }
#else
        for (uint b0 = 0u; b0 < nb; b0 += BPI) {
            uint b = b0 + ib;
            if (b < nb) {
                uint bi = row * nb + b;
                uint ebase = pc.x_offset + b * 32u + wq * 4u; /* element of the low part */
#if defined(DT_Q8_0)
                int qw = int(w[bi * 8u + wq]);
                float d = block_scale(bi, sc0);
                vec4 wv = d * vec4(float(bitfieldExtract(qw, 0, 8)), float(bitfieldExtract(qw, 8, 8)),
                                   float(bitfieldExtract(qw, 16, 8)), float(bitfieldExtract(qw, 24, 8)));
                for (uint t = 0u; t < TM; ++t) {
                    if (t >= tm) {
                        break;
                    }
                    uint xv = (ebase + (t0 + t) * pc.x_stride) >> 2u;
                    s[t] += dot(wv, x4[xv]);
                }
#else
                vec4 wlo;
                vec4 whi;
#if defined(DT_Q4_0)
                uint qw = w[bi * 4u + wq];
                float d = block_scale(bi, sc0);
                wlo = d * (vec4(unpack8(qw & 0x0F0F0F0Fu)) - vec4(8.0));
                whi = d * (vec4(unpack8((qw >> 4u) & 0x0F0F0F0Fu)) - vec4(8.0));
#else
                vec2 dm = unpackHalf2x16(w[bi * 5u]);
                uint qw = w[bi * 5u + 1u + wq];
                wlo = dm.x * vec4(unpack8(qw & 0x0F0F0F0Fu)) + vec4(dm.y);
                whi = dm.x * vec4(unpack8((qw >> 4u) & 0x0F0F0F0Fu)) + vec4(dm.y);
#endif
                for (uint t = 0u; t < TM; ++t) {
                    if (t >= tm) {
                        break;
                    }
                    uint xv = (ebase + (t0 + t) * pc.x_stride) >> 2u;
                    s[t] += dot(wlo, x4[xv]) + dot(whi, x4[xv + 4u]);
                }
#endif
            }
        }
#endif
    }

    for (uint t = 0u; t < TM; ++t) {
        if (t >= tm) {
            break;
        }
        float r = subgroupAdd(s[t]);
        if (subgroupElect() && row < pc.n_out) {
            y[pc.y_offset + (t0 + t) * pc.y_stride + row] = r;
        }
    }
}
