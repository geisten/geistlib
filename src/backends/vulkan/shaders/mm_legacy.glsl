/* Shared frame of the register-tiled GEMM kernels: one output row per
 * hardware subgroup, 8 rows x 32 batch rows per 256-thread workgroup. Each
 * lane dequantizes its weights ONCE per block into registers and reuses them
 * for every batch row of the tile. Assumes subgroup size 32 (the host loops
 * the size-agnostic matvec on other sizes).
 * Included after exactly one of:
 *   DT_Q4_0 / DT_Q4_1 / DT_Q8_0  32-element blocks; layouts as in
 *                                mv_legacy.glsl
 *   DT_TQ2_0  layout and lane mapping as in matvec_tq2_0.comp
 *   DT_PQ2_0  layout and lane mapping as in matvec_pq2_0.comp (16 lanes per
 *             128-element block, two blocks per step)
 *   DT_Q5K    native 176-byte superblocks (d, dmin, 12 scale bytes, 32 qh,
 *             128 qs); lane mapping and value formula as in matvec_q5k.comp
 * Dispatch: gx = ceil(n_out / 8), gy = ceil(rows / 32). */
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

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
#endif

void main() {
    uint row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint t0 = gl_WorkGroupID.y * 32u;
    if (t0 >= pc.rows) {
        return;
    }
    uint tm = min(32u, pc.rows - t0);
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
#elif !defined(DT_Q5K)
    uint nb = pc.blocks_per_row;
    uint ib = lane / LPB;
    uint wq = lane % LPB;
#endif

    float s[32];
    for (uint t = 0u; t < 32u; ++t) {
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
            for (uint t = 0u; t < 32u; ++t) {
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
                for (uint t = 0u; t < 32u; ++t) {
                    if (t >= tm) {
                        break;
                    }
                    uint xv = (e_base + (t0 + t) * pc.x_stride) >> 2u;
                    s[t] += dot(wv0, x4[xv]) + dot(wv1, x4[xv + 1u]);
                }
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
            for (uint t = 0u; t < 32u; ++t) {
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
                for (uint t = 0u; t < 32u; ++t) {
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
                for (uint t = 0u; t < 32u; ++t) {
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

    for (uint t = 0u; t < 32u; ++t) {
        if (t >= tm) {
            break;
        }
        float r = subgroupAdd(s[t]);
        if (subgroupElect() && row < pc.n_out) {
            y[pc.y_offset + (t0 + t) * pc.y_stride + row] = r;
        }
    }
}
