/* Shared body of the 32-element-block GEMM kernels (Q4_0, Q4_1, Q8_0);
 * layouts as in mv_legacy.glsl. Register-tiled like matmul_q4k: one output
 * row per hardware subgroup, 8 rows x 32 batch rows per 256-thread
 * workgroup. Each lane dequantizes its weights ONCE per block into registers
 * and reuses them for every batch row of the tile. Assumes subgroup size 32
 * (the host loops the size-agnostic matvec on other sizes).
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
#else
const uint LPB = 4u;
#endif
const uint BPI = 32u / LPB;

#if defined(DT_Q4_0) || defined(DT_Q8_0)
float block_scale(uint bi, uint sc0) {
    uint word = w[sc0 + (bi >> 1u)];
    return unpackHalf2x16(word >> ((bi & 1u) * 16u)).x;
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
    uint nb = pc.blocks_per_row;
    uint ib = lane / LPB;
    uint wq = lane % LPB;

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
