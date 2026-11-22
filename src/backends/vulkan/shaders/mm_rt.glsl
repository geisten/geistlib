#extension GL_EXT_control_flow_attributes : enable

/* GEMM in a register tile for GPUs without cooperative matrices (#588):
 * the bindings and the push block of the register-tiled GEMMs
 * (mm_legacy.glsl), the tile of the grouped expert GEMM (moe_mm_body.glsl):
 * BM = 128 weight rows x BN = 64 batch rows x BK = 32 per 256-thread
 * workgroup, the weights dequantized and the batch rows copied into shared
 * memory once per k-step, then multiplied in registers -- thread (tr, tc) =
 * (lid % 16, lid / 16) owns rows tr + 16 i (i < 8) x batch rows 4 tc ..
 * 4 tc + 3, and skips the multiplies when its 4 batch rows are past the
 * tile's count. mm_legacy.glsl reads every batch row from the scratch buffer
 * once per weight row; this reads it once per 128.
 *
 * The tiles are f32, as mm_legacy.glsl computes (the tensor-core tiles and
 * the grouped expert GEMM round to f16): 27 KiB of shared memory in one
 * buffer pair, so the next k-step's loads are in flight during the
 * multiplies and stored between two barriers. No subgroup operations: any
 * subgroup size.
 *
 * Included after exactly one of DT_Q4K, DT_Q5K, DT_Q6K (the 216-byte padded
 * blocks), DT_Q8_0 (struct-of-arrays, mv_legacy.glsl); the weight layouts
 * are those of the dense kernels. Weight rows past n_out read as zero and
 * are not written. n_in % 32 == 0, x offsets and strides are multiples of 4
 * floats, the weight's byte offset is a multiple of 16 (the host guards
 * them). Dispatch: gx = ceil(rows / BN), gy = ceil(n_out / BM). */

#include "vk_limits.h"

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
/* the weight, three views of one binding */
layout(set = 0, binding = 1) readonly buffer W1 { uint v[]; } w1;
layout(set = 0, binding = 1) readonly buffer W2 { uvec2 v[]; } w2;
layout(set = 0, binding = 1) readonly buffer W4 { uvec4 v[]; } w4;
layout(set = 0, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform Push {
    uint n_in;
    uint n_out;
    uint blocks_per_row;
    uint rows;
    uint x_offset;
    uint w_offset; /* bytes */
    uint y_offset;
    uint x_stride;
    uint y_stride;
} pc;

const uint BM = VK_MOE_MM_ROWS;
const uint BN = VK_MOE_MM_BN;
const uint BK = 32u;
/* 8 vectors of data + 1 of padding per 32-k row */
const uint STRIDE = BK / 4u + 1u;
const uint TM = BM / 16u; /* rows per thread */
const uint TN = BN / 16u; /* batch rows per thread */

shared vec4 Ash[BM * STRIDE];
shared vec4 Bsh[BN * STRIDE];

/* 4 quants of word qw (and their high bits) -> 4 weights */
#if defined(DT_Q4K)
vec4 deq4(uint qw, uint shift, float dsc, float dmn) {
    uvec4 q = (uvec4(qw) >> (uvec4(0u, 8u, 16u, 24u) + shift)) & 15u;
    return vec4(q) * dsc - dmn;
}
#elif defined(DT_Q5K)
vec4 deq4(uint qw, uint hw, uint shift, uint sub, float dsc, float dmn) {
    uvec4 sh = uvec4(0u, 8u, 16u, 24u);
    uvec4 q = ((uvec4(qw) >> (sh + shift)) & 15u) | (((uvec4(hw) >> (sh + sub)) & 1u) << 4u);
    return vec4(q) * dsc - dmn;
}
#elif defined(DT_Q6K)
vec4 deq4(uint l, uint h, uint qls, uint qhs, float dsc) {
    uvec4 sh = uvec4(0u, 8u, 16u, 24u);
    uvec4 q = ((uvec4(l) >> (sh + qls)) & 15u) | (((uvec4(h) >> (sh + qhs)) & 3u) << 4u);
    return dsc * (vec4(q) - 32.0);
}
#elif defined(DT_Q8_0)
vec4 deq4(uint qw, float d) {
    ivec4 q = ivec4(uvec4(qw) << uvec4(24u, 16u, 8u, 0u)) >> 24; /* sign-extending */
    return d * vec4(q);
}
#endif

/* Global -> registers for k-step ks (in flight during the multiplies of the
 * step before); registers -> shared afterwards (store_tiles). Thread lid
 * holds half a k-step (16 weights) of row lid / 2 and a quarter (8 values)
 * of batch row lid / 4. */
struct Fetch {
#if defined(DT_Q4K)
    uvec4 hdr;
    uvec4 qs;
    uint sub;
#elif defined(DT_Q5K)
    uvec4 hdr;
    uvec4 qh;
    uvec4 qs;
    uint sub;
#elif defined(DT_Q6K)
    uvec4 ql;
    uvec4 qh;
    uint scw;
    uint dw;
    uint sub;
#elif defined(DT_Q8_0)
    uvec4 qs;
    float d;
#endif
    vec4 x0;
    vec4 x1;
};

Fetch fetch_tiles(uint ks, uint lid, uint row0, uint t0, uint cnt) {
    uint k0 = ks * BK;
    uint r = lid >> 1u;
    uint hk = lid & 1u;
    uint row = min(row0 + r, pc.n_out - 1u); /* rows past n_out are zeroed at the store */
    Fetch f;
#if defined(DT_Q4K)
    uint bi = row * pc.blocks_per_row + (k0 >> 8u);
    uint b4 = (pc.w_offset >> 4u) + bi * 9u; /* 144-byte blocks = 9 uvec4 */
    f.sub = (k0 & 255u) >> 5u;
    f.hdr = w4.v[b4];
    f.qs = w4.v[b4 + 1u + (f.sub >> 1u) * 2u + hk];
#elif defined(DT_Q5K)
    uint bi = row * pc.blocks_per_row + (k0 >> 8u);
    uint b11 = (pc.w_offset >> 4u) + bi * 11u; /* 176 bytes: header, qh[32], qs[128] */
    f.sub = (k0 & 255u) >> 5u;
    f.hdr = w4.v[b11];
    f.qh = w4.v[b11 + 1u + hk];
    f.qs = w4.v[b11 + 3u + (f.sub >> 1u) * 2u + hk];
#elif defined(DT_Q6K)
    uint bi = row * pc.blocks_per_row + (k0 >> 8u);
    uint b2 = (pc.w_offset >> 3u) + bi * 27u; /* 216-byte padded blocks = 27 uvec2 */
    f.sub = (k0 & 255u) >> 5u;
    uint half_idx = f.sub >> 2u;
    uint stream = f.sub & 3u;
    uint qlv = b2 + half_idx * 8u + (stream & 1u) * 4u + hk * 2u;
    f.ql = uvec4(w2.v[qlv], w2.v[qlv + 1u]);
    uint qhv = b2 + 16u + half_idx * 4u + hk * 2u;
    f.qh = uvec4(w2.v[qhv], w2.v[qhv + 1u]);
    f.scw = w2.v[b2 + 24u + half_idx][stream >> 1u];
    f.dw = w2.v[b2 + 26u].x;
#elif defined(DT_Q8_0)
    uint bi = row * pc.blocks_per_row + ks;
    f.qs = w4.v[(pc.w_offset >> 4u) + bi * 2u + hk];
    uint sc0 = (pc.w_offset >> 2u) + pc.n_out * pc.blocks_per_row * 8u;
    f.d = unpackHalf2x16(w1.v[sc0 + (bi >> 1u)] >> ((bi & 1u) * 16u)).x;
#endif
    /* B: batch row c = lid >> 2, its 8 k from (lid & 3) * 8 */
    uint c = lid >> 2u;
    f.x0 = vec4(0.0);
    f.x1 = vec4(0.0);
    if (c < cnt) {
        uint xv = (pc.x_offset + (t0 + c) * pc.x_stride + k0 + (lid & 3u) * 8u) >> 2u;
        f.x0 = x4[xv];
        f.x1 = x4[xv + 1u];
    }
    return f;
}

void store_tiles(uint lid, uint row0, Fetch f) {
    uint r = lid >> 1u;
    uint hk = lid & 1u;
    uint abase = r * STRIDE + hk * 4u;
    vec4 a0, a1, a2, a3;
#if defined(DT_Q4K) || defined(DT_Q5K)
    uint sub = f.sub;
    vec2 dd = unpackHalf2x16(f.hdr.x);
    uint sc_u, mn_u;
    if (sub < 4u) {
        sc_u = (f.hdr.y >> (8u * sub)) & 63u;
        mn_u = (f.hdr.z >> (8u * sub)) & 63u;
    } else {
        uint j = sub - 4u;
        uint q_j4 = (f.hdr.w >> (8u * j)) & 0xffu;
        sc_u = (q_j4 & 15u) | ((((f.hdr.y >> (8u * j)) & 0xffu) >> 6u) << 4u);
        mn_u = (q_j4 >> 4u) | ((((f.hdr.z >> (8u * j)) & 0xffu) >> 6u) << 4u);
    }
    float dsc = dd.x * float(sc_u);
    float dmn = dd.y * float(mn_u);
    uint shift = (sub & 1u) * 4u;
#if defined(DT_Q4K)
    a0 = deq4(f.qs.x, shift, dsc, dmn);
    a1 = deq4(f.qs.y, shift, dsc, dmn);
    a2 = deq4(f.qs.z, shift, dsc, dmn);
    a3 = deq4(f.qs.w, shift, dsc, dmn);
#else
    a0 = deq4(f.qs.x, f.qh.x, shift, sub, dsc, dmn);
    a1 = deq4(f.qs.y, f.qh.y, shift, sub, dsc, dmn);
    a2 = deq4(f.qs.z, f.qh.z, shift, sub, dsc, dmn);
    a3 = deq4(f.qs.w, f.qh.w, shift, sub, dsc, dmn);
#endif
#elif defined(DT_Q6K)
    uint stream = f.sub & 3u;
    float dsc = unpackHalf2x16(f.dw).x *
                float(bitfieldExtract(int(f.scw), int(((stream * 2u + hk) & 3u) * 8u), 8));
    uint qls = stream >= 2u ? 4u : 0u;
    uint qhs = stream * 2u;
    a0 = deq4(f.ql.x, f.qh.x, qls, qhs, dsc);
    a1 = deq4(f.ql.y, f.qh.y, qls, qhs, dsc);
    a2 = deq4(f.ql.z, f.qh.z, qls, qhs, dsc);
    a3 = deq4(f.ql.w, f.qh.w, qls, qhs, dsc);
#elif defined(DT_Q8_0)
    a0 = deq4(f.qs.x, f.d);
    a1 = deq4(f.qs.y, f.d);
    a2 = deq4(f.qs.z, f.d);
    a3 = deq4(f.qs.w, f.d);
#endif
    if (row0 + r >= pc.n_out) {
        a0 = vec4(0.0);
        a1 = vec4(0.0);
        a2 = vec4(0.0);
        a3 = vec4(0.0);
    }
    Ash[abase] = a0;
    Ash[abase + 1u] = a1;
    Ash[abase + 2u] = a2;
    Ash[abase + 3u] = a3;
    uint bbase = (lid >> 2u) * STRIDE + (lid & 3u) * 2u;
    Bsh[bbase] = f.x0;
    Bsh[bbase + 1u] = f.x1;
}

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint t0 = gl_WorkGroupID.x * BN;
    if (t0 >= pc.rows) {
        return;
    }
    uint cnt = min(BN, pc.rows - t0);
    uint row0 = gl_WorkGroupID.y * BM;
    uint tr = lid & 15u;
    uint tc = lid >> 4u;
    bool busy = tc * TN < cnt; /* the same for the 16 lanes of a row group */
    float acc[TM * TN];
    [[unroll]] for (uint i = 0u; i < TM * TN; i++) {
        acc[i] = 0.0;
    }

    uint ksteps = pc.n_in / BK;
    store_tiles(lid, row0, fetch_tiles(0u, lid, row0, t0, cnt));
    barrier();

    for (uint ks = 0u; ks < ksteps; ks++) {
        Fetch nxt;
        bool more = ks + 1u < ksteps;
        if (more) {
            nxt = fetch_tiles(ks + 1u, lid, row0, t0, cnt);
        }
        if (busy) {
            [[unroll]] for (uint q = 0u; q < BK / 4u; q++) {
                vec4 b[TN];
                [[unroll]] for (uint j = 0u; j < TN; j++) {
                    b[j] = Bsh[(tc * TN + j) * STRIDE + q];
                }
                [[unroll]] for (uint i = 0u; i < TM; i++) {
                    vec4 a = Ash[(tr + 16u * i) * STRIDE + q];
                    [[unroll]] for (uint j = 0u; j < TN; j++) {
                        acc[i * TN + j] += dot(a, b[j]);
                    }
                }
            }
        }
        if (more) {
            barrier();
            store_tiles(lid, row0, nxt);
        }
        barrier();
    }

    [[unroll]] for (uint j = 0u; j < TN; j++) {
        uint c = tc * TN + j;
        if (c < cnt) {
            uint yb = pc.y_offset + (t0 + c) * pc.y_stride;
            [[unroll]] for (uint i = 0u; i < TM; i++) {
                uint r = row0 + tr + 16u * i;
                if (r < pc.n_out) {
                    y[yb + r] = acc[i * TN + j];
                }
            }
        }
    }
}
