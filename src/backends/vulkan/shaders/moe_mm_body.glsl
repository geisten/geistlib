#if defined(MOE_MM_CM)
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#else
#extension GL_EXT_control_flow_attributes : enable
#endif
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require

/* Grouped expert GEMM (#499), llama.cpp's mul_mm_id and Metal's moe_mm
 * (#482): the routed slots of a prefill chunk, bucketed by
 * expert (moe_map.comp), each expert's tiles multiplied by its weight in one
 * dispatch. Workgroup (i, y) takes tile i of moe_map's list (past the count:
 * nothing), e | (j << 16): sorted rows j*BN .. j*BN + BN-1 of expert e, and
 * output rows y*BM .. y*BM + BM-1 of expert e's weight, read by device
 * address (tab[tab_off + e]). Slot s of a row reads x row s / x_div (k: one
 * row per token; 1: one per slot) and writes y row s.
 *
 * The frame of matmul_pq2_0_cm_body.glsl with BN = VK_MOE_MM_BN slots:
 * BM = 128 weight rows x BN slots x BK = 32 per 256-thread workgroup, the
 * next k-step's loads in flight during the multiplies and stored after them
 * (one barrier per step), weights and x rounded to f16 in shared memory,
 * f32 accumulation. Two ways to multiply the staged tiles:
 *   MOE_MM_CM (moe_mm_*_cm.comp) on the tensor cores: 8 subgroups of 32
 *     lanes in a 4 x 2 layout, each owning 32 rows x BN / 2 slots (the four
 *     accumulators unrolled by hand: an array of them indexed in a loop
 *     leaves registers, #658);
 *   otherwise (moe_mm_*.comp, #249: GPUs without cooperative matrices, any
 *     subgroup size) in registers, as llama.cpp's non-coopmat mul_mm_id:
 *     thread (tr, tc) = (lid % 16, lid / 16) owns rows tr + 16 i (i < 8) x
 *     slots 4 tc .. 4 tc + 3 -- each 8-k vector of a weight row read once
 *     per thread for its 4 slots, the 16 lanes of a row group reading 16
 *     rows without bank conflicts (the padded stride) -- and skips the
 *     multiplies when all 4 slots are past the tile's count.
 * The A stage per DT_* reads the weights as the dense
 * GEMMs lay them out: Q4_K / Q5_K superblocks, Q6_K's 216-byte padded
 * blocks, and the struct-of-arrays Q8_0, Q5_0 and MXFP4 (mv_legacy.glsl).
 * The tensor cores' outputs leave through a 16 x 16 shared tile per subgroup
 * (their y rows are scattered). Weight rows past n_out read as zero and are not
 * written; x offsets and strides are multiples of 4.
 * Dispatch: gx = the tiles' bound mk / BN + min(n, mk), gy = ceil(n_out / BM). */

#include "vk_limits.h"

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(set = 0, binding = 2) readonly buffer T { uvec2 tab[]; };
layout(set = 0, binding = 3) readonly buffer M { uint map[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WU { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer WU2 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer WU4 { uvec4 v[]; };

layout(push_constant) uniform Push {
    uint n_in;
    uint n_out;
    uint blocks_per_row;
    uint x_div;
    uint x_offset;
    uint y_offset;
    uint x_stride;
    uint y_stride;
    uint tab_off; /* entries into tab */
    uint mo;      /* words into map */
    uint n;       /* experts: map[mo + n + 1 + r] is sorted row r's slot */
    uint tb;      /* the tile list, words from mo */
} pc;

const uint BM = VK_MOE_MM_ROWS;
const uint BN = VK_MOE_MM_BN;
const uint BK = 32u;
/* 4 vectors of data + 1 of padding per 32-k row */
const uint STRIDE4 = BK / 8u + 1u;

shared uvec4 Ash[2][BM * STRIDE4];
shared uvec4 Bsh[2][BN * STRIDE4];
#if defined(MOE_MM_CM)
shared float Csh[8][256]; /* one 16 x 16 output tile per subgroup */
#endif
shared uint slot_sh[BN];

/* The expert's weight (one address, three views) */
WU w1;
WU2 w2;
WU4 w4;

#if defined(DT_Q4K)
uvec2 deq4_q4k(uint qw, uint shift, float dsc, float dmn) {
    uvec4 q = (uvec4(qw) >> (uvec4(0u, 8u, 16u, 24u) + shift)) & 15u;
    vec4 v = vec4(q) * dsc - dmn;
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#elif defined(DT_Q6K)
uvec2 deq4_q6k(uint l, uint h, uint qls, uint qhs, float dsc) {
    uvec4 sh = uvec4(0u, 8u, 16u, 24u);
    uvec4 q = ((uvec4(l) >> (sh + qls)) & 15u) | (((uvec4(h) >> (sh + qhs)) & 3u) << 4u);
    vec4 v = dsc * (vec4(q) - 32.0);
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#elif defined(DT_Q8_0)
uvec2 deq4_q8(uint qw, float d) {
    ivec4 q = ivec4(uvec4(qw) << uvec4(24u, 16u, 8u, 0u)) >> 24; /* sign-extending */
    vec4 v = d * vec4(q);
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#elif defined(DT_Q5K)
uvec2 deq4_q5k(uint qw, uint hw, uint shift, uint sub, float dsc, float dmn) {
    uvec4 sh = uvec4(0u, 8u, 16u, 24u);
    uvec4 q = ((uvec4(qw) >> (sh + shift)) & 15u) | (((uvec4(hw) >> (sh + sub)) & 1u) << 4u);
    vec4 v = vec4(q) * dsc - dmn;
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#elif defined(DT_Q5_0)
/* 4 quant bytes, nibble at `shift`, and their 4 high bits (h4, bit i for
 * byte i) -> 4 packed f16 values d * (q - 16) */
uvec2 deq4_q50(uint qw, uint shift, uint h4, float d) {
    uvec4 q = ((uvec4(qw) >> (uvec4(0u, 8u, 16u, 24u) + shift)) & 15u) |
              (((uvec4(h4) >> uvec4(0u, 1u, 2u, 3u)) & 1u) << 4u);
    vec4 v = d * (vec4(q) - 16.0);
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#elif defined(DT_MXFP4)
/* kvalues_mxfp4 and mxfp4_e8m0_half as in mv_legacy.glsl */
vec4 mx4(uint q) {
    uvec4 c = (uvec4(q) >> uvec4(0u, 8u, 16u, 24u)) & 15u;
    vec4 m = vec4((uvec4(0xC8643210u) >> ((c & 7u) * 4u)) & 15u);
    return mix(m, -m, greaterThanEqual(c, uvec4(8u)));
}
float e8m0_half(uint e) {
    return uintBitsToFloat(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23u));
}
uvec2 deq4_mx(uint qw, uint shift, float d) {
    vec4 v = d * mx4(qw >> shift);
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#endif

/* Global -> registers for k-step ks (in flight during the MMAs of the step
 * before); registers -> shared afterwards (store_tiles). */
struct Fetch {
#if defined(DT_Q4K)
    uvec4 hdr;
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
#elif defined(DT_Q5K)
    uvec4 hdr;
    uvec4 qh;
    uvec4 qs;
    uint sub;
#elif defined(DT_Q5_0)
    uvec4 qs;
    uint qh;
    float d;
#elif defined(DT_MXFP4)
    uvec4 qs;
    float d;
#endif
    vec4 x0;
    vec4 x1;
};

Fetch fetch_tiles(uint ks, uint lid, uint row0) {
    uint k0 = ks * BK;
    uint r = lid >> 1u;
    uint hk = lid & 1u;
    uint row = min(row0 + r, pc.n_out - 1u); /* rows past n_out are zeroed at the store */
    Fetch f;
#if defined(DT_Q4K)
    uint bi = row * pc.blocks_per_row + (k0 >> 8u);
    uint b4 = bi * 9u; /* 144-byte blocks = 9 uvec4 */
    f.sub = (k0 & 255u) >> 5u;
    f.hdr = w4.v[b4];
    f.qs = w4.v[b4 + 1u + (f.sub >> 1u) * 2u + hk];
#elif defined(DT_Q6K)
    uint bi = row * pc.blocks_per_row + (k0 >> 8u);
    uint b2 = bi * 27u; /* 216-byte padded blocks = 27 uvec2 */
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
    f.qs = w4.v[bi * 2u + hk];
    uint sc0 = pc.n_out * pc.blocks_per_row * 8u;
    f.d = unpackHalf2x16(w1.v[sc0 + (bi >> 1u)] >> ((bi & 1u) * 16u)).x;
#elif defined(DT_Q5K)
    uint bi = row * pc.blocks_per_row + (k0 >> 8u);
    uint b11 = bi * 11u; /* 176-byte blocks = 11 uvec4: header, qh[32], qs[128] */
    f.sub = (k0 & 255u) >> 5u;
    f.hdr = w4.v[b11];
    f.qh = w4.v[b11 + 1u + hk];
    f.qs = w4.v[b11 + 3u + (f.sub >> 1u) * 2u + hk];
#elif defined(DT_Q5_0)
    uint bi = row * pc.blocks_per_row + ks;
    uint b5 = bi * 5u; /* 20 bytes: the high bits, then the quants */
    f.qh = w1.v[b5];
    f.qs = uvec4(w1.v[b5 + 1u], w1.v[b5 + 2u], w1.v[b5 + 3u], w1.v[b5 + 4u]);
    uint sc0 = pc.n_out * pc.blocks_per_row * 5u;
    f.d = unpackHalf2x16(w1.v[sc0 + (bi >> 1u)] >> ((bi & 1u) * 16u)).x;
#elif defined(DT_MXFP4)
    uint bi = row * pc.blocks_per_row + ks;
    f.qs = w4.v[bi];
    uint sc0 = pc.n_out * pc.blocks_per_row * 4u;
    f.d = e8m0_half((w1.v[sc0 + (bi >> 2u)] >> ((bi & 3u) * 8u)) & 0xffu);
#endif
    /* B: slot column c = lid >> 2, its 8 k from (lid & 3) * 8 */
    uint c = lid >> 2u;
    uint s = c < BN ? slot_sh[c] : 0xffffffffu;
    f.x0 = vec4(0.0);
    f.x1 = vec4(0.0);
    if (s != 0xffffffffu) {
        uint xv = (pc.x_offset + (s / pc.x_div) * pc.x_stride + k0 + (lid & 3u) * 8u) >> 2u;
        f.x0 = x4[xv];
        f.x1 = x4[xv + 1u];
    }
    return f;
}

void store_tiles(uint buf, uint lid, uint row0, Fetch f) {
    uint r = lid >> 1u;
    uint hk = lid & 1u;
    uint abase = r * STRIDE4 + hk * 2u;
    uvec4 a0, a1;
#if defined(DT_Q4K)
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
    a0 = uvec4(deq4_q4k(f.qs.x, shift, dsc, dmn), deq4_q4k(f.qs.y, shift, dsc, dmn));
    a1 = uvec4(deq4_q4k(f.qs.z, shift, dsc, dmn), deq4_q4k(f.qs.w, shift, dsc, dmn));
#elif defined(DT_Q6K)
    uint stream = f.sub & 3u;
    float dsc = unpackHalf2x16(f.dw).x *
                float(bitfieldExtract(int(f.scw), int(((stream * 2u + hk) & 3u) * 8u), 8));
    uint qls = stream >= 2u ? 4u : 0u;
    uint qhs = stream * 2u;
    a0 = uvec4(deq4_q6k(f.ql.x, f.qh.x, qls, qhs, dsc), deq4_q6k(f.ql.y, f.qh.y, qls, qhs, dsc));
    a1 = uvec4(deq4_q6k(f.ql.z, f.qh.z, qls, qhs, dsc), deq4_q6k(f.ql.w, f.qh.w, qls, qhs, dsc));
#elif defined(DT_Q8_0)
    a0 = uvec4(deq4_q8(f.qs.x, f.d), deq4_q8(f.qs.y, f.d));
    a1 = uvec4(deq4_q8(f.qs.z, f.d), deq4_q8(f.qs.w, f.d));
#elif defined(DT_Q5K)
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
    a0 = uvec4(deq4_q5k(f.qs.x, f.qh.x, shift, sub, dsc, dmn),
               deq4_q5k(f.qs.y, f.qh.y, shift, sub, dsc, dmn));
    a1 = uvec4(deq4_q5k(f.qs.z, f.qh.z, shift, sub, dsc, dmn),
               deq4_q5k(f.qs.w, f.qh.w, shift, sub, dsc, dmn));
#elif defined(DT_Q5_0)
    /* half hk: nibble hk of the 16 bytes, elements hk * 16 + 4u + i */
    uint h = f.qh >> (hk * 16u);
    uint shift = hk * 4u;
    a0 = uvec4(deq4_q50(f.qs.x, shift, h, f.d), deq4_q50(f.qs.y, shift, h >> 4u, f.d));
    a1 = uvec4(deq4_q50(f.qs.z, shift, h >> 8u, f.d), deq4_q50(f.qs.w, shift, h >> 12u, f.d));
#elif defined(DT_MXFP4)
    uint shift = hk * 4u;
    a0 = uvec4(deq4_mx(f.qs.x, shift, f.d), deq4_mx(f.qs.y, shift, f.d));
    a1 = uvec4(deq4_mx(f.qs.z, shift, f.d), deq4_mx(f.qs.w, shift, f.d));
#endif
    if (row0 + r >= pc.n_out) {
        a0 = uvec4(0u);
        a1 = uvec4(0u);
    }
    Ash[buf][abase] = a0;
    Ash[buf][abase + 1u] = a1;
    uint c = lid >> 2u;
    if (c < BN) {
        Bsh[buf][c * STRIDE4 + (lid & 3u)] = uvec4(packHalf2x16(f.x0.xy), packHalf2x16(f.x0.zw),
                                                   packHalf2x16(f.x1.xy), packHalf2x16(f.x1.zw));
    }
}

#if defined(MOE_MM_CM)
/* The subgroup's 16 x 16 tile acc of weight rows rb.. and slot columns cb..
 * to y: through its shared tile, each lane 8 of the 256 values. */
void store_out(coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc,
               uint sg, uint rb, uint cb, uint cnt) {
    coopMatStore(acc, Csh[sg], 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    for (uint i = gl_SubgroupInvocationID; i < 256u; i += gl_SubgroupSize) {
        uint c = cb + i / 16u;
        uint r = rb + i % 16u;
        if (c < cnt && r < pc.n_out) {
            y[pc.y_offset + slot_sh[c] * pc.y_stride + r] = Csh[sg][i];
        }
    }
    subgroupBarrier();
}
#else
const uint TM = BM / 16u; /* rows per thread */
const uint TN = BN / 16u; /* slots per thread */

/* 8 f16 values (one shared-memory vector) as two vec4 */
void unpack8(uvec4 v, out vec4 lo, out vec4 hi) {
    lo = vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y));
    hi = vec4(unpackHalf2x16(v.z), unpackHalf2x16(v.w));
}
#endif

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint ti = gl_WorkGroupID.x;
    if (ti >= map[pc.mo + pc.tb]) {
        return;
    }
    uint te = map[pc.mo + pc.tb + 1u + ti];
    uint e = te & 0xffffu;
    uint r0 = map[pc.mo + e] + (te >> 16u) * BN;
    uint cnt = min(BN, map[pc.mo + e + 1u] - r0);
    if (lid < BN) {
        slot_sh[lid] = lid < cnt ? map[pc.mo + pc.n + 1u + r0 + lid] : 0xffffffffu;
    }
    uvec2 wa = tab[pc.tab_off + e];
    w1 = WU(wa);
    w2 = WU2(wa);
    w4 = WU4(wa);
    uint row0 = gl_WorkGroupID.y * BM;
#if defined(MOE_MM_CM)
    uint sg = gl_SubgroupID;
    uint sgr = sg >> 1u; /* 32-row quarter */
    uint sgc = sg & 1u;  /* BN / 2-slot half */
#else
    uint tr = lid & 15u;
    uint tc = lid >> 4u;
    bool busy = tc * TN < cnt; /* the same for the 16 lanes of a row group */
    float acc[TM * TN];
    [[unroll]] for (uint i = 0u; i < TM * TN; i++) {
        acc[i] = 0.0;
    }
#endif
    barrier();

#if defined(MOE_MM_CM)
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc00 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc01 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc10 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc11 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif

    uint ksteps = pc.n_in / BK;
    store_tiles(0u, lid, row0, fetch_tiles(0u, lid, row0));
    barrier();

    for (uint ks = 0u; ks < ksteps; ks++) {
        uint cur = ks & 1u;
        Fetch nxt;
        bool more = ks + 1u < ksteps;
        if (more) {
            nxt = fetch_tiles(ks + 1u, lid, row0);
        }
#if defined(MOE_MM_CM)
        for (uint kk = 0u; kk < BK; kk += 16u) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB0;
            coopMatLoad(matB0, Bsh[cur], (sgc * (BN / 2u) + 0u) * STRIDE4 + kk / 8u, STRIDE4,
                        gl_CooperativeMatrixLayoutColumnMajor);
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB1;
            coopMatLoad(matB1, Bsh[cur], (sgc * (BN / 2u) + 16u) * STRIDE4 + kk / 8u, STRIDE4,
                        gl_CooperativeMatrixLayoutColumnMajor);
            {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
                coopMatLoad(matA, Ash[cur], (sgr * 32u + 0u) * STRIDE4 + kk / 8u, STRIDE4,
                            gl_CooperativeMatrixLayoutRowMajor);
                acc00 = coopMatMulAdd(matA, matB0, acc00);
                acc01 = coopMatMulAdd(matA, matB1, acc01);
            }
            {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
                coopMatLoad(matA, Ash[cur], (sgr * 32u + 16u) * STRIDE4 + kk / 8u, STRIDE4,
                            gl_CooperativeMatrixLayoutRowMajor);
                acc10 = coopMatMulAdd(matA, matB0, acc10);
                acc11 = coopMatMulAdd(matA, matB1, acc11);
            }
        }
#else
        if (busy) {
            [[unroll]] for (uint q = 0u; q < BK / 8u; q++) {
                vec4 b[TN][2];
                [[unroll]] for (uint j = 0u; j < TN; j++) {
                    unpack8(Bsh[cur][(tc * TN + j) * STRIDE4 + q], b[j][0], b[j][1]);
                }
                [[unroll]] for (uint i = 0u; i < TM; i++) {
                    vec4 a0, a1;
                    unpack8(Ash[cur][(tr + 16u * i) * STRIDE4 + q], a0, a1);
                    [[unroll]] for (uint j = 0u; j < TN; j++) {
                        acc[i * TN + j] += dot(a0, b[j][0]) + dot(a1, b[j][1]);
                    }
                }
            }
        }
#endif
        if (more) {
            store_tiles(cur ^ 1u, lid, row0, nxt);
        }
        barrier();
    }

#if defined(MOE_MM_CM)
    uint rb = row0 + sgr * 32u;
    uint cb = sgc * (BN / 2u);
    store_out(acc00, sg, rb, cb, cnt);
    store_out(acc01, sg, rb, cb + 16u, cnt);
    store_out(acc10, sg, rb + 16u, cb, cnt);
    store_out(acc11, sg, rb + 16u, cb + 16u, cnt);
#else
    [[unroll]] for (uint j = 0u; j < TN; j++) {
        uint c = tc * TN + j;
        if (c < cnt) {
            uint yb = pc.y_offset + slot_sh[c] * pc.y_stride;
            [[unroll]] for (uint i = 0u; i < TM; i++) {
                uint r = row0 + tr + 16u * i;
                if (r < pc.n_out) {
                    y[yb + r] = acc[i * TN + j];
                }
            }
        }
    }
#endif
}
