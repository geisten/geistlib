#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

/* PQ2_0 (Ternary-Bonsai) GEMM on tensor cores, double-buffered: a 128 weight
 * rows x 128 tokens x BK = 32 tile per workgroup, 256 threads = 8 subgroups in
 * a 4 x 2 layout, each subgroup owning a 32 x 64 block (8 accumulators, 2 A +
 * 4 B fragments per 16-k step for 8 MMAs; 64 x 64 per subgroup with only four
 * subgroups measured slower: too few warps to hide the loads). The cooperative
 * matrix arrays are unrolled by hand: an array indexed in a loop is not kept
 * in registers by the driver (4x slower). While the MMAs consume k-step
 * ks from one shared buffer pair the loads of step ks + 1 are in flight and are
 * stored into the other pair after the MMAs (one barrier per step).
 *
 * A tile: two threads per weight row, 16 codes = 1 quant word each -> 2 vectors
 * of 8 f16 (the ternary values (code - 1) * d are exact in f16). B tile: two
 * threads per token, 16 activations each -> 2 vectors of 8 f16 (the only
 * rounding). Both tiles
 * are stored k-contiguous (128-bit shared stores); B is loaded ColumnMajor.
 * f32 accumulation.
 *
 * GPU layout = struct-of-arrays (see matvec_pq2_0.comp): 8 quant words per
 * 128-element block, all blocks first, then one f16 scale per block.
 * Requires n_out % 128 == 0, rows % 16 == 0, n_in % 128 == 0.
 * Dispatch: gx = n_out / 128, gy = ceil(rows / 128).
 *
 * The same frame serves the k-quants (#658): DT_Q4K or DT_Q6K replaces the A
 * stage only. Each thread dequantizes 16 consecutive k of one weight row (one
 * half of a 32-element sub-block, so one scale per thread and k-step) from the
 * native superblocks: Q4_K's 144-byte blocks (16-byte aligned: header and
 * quants come as two 128-bit loads), Q6_K's 216-byte padded blocks (8-byte
 * aligned: ql and qh as 64-bit loads). The dequantized weights are rounded to
 * f16, as in matmul_q4k_cm.comp. Requires n_in % 256 == 0 for those. Without a
 * DT_* macro the body is PQ2_0.
 *
 * Likewise DT_TQ2_0 replaces the A stage with TQ2_0's (#467), in the SOA layout of
 * matvec_tq2_0.comp: 16 quant words per 256-element block, then the f16
 * scales. A k-step is one of the block's eight 32-element runs
 * e = g*128 + l*32 + m (bytes g*32 + m, bits 2l); each thread loads its 16
 * bytes as one 128-bit word. Requires n_in % 256 == 0 for it.
 *
 * DT_F32 reads dense row-major f32 weights (the PLE projections, #658): each
 * thread loads 16 consecutive weights of its row like its 16 activations and
 * rounds them to f16 the same way. Requires n_in % 32 == 0 for it.
 *
 * SINGLE_BUF keeps one shared buffer pair instead of two (#658): the next
 * k-step's loads still fly during the MMAs, but are stored only after a
 * second barrier. That halves the ~40 KiB of shared memory, so Turing runs
 * two workgroups per SM instead of one; it pays once a GEMM has more
 * workgroups than the GPU has SMs, and costs the extra barrier below that
 * (vk_linear_cm_route picks). */

#ifdef ACC_F16
#define ACCUM(i, j) hac##i##j
#else
#define ACCUM(i, j) acc##i##j
#endif

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
layout(set = 0, binding = 1) readonly buffer W { uint w[]; };
layout(set = 0, binding = 2) writeonly buffer Y { float y[]; };
#if defined(DT_Q4K)
layout(set = 0, binding = 1) readonly buffer W4 { uvec4 w4[]; };
#elif defined(DT_Q6K)
layout(set = 0, binding = 1) readonly buffer W2 { uvec2 w2[]; };
#elif defined(DT_TQ2_0)
layout(set = 0, binding = 1) readonly buffer W4 { uvec4 w4[]; };
#elif defined(DT_F32)
layout(set = 0, binding = 1) readonly buffer WF { vec4 wf4[]; };
#endif

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

const uint BM = 128;
const uint BN = 128;
const uint BK = 32;
/* 4 vectors of data + 1 of padding per 32-k row (128-bit rows stay
 * 16-byte aligned for coopMatLoad) */
const uint STRIDE4 = BK / 8u + 1u;

#ifdef SINGLE_BUF
shared uvec4 Ash[1][BM * STRIDE4];
shared uvec4 Bsh[1][BN * STRIDE4];
#else
shared uvec4 Ash[2][BM * STRIDE4];
shared uvec4 Bsh[2][BN * STRIDE4];
#endif

#if defined(DT_Q4K)
/* 4 quant bytes, nibble at `shift` -> 4 packed f16 values dsc * q - dmn */
uvec2 deq4_q4k(uint qw, uint shift, float dsc, float dmn) {
    uvec4 q = (uvec4(qw) >> (uvec4(0u, 8u, 16u, 24u) + shift)) & 15u;
    vec4 v = vec4(q) * dsc - dmn;
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#elif defined(DT_Q6K)
/* 4 ql bytes (nibble at qls) and 4 qh bytes (2 bits at qhs) -> 4 packed f16
 * values dsc * (q - 32) */
uvec2 deq4_q6k(uint l, uint h, uint qls, uint qhs, float dsc) {
    uvec4 sh = uvec4(0u, 8u, 16u, 24u);
    uvec4 q = ((uvec4(l) >> (sh + qls)) & 15u) | (((uvec4(h) >> (sh + qhs)) & 3u) << 4u);
    vec4 v = dsc * (vec4(q) - 32.0);
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#elif defined(DT_TQ2_0)
/* 4 quant bytes, 2-bit code at `sh` in each -> 4 packed f16 values
 * (code - 1) * d, exact in f16 */
uvec2 deq4_tq2(uint qw, uint sh, float d) {
    uvec4 c = (uvec4(qw) >> (uvec4(0u, 8u, 16u, 24u) + sh)) & 3u;
    vec4 v = (vec4(c) - 1.0) * d;
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#else
/* 8 ternary codes (16 bits) -> 8 packed f16 values (code - 1) * d, exact in
 * f16 for -d, 0, d and 2d */
uvec4 expand8(uint bits, float d) {
    uvec4 r;
    for (uint p = 0u; p < 4u; p++) {
        uvec2 c = uvec2(bits >> (4u * p), bits >> (4u * p + 2u)) & 3u;
        r[p] = packHalf2x16((vec2(c) - 1.0) * d);
    }
    return r;
}
#endif

/* Global -> registers for k-step ks (the loads stay in flight while the MMAs
 * of the previous step run); registers -> shared afterwards (store_tiles). */
struct Fetch {
#if defined(DT_Q4K)
    uvec4 hdr; /* d, dmin, 12 bytes of packed 6-bit scales and mins */
    uvec4 qs;  /* 16 quant bytes: this thread's half of the sub-block */
    uint sub;  /* 32-element sub-block of k-step ks within the superblock */
#elif defined(DT_Q6K)
    uvec4 ql;  /* 16 low-nibble bytes */
    uvec4 qh;  /* 16 high-bit bytes */
    uint scw;  /* the word holding this thread's int8 scale */
    uint dw;   /* d in the low half */
    uint sub;
#elif defined(DT_TQ2_0)
    uvec4 qs; /* this thread's 16 quant bytes */
    uint sh;  /* 2 * l of k-step ks */
    float d;
#elif defined(DT_F32)
    vec4 w0;
    vec4 w1;
    vec4 w2;
    vec4 w3;
#else
    uint qw;
    float d;
#endif
    vec4 x0;
    vec4 x1;
    vec4 x2;
    vec4 x3;
};

Fetch fetch_tiles(uint ks, uint lid, uint row0, uint tb0) {
    uint k0 = ks * BK;
    uint r = lid >> 1u;
    uint hk = lid & 1u;
    Fetch f;
#if defined(DT_Q4K)
    uint bi = (row0 + r) * pc.blocks_per_row + (k0 >> 8u);
    uint b4 = bi * 9u; /* 144-byte blocks = 9 uvec4 */
    f.sub = (k0 & 255u) >> 5u;
    f.hdr = w4[b4];
    /* qs: 32 bytes per pair of sub-blocks (low nibbles even, high odd) */
    f.qs = w4[b4 + 1u + (f.sub >> 1u) * 2u + hk];
#elif defined(DT_Q6K)
    uint bi = (row0 + r) * pc.blocks_per_row + (k0 >> 8u);
    uint b2 = bi * 27u; /* 216-byte padded blocks = 27 uvec2 */
    f.sub = (k0 & 255u) >> 5u;
    uint half_idx = f.sub >> 2u; /* 128-element half of the block */
    uint stream = f.sub & 3u;    /* 32-element stream within the half */
    /* ql at byte half * 64 + (stream & 1) * 32, qh at 128 + half * 32,
     * scales at 192 + half * 8 + stream * 2 + hk, d at 208 */
    uint qlv = b2 + half_idx * 8u + (stream & 1u) * 4u + hk * 2u;
    f.ql = uvec4(w2[qlv], w2[qlv + 1u]);
    uint qhv = b2 + 16u + half_idx * 4u + hk * 2u;
    f.qh = uvec4(w2[qhv], w2[qhv + 1u]);
    f.scw = w2[b2 + 24u + half_idx][stream >> 1u];
    f.dw = w2[b2 + 26u].x;
#elif defined(DT_TQ2_0)
    uint bi = (row0 + r) * pc.blocks_per_row + (k0 >> 8u);
    uint sub = (k0 & 255u) >> 5u;
    f.qs = w4[bi * 4u + (sub >> 2u) * 2u + hk];
    f.sh = 2u * (sub & 3u);
    f.d = unpackHalf2x16(w[pc.n_out * pc.blocks_per_row * 16u + (bi >> 1u)] >>
                         ((bi & 1u) * 16u)).x;
#elif defined(DT_F32)
    uint wv = ((row0 + r) * pc.n_in + k0 + hk * 16u) >> 2u;
    f.w0 = wf4[wv];
    f.w1 = wf4[wv + 1u];
    f.w2 = wf4[wv + 2u];
    f.w3 = wf4[wv + 3u];
#else
    uint bi = (row0 + r) * pc.blocks_per_row + (k0 >> 7u);
    f.qw = w[bi * 8u + ((k0 & 127u) >> 4u) + hk];
    f.d = unpackHalf2x16(w[pc.n_out * pc.blocks_per_row * 8u + (bi >> 1u)] >>
                         ((bi & 1u) * 16u)).x;
#endif
    uint t = tb0 + r;
    f.x0 = vec4(0.0);
    f.x1 = vec4(0.0);
    f.x2 = vec4(0.0);
    f.x3 = vec4(0.0);
    if (t < pc.rows) {
        uint xv = (pc.x_offset + t * pc.x_stride + k0 + hk * 16u) >> 2u;
        f.x0 = x4[xv];
        f.x1 = x4[xv + 1u];
        f.x2 = x4[xv + 2u];
        f.x3 = x4[xv + 3u];
    }
    return f;
}

void store_tiles(uint buf, uint lid, Fetch f) {
    uint r = lid >> 1u;
    uint hk = lid & 1u;
    uint abase = r * STRIDE4 + hk * 2u;
#if defined(DT_Q4K)
    /* the dequant math runs here, after the MMAs, not where the loads issue */
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
    Ash[buf][abase] = uvec4(deq4_q4k(f.qs.x, shift, dsc, dmn), deq4_q4k(f.qs.y, shift, dsc, dmn));
    Ash[buf][abase + 1u] =
            uvec4(deq4_q4k(f.qs.z, shift, dsc, dmn), deq4_q4k(f.qs.w, shift, dsc, dmn));
#elif defined(DT_Q6K)
    uint stream = f.sub & 3u;
    float dsc = unpackHalf2x16(f.dw).x *
                float(bitfieldExtract(int(f.scw), int(((stream * 2u + hk) & 3u) * 8u), 8));
    uint qls = stream >= 2u ? 4u : 0u;
    uint qhs = stream * 2u;
    Ash[buf][abase] = uvec4(deq4_q6k(f.ql.x, f.qh.x, qls, qhs, dsc),
                            deq4_q6k(f.ql.y, f.qh.y, qls, qhs, dsc));
    Ash[buf][abase + 1u] = uvec4(deq4_q6k(f.ql.z, f.qh.z, qls, qhs, dsc),
                                 deq4_q6k(f.ql.w, f.qh.w, qls, qhs, dsc));
#elif defined(DT_TQ2_0)
    Ash[buf][abase] = uvec4(deq4_tq2(f.qs.x, f.sh, f.d), deq4_tq2(f.qs.y, f.sh, f.d));
    Ash[buf][abase + 1u] = uvec4(deq4_tq2(f.qs.z, f.sh, f.d), deq4_tq2(f.qs.w, f.sh, f.d));
#elif defined(DT_F32)
    Ash[buf][abase] = uvec4(packHalf2x16(f.w0.xy), packHalf2x16(f.w0.zw), packHalf2x16(f.w1.xy),
                            packHalf2x16(f.w1.zw));
    Ash[buf][abase + 1u] = uvec4(packHalf2x16(f.w2.xy), packHalf2x16(f.w2.zw),
                                 packHalf2x16(f.w3.xy), packHalf2x16(f.w3.zw));
#else
    Ash[buf][abase] = expand8(f.qw & 0xffffu, f.d);
    Ash[buf][abase + 1u] = expand8(f.qw >> 16u, f.d);
#endif
    uint bbase = r * STRIDE4 + hk * 2u;
    Bsh[buf][bbase] = uvec4(packHalf2x16(f.x0.xy), packHalf2x16(f.x0.zw), packHalf2x16(f.x1.xy),
                            packHalf2x16(f.x1.zw));
    Bsh[buf][bbase + 1u] = uvec4(packHalf2x16(f.x2.xy), packHalf2x16(f.x2.zw),
                                 packHalf2x16(f.x3.xy), packHalf2x16(f.x3.zw));
}

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint row0 = gl_WorkGroupID.x * BM;
    uint tb0 = gl_WorkGroupID.y * BN;
    uint sg = gl_SubgroupID;
    uint sgr = sg >> 1u; /* 32-row quarter of the tile */
    uint sgc = sg & 1u;  /* 64-token half of the tile */

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc00 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc01 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc02 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc03 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc10 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc11 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc12 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc13 =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

#ifdef ACC_F16
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> hac00 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
#ifdef ACC_F16
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> hac01 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
#ifdef ACC_F16
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> hac02 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
#ifdef ACC_F16
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> hac03 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
#ifdef ACC_F16
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> hac10 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
#ifdef ACC_F16
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> hac11 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
#ifdef ACC_F16
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> hac12 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
#ifdef ACC_F16
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> hac13 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif

    uint ksteps = pc.n_in / BK;
    store_tiles(0u, lid, fetch_tiles(0u, lid, row0, tb0));
    barrier();

    for (uint ks = 0; ks < ksteps; ks++) {
#ifdef SINGLE_BUF
        uint cur = 0u;
#else
        uint cur = ks & 1u;
#endif
        Fetch nxt;
        bool more = ks + 1u < ksteps;
        if (more) {
            nxt = fetch_tiles(ks + 1u, lid, row0, tb0); /* in flight during the MMAs */
        }
        for (uint kk = 0; kk < BK; kk += 16u) {
            /* unrolled by hand: an array of cooperative matrices indexed in a loop
             * is not kept in registers by the driver (4x slower) */
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB0;
            coopMatLoad(matB0, Bsh[cur], (sgc * 64u + 0u) * STRIDE4 + kk / 8u, STRIDE4,
                        gl_CooperativeMatrixLayoutColumnMajor);
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB1;
            coopMatLoad(matB1, Bsh[cur], (sgc * 64u + 16u) * STRIDE4 + kk / 8u, STRIDE4,
                        gl_CooperativeMatrixLayoutColumnMajor);
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB2;
            coopMatLoad(matB2, Bsh[cur], (sgc * 64u + 32u) * STRIDE4 + kk / 8u, STRIDE4,
                        gl_CooperativeMatrixLayoutColumnMajor);
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB3;
            coopMatLoad(matB3, Bsh[cur], (sgc * 64u + 48u) * STRIDE4 + kk / 8u, STRIDE4,
                        gl_CooperativeMatrixLayoutColumnMajor);
            {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
                coopMatLoad(matA, Ash[cur], (sgr * 32u + 0u) * STRIDE4 + kk / 8u, STRIDE4,
                            gl_CooperativeMatrixLayoutRowMajor);
                ACCUM(0, 0) = coopMatMulAdd(matA, matB0, ACCUM(0, 0));
                ACCUM(0, 1) = coopMatMulAdd(matA, matB1, ACCUM(0, 1));
                ACCUM(0, 2) = coopMatMulAdd(matA, matB2, ACCUM(0, 2));
                ACCUM(0, 3) = coopMatMulAdd(matA, matB3, ACCUM(0, 3));
            }
            {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
                coopMatLoad(matA, Ash[cur], (sgr * 32u + 16u) * STRIDE4 + kk / 8u, STRIDE4,
                            gl_CooperativeMatrixLayoutRowMajor);
                ACCUM(1, 0) = coopMatMulAdd(matA, matB0, ACCUM(1, 0));
                ACCUM(1, 1) = coopMatMulAdd(matA, matB1, ACCUM(1, 1));
                ACCUM(1, 2) = coopMatMulAdd(matA, matB2, ACCUM(1, 2));
                ACCUM(1, 3) = coopMatMulAdd(matA, matB3, ACCUM(1, 3));
            }
        }
        if (more) {
#ifdef SINGLE_BUF
            barrier(); /* every subgroup is done reading the pair */
            store_tiles(0u, lid, nxt);
#else
            store_tiles(cur ^ 1u, lid, nxt);
#endif
        }
        barrier();
#ifdef ACC_F16
        /* f16 accumulation runs at full rate on the tensor cores (f32
         * accumulation is half rate on GeForce Turing): sum two k-steps (64 k)
         * in f16, then fold into the f32 accumulators */
        if ((ks & 1u) == 1u || ks + 1u == ksteps) {
            acc00 += coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(hac00);
            hac00 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            acc01 += coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(hac01);
            hac01 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            acc02 += coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(hac02);
            hac02 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            acc03 += coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(hac03);
            hac03 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            acc10 += coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(hac10);
            hac10 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            acc11 += coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(hac11);
            hac11 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            acc12 += coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(hac12);
            hac12 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            acc13 += coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(hac13);
            hac13 = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
        }
#endif
    }

    if (tb0 + sgc * 64u + 0u < pc.rows) {
        uint t = tb0 + sgc * 64u + 0u;
        coopMatStore(acc00, y, pc.y_offset + t * pc.y_stride + row0 + sgr * 32u + 0u, pc.y_stride,
                     gl_CooperativeMatrixLayoutColumnMajor);
        coopMatStore(acc10, y, pc.y_offset + t * pc.y_stride + row0 + sgr * 32u + 16u, pc.y_stride,
                     gl_CooperativeMatrixLayoutColumnMajor);
    }
    if (tb0 + sgc * 64u + 16u < pc.rows) {
        uint t = tb0 + sgc * 64u + 16u;
        coopMatStore(acc01, y, pc.y_offset + t * pc.y_stride + row0 + sgr * 32u + 0u, pc.y_stride,
                     gl_CooperativeMatrixLayoutColumnMajor);
        coopMatStore(acc11, y, pc.y_offset + t * pc.y_stride + row0 + sgr * 32u + 16u, pc.y_stride,
                     gl_CooperativeMatrixLayoutColumnMajor);
    }
    if (tb0 + sgc * 64u + 32u < pc.rows) {
        uint t = tb0 + sgc * 64u + 32u;
        coopMatStore(acc02, y, pc.y_offset + t * pc.y_stride + row0 + sgr * 32u + 0u, pc.y_stride,
                     gl_CooperativeMatrixLayoutColumnMajor);
        coopMatStore(acc12, y, pc.y_offset + t * pc.y_stride + row0 + sgr * 32u + 16u, pc.y_stride,
                     gl_CooperativeMatrixLayoutColumnMajor);
    }
    if (tb0 + sgc * 64u + 48u < pc.rows) {
        uint t = tb0 + sgc * 64u + 48u;
        coopMatStore(acc03, y, pc.y_offset + t * pc.y_stride + row0 + sgr * 32u + 0u, pc.y_stride,
                     gl_CooperativeMatrixLayoutColumnMajor);
        coopMatStore(acc13, y, pc.y_offset + t * pc.y_stride + row0 + sgr * 32u + 16u, pc.y_stride,
                     gl_CooperativeMatrixLayoutColumnMajor);
    }
}
