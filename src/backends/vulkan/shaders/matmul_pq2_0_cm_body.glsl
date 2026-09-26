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
 * Dispatch: gx = n_out / 128, gy = ceil(rows / 128). */

#ifdef ACC_F16
#define ACCUM(i, j) hac##i##j
#else
#define ACCUM(i, j) acc##i##j
#endif

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

const uint BM = 128;
const uint BN = 128;
const uint BK = 32;
/* 4 vectors of data + 1 of padding per 32-k row (128-bit rows stay
 * 16-byte aligned for coopMatLoad) */
const uint STRIDE4 = BK / 8u + 1u;

shared uvec4 Ash[2][BM * STRIDE4];
shared uvec4 Bsh[2][BN * STRIDE4];

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

/* Global -> registers for k-step ks (the loads stay in flight while the MMAs
 * of the previous step run); registers -> shared afterwards (store_tiles). */
struct Fetch {
    uint qw;
    float d;
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
    uint bi = (row0 + r) * pc.blocks_per_row + (k0 >> 7u);
    f.qw = w[bi * 8u + ((k0 & 127u) >> 4u) + hk];
    f.d = unpackHalf2x16(w[pc.n_out * pc.blocks_per_row * 8u + (bi >> 1u)] >>
                         ((bi & 1u) * 16u)).x;
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
    Ash[buf][abase] = expand8(f.qw & 0xffffu, f.d);
    Ash[buf][abase + 1u] = expand8(f.qw >> 16u, f.d);
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
        uint cur = ks & 1u;
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
            store_tiles(cur ^ 1u, lid, nxt);
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
