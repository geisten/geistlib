#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

/* Q8_0 / Q4_0 GEMM on tensor cores — the frame of matmul_q4k_cm.comp (64x64
 * tile, BK = 32, double-buffered: one barrier per step) with a legacy-block A
 * stage, picked by DT_Q8_0 or DT_Q4_0 in the including .comp. One k-step is
 * exactly one 32-element block, so each row needs one scale per step. Weight
 * layout as in mm_legacy.glsl: LPB quant words per block (8 for Q8_0, 4 for
 * Q4_0), then all fp16 scales packed two per word after the n_out * nb * LPB
 * quant words. Q4_0 byte j holds element j (low nibble) and j + 16 (high).
 * Requires n_out % 64 == 0, rows % 16 == 0 (n_in % 32 == 0 is the format's).
 * Dispatch: gx = n_out/64, gy = ceil(rows/64). */

layout(local_size_x = 128) in;

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
#elif defined(DT_Q4_0)
const uint LPB = 4u;
#endif
const uint BM = 64;
const uint BN = 64;
const uint BK = 32;
const uint ASTRIDE = BK + 8u;
const uint BSTRIDE = BN + 8u;

shared float16_t Ash[2][BM * ASTRIDE];
shared float16_t Bsh[2][BK * BSTRIDE];

void stage(uint buf, uint ks, uint lid, uint row0, uint tb0) {
    uint k0 = ks * BK;
    /* A: 64 rows x 32 k — thread -> (row = lid/2, half-block = lid&1) */
    {
        uint row = lid >> 1u;
        uint hk = lid & 1u;
        uint bi = (row0 + row) * pc.blocks_per_row + ks;
        uint sw = w[pc.n_out * pc.blocks_per_row * LPB + (bi >> 1u)];
        float d = unpackHalf2x16(sw >> ((bi & 1u) * 16u)).x;
        uint abase = row * ASTRIDE + hk * 16u;
#if defined(DT_Q8_0)
        uint qw0 = bi * 8u + hk * 4u;
        for (uint u = 0; u < 4u; u++) {
            int qw = int(w[qw0 + u]);
            for (uint j = 0; j < 4u; j++) {
                Ash[buf][abase + u * 4u + j] = float16_t(d * float(bitfieldExtract(qw, int(8u * j), 8)));
            }
        }
#elif defined(DT_Q4_0)
        /* both halves read the block's 4 words: hk 0 the low nibbles, 1 the high */
        uint qw0 = bi * 4u;
        for (uint u = 0; u < 4u; u++) {
            uint qw = w[qw0 + u] >> (hk * 4u);
            for (uint j = 0; j < 4u; j++) {
                Ash[buf][abase + u * 4u + j] = float16_t(d * (float((qw >> (8u * j)) & 15u) - 8.0));
            }
        }
#endif
    }
    /* B: 32 k x 64 cols — thread -> (col = lid&63, half-k = lid>>6) */
    {
        uint col = lid & 63u;
        uint hk = lid >> 6u;
        uint t = tb0 + col;
        if (t < pc.rows) {
            uint xv = (pc.x_offset + t * pc.x_stride + k0 + hk * 16u) >> 2u;
            for (uint u = 0; u < 4u; u++) {
                vec4 xw = x4[xv + u];
                uint kb = hk * 16u + u * 4u;
                Bsh[buf][(kb + 0u) * BSTRIDE + col] = float16_t(xw.x);
                Bsh[buf][(kb + 1u) * BSTRIDE + col] = float16_t(xw.y);
                Bsh[buf][(kb + 2u) * BSTRIDE + col] = float16_t(xw.z);
                Bsh[buf][(kb + 3u) * BSTRIDE + col] = float16_t(xw.w);
            }
        } else {
            for (uint u = 0; u < 16u; u++) {
                Bsh[buf][(hk * 16u + u) * BSTRIDE + col] = float16_t(0.0);
            }
        }
    }
}

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint row0 = gl_WorkGroupID.x * BM;
    uint tb0 = gl_WorkGroupID.y * BN;
    uint sg = gl_SubgroupID;

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[4];
    for (uint j = 0; j < 4u; j++) {
        acc[j] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    }

    uint ksteps = pc.n_in / BK;
    stage(0u, 0u, lid, row0, tb0);
    barrier();

    for (uint ks = 0; ks < ksteps; ks++) {
        uint cur = ks & 1u;
        if (ks + 1u < ksteps) {
            stage(cur ^ 1u, ks + 1u, lid, row0, tb0);
        }
        for (uint kk = 0; kk < BK; kk += 16u) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
            coopMatLoad(matA, Ash[cur], sg * 16u * ASTRIDE + kk, ASTRIDE,
                        gl_CooperativeMatrixLayoutRowMajor);
            for (uint j = 0; j < 4u; j++) {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
                coopMatLoad(matB, Bsh[cur], kk * BSTRIDE + j * 16u, BSTRIDE,
                            gl_CooperativeMatrixLayoutRowMajor);
                acc[j] = coopMatMulAdd(matA, matB, acc[j]);
            }
        }
        barrier();
    }

    uint out_row = row0 + sg * 16u;
    for (uint j = 0; j < 4u; j++) {
        uint t = tb0 + j * 16u;
        if (t < pc.rows) {
            coopMatStore(acc[j], y, pc.y_offset + t * pc.y_stride + out_row, pc.y_stride,
                         gl_CooperativeMatrixLayoutColumnMajor);
        }
    }
}
