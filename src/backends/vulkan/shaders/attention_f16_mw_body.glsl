#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_clustered : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

/* Tensor-core causal MQA/GQA prefill attention, multi-subgroup version of
 * attention_f16_cm_body.glsl (#475, #658). Same bindings, push constants
 * and semantics (causal, sliding window, q_offset_abs for chunked prefill,
 * f32 Q, f16 K/V); the host routes here only on 32-lane subgroups.
 *
 * One workgroup = NSG (4) subgroups and 16 query rows of one head. Every
 * step covers BC = 16 * NSG keys, one pass, online softmax:
 *   S:  subgroup s computes the 16x16 S tile of keys [16s, 16s + 16) over
 *       the whole head_dim, K read straight from global memory as the B
 *       operand (no shared K tile: the subgroups' keys are disjoint).
 *   softmax: 8 lanes per row over the 16 x BC S tile in shared memory,
 *       clustered max/sum; P goes to shared memory as f16.
 *   P@V: subgroup s owns head_dim / NSG output columns (OT accumulators) and
 *       multiplies the whole 16 x BC P by its V columns, again read from
 *       global memory.
 * Two workgroup barriers per BC keys (the 16-key kernel needs four per 16
 * keys and runs the QK^T product twice).
 *
 * Online softmax without per-row accumulator scaling (KHR coopmat has none):
 * a row's running max m only moves when a tile's max exceeds it by more
 * than RESCALE_T, so P = exp(s - m) <= exp(RESCALE_T) stays well inside f16
 * and l / O (f32) absorb it. When some row does move after its first tile,
 * every subgroup stores its O tiles to shared memory, scales the rows by
 * exp(m_old - m_new) and reloads them; that is rare after the first few
 * tiles of a row. (RESCALE_T = 0, an exact running max, costs ~20 % more on
 * Gemma 4 E2B. The stale max means the largest P of a row is not exactly
 * 1.0 in f16, so very peaked rows see about twice the old kernel's rounding
 * error, still < 1e-3 in test_backend_vulkan_ops_unit.)
 *
 * Out-of-range keys: a 16-key chunk that lies entirely outside every row's
 * window/causal range is skipped (its S entries are masked, its P is 0); a
 * chunk that reaches past n_kv is staged through shared memory with rows
 * clamped to n_kv - 1, so nothing reads past the K/V tensors and V never
 * contributes a non-finite value (P = 0 there, V finite).
 *
 * Dispatch: gx = ceil(n_q / 16), gy = n_q_heads, gz = 1. Workgroup x walks
 * the query blocks from the last (longest causal range) to the first so the
 * heavy blocks start first. */

const uint NSG = 4u;
#define OT_MAX (HD / 64) /* == OT, for the named accumulators */
layout(local_size_x = 128) in; /* NSG * 32 */

layout(set = 0, binding = 0) readonly buffer Q { vec4 q4[]; };
layout(set = 0, binding = 1) readonly buffer K { f16vec4 k4[]; };
layout(set = 0, binding = 2) readonly buffer V { f16vec4 v4[]; };
layout(set = 0, binding = 3) writeonly buffer Out { float out_v[]; };

layout(push_constant) uniform Push {
    uint n_q;
    uint n_kv;
    uint n_q_heads;
    uint n_kv_heads;
    uint head_dim;       /* == HD */
    uint q_offset_abs;
    uint sliding_window; /* 0: full causal */
    uint q_offset;
    uint k_offset;
    uint v_offset;
    uint out_offset;
} pc;

const uint  BR        = 16u;
const uint  BC        = 16u * NSG;
const uint  HD_U4     = uint(HD) / 8u;       /* uvec4 (8 f16) per Q row */
const uint  QSTR      = HD_U4 + 1u;          /* padded: no bank conflicts */
const uint  SSTR      = BC + 8u;             /* floats per S row */
const uint  PSTR      = BC / 2u + 4u;        /* uints (f16 pairs) per P row */
const uint  OCOLS     = uint(HD) / NSG;      /* O columns per subgroup */
const uint  OT        = OCOLS / 16u;         /* O accumulators per subgroup */
const uint  TPR       = NSG * 32u / BR;      /* softmax lanes per row */
const uint  PPT       = BC / 2u / TPR;       /* column pairs per lane */
const float RESCALE_T = 8.0;

shared uvec4 Qsh[BR * QSTR];
shared float Ssh[BR * SSTR];
shared uint  Psh[BR * PSTR];
shared float Osc[NSG * 256u]; /* per-subgroup 16x16 f32 staging */
shared uvec4 Ksc[NSG * 32u];  /* per-subgroup 16x16 f16 staging (tail) */
shared float alpha_sh[BR];
shared float l_sh[BR];
shared uint  resc_sh;

uvec4 pack_f32x8(vec4 a, vec4 b) {
    return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}

uvec4 pack_f16x8(f16vec4 a, f16vec4 b) {
    return uvec4(packHalf2x16(vec2(a.xy)),
                 packHalf2x16(vec2(a.zw)),
                 packHalf2x16(vec2(b.xy)),
                 packHalf2x16(vec2(b.zw)));
}

void sg_sync() {
    subgroupMemoryBarrierShared();
    subgroupBarrier();
}

void main() {
    uint tid     = gl_LocalInvocationID.x;
    uint sg      = gl_SubgroupID;
    uint lane    = gl_SubgroupInvocationID;
    uint q_block = (gl_NumWorkGroups.x - 1u - gl_WorkGroupID.x) * BR;
    uint q_head  = gl_WorkGroupID.y;
    if (q_block >= pc.n_q || q_head >= pc.n_q_heads || pc.n_kv == 0u) {
        return;
    }
    uint kv_group = pc.n_q_heads / pc.n_kv_heads;
    uint kv_head  = q_head / kv_group;
    uint hd4      = pc.head_dim >> 2u; /* vec4s per row */
    uint kv_row4  = pc.n_kv_heads * hd4;
    uint kbase    = (pc.k_offset >> 2u) + kv_head * hd4;
    uint vbase    = (pc.v_offset >> 2u) + kv_head * hd4 + sg * (OCOLS / 4u);

    /* 16-row Q tile, f16 (rows past n_q repeat the last row; never stored) */
    for (uint idx = tid; idx < BR * HD_U4; idx += NSG * 32u) {
        uint r    = idx / HD_U4;
        uint j    = idx % HD_U4;
        uint qpos = min(q_block + r, pc.n_q - 1u);
        uint base = (pc.q_offset >> 2u) + (qpos * pc.n_q_heads + q_head) * hd4 + j * 2u;
        Qsh[r * QSTR + j] = pack_f32x8(q4[base], q4[base + 1u]);
    }
    if (tid == 0u) {
        resc_sh = 0u;
    }
    barrier();

    uint qabs0  = pc.q_offset_abs + q_block;
    uint slo0   = pc.sliding_window > 0u && qabs0 + 1u > pc.sliding_window
                          ? qabs0 + 1u - pc.sliding_window
                          : 0u;
    uint hi_all = min(qabs0 + BR - 1u, pc.n_kv - 1u); /* last key any row sees */
    uint first_kb = slo0 / BC;
    uint last_kb  = hi_all / BC;

    /* softmax role: row `row`, lanes `part` of TPR (same subgroup) */
    uint  row  = tid / TPR;
    uint  part = tid % TPR;
    uint  qabs = qabs0 + row;
    uint  shi  = min(qabs, pc.n_kv - 1u);
    uint  slo  = pc.sliding_window > 0u && qabs + 1u > pc.sliding_window
                         ? qabs + 1u - pc.sliding_window
                         : 0u;
    float m    = -1.0 / 0.0;
    float l    = 0.0; /* this lane's share of the row sum */

#define GEIST_ATTN_MW_DECLO(N)                                                        \
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> O##N =          \
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    GEIST_ATTN_MW_DECLO(0)
    GEIST_ATTN_MW_DECLO(1)
#if OT_MAX > 2
    GEIST_ATTN_MW_DECLO(2)
    GEIST_ATTN_MW_DECLO(3)
#endif
#if OT_MAX > 4
    GEIST_ATTN_MW_DECLO(4)
    GEIST_ATTN_MW_DECLO(5)
    GEIST_ATTN_MW_DECLO(6)
    GEIST_ATTN_MW_DECLO(7)
#endif
#undef GEIST_ATTN_MW_DECLO

    uint tag = 0u;
    for (uint kb = first_kb; kb <= last_kb; kb++) {
        tag++;
        uint kv0 = kb * BC;

        /* ---- S = Q K^T, subgroup sg: keys [kc0, kc0 + 16) ---- */
        uint kc0 = kv0 + sg * 16u;
        if (kc0 <= hi_all && kc0 + 15u >= slo0) {
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> accS =
                    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matQ;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matK;
            if (kc0 + 16u <= pc.n_kv) {
                uint kofs = kbase + kc0 * kv_row4;
                for (uint j = 0u; j < HD_U4; j += 2u) {
                    coopMatLoad(matQ, Qsh, j, QSTR, gl_CooperativeMatrixLayoutRowMajor);
                    coopMatLoad(matK, k4, kofs + j * 2u, kv_row4, gl_CooperativeMatrixLayoutColumnMajor);
                    accS = coopMatMulAdd(matQ, matK, accS);
                }
            } else {
                /* tail chunk: stage 16 keys x 16 dims, rows clamped */
                uint kpos = min(kc0 + (lane >> 1u), pc.n_kv - 1u);
                uint src  = kbase + kpos * kv_row4 + (lane & 1u) * 2u;
                for (uint j = 0u; j < HD_U4; j += 2u) {
                    Ksc[sg * 32u + lane] = pack_f16x8(k4[src + j * 2u], k4[src + j * 2u + 1u]);
                    sg_sync();
                    coopMatLoad(matQ, Qsh, j, QSTR, gl_CooperativeMatrixLayoutRowMajor);
                    coopMatLoad(matK, Ksc, sg * 32u, 2u, gl_CooperativeMatrixLayoutColumnMajor);
                    accS = coopMatMulAdd(matQ, matK, accS);
                    sg_sync();
                }
            }
            coopMatStore(accS, Ssh, sg * 16u, SSTR, gl_CooperativeMatrixLayoutRowMajor);
        }
        barrier();

        /* ---- online softmax over the 16 x BC tile ---- */
        {
            float mx = -1.0 / 0.0;
            for (uint i = 0u; i < PPT; i++) {
                uint c  = (i * TPR + part) * 2u;
                uint k0 = kv0 + c;
                vec2 s  = vec2(Ssh[row * SSTR + c], Ssh[row * SSTR + c + 1u]);
                if (k0 >= slo && k0 <= shi) {
                    mx = max(mx, s.x);
                }
                if (k0 + 1u >= slo && k0 + 1u <= shi) {
                    mx = max(mx, s.y);
                }
            }
            mx          = subgroupClusteredMax(mx, TPR);
            float alpha = 1.0;
            if (mx > m + RESCALE_T) { /* also the first finite max (m = -inf) */
                if (m > -1.0 / 0.0) {
                    alpha   = exp(m - mx);
                    l      *= alpha;
                    resc_sh = tag;
                }
                m = mx;
            }
            if (part == 0u) {
                alpha_sh[row] = alpha;
            }
            for (uint i = 0u; i < PPT; i++) {
                uint  pr = i * TPR + part;
                uint  c  = pr * 2u;
                uint  k0 = kv0 + c;
                float w0 = 0.0;
                float w1 = 0.0;
                if (k0 >= slo && k0 <= shi) {
                    w0 = exp(Ssh[row * SSTR + c] - m);
                }
                if (k0 + 1u >= slo && k0 + 1u <= shi) {
                    w1 = exp(Ssh[row * SSTR + c + 1u] - m);
                }
                /* l sums the same f16-rounded P that P@V multiplies */
                uint pk = packHalf2x16(vec2(w0, w1));
                vec2 wr = unpackHalf2x16(pk);
                l      += wr.x + wr.y;
                Psh[row * PSTR + pr] = pk;
            }
        }
        barrier();

        /* ---- rescale O rows whose max moved (workgroup-uniform) ---- */
        if (resc_sh == tag) {
#define GEIST_ATTN_MW_RESCALE(N)                                                      \
    coopMatStore(O##N, Osc, sg * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);      \
    sg_sync();                                                                        \
    for (uint e = lane; e < 256u; e += 32u) {                                         \
        Osc[sg * 256u + e] *= alpha_sh[e / 16u];                                      \
    }                                                                                 \
    sg_sync();                                                                        \
    coopMatLoad(O##N, Osc, sg * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);       \
    sg_sync();
            GEIST_ATTN_MW_RESCALE(0)
            GEIST_ATTN_MW_RESCALE(1)
#if OT_MAX > 2
            GEIST_ATTN_MW_RESCALE(2)
            GEIST_ATTN_MW_RESCALE(3)
#endif
#if OT_MAX > 4
            GEIST_ATTN_MW_RESCALE(4)
            GEIST_ATTN_MW_RESCALE(5)
            GEIST_ATTN_MW_RESCALE(6)
            GEIST_ATTN_MW_RESCALE(7)
#endif
#undef GEIST_ATTN_MW_RESCALE
        }

        /* ---- O[:, sg's columns] += P V ---- */
        for (uint c = 0u; c < NSG; c++) {
            uint kcc = kv0 + c * 16u;
            if (kcc > hi_all || kcc + 15u < slo0) {
                continue; /* P is 0 for every row here */
            }
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matP;
            coopMatLoad(matP, Psh, c * 8u, PSTR, gl_CooperativeMatrixLayoutRowMajor);
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matV;
            if (kcc + 16u <= pc.n_kv) {
                uint vofs = vbase + kcc * kv_row4;
#define GEIST_ATTN_MW_PV(N)                                                            \
    coopMatLoad(matV, v4, vofs + (N) * 4u, kv_row4, gl_CooperativeMatrixLayoutRowMajor); \
    O##N = coopMatMulAdd(matP, matV, O##N);
                GEIST_ATTN_MW_PV(0)
                GEIST_ATTN_MW_PV(1)
#if OT_MAX > 2
                GEIST_ATTN_MW_PV(2)
                GEIST_ATTN_MW_PV(3)
#endif
#if OT_MAX > 4
                GEIST_ATTN_MW_PV(4)
                GEIST_ATTN_MW_PV(5)
                GEIST_ATTN_MW_PV(6)
                GEIST_ATTN_MW_PV(7)
#endif
#undef GEIST_ATTN_MW_PV
            } else {
                uint kpos = min(kcc + (lane >> 1u), pc.n_kv - 1u);
                uint src  = vbase + kpos * kv_row4 + (lane & 1u) * 2u;
#define GEIST_ATTN_MW_PVT(N)                                                           \
    Ksc[sg * 32u + lane] = pack_f16x8(v4[src + (N) * 4u], v4[src + (N) * 4u + 1u]);      \
    sg_sync();                                                                         \
    coopMatLoad(matV, Ksc, sg * 32u, 2u, gl_CooperativeMatrixLayoutRowMajor);            \
    O##N = coopMatMulAdd(matP, matV, O##N);                                             \
    sg_sync();
                GEIST_ATTN_MW_PVT(0)
                GEIST_ATTN_MW_PVT(1)
#if OT_MAX > 2
                GEIST_ATTN_MW_PVT(2)
                GEIST_ATTN_MW_PVT(3)
#endif
#if OT_MAX > 4
                GEIST_ATTN_MW_PVT(4)
                GEIST_ATTN_MW_PVT(5)
                GEIST_ATTN_MW_PVT(6)
                GEIST_ATTN_MW_PVT(7)
#endif
#undef GEIST_ATTN_MW_PVT
            }
        }
    }

    l = subgroupClusteredAdd(l, TPR);
    if (part == 0u) {
        l_sh[row] = l;
    }
    barrier();

    /* O / l, one 16x16 tile at a time through this subgroup's staging */
#define GEIST_ATTN_MW_FINISH(N)                                                       \
    coopMatStore(O##N, Osc, sg * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);      \
    sg_sync();                                                                        \
    for (uint e = lane; e < 256u; e += 32u) {                                         \
        uint r    = e / 16u;                                                          \
        uint qpos = q_block + r;                                                      \
        if (qpos < pc.n_q) {                                                          \
            float lr    = l_sh[r];                                                    \
            uint  d     = sg * OCOLS + (N) * 16u + e % 16u;                           \
            out_v[pc.out_offset + (qpos * pc.n_q_heads + q_head) * pc.head_dim + d] = \
                    lr > 0.0 ? Osc[sg * 256u + e] / lr : 0.0;                         \
        }                                                                             \
    }                                                                                 \
    sg_sync();
    GEIST_ATTN_MW_FINISH(0)
    GEIST_ATTN_MW_FINISH(1)
#if OT_MAX > 2
    GEIST_ATTN_MW_FINISH(2)
    GEIST_ATTN_MW_FINISH(3)
#endif
#if OT_MAX > 4
    GEIST_ATTN_MW_FINISH(4)
    GEIST_ATTN_MW_FINISH(5)
    GEIST_ATTN_MW_FINISH(6)
    GEIST_ATTN_MW_FINISH(7)
#endif
#undef GEIST_ATTN_MW_FINISH
}
