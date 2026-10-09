#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable
#include "vk_limits.h"

/* Flash-decoding partial pass (n_q == 1, f16 KV), GQA-shared (#475): one
 * workgroup per (span, kv-head, q-head batch). The workgroup reads each K/V
 * row of its span ONCE and scores it against all `gb` q-heads of the batch
 * (the q-heads are the inner dimension), so a GQA group no longer re-reads
 * its kv-head's cache once per q-head. A span is `n_tiles` tiles of `tile`
 * positions (16, 32 or 64), walked with an online softmax; the host sizes
 * spans so the dispatch still fills the GPU when there are few kv-heads.
 *
 * Partials for attn_comb: acc as [head][span][hd] at p_offset (a multiple
 * of 4), then (m, l) as [head][span][2].
 * Spans start at s_lo (the sliding window's first key), so every span holds
 * at least one key unless the whole range is empty (n_spans == 1 then).
 * K/V/Q stream as 4-wide vectors: q/k/v element offsets and hd % 4 == 0
 * (the host falls back to the direct kernel otherwise).
 *
 * GROUP (2 or 4: the most q-heads per workgroup) is fixed per variant by the
 * including .comp; the per-head registers and the shared q / merge buffer
 * scale with it. Wider costs more than the K/V re-reads it saves (RTX 2080
 * Ti, 3072 keys): a 4-wide variant serving Qwen3's 2-head groups ran 10 %
 * slower than the 2-wide one, an 8-wide one 1.8x slower, and Gemma 4's
 * 8-head group runs 1.4x faster as two batches of 4 than as one of 8. */

layout(local_size_x = VK_ATTN_PART_WG) in;

layout(set = 0, binding = 0) readonly buffer Q { vec4 q4[]; };
layout(set = 0, binding = 1) readonly buffer K { f16vec4 k4[]; };
layout(set = 0, binding = 2) readonly buffer V { f16vec4 v4[]; };
layout(set = 0, binding = 3) writeonly buffer P { float part[]; };
layout(set = 0, binding = 3) writeonly buffer P4 { vec4 part4[]; };

layout(push_constant) uniform Push {
    uint n_q_heads;
    uint n_kv_heads;
    uint hd;
    uint s_lo;    /* first key (sliding window start) */
    uint s_end;   /* one past the last key */
    uint tile;    /* positions per tile: 16, 32 or 64 */
    uint n_tiles; /* tiles per span */
    uint gb;      /* q-heads per workgroup, <= VK_ATTN_PART_MAX_GROUP */
    uint q_offset;
    uint k_offset;
    uint v_offset;
    uint p_offset;
    uint n_spans;
} pc;

const uint WG = VK_ATTN_PART_WG;
const uint TMAX = VK_ATTN_PART_TILE_MAX;
const uint GMAX = GROUP; /* q-heads one workgroup serves, fixed per variant */
const uint UNR = 8u; /* loads in flight per lane */
const float NEG_INF = -1.0 / 0.0;

/* q of the batch's heads; at the end, the V position-group merge. */
shared vec4 big[GROUP * (VK_ATTN_MAX_HEAD_DIM / 4u)];
shared float qk[GMAX * WG];   /* QK lane partials */
shared float wsh[GMAX * TMAX]; /* scores, then weights */
shared float red[GMAX * TMAX]; /* reduction scratch */
shared float m_sh[GMAX];
shared float l_sh[GMAX];
shared float alpha_sh[GMAX];

void main() {
    uint span = gl_WorkGroupID.x;
    uint kvh = gl_WorkGroupID.y;
    uint lid = gl_LocalInvocationID.x;
    uint grp = pc.n_q_heads / pc.n_kv_heads;
    uint h0 = kvh * grp + gl_WorkGroupID.z * pc.gb;
    if (span >= pc.n_spans || kvh >= pc.n_kv_heads || h0 >= (kvh + 1u) * grp ||
        pc.hd < 4u || pc.hd > VK_ATTN_MAX_HEAD_DIM || pc.gb > GMAX || pc.tile == 0u ||
        pc.tile > TMAX) {
        return;
    }
    uint ng = min(pc.gb, (kvh + 1u) * grp - h0);
    uint hd = pc.hd;
    uint hd4 = hd >> 2u;
    uint kv_row4 = (pc.n_kv_heads * hd) >> 2u;
    uint T = pc.tile;
    uint span_len = T * pc.n_tiles;

    uint c_lo = pc.s_lo + span * span_len;
    uint c_hi = min(pc.s_end, c_lo + span_len);

    uint ml = pc.p_offset + pc.n_q_heads * pc.n_spans * hd; /* (m, l) region */
    if (c_lo >= c_hi) {
        for (uint i = lid; i < ng * hd4; i += WG) {
            uint g = i / hd4;
            part4[(pc.p_offset >> 2u) + ((h0 + g) * pc.n_spans + span) * hd4 + (i - g * hd4)] =
                    vec4(0.0);
        }
        if (lid < ng) {
            part[ml + 2u * ((h0 + lid) * pc.n_spans + span)] = NEG_INF; /* m */
            part[ml + 2u * ((h0 + lid) * pc.n_spans + span) + 1u] = 0.0; /* l */
        }
        return;
    }

    /* the batch's q-heads are contiguous in q */
    uint q_base4 = (pc.q_offset >> 2u) + h0 * hd4;
    for (uint i = lid; i < ng * hd4; i += WG) {
        big[i] = q4[q_base4 + i];
    }
    if (lid < GMAX) {
        m_sh[lid] = NEG_INF;
        l_sh[lid] = 0.0;
    }

    /* QK: lp lanes per position, interleaved over hd4 */
    uint lp = WG / T;
    uint qp = lid / lp;
    uint qpart = lid - qp * lp;
    /* V: one 4-wide dim slice per lane, npg position groups */
    uint npg = WG / hd4;
    uint vpg = lid / hd4;
    uint vsl = lid - vpg * hd4;
    uint k_head4 = (pc.k_offset >> 2u) + kvh * hd4;
    uint v_head4 = (pc.v_offset >> 2u) + kvh * hd4;

    vec4 a[GMAX];
    [[unroll]] for (uint g = 0; g < GMAX; g++) {
        a[g] = vec4(0.0);
    }

    for (uint t0 = c_lo; t0 < c_hi; t0 += T) {
        uint nn = min(T, c_hi - t0);
        barrier(); /* q staged / previous tile's wsh consumed */

        float acc[GMAX];
        [[unroll]] for (uint g = 0; g < GMAX; g++) {
            acc[g] = 0.0;
        }
        if (qp < nn) {
            /* UNR independent loads in flight per lane */
            uint kb = k_head4 + (t0 + qp) * kv_row4;
            for (uint i0 = qpart; i0 < hd4; i0 += UNR * lp) {
                vec4 kk[UNR];
                [[unroll]] for (uint u = 0; u < UNR; u++) {
                    uint i = i0 + u * lp;
                    kk[u] = i < hd4 ? vec4(k4[kb + i]) : vec4(0.0);
                }
                [[unroll]] for (uint u = 0; u < UNR; u++) {
                    uint i = min(i0 + u * lp, hd4 - 1u); /* kk[u] is 0 past hd4 */
                    [[unroll]] for (uint g = 0; g < GMAX; g++) {
                        if (g < ng) {
                            acc[g] += dot(kk[u], big[g * hd4 + i]);
                        }
                    }
                }
            }
        }
        [[unroll]] for (uint g = 0; g < GMAX; g++) {
            if (g < ng) {
                qk[g * WG + lid] = acc[g];
            }
        }
        barrier();
        for (uint i = lid; i < ng * T; i += WG) {
            uint g = i / T;
            uint p = i - g * T;
            float s = 0.0;
            for (uint j = 0; j < lp; j++) {
                s += qk[g * WG + p * lp + j];
            }
            s = p < nn ? s : NEG_INF;
            wsh[g * TMAX + p] = s;
            red[g * TMAX + p] = s;
        }
        barrier();
        for (uint st = T >> 1u; st > 0u; st >>= 1u) {
            for (uint i = lid; i < ng * st; i += WG) {
                uint g = i / st;
                uint p = i - g * st;
                red[g * TMAX + p] = max(red[g * TMAX + p], red[g * TMAX + p + st]);
            }
            barrier();
        }
        if (lid < ng) {
            float m_old = m_sh[lid];
            float m_new = max(m_old, red[lid * TMAX]); /* finite: nn >= 1 */
            alpha_sh[lid] = exp(m_old - m_new);
            m_sh[lid] = m_new;
        }
        barrier();
        for (uint i = lid; i < ng * T; i += WG) {
            uint g = i / T;
            uint p = i - g * T;
            float w = p < nn ? exp(wsh[g * TMAX + p] - m_sh[g]) : 0.0;
            wsh[g * TMAX + p] = w;
            red[g * TMAX + p] = w;
        }
        barrier();
        for (uint st = T >> 1u; st > 0u; st >>= 1u) {
            for (uint i = lid; i < ng * st; i += WG) {
                uint g = i / st;
                uint p = i - g * st;
                red[g * TMAX + p] += red[g * TMAX + p + st];
            }
            barrier();
        }
        if (lid < ng) {
            l_sh[lid] = l_sh[lid] * alpha_sh[lid] + red[lid * TMAX];
        }

        /* weighted V, each row read once for all ng heads */
        if (vpg < npg) {
            [[unroll]] for (uint g = 0; g < GMAX; g++) {
                if (g < ng) {
                    a[g] *= alpha_sh[g];
                }
            }
            uint vb = v_head4 + t0 * kv_row4 + vsl;
            for (uint j0 = vpg; j0 < nn; j0 += UNR * npg) {
                vec4 vv[UNR];
                [[unroll]] for (uint u = 0; u < UNR; u++) {
                    uint j = j0 + u * npg;
                    vv[u] = j < nn ? vec4(v4[vb + j * kv_row4]) : vec4(0.0);
                }
                [[unroll]] for (uint u = 0; u < UNR; u++) {
                    uint j = min(j0 + u * npg, nn - 1u); /* vv[u] is 0 past nn */
                    [[unroll]] for (uint g = 0; g < GMAX; g++) {
                        if (g < ng) {
                            a[g] += wsh[g * TMAX + j] * vv[u];
                        }
                    }
                }
            }
        }
    }
    barrier(); /* q no longer read: big becomes the merge buffer */

    if (vpg < npg) {
        [[unroll]] for (uint g = 0; g < GMAX; g++) {
            if (g < ng) {
                big[(vpg * ng + g) * hd4 + vsl] = a[g];
            }
        }
    }
    barrier();
    for (uint i = lid; i < ng * hd4; i += WG) {
        uint g = i / hd4;
        uint sl = i - g * hd4;
        vec4 s = vec4(0.0);
        for (uint pg = 0; pg < npg; pg++) {
            s += big[(pg * ng + g) * hd4 + sl];
        }
        part4[(pc.p_offset >> 2u) + ((h0 + g) * pc.n_spans + span) * hd4 + sl] = s;
    }
    if (lid < ng) {
        part[ml + 2u * ((h0 + lid) * pc.n_spans + span)] = m_sh[lid];
        part[ml + 2u * ((h0 + lid) * pc.n_spans + span) + 1u] = l_sh[lid];
    }
}
