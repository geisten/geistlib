/* Grouped Q6_K matvec (#78 phase 3): matvec_q6k with workgroup y picking
 * the slot, whose weight (the 216-byte GPU blocks) is read by device
 * address. Dispatch: gx = ceil(n_out / 8), gy = slots.
 *
 * Q6_K matvec, 8 rows per 256-thread workgroup (8 virtual warps of 32).
 * Lane owns 8 consecutive elements — one (half, stream, l-run), so scale,
 * d and the shift constants are per-lane-per-block, and ql/qh arrive as
 * aligned u32 pairs. Dispatch: gx = ceil(n_out / 8). */

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer WU2 { uvec2 v[]; };

layout(push_constant) uniform Push {
    uint n_in;
    uint n_out;
    uint blocks_per_row;
    uint pad;
    uint x_offset; /* elements, slot 0's row */
    uint y_offset;
    uint x_stride; /* elements between the slots' rows */
    uint y_stride;
#ifdef MV_SEL
    uint ids_off; /* words into ids: slot 0's expert id */
    uint tab_off; /* entries into tab */
#else
    uvec2 addr[VK_GROUP_MAX]; /* device address of each slot's weight */
#endif
} pc;
#ifdef MV_SEL
/* The _sel variant (#162): slot s multiplies expert ids[ids_off + s]'s
 * weight, whose device address is tab[tab_off + id]; a zero address (an
 * expert without a copy here) skips the slot, leaving its y row as it was. */
layout(set = 0, binding = 2) readonly buffer T { uvec2 tab[]; };
layout(set = 0, binding = 3) readonly buffer I { uint ids[]; };
#define MV_ADDR(s) tab[pc.tab_off + ids[pc.ids_off + (s)]]
#else
#define MV_ADDR(s) pc.addr[s]
#endif

shared float red[256];
shared vec4 xsh[1536];

void main() {
    uint slot = gl_WorkGroupID.y;
#ifdef MV_SEL
    if (MV_ADDR(slot) == uvec2(0u)) {
        return; /* a cold expert: the host runs it */
    }
#endif
    WU2 w2 = WU2(MV_ADDR(slot));
    uint lid = gl_LocalInvocationID.x;
    uint warp = lid >> 5;
    uint lane = lid & 31u;
    uint row = gl_WorkGroupID.x * VK_MV_ROWS_PER_WG + warp;

    uint nv = pc.n_in >> 2u;
    bool x_shared = nv <= 1536u;
    if (x_shared) {
        uint xbase = (pc.x_offset + slot * pc.x_stride) >> 2u;
        for (uint i = lid; i < nv; i += 256u) {
            xsh[i] = x4[xbase + i];
        }
        barrier();
    }

    float sum = 0.0;
    if (row < pc.n_out) {
        uint e0 = lane * 8u;             /* first owned element in block */
        uint half_idx = e0 >> 7;
        uint in_half = e0 & 127u;
        uint stream = in_half >> 5;      /* quadrant 0..3 */
        uint l0 = in_half & 31u;         /* 0, 8, 16, 24 */
        uint ql_shift = (stream >= 2u) ? 4u : 0u;
        uint qh_shift = stream * 2u;
        uint sc_idx = stream * 2u + (l0 >> 4);
        uint row_byte = row * pc.blocks_per_row * 216u;

        for (uint b = 0; b < pc.blocks_per_row; b++) {
            uint block_byte = row_byte + b * 216u; /* GPU copy: 216-B aligned blocks */
            /* all four offsets are % 8 (l0 in {0,8,16,24}, 216-B blocks
             * are 8-aligned) — pair loads halve the transactions */
            uvec2 qlp = w2.v[(block_byte + half_idx * 64u + (stream & 1u) * 32u + l0) >> 3u];
            uvec2 qhp = w2.v[(block_byte + 128u + half_idx * 32u + l0) >> 3u];
            uint ql0 = qlp.x;
            uint ql1 = qlp.y;
            uint qh0 = qhp.x;
            uint qh1 = qhp.y;

            float d = unpackHalf2x16(w2.v[(block_byte + 208u) >> 3u].x).x;
            uvec2 scp = w2.v[(block_byte + 192u + half_idx * 8u) >> 3u];
            uint sc_b = (sc_idx < 4u ? scp.x >> (8u * sc_idx)
                                     : scp.y >> (8u * (sc_idx - 4u))) & 0xffu;
            int sc = int(sc_b ^ 0x80u) - 0x80;

            vec4 q_lo, q_hi;
            for (uint j = 0; j < 4u; j++) {
                q_lo[j] = float((((ql0 >> (8u * j)) >> ql_shift) & 15u) |
                                ((((qh0 >> (8u * j)) >> qh_shift) & 3u) << 4u));
                q_hi[j] = float((((ql1 >> (8u * j)) >> ql_shift) & 15u) |
                                ((((qh1 >> (8u * j)) >> qh_shift) & 3u) << 4u));
            }
            uint xe = (b * 256u + e0) >> 2u;
            vec4 xa = x_shared ? xsh[xe] : x4[((pc.x_offset + slot * pc.x_stride) >> 2u) + xe];
            vec4 xb = x_shared ? xsh[xe + 1u] : x4[((pc.x_offset + slot * pc.x_stride) >> 2u) + xe + 1u];
            float xq = dot(q_lo, xa) + dot(q_hi, xb);
            float xs = dot(xa + xb, vec4(1.0));
            sum += d * float(sc) * (xq - 32.0 * xs);
        }
    }

    red[lid] = sum;
    barrier();
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            red[lid] += red[lid + stride];
        }
        barrier();
    }
    if (lane == 0u && row < pc.n_out) {
        y[pc.y_offset + slot * pc.y_stride + row] = red[warp * 32u];
    }
}
