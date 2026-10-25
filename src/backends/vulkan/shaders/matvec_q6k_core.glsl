/* Q6_K matvec core, shared by matvec_q6k.comp and the grouped
 * matvec_q6k_id_body.glsl. The includer declares x4, y and pc and defines
 * MV_BEGIN (statements opening main), W2_LOAD(i) (weight uvec2 i), W_ROW(e)
 * (byte e of the weight rows), X_BASE and Y_BASE (element offsets of the
 * x and y rows). Dispatch: gx = ceil(n_out / 8). */

layout(local_size_x = 256) in;

shared float red[256];
shared vec4 xsh[1536];

void main() {
    MV_BEGIN
    uint lid = gl_LocalInvocationID.x;
    uint warp = lid >> 5;
    uint lane = lid & 31u;
    uint row = gl_WorkGroupID.x * VK_MV_ROWS_PER_WG + warp;

    uint nv = pc.n_in >> 2u;
    bool x_shared = nv <= 1536u;
    if (x_shared) {
        uint xbase = (X_BASE) >> 2u;
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
        uint row_byte = W_ROW(row * pc.blocks_per_row * 216u);

        for (uint b = 0; b < pc.blocks_per_row; b++) {
            uint block_byte = row_byte + b * 216u; /* GPU copy: 216-B aligned blocks */
            /* all four offsets are % 8 (l0 in {0,8,16,24}, 216-B blocks
             * are 8-aligned) — pair loads halve the transactions */
            uvec2 qlp = W2_LOAD((block_byte + half_idx * 64u + (stream & 1u) * 32u + l0) >> 3u);
            uvec2 qhp = W2_LOAD((block_byte + 128u + half_idx * 32u + l0) >> 3u);
            uint ql0 = qlp.x;
            uint ql1 = qlp.y;
            uint qh0 = qhp.x;
            uint qh1 = qhp.y;

            float d = unpackHalf2x16(W2_LOAD((block_byte + 208u) >> 3u).x).x;
            uvec2 scp = W2_LOAD((block_byte + 192u + half_idx * 8u) >> 3u);
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
            vec4 xa = x_shared ? xsh[xe] : x4[((X_BASE) >> 2u) + xe];
            vec4 xb = x_shared ? xsh[xe + 1u] : x4[((X_BASE) >> 2u) + xe + 1u];
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
        y[(Y_BASE) + row] = red[warp * 32u];
    }
}
