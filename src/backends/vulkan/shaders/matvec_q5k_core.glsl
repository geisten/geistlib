/* Q5_K matvec core, shared by matvec_q5k.comp and the grouped
 * matvec_q5k_id_body.glsl. The includer declares x4, y and pc and defines
 * MV_BEGIN (statements opening main), W_LOAD(i) (weight word i), W_ROW(e)
 * (word e of the weight rows), X_BASE and Y_BASE (element offsets of the
 * x and y rows). Dispatch: gx = ceil(n_out / 8). */

layout(local_size_x = 256) in;

shared float tmpsh[8][32];

void main() {
    MV_BEGIN
    uint tid = gl_LocalInvocationID.x;
    uint wrp = tid >> 5u;
    uint lane = tid & 31u;
    uint row = gl_WorkGroupID.x * VK_MV_ROWS_PER_WG + wrp;
    uint sub = lane >> 2u;
    uint idx8 = (lane & 3u) * 8u;
    uint shift = (sub & 1u) * 4u;
    uint nb = pc.blocks_per_row;

    float acc = 0.0;
    if (row < pc.n_out) {
        uint row_word = W_ROW(row * nb * 44u);
        uint xbase = (X_BASE) >> 2u;
        for (uint b = 0u; b < nb; ++b) {
            uint blk = row_word + b * 44u;
            uint h0 = W_LOAD(blk);
            uint h1 = W_LOAD(blk + 1u);
            uint h2 = W_LOAD(blk + 2u);
            uint h3 = W_LOAD(blk + 3u);
            vec2 dd = unpackHalf2x16(h0);
            uint sc_u;
            uint mn_u;
            if (sub < 4u) {
                sc_u = (h1 >> (8u * sub)) & 63u;
                mn_u = (h2 >> (8u * sub)) & 63u;
            } else {
                uint j = sub - 4u;
                uint q_j4 = (h3 >> (8u * j)) & 0xffu;
                sc_u = (q_j4 & 15u) | ((((h1 >> (8u * j)) & 0xffu) >> 6u) << 4u);
                mn_u = (q_j4 >> 4u) | ((((h2 >> (8u * j)) & 0xffu) >> 6u) << 4u);
            }
            uint qh0 = W_LOAD(blk + 4u + (idx8 >> 2u));
            uint qh1 = W_LOAD(blk + 5u + (idx8 >> 2u));
            uint qs = blk + 12u + (sub >> 1u) * 8u + (idx8 >> 2u);
            uint q0 = W_LOAD(qs);
            uint q1 = W_LOAD(qs + 1u);

            float dsc = dd.x * float(sc_u);
            float dmn = dd.y * float(mn_u);
            vec4 v0 = vec4(float(((q0 >> shift) & 15u) | (((qh0 >> sub) & 1u) << 4u)),
                           float(((q0 >> (8u + shift)) & 15u) | (((qh0 >> (8u + sub)) & 1u) << 4u)),
                           float(((q0 >> (16u + shift)) & 15u) | (((qh0 >> (16u + sub)) & 1u) << 4u)),
                           float(((q0 >> (24u + shift)) & 15u) | (((qh0 >> (24u + sub)) & 1u) << 4u)));
            vec4 v1 = vec4(float(((q1 >> shift) & 15u) | (((qh1 >> sub) & 1u) << 4u)),
                           float(((q1 >> (8u + shift)) & 15u) | (((qh1 >> (8u + sub)) & 1u) << 4u)),
                           float(((q1 >> (16u + shift)) & 15u) | (((qh1 >> (16u + sub)) & 1u) << 4u)),
                           float(((q1 >> (24u + shift)) & 15u) | (((qh1 >> (24u + sub)) & 1u) << 4u)));
            uint xi = xbase + ((b * 256u + sub * 32u + idx8) >> 2u);
            acc += dot(dsc * v0 - vec4(dmn), x4[xi]) + dot(dsc * v1 - vec4(dmn), x4[xi + 1u]);
        }
    }

    tmpsh[wrp][lane] = acc;
    barrier();
    for (uint s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            tmpsh[wrp][lane] += tmpsh[wrp][lane + s];
        }
        barrier();
    }
    if (lane == 0u && row < pc.n_out) {
        y[(Y_BASE) + row] = tmpsh[wrp][0];
    }
}
