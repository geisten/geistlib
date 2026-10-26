/* Q4_K matvec core, shared by matvec_q4k.comp and the grouped
 * matvec_q4k_id_body.glsl. The includer declares x4, y and pc and defines
 * MV_BEGIN (statements opening main), W_LOAD(i) / W4_LOAD(i) (weight word /
 * uvec4 i), X_BASE and Y_BASE (element offsets of the x and y rows).
 * Dispatch: gx = ceil(n_out / 8). */

layout(local_size_x = 64) in;

const uint NUM_ROWS = 4; /* per warp */
shared float tmpsh[2][NUM_ROWS][32];

void main() {
    MV_BEGIN
    uint tid = gl_LocalInvocationID.x;
    uint wrp = tid >> 5u;  /* warp 0/1 */
    uint lane = tid & 31u;
    uint first_row = gl_WorkGroupID.x * VK_MV_ROWS_PER_WG + wrp * NUM_ROWS;
    uint itid = lane & 15u; /* lane within superblock */
    uint ix = lane >> 4u;   /* block stream 0/1 */

    uint il = itid >> 2u;      /* 0..3 */
    uint ir = itid & 3u;       /* 0..3 */
    uint v_im = il >> 1u;      /* half selector */
    uint v_in = il & 1u;
    uint l0 = 4u * (2u * ir + v_in); /* 0..28 */
    uint q_offset = 32u * v_im + l0; /* byte into qs */
    uint y_offset = 64u * v_im + l0; /* element into x */

    uint num_rows = first_row < pc.n_out ? min(NUM_ROWS, pc.n_out - first_row) : 0u;
    float temp[NUM_ROWS];
    [[unroll]] for (uint n = 0; n < NUM_ROWS; ++n) {
        temp[n] = 0.0;
    }

    uint xbase = (X_BASE) >> 2u;
    for (uint i = ix; i < pc.blocks_per_row; i += 2u) {
        uint y1 = i * 256u + y_offset; /* element index */
        uint y2 = y1 + 128u;
        vec4 by10 = x4[xbase + (y1 >> 2u)];
        vec4 by132 = x4[xbase + (y1 >> 2u) + 8u];
        vec4 by20 = x4[xbase + (y2 >> 2u)];
        vec4 by232 = x4[xbase + (y2 >> 2u) + 8u];

        [[unroll]] for (uint n = 0; n < NUM_ROWS; ++n) {
            /* clamp keeps tail-row weight reads in bounds */
            uint blk = (min(first_row + n, pc.n_out - 1u) * pc.blocks_per_row + i) * 36u;
            uvec4 hdr = W4_LOAD(blk >> 2u); /* d/dmin + 3 scale words, one load */
            vec2 dm = unpackHalf2x16(hdr.x);

            uint s01 = hdr.y; /* scale u16s 0,1 */
            uint s23 = hdr.z;
            uint s45 = hdr.w;
            uint scale0_u32 = v_im == 0u ? (s01 & 0xffffu) : (s01 >> 16u);
            uint scale4_u32 = v_im == 0u ? (s23 & 0xffffu) : (s23 >> 16u);
            uint scale8_u32 = v_im == 0u ? (s45 & 0xffffu) : (s45 >> 16u);
            uint scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
            uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2u;
            vec4 scl = vec4(unpack8(scale_0_4_l & 0x3F3F3F3Fu));
            vec4 sch = vec4(unpack8((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0Fu) |
                                    scale_0_4_h));

            uint qsw = blk + 4u + (q_offset >> 2u);
            uint qs0 = W_LOAD(qsw);
            uint qs64 = W_LOAD(qsw + 16u);
            vec4 q0l = vec4(unpack8(qs0 & 0x0F0F0F0Fu));
            vec4 q0h = vec4(unpack8((qs0 >> 4u) & 0x0F0F0F0Fu));
            vec4 q6l = vec4(unpack8(qs64 & 0x0F0F0F0Fu));
            vec4 q6h = vec4(unpack8((qs64 >> 4u) & 0x0F0F0F0Fu));

            float sx = dot(by10, q0l);
            float sy = dot(by132, q0h);
            float sz = dot(by20, q6l);
            float sw = dot(by232, q6h);
            float smin = (by10.x + by10.y + by10.z + by10.w) * scl.z +
                         (by132.x + by132.y + by132.z + by132.w) * scl.w +
                         (by20.x + by20.y + by20.z + by20.w) * sch.z +
                         (by232.x + by232.y + by232.z + by232.w) * sch.w;
            temp[n] = fma(dm.x, fma(sx, scl.x, fma(sy, scl.y, fma(sz, sch.x, sw * sch.y))),
                          fma(-dm.y, smin, temp[n]));
        }
    }

    [[unroll]] for (uint n = 0; n < NUM_ROWS; ++n) {
        tmpsh[wrp][n][lane] = temp[n];
    }
    barrier();
    [[unroll]] for (uint s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            [[unroll]] for (uint n = 0; n < NUM_ROWS; ++n) {
                tmpsh[wrp][n][lane] += tmpsh[wrp][n][lane + s];
            }
        }
        barrier();
    }
    if (lane < num_rows) {
        y[(Y_BASE) + first_row + lane] = tmpsh[wrp][lane][0];
    }
}
