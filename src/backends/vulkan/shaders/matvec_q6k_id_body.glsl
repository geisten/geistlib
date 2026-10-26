/* Grouped Q6_K matvec (#78 phase 3): matvec_q6k with workgroup y picking
 * the slot (mv_group_abi.glsl), whose weight (the 216-byte GPU blocks) is
 * read by device address. Dispatch: gx = ceil(n_out / 8), gy = slots. */

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer WU2 { uvec2 v[]; };

#include "mv_group_abi.glsl"

#define MV_BEGIN                                                              \
    uint slot = gl_WorkGroupID.y;                                             \
    MV_SKIP_COLD(slot)                                                        \
    WU2 w2 = WU2(MV_ADDR(slot));
#define W2_LOAD(i) w2.v[i]
#include "matvec_q6k_core.glsl"
