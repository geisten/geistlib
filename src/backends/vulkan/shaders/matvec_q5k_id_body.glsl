/* Grouped Q5_K matvec (#78 phase 3): matvec_q5k with workgroup y picking
 * the slot (mv_group_abi.glsl), whose weight (native blocks) is read by
 * device address. Dispatch: gx = ceil(n_out / 8), gy = slots. */

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WU { uint v[]; };

#include "mv_group_abi.glsl"

#define MV_BEGIN                                                              \
    uint slot = gl_WorkGroupID.y;                                             \
    MV_SKIP_COLD(slot)                                                        \
    WU w = WU(MV_ADDR(slot));
#define W_LOAD(i) w.v[i]
#include "matvec_q5k_core.glsl"
