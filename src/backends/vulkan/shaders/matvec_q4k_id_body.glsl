/* Grouped Q4_K matvec (#78 phase 3): matvec_q4k with workgroup y picking
 * the slot (mv_group_abi.glsl). Dispatch: gx = ceil(n_out / 8),
 * gy = slots. */

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WU { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer WU4 { uvec4 v[]; };

#include "mv_group_abi.glsl"

#define MV_BEGIN                                                              \
    uint slot = gl_WorkGroupID.y;                                             \
    MV_SKIP_COLD(slot)                                                        \
    WU w = WU(MV_ADDR(slot));                                                 \
    WU4 wv4 = WU4(MV_ADDR(slot));
#define W_LOAD(i) w.v[i]
#define W4_LOAD(i) wv4.v[i]
#include "matvec_q4k_core.glsl"
