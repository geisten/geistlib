/* Grouped F32 matvec (#78 phase 3): workgroup (row, slot) computes row `row`
 * of slot `slot`'s weight against slot's x row (mv_group_abi.glsl). The
 * weights are reached by device address, so one dispatch covers up to
 * VK_GROUP_MAX of them. Dispatch: gx = n_out, gy = slots. */

layout(set = 0, binding = 0) readonly buffer X { float x[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WF { float v[]; };

#include "mv_group_abi.glsl"

#define MV_BEGIN                                                              \
    uint slot = gl_WorkGroupID.y;                                             \
    MV_SKIP_COLD(slot)
#define MV_BIND                                                               \
    WF w = WF(MV_ADDR(slot));                                                 \
    uint xb = X_BASE;
#define X_ROW xb
#define W_LOAD(i) w.v[i]
#include "matvec_f32_core.glsl"
