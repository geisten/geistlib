/* Grouped 32-element-block matvecs (#499: Q8_0, Q5_0, MXFP4): mv_legacy.glsl
 * with workgroup y picking the slot (mv_group_abi.glsl), whose weight (the
 * struct-of-arrays GPU layout) is read by device address. The includer
 * defines one DT_* and, for the _sel variant, MV_SEL.
 * Dispatch: gx = ceil(n_out / 8), gy = slots. */

layout(set = 0, binding = 0) readonly buffer X { vec4 x4[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WU { uint v[]; };

#include "mv_group_abi.glsl"

#define MV_ID 1
#define MV_BEGIN                                                              \
    uint slot = gl_WorkGroupID.y;                                             \
    MV_SKIP_COLD(slot)                                                        \
    WU w = WU(MV_ADDR(slot));
#define W_LOAD(i) w.v[i]
#include "mv_legacy.glsl"
