/* Grouped F32 matvec (#78 phase 3): workgroup (row, slot) computes row `row`
 * of slot `slot`'s weight against slot's x row. The weights are reached by
 * device address, so one dispatch covers up to VK_GROUP_MAX of them.
 * Dispatch: gx = n_out, gy = slots. */

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { float x[]; };
layout(set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WF { float v[]; };

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

shared float partial[256];

void main() {
    uint row = gl_WorkGroupID.x;
    uint slot = gl_WorkGroupID.y;
#ifdef MV_SEL
    if (MV_ADDR(slot) == uvec2(0u)) {
        return; /* a cold expert: the host runs it */
    }
#endif
    uint lid = gl_LocalInvocationID.x;
    if (row >= pc.n_out) {
        return;
    }
    WF w = WF(MV_ADDR(slot));
    uint xb = pc.x_offset + slot * pc.x_stride;
    float sum = 0.0;
    uint w_base = row * pc.n_in;
    for (uint k = lid; k < pc.n_in; k += 256) {
        sum += x[xb + k] * w.v[w_base + k];
    }
    partial[lid] = sum;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        barrier();
    }
    if (lid == 0) {
        y[pc.y_offset + slot * pc.y_stride + row] = partial[0];
    }
}
