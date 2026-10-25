/* The grouped matvecs' ABI (#78 phase 3), shared by matvec_*_id_body.glsl:
 * workgroup y picks the slot, whose weight is read by device address and
 * whose x and y rows are slot * stride past the offsets. Defines the push
 * block, the _sel variant's bindings, MV_ADDR(s) (slot s's weight address)
 * and MV_SKIP_COLD(s), plus the slot-relative X_BASE / Y_BASE / W_ROW the
 * matvec_*_core.glsl bodies use. Bindings 0/1 (x, y) are the includer's. */

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
/* a cold expert: the host runs it */
#define MV_SKIP_COLD(s)                                                       \
    if (MV_ADDR(s) == uvec2(0u)) {                                            \
        return;                                                               \
    }
#else
#define MV_ADDR(s) pc.addr[s]
#define MV_SKIP_COLD(s)
#endif

/* x / y element offset of this slot's row; a weight row offset needs no
 * base (each slot's address points at its weight). */
#define X_BASE (pc.x_offset + slot * pc.x_stride)
#define Y_BASE (pc.y_offset + slot * pc.y_stride)
#define W_ROW(e) (e)
