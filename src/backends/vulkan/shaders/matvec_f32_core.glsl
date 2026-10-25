/* F32 matvec core, shared by matvec_f32.comp and the grouped
 * matvec_f32_id_body.glsl: workgroup `row` computes one output row. The
 * includer declares x, y and pc and defines MV_BEGIN (statements opening
 * main), MV_BIND (statements once the row is known to be in range),
 * W_LOAD(i) (weight element i), W_ROW(e) (element e of the weight rows),
 * X_ROW and Y_BASE (element offsets of the x and y rows; X_ROW is a plain
 * name or member, as it sits unparenthesized in the loop).
 * Dispatch: gx = n_out. */

layout(local_size_x = 256) in;

shared float partial[256];

void main() {
    uint row = gl_WorkGroupID.x;
    MV_BEGIN
    uint lid = gl_LocalInvocationID.x;

    if (row >= pc.n_out) {
        return;
    }
    MV_BIND

    float sum = 0.0;
    uint w_base = W_ROW(row * pc.n_in);
    for (uint k = lid; k < pc.n_in; k += 256) {
        sum += x[X_ROW + k] * W_LOAD(w_base + k);
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
        y[(Y_BASE) + row] = partial[0];
    }
}
