/* Shared body of the unary elementwise kernels: y[i] = ew_op(x[i]) over n
 * contiguous elements. The includer defines float ew_op(float) first.
 * In place is fine. Dispatch: gx = ceil(n / 256). */

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X {
    float x[];
};

layout(set = 0, binding = 1) writeonly buffer Y {
    float y[];
};

layout(push_constant) uniform Push {
    uint n;
    uint x_offset;
    uint y_offset;
    uint pad;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) {
        return;
    }
    float v = x[pc.x_offset + i];
    y[pc.y_offset + i] = ew_op(v);
}
