/* Shared body of the binary elementwise kernels: y = ew_op(a, b). The
 * includer defines float ew_op(float a, float b) first. y may alias a.
 * cols == 0: all operands contiguous; otherwise (row, col) addressing with
 * per-operand row strides (slab views). Dispatch: gx = ceil(n / 256). */

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X {
    float x[];
};

layout(set = 0, binding = 1) readonly buffer Z {
    float z[];
};

layout(set = 0, binding = 2) writeonly buffer Y {
    float y[];
};

layout(push_constant) uniform Push {
    uint n;
    uint a_offset;
    uint b_offset;
    uint y_offset;
    uint cols;
    uint a_stride;
    uint b_stride;
    uint y_stride;
} pc;

uint addr(uint base, uint stride, uint i) {
    if (pc.cols == 0u) {
        return base + i;
    }
    return base + (i / pc.cols) * stride + (i % pc.cols);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) {
        return;
    }
    float a = x[addr(pc.a_offset, pc.a_stride, i)];
    float b = z[addr(pc.b_offset, pc.b_stride, i)];
    y[addr(pc.y_offset, pc.y_stride, i)] = ew_op(a, b);
}
