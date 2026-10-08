/* silu(x) = x * sigmoid(x), overflow-safe form (same as the host silu and
 * the DeltaNet oracle). Shared by silu, silu_mul and the DeltaNet kernels. */
float silu(float v) {
    float e = exp(-abs(v));
    return v >= 0.0 ? v / (1.0 + e) : (v * e) / (1.0 + e);
}
