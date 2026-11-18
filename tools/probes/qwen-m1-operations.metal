inline float q1_simd_sum(float value) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    value += simd_shuffle_xor(value, ushort(1));
    value += simd_shuffle_xor(value, ushort(2));
    value += simd_shuffle_xor(value, ushort(4));
    value += simd_shuffle_xor(value, ushort(8));
    return value + simd_shuffle_xor(value, ushort(16));
}

inline float q1_nonpositive_exp(float value, device const uint* table) {
    return as_type<float>(q1_exp_bits(as_type<uint>(value) & 0x7fffffffu, table));
}

inline float q1_positive_rcp(float value, device const uint* table) {
    const uint bits = as_type<uint>(value);
    const uint exponent = bits >> 23;
    const uint key = (bits & 0x7fffffu) | 0x3f800000u;
    const uint rcp = q1_lookup(key, table);
    // Attention sums are positive normal FP32 values in [1, context length].
    return as_type<float>(rcp + ((127u - exponent) << 23));
}
