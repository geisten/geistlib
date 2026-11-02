/* Independent fixture and arithmetic helpers for PQ2 GPU tests/benchmarks. */
#pragma once

static uint32_t random_state = 0x1731u;
static uint8_t  random_byte(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return (uint8_t) (random_state >> 24);
}

static void fill_weights(size_t bytes, uint8_t out[static bytes], enum geist_dtype dtype) {
    const uint16_t scales[] = {0x3555, 0x3800, 0x3c00, 0xbc00, 0x4000};
    for (size_t off = 0; off < bytes; off += 34u) {
        for (size_t i = 0; i < 34u; i++)
            out[off + i] = random_byte();
        const uint16_t d = dtype == GEIST_DTYPE_PQ2_0 ? scales[(off / 34u) % 5u] : 0x3c00;
        memcpy(out + off, &d, 2);
    }
}

static float round_block(unsigned mode, float amax, float value) {
    if (mode == 7)
        mode = 1;
    if (mode == 8)
        mode = 2;
    if (mode == 5)
        return value;
    if (amax == 0 || value == 0)
        return 0;
    const double limit    = mode == 2 ? 448.0 : mode == 4 ? 32752.0 : 127.0;
    int          exponent = (int) ceil(log2((double) amax / limit));
    if (exponent < -126)
        exponent = -126;
    const float scale = ldexpf(1.0f, exponent);
    if (mode == 2) {
        const int    ev   = (int) floor(log2(fabs((double) value)));
        const double step = ldexp(1.0, ev - 3 > exponent - 9 ? ev - 3 : exponent - 9);
        return (float) (nearbyint((double) value / step) * step);
    }
    if (mode == 4)
        return fp16_to_fp32(fp32_to_fp16_rne(value / scale)) * scale;
    volatile float normalized = value / scale;
    return nearbyintf(normalized) * scale;
}
