/*
 * test_int4_kv_unit — verifies the symmetric 4-bit pack/unpack used by the
 * packed-INT4 KV cache (issue #61). The failure-prone part is nibble
 * sign-extension: a wrong shift silently turns -1 (0xF) into +15.
 *
 * Deterministic — no model needed.
 */
#include "int4_kv.h"
#include "test_helpers.h"

#include <math.h>

/* Scenario 1: every representable level round-trips exactly (inv=1 → q=x). */
static int test_levels_exact(void) {
    /* All 15 symmetric levels [-7,7], padded to an even count. */
    float x[16];
    for (int i = 0; i < 15; i++)
        x[i] = (float) (i - 7);
    x[15] = 0.0f;
    uint8_t packed[8];
    int8_t  out[16];
    int4_pack_row(x, 1.0f, packed, 16);
    int4_unpack_row(packed, out, 16);
    for (size_t i = 0; i < 16; i++) {
        if (out[i] != (int8_t) lrintf(x[i])) {
            printf("level %zu: got %d want %d (packed sign-extension wrong?)\n",
                   i,
                   out[i],
                   (int) lrintf(x[i]));
            return 1;
        }
    }
    return 0;
}

/* Scenario 2: nibble ordering — low nibble is the even index, high the odd. */
static int test_nibble_order(void) {
    float   x[2] = {3.0f, -5.0f};
    uint8_t b;
    int4_pack_row(x, 1.0f, &b, 2);
    /* low = 3 (0x3), high = -5 (0xB) → byte 0xB3 */
    if (b != 0xB3) {
        printf("nibble order: byte=0x%02X want 0xB3\n", b);
        return 1;
    }
    int8_t out[2];
    int4_unpack_row(&b, out, 2);
    if (out[0] != 3 || out[1] != -5) {
        printf("nibble order unpack: %d,%d want 3,-5\n", out[0], out[1]);
        return 1;
    }
    return 0;
}

/* Scenario 3: scaled round-trip — dequant error stays within half a step. */
static int test_scaled_roundtrip(void) {
    enum { N = 256 };
    float    x[N];
    uint32_t seed = 0x1234u;
    float    amax = 0.0f;
    for (size_t i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        x[i] = ((float) (seed >> 8) / (float) 0xFFFFFF - 0.5f) * 4.0f; /* ~[-2,2] */
        if (fabsf(x[i]) > amax)
            amax = fabsf(x[i]);
    }
    const float scale = amax / 7.0f;
    uint8_t     packed[N / 2];
    int8_t      q[N];
    int4_pack_row(x, 1.0f / scale, packed, N);
    int4_unpack_row(packed, q, N);
    for (size_t i = 0; i < N; i++) {
        const float deq = (float) q[i] * scale;
        if (fabsf(deq - x[i]) > 0.5f * scale + 1e-4f) {
            printf("scaled[%zu]: x=%.4f deq=%.4f err=%.4f > %.4f\n",
                   i,
                   x[i],
                   deq,
                   fabsf(deq - x[i]),
                   0.5f * scale);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    if (test_levels_exact() == 0 && test_nibble_order() == 0 && test_scaled_roundtrip() == 0) {
        printf("PASS: int4 pack/unpack round-trips with correct sign-extension\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
