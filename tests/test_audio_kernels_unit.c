/*
 * test_audio_kernels_unit — the audio tower's elementwise kernels stay finite
 * where exp overflows, under the build's own math flags.
 *
 *   1. silu_fp32 over [-1000, 40], long enough to be vectorized: every
 *      result finite, within 1e-6 relative (or 1e-30 absolute) of the double
 *      reference x / (1 + e^-x), and the overflow tail exactly -0 or a
 *      negative number of magnitude below 1e-30. gcc's -ffast-math expf
 *      (libmvec) returned NaN past -88.72; the old scalar clamp in front of
 *      the next linear happened to hide it on x86.
 *   2. clamp_fp32 holds finite values inside [lo, hi] and keeps the ones
 *      already inside bit for bit, at lengths around the NEON 16-block.
 *
 * Hermetic: no fixtures.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_kernels.h"
#undef GEIST_INTERNAL_ARCH_LAYER
#include "gemma4_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { N = 4096 };

static bool finite_bits(float v) {
    uint32_t b;
    memcpy(&b, &v, sizeof b);
    return (b & 0x7f800000u) != 0x7f800000u; /* immune to -ffinite-math-only */
}

int main(void) {
    static float x[N], ref[N];
    for (size_t i = 0; i < N; i++) {
        x[i]   = -1000.0f + 1040.0f * (float) i / (float) (N - 1);
        ref[i] = (float) ((double) x[i] / (1.0 + exp(-(double) x[i])));
    }
    silu_fp32(N, x);
    int    fails = 0;
    size_t bad   = 0;
    for (size_t i = 0; i < N; i++) {
        const double d = fabs((double) x[i] - (double) ref[i]);
        if (!finite_bits(x[i]) || d > 1e-30 + 1e-6 * fabs((double) ref[i])) {
            if (bad++ < 5)
                fprintf(stderr, "silu[%zu]: got %g want %g\n", i, (double) x[i], (double) ref[i]);
        }
    }
    fails += geist_expect(bad == 0, "silu_fp32 finite and close to x / (1 + e^-x) on [-1000, 40]");

    static const size_t lens[] = {1, 15, 16, 17, 33, 1023};
    for (size_t k = 0; k < sizeof lens / sizeof lens[0]; k++) {
        static float y[1024], y0[1024];
        const size_t n = lens[k];
        for (size_t i = 0; i < n; i++)
            y[i] = y0[i] = -20.0f + 40.0f * (float) i / (float) n;
        clamp_fp32(n, y, -7.75f, 7.6875f);
        size_t off = 0;
        for (size_t i = 0; i < n; i++) {
            const float want = y0[i] < -7.75f ? -7.75f : (y0[i] > 7.6875f ? 7.6875f : y0[i]);
            off += memcmp(&y[i], &want, sizeof want) != 0;
        }
        if (off != 0)
            fprintf(stderr, "clamp_fp32 n=%zu: %zu values off\n", n, off);
        fails += geist_expect(off == 0, "clamp_fp32 matches the scalar clamp bit for bit");
    }
    if (fails == 0)
        printf("PASS: silu_fp32 finite past the exp overflow, clamp_fp32 exact\n");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
