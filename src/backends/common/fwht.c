/*
 * fwht.c — orthonormal Fast Walsh–Hadamard Transform. See fwht.h.
 */
#include "fwht.h"

#include <math.h>

void fwht_orthonormal(size_t n, float a[static n]) {
    if (n < 2) {
        return; /* H_1 = [1], and 1/sqrt(1) == 1 */
    }
    /* Butterfly passes: unnormalized Hadamard (H_n = H_2 ⊗ H_{n/2}), all
     * but the last. The passes of len 1, 2 and 4 have inner loops of 1, 2
     * and 4 butterflies, too short for the compiler to vectorize, and they
     * ran scalar: 2.0 ns per float of a 1024-float block on x86-64 (AVX2),
     * against 0.63 ns with each spelled out per group of 2 len as one loop
     * over the block, which it vectorizes. Same butterflies in the same
     * order as the general loop, so the result does not move by a bit; the
     * Vulkan and Metal ports run that order too (test_fwht_unit pins it). */
    const size_t half = n >> 1;
    if (half > 1) {
        for (size_t j = 0; j < n; j += 2) {
            const float x = a[j];
            const float y = a[j + 1];
            a[j]          = x + y;
            a[j + 1]      = x - y;
        }
    }
    if (half > 2) {
        for (size_t j = 0; j < n; j += 4) {
            const float x0 = a[j];
            const float x1 = a[j + 1];
            const float y0 = a[j + 2];
            const float y1 = a[j + 3];
            a[j]           = x0 + y0;
            a[j + 1]       = x1 + y1;
            a[j + 2]       = x0 - y0;
            a[j + 3]       = x1 - y1;
        }
    }
    if (half > 4) {
        for (size_t j = 0; j < n; j += 8) {
            float x[4], y[4];
            for (size_t k = 0; k < 4; k++) {
                x[k] = a[j + k];
                y[k] = a[j + 4 + k];
            }
            for (size_t k = 0; k < 4; k++) {
                a[j + k]     = x[k] + y[k];
                a[j + 4 + k] = x[k] - y[k];
            }
        }
    }
    for (size_t len = 8; len < half; len <<= 1) {
        for (size_t i = 0; i < n; i += (len << 1)) {
            for (size_t j = i; j < i + len; j++) {
                const float x = a[j];
                const float y = a[j + len];
                a[j]          = x + y;
                a[j + len]    = x - y;
            }
        }
    }
    /* The last stage spans the whole block (one iteration of the outer
     * loop above), so the 1/sqrt(n) that makes the transform orthonormal,
     * self-inverse and dot-preserving folds into it. It used to be its own
     * pass over a[], which on the prism.hadamard path is ~2.1 M floats per
     * token re-read and rewritten for one multiply. Same two operations
     * per element in the same order as before, so the result is unchanged
     * bit for bit (test_fwht_unit pins the properties; the rotation tests
     * pin the values). */
    const float s = 1.0f / sqrtf((float) n);
    for (size_t j = 0; j < half; j++) {
        const float x = a[j];
        const float y = a[j + half];
        a[j]          = (x + y) * s;
        a[j + half]   = (x - y) * s;
    }
}
