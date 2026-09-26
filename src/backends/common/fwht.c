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
     * but the last. */
    const size_t half = n >> 1;
    for (size_t len = 1; len < half; len <<= 1) {
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
