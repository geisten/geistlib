/*
 * fwht.c — orthonormal Fast Walsh–Hadamard Transform. See fwht.h.
 */
#include "fwht.h"

#include <math.h>

void fwht_orthonormal(float *a, size_t n) {
    /* Butterfly passes: unnormalized Hadamard (H_n = H_2 ⊗ H_{n/2}). */
    for (size_t len = 1; len < n; len <<= 1) {
        for (size_t i = 0; i < n; i += (len << 1)) {
            for (size_t j = i; j < i + len; j++) {
                const float x = a[j];
                const float y = a[j + len];
                a[j]          = x + y;
                a[j + len]    = x - y;
            }
        }
    }
    /* Normalize by 1/sqrt(n) → orthonormal, self-inverse, dot-preserving. */
    const float s = 1.0f / sqrtf((float) n);
    for (size_t i = 0; i < n; i++) {
        a[i] *= s;
    }
}
