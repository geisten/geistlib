/*
 * fwht.h — in-place orthonormal Fast Walsh–Hadamard Transform.
 *
 * Rotation used to suppress activation outliers before INT8 KV-cache
 * quantization (issue #61, after llama.cpp#21038 / QuaRot). The Hadamard
 * matrix normalized by 1/sqrt(n) is orthonormal, so:
 *   - it is its own inverse:  fwht(fwht(x)) == x  (modulo fp rounding);
 *   - it preserves dot products:  (Hx)·(Hy) == x·y.
 * The first property lets one transform serve both the forward rotation
 * (of Q/K/V) and the backward rotation (of the attention output); the
 * second is why rotating Q and K by the same H leaves the QK scores intact.
 */
#ifndef GEIST_FWHT_H
#define GEIST_FWHT_H

#include <stdbool.h>
#include <stddef.h>

/* True iff n is a nonzero power of two — the FWHT precondition. Callers
 * gate on this and fall back to the unrotated path when it fails. */
static inline bool fwht_supported(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

/* In-place orthonormal FWHT over a[0..n). n MUST be a power of two
 * (see fwht_supported). O(n log n). */
void fwht_orthonormal(float *a, size_t n);

#endif /* GEIST_FWHT_H */
