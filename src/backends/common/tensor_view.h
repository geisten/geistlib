/*
 * src/backends/common/tensor_view.h — validating a geist_tensor view
 * against the buffer it points into.
 *
 * Layer: BACKEND (shared by cpu_scalar, cpu_neon, cpu_x86).
 *
 * Checks ndim, each dimension, the element-count and byte-range overflow,
 * the extent against the buffer, and alignment. Takes the host pointer and
 * its extent rather than the buffer, because struct geist_buffer is
 * defined privately by each backend.
 */
#pragma once

#include "checked.h"

#include <geist_types.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Number of shape/stride slots in struct geist_tensor. */
constexpr int GEIST_TENSOR_MAX_DIMS = 8;

/* Element count of a tensor view: the product of its dimensions, with
 * every step checked. Returns true (failure) on a bad ndim, a
 * non-positive dimension, or overflow; *out is written only on success. */
[[nodiscard]] static inline bool geist_tensor_elems(const struct geist_tensor *t, size_t *out) {
    if (t == nullptr || t->ndim < 1 || t->ndim > GEIST_TENSOR_MAX_DIMS) {
        return true;
    }
    size_t n = 1;
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0) {
            return true;
        }
        if (ckd_mul(&n, n, (size_t) t->shape[d])) {
            return true;
        }
    }
    *out = n;
    return false;
}

/* Validate a contiguous DENSE view of `ndim` dimensions (exactly) and
 * `dtype`, whose elements are `elem` bytes, against `host` / `host_bytes`
 * and return the pointer to its first element, or nullptr if anything
 * about the view does not hold: wrong dtype, layout or ndim, a non-positive
 * or unrepresentable dimension, an element count that overflows, a byte
 * range that runs past the buffer, or a start address that is not aligned
 * to `elem`.
 *
 * `host_bytes` is the buffer's own size, so this catches a view that is
 * internally consistent but larger than the memory behind it.
 *
 * Call once per op, outside the element loop: for the tensors these ops
 * run on (thousands of elements) the cost is not measurable, and the
 * alternative is per-element bounds checking or none at all. */
[[nodiscard]] static inline void *geist_tensor_dense(const struct geist_tensor *t,
                                                     enum geist_dtype           dtype,
                                                     size_t                     elem,
                                                     int                        ndim,
                                                     void                      *host,
                                                     size_t                     host_bytes,
                                                     size_t                    *out_n) {
    if (t == nullptr || host == nullptr || t->dtype != dtype || t->layout != GEIST_LAYOUT_DENSE ||
        t->ndim != ndim) {
        return nullptr;
    }
    size_t n = 0, bytes = 0, end = 0;
    if (geist_tensor_elems(t, &n) || ckd_mul(&bytes, n, elem) || ckd_add(&end, t->offset, bytes) ||
        end > host_bytes) {
        return nullptr;
    }
    uint8_t *p = (uint8_t *) host + t->offset;
    if ((uintptr_t) p % elem != 0u) {
        return nullptr;
    }
    *out_n = n;
    return p;
}

/* geist_tensor_dense for an F32 view of any rank. */
[[nodiscard]] static inline float *
geist_tensor_f32_dense(const struct geist_tensor *t, void *host, size_t host_bytes, size_t *out_n) {
    return t == nullptr
                   ? nullptr
                   : (float *) geist_tensor_dense(
                             t, GEIST_DTYPE_F32, sizeof(float), t->ndim, host, host_bytes, out_n);
}
