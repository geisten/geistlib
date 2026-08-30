/*
 * src/backends/common/tensor_view.h — validating a geist_tensor view
 * against the buffer it points into.
 *
 * Layer: BACKEND (shared by cpu_scalar, cpu_neon, cpu_x86).
 *
 * Every elementwise op used to unpack a view the same way and check none
 * of it: fold shape[0..ndim) into an element count with an unchecked
 * multiply, add `offset` to the host pointer, and start writing. `ndim`
 * was only tested for >= 1 although shape[] holds eight entries, the
 * product could wrap, the byte count could exceed the buffer, and nothing
 * looked at alignment. Three near-identical copies of that, one per CPU
 * backend.
 *
 * This is the one copy, and it checks. It takes the host pointer and its
 * extent rather than the buffer, because struct geist_buffer is defined
 * privately by each backend.
 */
#pragma once

#include "checked.h"

#include <geist_types.h>

#include <stdalign.h>
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

/* Validate a contiguous F32 DENSE view against `host` / `host_bytes` and
 * return the pointer to its first element, or nullptr if anything about
 * the view does not hold: wrong dtype or layout, bad ndim, a non-positive
 * or unrepresentable dimension, an element count that overflows, a byte
 * range that runs past the buffer, or a start address that is not aligned
 * for a float.
 *
 * `host_bytes` is the buffer's own size, so this catches a view that is
 * internally consistent but larger than the memory behind it.
 *
 * Call once per op, outside the element loop: for the tensors these ops
 * run on (thousands of elements) the cost is not measurable, and the
 * alternative is per-element bounds checking or none at all. */
[[nodiscard]] static inline float *
geist_tensor_f32_dense(const struct geist_tensor *t, void *host, size_t host_bytes, size_t *out_n) {
    if (t == nullptr || host == nullptr || t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE) {
        return nullptr;
    }
    size_t n = 0;
    if (geist_tensor_elems(t, &n)) {
        return nullptr;
    }
    size_t bytes = 0;
    size_t end   = 0;
    if (ckd_mul(&bytes, n, sizeof(float)) || ckd_add(&end, t->offset, bytes) || end > host_bytes) {
        return nullptr;
    }
    uint8_t *p = (uint8_t *) host + t->offset;
    if (((uintptr_t) p % alignof(float)) != 0u) {
        return nullptr;
    }
    *out_n = n;
    return (float *) p;
}
