//
// Checked size_t arithmetic — the one place the engine does overflow-safe
// multiply/add/round-up.
//
// Model-controlled dimensions (GGUF tensor dims, image/video geometry, KV
// capacities) reach allocation and indexing math. An unchecked `a * b` there
// wraps silently: a huge logical request becomes a small allocation, and the
// write that follows runs off the end. Every size computed from untrusted
// input goes through these.
//
// Convention is C23's, not this file's invention: the helpers return
// **true on overflow** and write the result only when they return false,
// exactly like `ckd_mul`/`ckd_add`. Where the toolchain ships
// <stdckdint.h> those are the real thing; older GCC/clang get the builtins
// they have always had.
//
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__has_include)
#if __has_include(<stdckdint.h>)
#include <stdckdint.h>
#define GEIST_HAVE_STDCKDINT 1
#endif
#endif

#ifndef GEIST_HAVE_STDCKDINT
/* GCC/clang have had these builtins since long before C23; the header is the
 * only new part. Same argument order and same true-on-overflow result. */
#define ckd_add(r, a, b) __builtin_add_overflow((a), (b), (r))
#define ckd_sub(r, a, b) __builtin_sub_overflow((a), (b), (r))
#define ckd_mul(r, a, b) __builtin_mul_overflow((a), (b), (r))
#endif

/* Round `v` up to a multiple of `align`, which must be a non-zero power of
 * two. Returns true (and leaves *out untouched) on overflow or on an invalid
 * alignment — a wrapped round-up under-allocates, which is the bug this
 * exists to prevent. */
[[nodiscard]] static inline bool
geist_ckd_round_up_pow2(const size_t v, const size_t align, size_t *out) {
    if (align == 0u || (align & (align - 1u)) != 0u) {
        return true;
    }
    if (v > SIZE_MAX - (align - 1u)) {
        return true;
    }
    *out = (v + align - 1u) & ~(align - 1u);
    return false;
}

/* Product of `n` size_t terms. Returns true on overflow; *out is written only
 * on success. An empty product is 1 (the identity), matching how tensor
 * element counts fold over dimensions. */
[[nodiscard]] static inline bool
geist_ckd_mul_n(const size_t n, const size_t terms[static 1], size_t *out) {
    size_t acc = 1u;
    for (size_t i = 0; i < n; i++) {
        if (ckd_mul(&acc, acc, terms[i])) {
            return true;
        }
    }
    *out = acc;
    return false;
}
