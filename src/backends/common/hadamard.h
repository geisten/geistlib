/*
 * hadamard.h — blockwise Walsh-Hadamard activation transform for models
 * stored in a rotated weight basis (prism.hadamard GGUF keys).
 *
 * Layer: BACKEND (shared host implementation behind fused->hadamard_rotate;
 * cpu_scalar and cpu_neon install it, and cpu_x86 inherits cpu_scalar's
 * fused table; the per-backend glue stays per backend because struct
 * geist_buffer is backend-private). The math is spelled out on
 * struct geist_hadamard_args in geist_backend.h; this file implements it
 * on host rows.
 */
#pragma once

#include <geist_backend.h>
#include <geist_types.h>

#include <stdbool.h>
#include <stddef.h>

/* Transform `rows` rows of `width` floats from x into y. Parameters mirror
 * struct geist_hadamard_args. y may equal x when perm_rep <= 1; any other
 * overlap between x and y is rejected. signs is nullable (all +1).
 *
 * Returns GEIST_E_INVALID_ARG for a block that is not a power of two or
 * does not divide width, a permutation whose geometry does not multiply
 * out to width, a permutation on the inverse transform, or overlapping
 * buffers under a permutation. y is untouched on failure. */
[[nodiscard]] enum geist_status geist_hadamard_rows(size_t       rows,
                                                    size_t       width,
                                                    size_t       block,
                                                    size_t       perm_hd,
                                                    size_t       perm_nk,
                                                    size_t       perm_rep,
                                                    bool         inverse,
                                                    const float *x,
                                                    const float *signs,
                                                    float       *y);

/* Backend glue: validate the host views a backend resolved for args->x,
 * args->signs and args->y (element counts nx / ns / ny; ns ignored when
 * signs is nullptr) against args, then run geist_hadamard_rows. rows and
 * width come from args->x's leading and last dimensions. */
[[nodiscard]] enum geist_status geist_hadamard_apply(const struct geist_hadamard_args *args,
                                                     size_t                            nx,
                                                     size_t                            ns,
                                                     size_t                            ny,
                                                     const float                      *x,
                                                     const float                      *signs,
                                                     float                            *y);
