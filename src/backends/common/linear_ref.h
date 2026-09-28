/*
 * src/backends/common/linear_ref.h — the linear kernel that cannot fail.
 *
 * y = x · Wᵀ for m rows of x, for any weight whose dtype it decodes
 * (geist_linear_ref_decodes). Each row of W is decoded a tile at a time
 * into a stack buffer and dotted in double, then rounded to float. It
 * allocates nothing, so it has no failure path: it is cpu_scalar's kernel
 * for block-quantized and half-precision weights, and what a CPU backend's
 * fast kernel computes instead when that kernel's scratch cannot be had,
 * or when it is given more rows than it holds accumulators for. Such a
 * kernel returns void (geist_weight.h), so before this the caller got a y
 * of zeros, or an unwritten one, and no error.
 *
 * The weight must have passed quant_weight_extent_ok, as every
 * resolve_weight checks: rows of whole blocks, a source long enough. x is
 * [m, n_in] and y [m, n_out], both row-major, with the dimensions in w.
 */
#ifndef GEIST_BACKENDS_COMMON_LINEAR_REF_H
#define GEIST_BACKENDS_COMMON_LINEAR_REF_H

#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>

/* Whether geist_linear_ref can decode `dtype`: F32, F16, BF16 and every
 * block format with a row codec in quant.h, I2_S included. */
[[nodiscard]] bool geist_linear_ref_decodes(uint16_t dtype);

/* y = x · Wᵀ. For a dtype it cannot decode it writes zeros; resolvers bind
 * it only for the ones it can. */
void geist_linear_ref(size_t m, const float *x, const struct geist_weight *w, float *y);

/* Output columns [j0, j0 + nj) of the same product: y[i * ldy + (j - j0)]
 * for each of the m rows of x. For a kernel that splits the output rows of
 * W over threads and has to finish one thread's share without scratch. */
void geist_linear_ref_rows(size_t                     m,
                           size_t                     j0,
                           size_t                     nj,
                           size_t                     ldy,
                           const float               *x,
                           const struct geist_weight *w,
                           float                     *y);

#endif /* GEIST_BACKENDS_COMMON_LINEAR_REF_H */
