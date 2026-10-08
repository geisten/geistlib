/*
 * hadamard.c — blockwise Walsh-Hadamard activation transform. See
 * hadamard.h and struct geist_hadamard_args.
 */
#include "hadamard.h"

#include "checked.h"
#include "fwht.h"
#include "par.h"

#include <stdint.h>
#include <string.h>

/* Copy n floats, scaling by sg when there is a sign vector. The forward
 * direction folds S into the copy or permute it has to do anyway, instead
 * of walking the row a second time for one multiply per element. */
static void copy_maybe_scaled(size_t n, const float *src, const float *sg, float *dst) {
    if (sg == nullptr) {
        memcpy(dst, src, n * sizeof *dst);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] * sg[i];
    }
}

static bool ranges_overlap(const float *a, const float *b, size_t n) {
    const uintptr_t pa = (uintptr_t) a;
    const uintptr_t pb = (uintptr_t) b;
    const uintptr_t nb = (uintptr_t) n * sizeof(float);
    return pa < pb + nb && pb < pa + nb;
}

bool geist_hadamard_geometry_ok(
        size_t width, size_t block, size_t perm_hd, size_t perm_nk, size_t perm_rep, bool inverse) {
    if (!fwht_supported(block) || width == 0 || width % block != 0) {
        return false;
    }
    if (perm_rep <= 1) {
        return true;
    }
    size_t n = 0;
    return !inverse && !ckd_mul(&n, perm_hd, perm_nk) && !ckd_mul(&n, n, perm_rep) && n == width;
}

/* geist_hadamard_rows's operands, for geist_par_for. */
struct hadamard_job {
    size_t       width;
    size_t       block;
    size_t       perm_hd;
    size_t       perm_nk;
    size_t       perm_rep;
    bool         inverse;
    bool         permute;
    const float *x;
    const float *signs;
    float       *y;
};

/* Rows [r0, r1). */
static void hadamard_row_range(void *ctx, size_t r0, size_t r1) {
    const struct hadamard_job *job      = ctx;
    const size_t               width    = job->width;
    const size_t               block    = job->block;
    const size_t               perm_hd  = job->perm_hd;
    const size_t               perm_nk  = job->perm_nk;
    const size_t               perm_rep = job->perm_rep;
    const bool                 inverse  = job->inverse;
    const bool                 permute  = job->permute;
    const float               *x        = job->x;
    const float               *signs    = job->signs;
    float                     *y        = job->y;
    for (size_t r = r0; r < r1; r++) {
        const float *xr = x + r * width;
        float       *yr = y + r * width;
        /* S is folded into the gather/copy on the forward path; on the
         * inverse it comes after H, where it rides along with each
         * block while that block is still in L1. */
        const float *fwd_signs = inverse ? nullptr : signs;
        if (permute) {
            for (size_t rep = 0; rep < perm_rep; rep++) {
                for (size_t k = 0; k < perm_nk; k++) {
                    const size_t dst = perm_hd * (rep + perm_rep * k);
                    copy_maybe_scaled(perm_hd,
                                      xr + perm_hd * (k + perm_nk * rep),
                                      fwd_signs != nullptr ? fwd_signs + dst : nullptr,
                                      yr + dst);
                }
            }
        } else if (yr != xr) {
            copy_maybe_scaled(width, xr, fwd_signs, yr);
        } else if (fwd_signs != nullptr) {
            /* In place: there is no copy to fold into. */
            for (size_t i = 0; i < width; i++) {
                yr[i] *= fwd_signs[i];
            }
        }
        for (size_t b = 0; b < width; b += block) {
            fwht_orthonormal(block, yr + b);
            if (signs != nullptr && inverse) {
                for (size_t i = b; i < b + block; i++) {
                    yr[i] *= signs[i];
                }
            }
        }
    }
}

enum geist_status geist_hadamard_rows(size_t       rows,
                                      size_t       width,
                                      size_t       block,
                                      size_t       perm_hd,
                                      size_t       perm_nk,
                                      size_t       perm_rep,
                                      bool         inverse,
                                      const float *x,
                                      const float *signs,
                                      float       *y) {
    const bool permute = perm_rep > 1;
    if (!geist_hadamard_geometry_ok(width, block, perm_hd, perm_nk, perm_rep, inverse)) {
        return GEIST_E_INVALID_ARG;
    }
    if (rows == 0) {
        return GEIST_OK;
    }
    size_t total = 0;
    if (x == nullptr || y == nullptr || ckd_mul(&total, rows, width)) {
        return GEIST_E_INVALID_ARG;
    }
    if (x != y && ranges_overlap(x, y, total)) {
        return GEIST_E_INVALID_ARG;
    }
    if (permute && x == y) {
        return GEIST_E_INVALID_ARG;
    }

    /* Rows are independent: split them across threads for prefill (decode
     * is one row, which geist_par_for runs on the caller). */
    const struct hadamard_job job = {.width    = width,
                                     .block    = block,
                                     .perm_hd  = perm_hd,
                                     .perm_nk  = perm_nk,
                                     .perm_rep = perm_rep,
                                     .inverse  = inverse,
                                     .permute  = permute,
                                     .x        = x,
                                     .signs    = signs,
                                     .y        = y};
    geist_par_for(rows, hadamard_row_range, (void *) &job);
    return GEIST_OK;
}

enum geist_status geist_hadamard_apply(const struct geist_hadamard_args *args,
                                       size_t                            nx,
                                       size_t                            ns,
                                       size_t                            ny,
                                       const float                      *x,
                                       const float                      *signs,
                                       float                            *y) {
    if (args == nullptr || args->x == nullptr || x == nullptr || y == nullptr ||
        args->x->ndim < 1 || args->x->shape[args->x->ndim - 1] <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t width = (size_t) args->x->shape[args->x->ndim - 1];
    if (nx != ny || nx % width != 0 || (args->signs != nullptr && ns != width)) {
        return GEIST_E_INVALID_ARG;
    }
    return geist_hadamard_rows(nx / width,
                               width,
                               args->block,
                               args->perm_hd,
                               args->perm_nk,
                               args->perm_rep,
                               args->inverse,
                               x,
                               args->signs != nullptr ? signs : nullptr,
                               y);
}
