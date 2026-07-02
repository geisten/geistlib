/*
 * src/backends/cpu_neon/elementwise.c — wraps gemma4_kernels.c for the
 * elementwise + rmsnorm ops in the backend vtable.
 *
 * Layer: BACKEND.
 *
 * These ops are FP32 and not currently NEON-specialized in
 * gemma4_kernels.c (they're scalar with potential gcc auto-vectorization);
 * the cpu_neon backend reuses the same kernels for now. When a true
 * NEON specialization lands (e.g. vrsqrteq_f32 for rmsnorm), it slots
 * in here without touching the engine.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "gemma4_kernels.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
extern void vvtanhf(float *y, const float *x, const int *n);
#endif

static float *get_f32_dense_ptr(const struct geist_tensor *t, size_t *out_n) {
    if (t == nullptr || t->dtype != GEIST_DTYPE_F32 || t->layout != GEIST_LAYOUT_DENSE ||
        t->buffer == nullptr || t->ndim < 1) {
        return nullptr;
    }
    size_t n = 1;
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0) {
            return nullptr;
        }
        n *= (size_t) t->shape[d];
    }
    *out_n = n;
    return (float *) ((uint8_t *) t->buffer->host + t->offset);
}

[[nodiscard]] enum geist_status cpu_neon_add(struct geist_backend      *be,
                                             const struct geist_tensor *a,
                                             const struct geist_tensor *b,
                                             struct geist_tensor       *y) {
    size_t       na = 0, nb = 0, ny = 0;
    const float *ap = get_f32_dense_ptr(a, &na);
    const float *bp = get_f32_dense_ptr(b, &nb);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (ap == nullptr || bp == nullptr || yp == nullptr || na != nb || na != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon add: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    add_fp32(ap, bp, na, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_mul(struct geist_backend      *be,
                                             const struct geist_tensor *a,
                                             const struct geist_tensor *b,
                                             struct geist_tensor       *y) {
    size_t       na = 0, nb = 0, ny = 0;
    const float *ap = get_f32_dense_ptr(a, &na);
    const float *bp = get_f32_dense_ptr(b, &nb);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (ap == nullptr || bp == nullptr || yp == nullptr || na != nb || na != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    mul_fp32(ap, bp, na, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
cpu_neon_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon gelu_tanh: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    gelu_tanh_fp32(xp, nx, yp);
    return GEIST_OK;
}

#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
/* Shared core of the Accelerate gelu-tanh fast paths: grow the elementwise
 * scratch and fill it with tanh(kAlpha*(x + kBeta*x^3)) via vvtanhf.
 * Returns the scratch, or nullptr when the fast path can't run (no state,
 * n too large, OOM) — caller falls back to the scalar kernel. */
static float *gelu_vvtanh_inner(struct geist_backend *be, const float *xp, size_t nx) {
    if (be == nullptr || be->state == nullptr || nx > (size_t) INT32_MAX) {
        return nullptr;
    }
    struct cpu_neon_state     *st = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws = &st->workspace;
    if (ws->elt_f32_cap < nx) {
        safe_free((void **) &ws->elt_f32);
        ws->elt_f32 = heap_alloc_array_aligned(float, nx);
        if (ws->elt_f32 == nullptr) {
            ws->elt_f32_cap = 0;
            return nullptr;
        }
        ws->elt_f32_cap = nx;
    }
    const float kAlpha = 0.7978845608028654f;
    const float kBeta  = 0.044715f;
    float      *tmp    = ws->elt_f32;
    for (size_t i = 0; i < nx; i++) {
        const float xi = xp[i];
        tmp[i]         = kAlpha * (xi + kBeta * xi * xi * xi);
    }
    int n_i32 = (int) nx;
    vvtanhf(tmp, tmp, &n_i32);
    return tmp;
}
#endif

[[nodiscard]] enum geist_status cpu_neon_gelu_tanh_mul(struct geist_backend      *be,
                                                       const struct geist_tensor *x,
                                                       const struct geist_tensor *z,
                                                       struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    const float *zp = get_f32_dense_ptr(z, &nz);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || nx != nz || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon gelu_tanh_mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    const float *tmp = gelu_vvtanh_inner(be, xp, nx);
    if (tmp != nullptr) {
        for (size_t i = 0; i < nx; i++) {
            yp[i] = (0.5f * xp[i] * (1.0f + tmp[i])) * zp[i];
        }
        return GEIST_OK;
    }
#endif
    gelu_tanh_mul_fp32(xp, zp, nx, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *z,
                                                              const float               *scale,
                                                              struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    const float *zp = get_f32_dense_ptr(z, &nz);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || scale == nullptr || nx != nz ||
        nx != ny || x->ndim < 1 || z->ndim < 1 || y->ndim < 1) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_neon gelu_tanh_mul_scaled: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    const size_t feat = (size_t) y->shape[y->ndim - 1];
    if (feat == 0 || nx % feat != 0) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_neon gelu_tanh_mul_scaled: feature mismatch");
        return GEIST_E_INVALID_ARG;
    }
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    const float *tmp = gelu_vvtanh_inner(be, xp, nx);
    if (tmp != nullptr) {
        for (size_t i = 0; i < nx; i++) {
            yp[i] = (0.5f * xp[i] * (1.0f + tmp[i])) * zp[i] * scale[i % feat];
        }
        return GEIST_OK;
    }
#endif
    const float  kAlpha = 0.7978845608028654f;
    const float  kBeta  = 0.044715f;
    const size_t rows   = nx / feat;
    for (size_t r = 0; r < rows; r++) {
        const size_t base = r * feat;
        for (size_t j = 0; j < feat; j++) {
            const size_t i     = base + j;
            const float  xi    = xp[i];
            const float  inner = kAlpha * (xi + kBeta * xi * xi * xi);
            yp[i]              = (0.5f * xi * (1.0f + tanhf(inner))) * zp[i] * scale[j];
        }
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_relu_squared(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      struct geist_tensor       *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon relu_squared: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    relu_squared_fp32(xp, nx, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
cpu_neon_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon silu: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    silu_fp32_ooo(xp, nx, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_rmsnorm(struct geist_backend      *be,
                                                 const struct geist_tensor *x,
                                                 const struct geist_tensor *w,
                                                 float                      eps,
                                                 struct geist_tensor       *y) {
    size_t       nx = 0, nw = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    const float *wp = get_f32_dense_ptr(w, &nw);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || wp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon rmsnorm: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    size_t feat = (size_t) x->shape[x->ndim - 1];
    if (feat == 0 || nw != feat || nx % feat != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon rmsnorm: feat mismatch");
        return GEIST_E_INVALID_ARG;
    }
    size_t n_rows = nx / feat;
    rmsnorm_fp32(xp, wp, n_rows, feat, eps, yp);
    return GEIST_OK;
}
