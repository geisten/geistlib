/*
 * src/backends/cpu_scalar/elementwise.c — pure-C reference for the
 * basic per-element ops used by transformer forward passes.
 *
 * Layer: BACKEND.
 *
 * Provides scalar reference implementations of rmsnorm, add, mul, and
 * gelu_tanh. cpu_neon wraps the same gemma4_kernels.c entry points; the
 * scalar version here uses straightforward C loops with double-precision
 * reduction for rmsnorm.
 *
 * Tensor contract for all ops: F32 DENSE, contiguous (stride = shape-tail).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Unpack a contiguous F32 DENSE tensor as a host float pointer + element
 * count. Returns nullptr on layout mismatch. */
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

/* ---- add: y = a + b ---- */

[[nodiscard]] enum geist_status cpu_scalar_add(struct geist_backend      *be,
                                               const struct geist_tensor *a,
                                               const struct geist_tensor *b,
                                               struct geist_tensor       *y) {
    if (be == nullptr || a == nullptr || b == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       na = 0, nb = 0, ny = 0;
    const float *ap = get_f32_dense_ptr(a, &na);
    const float *bp = get_f32_dense_ptr(b, &nb);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (ap == nullptr || bp == nullptr || yp == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "cpu_scalar add: all tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (na != nb || na != ny) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_scalar add: shape mismatch (a=%zu b=%zu y=%zu)",
                                na,
                                nb,
                                ny);
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < na; i++) {
        yp[i] = ap[i] + bp[i];
    }
    return GEIST_OK;
}

/* ---- mul: y = a * b (element-wise) ---- */

[[nodiscard]] enum geist_status cpu_scalar_mul(struct geist_backend      *be,
                                               const struct geist_tensor *a,
                                               const struct geist_tensor *b,
                                               struct geist_tensor       *y) {
    if (be == nullptr || a == nullptr || b == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       na = 0, nb = 0, ny = 0;
    const float *ap = get_f32_dense_ptr(a, &na);
    const float *bp = get_f32_dense_ptr(b, &nb);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (ap == nullptr || bp == nullptr || yp == nullptr || na != nb || na != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < na; i++) {
        yp[i] = ap[i] * bp[i];
    }
    return GEIST_OK;
}

/* ---- gelu_tanh: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) ---- */

[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh(struct geist_backend      *be,
                                                     const struct geist_tensor *x,
                                                     struct geist_tensor       *y) {
    if (be == nullptr || x == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar gelu_tanh: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    /* HF Gemma uses gelu_tanh (NewGELU). Constants match exactly. */
    static constexpr float K0 = 0.7978845608028654f; /* sqrt(2/pi) */
    static constexpr float K1 = 0.044715f;
    for (size_t i = 0; i < nx; i++) {
        float v = xp[i];
        float u = K0 * (v + K1 * v * v * v);
        yp[i]   = 0.5f * v * (1.0f + tanhf(u));
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh_mul(struct geist_backend      *be,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *z,
                                                         struct geist_tensor       *y) {
    if (be == nullptr || x == nullptr || z == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    const float *zp = get_f32_dense_ptr(z, &nz);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || nx != nz || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar gelu_tanh_mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    static constexpr float K0 = 0.7978845608028654f;
    static constexpr float K1 = 0.044715f;
    for (size_t i = 0; i < nx; i++) {
        const float v = xp[i];
        const float u = K0 * (v + K1 * v * v * v);
        yp[i]         = (0.5f * v * (1.0f + tanhf(u))) * zp[i];
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                                const struct geist_tensor *x,
                                                                const struct geist_tensor *z,
                                                                const float               *scale,
                                                                struct geist_tensor       *y) {
    if (be == nullptr || x == nullptr || z == nullptr || y == nullptr || scale == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    const float *zp = get_f32_dense_ptr(z, &nz);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || nx != nz || nx != ny || y->ndim < 1) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_scalar gelu_tanh_mul_scaled: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    const size_t feat = (size_t) y->shape[y->ndim - 1];
    if (feat == 0 || nx % feat != 0) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_scalar gelu_tanh_mul_scaled: feature mismatch");
        return GEIST_E_INVALID_ARG;
    }
    static constexpr float K0   = 0.7978845608028654f;
    static constexpr float K1   = 0.044715f;
    const size_t           rows = nx / feat;
    for (size_t r = 0; r < rows; r++) {
        const size_t base = r * feat;
        for (size_t j = 0; j < feat; j++) {
            const size_t i = base + j;
            const float  v = xp[i];
            const float  u = K0 * (v + K1 * v * v * v);
            yp[i]          = (0.5f * v * (1.0f + tanhf(u))) * zp[i] * scale[j];
        }
    }
    return GEIST_OK;
}

/* ---- relu_squared: y = max(x, 0)^2  (BitNet b1.58 2B-4T FFN) ---- */

[[nodiscard]] enum geist_status cpu_scalar_relu_squared(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        struct geist_tensor       *y) {
    if (be == nullptr || x == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar relu_squared: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        float v = xp[i] > 0.0f ? xp[i] : 0.0f;
        yp[i]   = v * v;
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
cpu_scalar_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    if (be == nullptr || x == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar silu: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        /* Overflow-safe sigmoid form: the exp argument is always <= 0, so
         * expf never returns inf. The naive v / (1 + expf(-v)) NaNs for
         * v < -88 under gcc's vectorized finite-math expf (libmvec skips
         * the overflow saturation the scalar libm does) — Qwen3-0.6B's
         * gate activations actually reach that range (#275). */
        const float v = xp[i];
        const float e = expf(-fabsf(v));
        yp[i]         = (v >= 0.0f) ? v / (1.0f + e) : (v * e) / (1.0f + e);
    }
    return GEIST_OK;
}

/* ---- rmsnorm: y = x * weight * rsqrt(mean(x^2) + eps) ----
 *
 * Operates per-row for 2D tensors (last dimension is the feature axis,
 * earlier dimensions are batch/sequence). For 1D x, single-row case.
 * weight has shape == x.shape[-1]. y can alias x. */

[[nodiscard]] enum geist_status cpu_scalar_rmsnorm(struct geist_backend      *be,
                                                   const struct geist_tensor *x,
                                                   const struct geist_tensor *w,
                                                   float                      eps,
                                                   struct geist_tensor       *y) {
    if (be == nullptr || x == nullptr || w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, nw = 0, ny = 0;
    const float *xp = get_f32_dense_ptr(x, &nx);
    const float *wp = get_f32_dense_ptr(w, &nw);
    float       *yp = get_f32_dense_ptr(y, &ny);
    if (xp == nullptr || wp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar rmsnorm: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    size_t feat = (size_t) x->shape[x->ndim - 1];
    if (feat == 0 || nw != feat || nx % feat != 0) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_scalar rmsnorm: feature size %zu mismatch (w=%zu)",
                                feat,
                                nw);
        return GEIST_E_INVALID_ARG;
    }
    size_t n_rows = nx / feat;
    for (size_t r = 0; r < n_rows; r++) {
        const float *row_x = xp + r * feat;
        float       *row_y = yp + r * feat;
        double       sumsq = 0.0;
        for (size_t i = 0; i < feat; i++) {
            sumsq += (double) row_x[i] * (double) row_x[i];
        }
        float inv = (float) (1.0 / sqrt(sumsq / (double) feat + (double) eps));
        for (size_t i = 0; i < feat; i++) {
            row_y[i] = row_x[i] * inv * wp[i];
        }
    }
    return GEIST_OK;
}
