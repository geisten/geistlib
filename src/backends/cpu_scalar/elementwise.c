/*
 * src/backends/cpu_scalar/elementwise.c — pure-C reference for the
 * basic per-element ops used by transformer forward passes.
 *
 * Layer: BACKEND.
 *
 * Plain C loops; rmsnorm reduces in double.
 *
 * Tensor contract for all ops: F32 DENSE, contiguous (stride = shape-tail).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"
#include "gemma4_kernels.h"
#include "host_ops.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

/* add, mul, gelu_tanh, relu_squared and silu are the shared loops of
 * gemma4_kernels.c, which cpu_neon also runs. */

[[nodiscard]] enum geist_status cpu_scalar_add(struct geist_backend      *be,
                                               const struct geist_tensor *a,
                                               const struct geist_tensor *b,
                                               struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s = geist_host_ew_views(be, "add", GEIST_EW_Z, a, b, nullptr, y, &v);
    if (s == GEIST_OK) {
        add_fp32(v.n, v.x, v.z, v.y);
    }
    return s;
}

[[nodiscard]] enum geist_status cpu_scalar_mul(struct geist_backend      *be,
                                               const struct geist_tensor *a,
                                               const struct geist_tensor *b,
                                               struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s = geist_host_ew_views(be, "mul", GEIST_EW_Z, a, b, nullptr, y, &v);
    if (s == GEIST_OK) {
        mul_fp32(v.n, v.x, v.z, v.y);
    }
    return s;
}

[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh(struct geist_backend      *be,
                                                     const struct geist_tensor *x,
                                                     struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "gelu_tanh", GEIST_EW_XY, x, nullptr, nullptr, y, &v);
    if (s == GEIST_OK) {
        gelu_tanh_fp32(v.n, v.x, v.y);
    }
    return s;
}

/* gelu_tanh(v) = 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3))),
 * HF Gemma's gelu_tanh (NewGELU), with libm's tanhf: gelu_tanh_mul_fp32
 * would take Accelerate's under GEIST_FAST_TANH. */
static inline float gelu_tanh1(float v) {
    static constexpr float K0 = 0.7978845608028654f; /* sqrt(2/pi) */
    static constexpr float K1 = 0.044715f;
    const float            u  = K0 * (v + K1 * v * v * v);
    return 0.5f * v * (1.0f + tanhf(u));
}

[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh_mul(struct geist_backend      *be,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *z,
                                                         struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "gelu_tanh_mul", GEIST_EW_Z, x, z, nullptr, y, &v);
    if (s != GEIST_OK) {
        return s;
    }
    for (size_t i = 0; i < v.n; i++) {
        v.y[i] = gelu_tanh1(v.x[i]) * v.z[i];
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                                const struct geist_tensor *x,
                                                                const struct geist_tensor *z,
                                                                const float               *scale,
                                                                struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s = geist_host_ew_views(
            be, "gelu_tanh_mul_scaled", GEIST_EW_Z | GEIST_EW_FEAT, x, z, nullptr, y, &v);
    if (s != GEIST_OK) {
        return s;
    }
    if (scale == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_scalar gelu_tanh_mul_scaled: null scale");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t r = 0; r < v.n / v.feat; r++) {
        const size_t base = r * v.feat;
        for (size_t j = 0; j < v.feat; j++) {
            v.y[base + j] = gelu_tanh1(v.x[base + j]) * v.z[base + j] * scale[j];
        }
    }
    return GEIST_OK;
}

/* relu_squared: y = max(x, 0)^2 (BitNet b1.58 2B-4T FFN). */
[[nodiscard]] enum geist_status cpu_scalar_relu_squared(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "relu_squared", GEIST_EW_XY, x, nullptr, nullptr, y, &v);
    if (s == GEIST_OK) {
        relu_squared_fp32(v.n, v.x, v.y);
    }
    return s;
}

/* silu in silu_fp32_ooo's overflow-safe form (#275). */
[[nodiscard]] enum geist_status
cpu_scalar_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "silu", GEIST_EW_XY, x, nullptr, nullptr, y, &v);
    if (s == GEIST_OK) {
        silu_fp32_ooo(v.n, v.x, v.y);
    }
    return s;
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
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "rmsnorm", GEIST_EW_W, x, nullptr, w, y, &v);
    if (s != GEIST_OK) {
        return s;
    }
    const size_t feat = v.feat;
    for (size_t r = 0; r < v.n / feat; r++) {
        const float *row_x = v.x + r * feat;
        float       *row_y = v.y + r * feat;
        double       sumsq = 0.0;
        for (size_t i = 0; i < feat; i++) {
            sumsq += (double) row_x[i] * (double) row_x[i];
        }
        float inv = (float) (1.0 / sqrt(sumsq / (double) feat + (double) eps));
        for (size_t i = 0; i < feat; i++) {
            row_y[i] = row_x[i] * inv * v.w[i];
        }
    }
    return GEIST_OK;
}
