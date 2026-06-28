/*
 * src/backends/cpu_x86/elementwise.c — cpu_x86 gelu_tanh overrides.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * The cpu_scalar gelu_tanh* are a single-threaded scalar `tanhf` per
 * element — the dominant FFN "act" cost at prefill once the matmuls are
 * fast (docs/LINUX_X86_PERF_PROFILE.md). These overrides (a) OMP-parallel
 * over the work and (b) compute tanh as 1 - 2/(e^2u+1) so the inner loop's
 * expf auto-vectorizes via glibc libmvec under -ffast-math -fopenmp (the
 * project's standard flags). u is clamped to ±10 (tanh(10) is 1 to float
 * precision) so e^2u can't overflow to inf. Same math as the scalar
 * reference within float epsilon; cross-checked in test_gelu_x86_unit.c.
 *
 * Only the three gelu entries are overridden; the rest of the elementwise
 * vtable stays on cpu_scalar (add/mul/rmsnorm are memory-bound and cheap).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "elementwise.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* CPU buffer layout, owned by cpu_scalar's buffer_create (cpu_x86 inherits
 * its buffer vtable). Mirrored here as cpu_neon does — geist.h keeps the
 * struct opaque, but the host pointer is needed for the dense fast path. */
struct geist_buffer {
    void                  *host;
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
};

static float *gelu_f32_ptr(const struct geist_tensor *t, size_t *out_n) {
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

/* gelu_tanh(v) = 0.5 * v * (1 + tanh(K0 * (v + K1 * v^3))). */
static inline float gelu1(float v) {
    static constexpr float K0 = 0.7978845608028654f; /* sqrt(2/pi) */
    static constexpr float K1 = 0.044715f;
    float                  u  = K0 * (v + K1 * v * v * v);
    u                         = fmaxf(-10.0f, fminf(10.0f, u));
    const float e             = expf(2.0f * u);
    const float t             = (e - 1.0f) / (e + 1.0f); /* tanh(u) */
    return 0.5f * v * (1.0f + t);
}

[[nodiscard]] enum geist_status
cpu_x86_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = gelu_f32_ptr(x, &nx);
    float       *yp = gelu_f32_ptr(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 gelu_tanh: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < nx; i++) {
        yp[i] = gelu1(xp[i]);
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_x86_gelu_tanh_mul(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      const struct geist_tensor *z,
                                                      struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = gelu_f32_ptr(x, &nx);
    const float *zp = gelu_f32_ptr(z, &nz);
    float       *yp = gelu_f32_ptr(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || nx != nz || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 gelu_tanh_mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < nx; i++) {
        yp[i] = gelu1(xp[i]) * zp[i];
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_x86_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                             const struct geist_tensor *x,
                                                             const struct geist_tensor *z,
                                                             const float               *scale,
                                                             struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = gelu_f32_ptr(x, &nx);
    const float *zp = gelu_f32_ptr(z, &nz);
    float       *yp = gelu_f32_ptr(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || scale == nullptr || nx != nz ||
        nx != ny || y->ndim < 1) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_x86 gelu_tanh_mul_scaled: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    const size_t feat = (size_t) y->shape[y->ndim - 1];
    if (feat == 0 || nx % feat != 0) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_x86 gelu_tanh_mul_scaled: feature mismatch");
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = nx / feat;
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; r++) {
        const float *xr = xp + r * feat;
        const float *zr = zp + r * feat;
        float       *yr = yp + r * feat;
#pragma omp simd
        for (size_t j = 0; j < feat; j++) {
            yr[j] = gelu1(xr[j]) * zr[j] * scale[j];
        }
    }
    return GEIST_OK;
}
