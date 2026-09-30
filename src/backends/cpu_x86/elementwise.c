/*
 * src/backends/cpu_x86/elementwise.c — cpu_x86 gelu_tanh overrides.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * The cpu_scalar gelu_tanh* are a single-threaded scalar `tanhf` per
 * element — the dominant FFN "act" cost at prefill once the matmuls are
 * fast. These overrides (a) OMP-parallel
 * over the work and (b) compute tanh as 1 - 2/(e^2u+1) so the inner loop's
 * expf auto-vectorizes via glibc libmvec under -ffast-math -fopenmp (the
 * project's standard flags). u is clamped to ±10 (tanh(10) is 1 to float
 * precision) so e^2u can't overflow to inf. Same math as the scalar
 * reference within float epsilon; cross-checked in test_gelu_x86_unit.c.
 *
 * The rest of the elementwise vtable stays on cpu_scalar (add/mul/rmsnorm
 * are memory-bound and cheap), except SiLU and the fused SiLU x mul below.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "elementwise.h"
#include "tensor_view.h"

#include <geist.h>
#include <geist_backend.h>

#include <immintrin.h>
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
    if (t == nullptr || t->buffer == nullptr) {
        return nullptr;
    }
    return geist_tensor_f32_dense(t, t->buffer->host, t->buffer->bytes, out_n);
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
    /* firstprivate: with the pointers shared, clang cannot perform the
     * requested simd vectorization of the outlined loop and -Werror stops
     * the build ("loop not vectorized: ... requested transformation");
     * private copies let it vectorize. gcc privatizes them either way. */
#pragma omp parallel for simd schedule(static) firstprivate(xp, yp)
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
#pragma omp parallel for simd schedule(static) firstprivate(xp, zp, yp) /* see gelu_tanh */
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

/* ---- SiLU ----------------------------------------------------------------
 *
 * cpu_scalar_silu is a libm expf and a division per element on one thread
 * (the branch between its two forms keeps it scalar): 56 ms of a 1.6 s
 * prefill of the synthetic Ternary-Bonsai-2-27B, and the mul after it 27 ms
 * more. Here eight lanes at a time on the whole team, in the same
 * overflow-safe form: e = exp(-|v|) <= 1, silu(v) = (v >= 0 ? v : v * e) /
 * (1 + e). The fused silu_mul makes the FFN's SwiGLU epilogue one pass.
 *
 * exp is Cephes' expf (Cody-Waite reduction by ln 2, a degree-5
 * polynomial, about 1 ulp) in AVX2 rather than libm's: a vectorized expf
 * is libmvec, which only glibc has and which runs every lane below -87.3
 * through a scalar slow path. Every element, the tail included (masked
 * loads), runs the same instructions, so silu_mul is silu then mul to the
 * bit: the empty asm hands the multiply the rounded silu, which -ffast-math
 * could otherwise fold into it. The argument is floored at -87, where 2^n
 * is still normal (n >= -126); past it silu is v (v > 0; 1 + e is 1 from
 * |v| > 17 on) or v * e^-87 instead of v * e^v (v < 0: both of magnitude
 * below 1e-35). */
static constexpr float  SILU_EXP_FLOOR = -87.0f;
static constexpr size_t SILU_CHUNK     = 1024; /* floats per work item, 4 KB */

/* exp(x) for x in [SILU_EXP_FLOOR, 0]. */
static inline __m256 exp8_nonpos(__m256 x) {
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256       r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r              = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p       = _mm256_set1_ps(1.9875691500e-4f);
    p              = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p              = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p              = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p              = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p              = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i two_n =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(two_n));
}

static inline __m256 silu8(__m256 v) {
    const __m256 neg = _mm256_or_ps(v, _mm256_set1_ps(-0.0f)); /* -|v| */
    const __m256 e   = exp8_nonpos(_mm256_max_ps(neg, _mm256_set1_ps(SILU_EXP_FLOOR)));
    const __m256 num = _mm256_blendv_ps(v, _mm256_mul_ps(v, e), v); /* sign set: v * e */
    return _mm256_div_ps(num, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

/* The first n lanes (1 <= n <= 7) of a maskload / maskstore. */
static inline __m256i lanes(size_t n) {
    static const int32_t ones_then_zeros[16] = {
            -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
    return _mm256_loadu_si256((const __m256i *) (ones_then_zeros + 8 - n));
}

/* y = silu(x), or silu(x) * z when z is non-null, over n floats. */
static void silu_span(size_t n, const float *x, const float *z, float *y) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 s = silu8(_mm256_loadu_ps(x + i));
        if (z != nullptr) {
            __asm__("" : "+x"(s)); /* silu rounded before the multiply */
            s = _mm256_mul_ps(s, _mm256_loadu_ps(z + i));
        }
        _mm256_storeu_ps(y + i, s);
    }
    if (i < n) {
        const __m256i m = lanes(n - i);
        __m256        s = silu8(_mm256_maskload_ps(x + i, m));
        if (z != nullptr) {
            __asm__("" : "+x"(s));
            s = _mm256_mul_ps(s, _mm256_maskload_ps(z + i, m));
        }
        _mm256_maskstore_ps(y + i, m, s);
    }
}

static void silu_all(size_t n, const float *x, const float *z, float *y) {
    const size_t chunks = (n + SILU_CHUNK - 1) / SILU_CHUNK;
#pragma omp parallel for schedule(static)
    for (size_t c = 0; c < chunks; c++) {
        const size_t i0  = c * SILU_CHUNK;
        const size_t len = n - i0 < SILU_CHUNK ? n - i0 : SILU_CHUNK;
        silu_span(len, x + i0, z != nullptr ? z + i0 : nullptr, y + i0);
    }
}

[[nodiscard]] enum geist_status
cpu_x86_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = gelu_f32_ptr(x, &nx);
    float       *yp = gelu_f32_ptr(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 silu: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    silu_all(nx, xp, nullptr, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_x86_silu_mul(struct geist_backend      *be,
                                                 const struct geist_tensor *x,
                                                 const struct geist_tensor *z,
                                                 struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = gelu_f32_ptr(x, &nx);
    const float *zp = gelu_f32_ptr(z, &nz);
    float       *yp = gelu_f32_ptr(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || nx != nz || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 silu_mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    silu_all(nx, xp, zp, yp);
    return GEIST_OK;
}
