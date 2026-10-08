/*
 * src/backends/cpu_x86/elementwise.c — cpu_x86 gelu_tanh, SiLU, RMSNorm and
 * add overrides.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * gelu_tanh*: in parallel, with tanh computed as (e^2u-1)/(e^2u+1) so the
 * inner loop's expf auto-vectorizes via glibc libmvec under -ffast-math
 * -fopenmp (the project's standard flags). u is clamped to ±10 (tanh(10) is 1
 * to float precision) so e^2u can't overflow to inf. Same math as the scalar
 * reference within float epsilon; cross-checked in test_gelu_x86_unit.c.
 *
 * SiLU and the fused SiLU x mul, RMSNorm and the residual add follow below;
 * mul stays on cpu_scalar (the fused silu_mul and gelu_tanh_mul take its
 * place in the FFN).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "elementwise.h"
#include "par.h"
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

/* One elementwise call for geist_par_for: y = op(x[, z][, scale]), over
 * elements, rows of `feat` or chunks of EW_CHUNK, as the body says. */
struct ew_call {
    const float *x;
    const float *z;
    const float *w; /* gelu_tanh_mul_scaled's scale, rmsnorm's weight */
    float       *y;
    size_t       n;    /* elements */
    size_t       feat; /* elements per row */
    float        eps;
};

/* Elements [i0, i1). The pointers are locals so the loop vectorizes (clang
 * refuses a `simd` loop over shared ones, and -Werror stops the build:
 * "loop not vectorized: ... requested transformation"). */
static void gelu_range(void *ctx, size_t i0, size_t i1) {
    const struct ew_call c  = *(const struct ew_call *) ctx;
    const float         *xp = c.x;
    float               *yp = c.y;
#pragma omp simd
    for (size_t i = i0; i < i1; i++) {
        yp[i] = gelu1(xp[i]);
    }
}

static void gelu_mul_range(void *ctx, size_t i0, size_t i1) {
    const struct ew_call c  = *(const struct ew_call *) ctx;
    const float         *xp = c.x;
    const float         *zp = c.z;
    float               *yp = c.y;
#pragma omp simd
    for (size_t i = i0; i < i1; i++) {
        yp[i] = gelu1(xp[i]) * zp[i];
    }
}

/* Rows [r0, r1) of feat elements, each times scale. */
static void gelu_mul_scaled_rows(void *ctx, size_t r0, size_t r1) {
    const struct ew_call c     = *(const struct ew_call *) ctx;
    const size_t         feat  = c.feat;
    const float         *scale = c.w;
    for (size_t r = r0; r < r1; r++) {
        const float *xr = c.x + r * feat;
        const float *zr = c.z + r * feat;
        float       *yr = c.y + r * feat;
#pragma omp simd
        for (size_t j = 0; j < feat; j++) {
            yr[j] = gelu1(xr[j]) * zr[j] * scale[j];
        }
    }
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
    struct ew_call c = {.x = xp, .y = yp};
    geist_par_for(nx, gelu_range, &c);
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
    struct ew_call c = {.x = xp, .z = zp, .y = yp};
    geist_par_for(nx, gelu_mul_range, &c);
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
    struct ew_call c = {.x = xp, .z = zp, .w = scale, .y = yp, .feat = feat};
    geist_par_for(nx / feat, gelu_mul_scaled_rows, &c);
    return GEIST_OK;
}

/* ---- SiLU ----------------------------------------------------------------
 *
 * Eight lanes at a time on the whole team (cpu_scalar_silu is scalar and
 * single-threaded), in the same overflow-safe form: e = exp(-|v|) <= 1,
 * silu(v) = (v >= 0 ? v : v * e) / (1 + e). The fused silu_mul makes the
 * FFN's SwiGLU epilogue one pass.
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
static constexpr size_t EW_CHUNK       = 1024; /* floats per work item of silu and add, 4 KB */

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

/* Chunks [c0, c1) of EW_CHUNK floats. */
static void silu_chunks(void *ctx, size_t c0, size_t c1) {
    const struct ew_call c = *(const struct ew_call *) ctx;
    for (size_t k = c0; k < c1; k++) {
        const size_t i0  = k * EW_CHUNK;
        const size_t len = c.n - i0 < EW_CHUNK ? c.n - i0 : EW_CHUNK;
        silu_span(len, c.x + i0, c.z != nullptr ? c.z + i0 : nullptr, c.y + i0);
    }
}

static void silu_all(size_t n, const float *x, const float *z, float *y) {
    struct ew_call c = {.x = x, .z = z, .y = y, .n = n};
    geist_par_for((n + EW_CHUNK - 1) / EW_CHUNK, silu_chunks, &c);
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

/* ---- RMSNorm and the residual add ----------------------------------------
 *
 * The rows of the norm, and 4 KB chunks of the add, are spread over the team
 * (cpu_scalar runs both on the calling thread). Below EW_PARALLEL_MIN floats,
 * a decode token's 5120 among them, the calling thread does it alone: waking
 * the team would cost more than it saves.
 *
 * The sum of squares is a double, as in cpu_scalar: a float's square is
 * exact in double, and four accumulators keep the FMA latency from bounding
 * the loop. The norm differs from cpu_scalar's by the order of that sum
 * alone, and a row whose squares would overflow a float (|x| > 1.8e19)
 * still normalizes. The add is exact float addition: cpu_scalar's bits. */
static constexpr size_t EW_PARALLEL_MIN = 16384; /* floats */

static double sumsq(size_t n, const float x[static n]) {
    __m256d a0 = _mm256_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    size_t  i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256d v0 = _mm256_cvtps_pd(_mm_loadu_ps(x + i));
        const __m256d v1 = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4));
        const __m256d v2 = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8));
        const __m256d v3 = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12));
        a0               = _mm256_fmadd_pd(v0, v0, a0);
        a1               = _mm256_fmadd_pd(v1, v1, a1);
        a2               = _mm256_fmadd_pd(v2, v2, a2);
        a3               = _mm256_fmadd_pd(v3, v3, a3);
    }
    const __m256d a = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
    const __m128d h = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
    double        s = _mm_cvtsd_f64(_mm_add_sd(h, _mm_unpackhi_pd(h, h)));
    for (; i < n; i++) {
        s += (double) x[i] * (double) x[i];
    }
    return s;
}

/* y = x / sqrt(mean(x^2) + eps) * w over one row of n floats; y may be x. */
static void rmsnorm_row(
        size_t n, const float x[static n], const float w[static n], float eps, float y[static n]) {
    const float  inv = (float) (1.0 / sqrt(sumsq(n, x) / (double) n + (double) eps));
    const __m256 vi  = _mm256_set1_ps(inv);
    size_t       i   = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + i), vi);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(v, _mm256_loadu_ps(w + i)));
    }
    if (i < n) {
        const __m256i m = lanes(n - i);
        const __m256  v = _mm256_mul_ps(_mm256_maskload_ps(x + i, m), vi);
        _mm256_maskstore_ps(y + i, m, _mm256_mul_ps(v, _mm256_maskload_ps(w + i, m)));
    }
}

/* Rows [r0, r1) of feat floats. */
static void rmsnorm_rows(void *ctx, size_t r0, size_t r1) {
    const struct ew_call c = *(const struct ew_call *) ctx;
    for (size_t r = r0; r < r1; r++) {
        rmsnorm_row(c.feat, c.x + r * c.feat, c.w, c.eps, c.y + r * c.feat);
    }
}

[[nodiscard]] enum geist_status cpu_x86_rmsnorm(struct geist_backend      *be,
                                                const struct geist_tensor *x,
                                                const struct geist_tensor *w,
                                                float                      eps,
                                                struct geist_tensor       *y) {
    if (be == nullptr || x == nullptr || w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, nw = 0, ny = 0;
    const float *xp = gelu_f32_ptr(x, &nx);
    const float *wp = gelu_f32_ptr(w, &nw);
    float       *yp = gelu_f32_ptr(y, &ny);
    if (xp == nullptr || wp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 rmsnorm: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    /* ndim >= 1 and every dimension >= 1, or the view had been refused */
    const size_t feat = (size_t) x->shape[x->ndim - 1];
    if (nw != feat || nx % feat != 0) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_x86 rmsnorm: feature size %zu mismatch (w=%zu)",
                                feat,
                                nw);
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = nx / feat;
    if (nx < EW_PARALLEL_MIN) {
        for (size_t r = 0; r < rows; r++) {
            rmsnorm_row(feat, xp + r * feat, wp, eps, yp + r * feat);
        }
        return GEIST_OK;
    }
    struct ew_call c = {.x = xp, .w = wp, .y = yp, .feat = feat, .eps = eps};
    geist_par_for(rows, rmsnorm_rows, &c);
    return GEIST_OK;
}

/* y = a + b over n floats. */
static void add_span(size_t n, const float *a, const float *b, float *y) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    if (i < n) {
        const __m256i m = lanes(n - i);
        _mm256_maskstore_ps(
                y + i,
                m,
                _mm256_add_ps(_mm256_maskload_ps(a + i, m), _mm256_maskload_ps(b + i, m)));
    }
}

/* Chunks [c0, c1) of EW_CHUNK floats: y = x + z. */
static void add_chunks(void *ctx, size_t c0, size_t c1) {
    const struct ew_call c = *(const struct ew_call *) ctx;
    for (size_t k = c0; k < c1; k++) {
        const size_t i0 = k * EW_CHUNK;
        add_span(c.n - i0 < EW_CHUNK ? c.n - i0 : EW_CHUNK, c.x + i0, c.z + i0, c.y + i0);
    }
}

[[nodiscard]] enum geist_status cpu_x86_add(struct geist_backend      *be,
                                            const struct geist_tensor *a,
                                            const struct geist_tensor *b,
                                            struct geist_tensor       *y) {
    if (be == nullptr || a == nullptr || b == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       na = 0, nb = 0, ny = 0;
    const float *ap = gelu_f32_ptr(a, &na);
    const float *bp = gelu_f32_ptr(b, &nb);
    float       *yp = gelu_f32_ptr(y, &ny);
    if (ap == nullptr || bp == nullptr || yp == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "cpu_x86 add: all tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (na != nb || na != ny) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_x86 add: shape mismatch (a=%zu b=%zu y=%zu)",
                                na,
                                nb,
                                ny);
        return GEIST_E_INVALID_ARG;
    }
    if (na < EW_PARALLEL_MIN) {
        add_span(na, ap, bp, yp);
        return GEIST_OK;
    }
    struct ew_call c = {.x = ap, .z = bp, .y = yp, .n = na};
    geist_par_for((na + EW_CHUNK - 1) / EW_CHUNK, add_chunks, &c);
    return GEIST_OK;
}
