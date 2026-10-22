/*
 * src/backends/cpu_neon/elementwise.c — wraps gemma4_kernels.c for the
 * elementwise + rmsnorm ops in the backend vtable.
 *
 * Layer: BACKEND.
 *
 * FP32 ops on the portable gemma4_kernels.c loops (left to
 * auto-vectorization); on Apple with Accelerate, the GELU-tanh products use
 * vvtanhf.
 */

#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"
#include "host_buffer.h"
#include "hadamard.h"

#include "gemma4_kernels.h"
#include "heap.h"

#include "par.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
extern void vvtanhf(float *y, const float *x, const int *n);
#endif

[[nodiscard]] enum geist_status cpu_neon_add(struct geist_backend      *be,
                                             const struct geist_tensor *a,
                                             const struct geist_tensor *b,
                                             struct geist_tensor       *y) {
    size_t       na = 0, nb = 0, ny = 0;
    const float *ap = geist_host_tensor_f32(a, &na);
    const float *bp = geist_host_tensor_f32(b, &nb);
    float       *yp = geist_host_tensor_f32(y, &ny);
    if (ap == nullptr || bp == nullptr || yp == nullptr || na != nb || na != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon add: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    add_fp32(na, ap, bp, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_mul(struct geist_backend      *be,
                                             const struct geist_tensor *a,
                                             const struct geist_tensor *b,
                                             struct geist_tensor       *y) {
    size_t       na = 0, nb = 0, ny = 0;
    const float *ap = geist_host_tensor_f32(a, &na);
    const float *bp = geist_host_tensor_f32(b, &nb);
    float       *yp = geist_host_tensor_f32(y, &ny);
    if (ap == nullptr || bp == nullptr || yp == nullptr || na != nb || na != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    mul_fp32(na, ap, bp, yp);
    return GEIST_OK;
}

#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
/* y = gelu_tanh(x) [* z] [* scale[i % feat]], the tanh through Accelerate's
 * vvtanhf, in blocks of GELU_BLOCK elements split across threads: a
 * prefill chunk is 64 tokens x the FFN width, and one thread took 8-9 % of
 * a Gemma 4 E2B prefill on the M1 Max. Each element gets the same
 * operations as before, whatever the split. y may alias x. */
enum { GELU_BLOCK = 2048 };

struct gelu_job {
    const float *x, *z, *scale;
    float       *y;
    size_t       n, feat;
};

static void gelu_blocks(void *ctx, size_t b0, size_t b1) {
    const struct gelu_job *j      = ctx;
    const float            kAlpha = 0.7978845608028654f;
    const float            kBeta  = 0.044715f;
    float                  tmp[GELU_BLOCK];
    for (size_t b = b0; b < b1; b++) {
        const size_t i0 = b * GELU_BLOCK;
        const size_t n  = j->n - i0 < GELU_BLOCK ? j->n - i0 : GELU_BLOCK;
        for (size_t i = 0; i < n; i++) {
            const float xi = j->x[i0 + i];
            tmp[i]         = kAlpha * (xi + kBeta * xi * xi * xi);
        }
        int n_i32 = (int) n;
        vvtanhf(tmp, tmp, &n_i32);
        for (size_t i = 0; i < n; i++) {
            float v = 0.5f * j->x[i0 + i] * (1.0f + tmp[i]);
            if (j->z != nullptr) {
                v *= j->z[i0 + i];
            }
            if (j->scale != nullptr) {
                v *= j->scale[(i0 + i) % j->feat];
            }
            j->y[i0 + i] = v;
        }
    }
}

static void gelu_accelerate(
        size_t n, const float *x, const float *z, const float *scale, size_t feat, float *y) {
    struct gelu_job job = {.x = x, .z = z, .scale = scale, .y = y, .n = n, .feat = feat};
    const size_t    nb  = (n + GELU_BLOCK - 1) / GELU_BLOCK;
    if (nb < 4) {
        gelu_blocks(&job, 0, nb);
    } else {
        geist_par_for(nb, gelu_blocks, &job);
    }
}
#endif

[[nodiscard]] enum geist_status
cpu_neon_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = geist_host_tensor_f32(x, &nx);
    float       *yp = geist_host_tensor_f32(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon gelu_tanh: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    gelu_accelerate(nx, xp, nullptr, nullptr, 1, yp);
    return GEIST_OK;
#endif
    gelu_tanh_fp32(nx, xp, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_gelu_tanh_mul(struct geist_backend      *be,
                                                       const struct geist_tensor *x,
                                                       const struct geist_tensor *z,
                                                       struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = geist_host_tensor_f32(x, &nx);
    const float *zp = geist_host_tensor_f32(z, &nz);
    float       *yp = geist_host_tensor_f32(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || nx != nz || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon gelu_tanh_mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    gelu_accelerate(nx, xp, zp, nullptr, 1, yp);
    return GEIST_OK;
#endif
    gelu_tanh_mul_fp32(nx, xp, zp, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *z,
                                                              const float               *scale,
                                                              struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = geist_host_tensor_f32(x, &nx);
    const float *zp = geist_host_tensor_f32(z, &nz);
    float       *yp = geist_host_tensor_f32(y, &ny);
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
    gelu_accelerate(nx, xp, zp, scale, feat, yp);
    return GEIST_OK;
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
    const float *xp = geist_host_tensor_f32(x, &nx);
    float       *yp = geist_host_tensor_f32(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon relu_squared: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    relu_squared_fp32(nx, xp, yp);
    return GEIST_OK;
}

/* y = silu(x), or the SwiGLU product silu(x) * z when z is non-null, in
 * blocks of SILU_BLOCK floats, split across threads from SILU_PAR_BLOCKS
 * blocks on: a prefill chunk of Llama 3.2 3B is 64 tokens x 8192, and on
 * one thread libm's expf took 11 % of the main thread's prefill on the M1
 * Max while the team waited (#16). A decode token's few thousand floats
 * stay on the calling thread, where a parallel region would cost more than
 * it saves. Every element gets silu_fp32_ooo's operations, and the product
 * multiplies the stored silu, as prims->silu then prims->mul do: the bits
 * are the same whatever the split. y may alias x. */
enum { SILU_BLOCK = 4096, SILU_PAR_BLOCKS = 4 };

struct silu_job {
    const float *x, *z;
    float       *y;
    size_t       n;
};

static void silu_blocks(void *ctx, size_t b0, size_t b1) {
    const struct silu_job *j = ctx;
    for (size_t b = b0; b < b1; b++) {
        const size_t i0 = b * SILU_BLOCK;
        const size_t n  = j->n - i0 < SILU_BLOCK ? j->n - i0 : SILU_BLOCK;
        silu_fp32_ooo(n, j->x + i0, j->y + i0);
        if (j->z != nullptr) {
            mul_fp32(n, j->y + i0, j->z + i0, j->y + i0);
        }
    }
}

static void silu_all(size_t n, const float *x, const float *z, float *y) {
    struct silu_job job = {.x = x, .z = z, .y = y, .n = n};
    const size_t    nb  = (n + SILU_BLOCK - 1) / SILU_BLOCK;
    if (nb < SILU_PAR_BLOCKS) {
        silu_blocks(&job, 0, nb);
    } else {
        geist_par_for(nb, silu_blocks, &job);
    }
}

[[nodiscard]] enum geist_status
cpu_neon_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = geist_host_tensor_f32(x, &nx);
    float       *yp = geist_host_tensor_f32(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon silu: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    silu_all(nx, xp, nullptr, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_silu_mul(struct geist_backend      *be,
                                                  const struct geist_tensor *x,
                                                  const struct geist_tensor *z,
                                                  struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = geist_host_tensor_f32(x, &nx);
    const float *zp = geist_host_tensor_f32(z, &nz);
    float       *yp = geist_host_tensor_f32(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || nx != nz || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon silu_mul: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    silu_all(nx, xp, zp, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_rmsnorm(struct geist_backend      *be,
                                                 const struct geist_tensor *x,
                                                 const struct geist_tensor *w,
                                                 float                      eps,
                                                 struct geist_tensor       *y) {
    size_t       nx = 0, nw = 0, ny = 0;
    const float *xp = geist_host_tensor_f32(x, &nx);
    const float *wp = geist_host_tensor_f32(w, &nw);
    float       *yp = geist_host_tensor_f32(y, &ny);
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
    rmsnorm_fp32(n_rows, feat, xp, wp, eps, yp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_hadamard_rotate(struct geist_backend             *be,
                                                         const struct geist_hadamard_args *args) {
    if (args == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon hadamard_rotate: null args");
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, ns = 0, ny = 0;
    const float *xp = geist_host_tensor_f32(args->x, &nx);
    const float *sp = args->signs != nullptr ? geist_host_tensor_f32(args->signs, &ns) : nullptr;
    float       *yp = geist_host_tensor_f32(args->y, &ny);
    enum geist_status s = (args->signs != nullptr && sp == nullptr)
                                  ? GEIST_E_INVALID_ARG
                                  : geist_hadamard_apply(args, nx, ns, ny, xp, sp, yp);
    if (s != GEIST_OK) {
        geist_backend_set_error(be, s, "cpu_neon hadamard_rotate: bad inputs");
    }
    return s;
}
