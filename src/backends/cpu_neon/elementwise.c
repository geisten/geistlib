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
#include "host_ops.h"

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
    struct geist_host_ew    v;
    const enum geist_status s = geist_host_ew_views(be, "add", GEIST_EW_Z, a, b, nullptr, y, &v);
    if (s == GEIST_OK) {
        add_fp32(v.n, v.x, v.z, v.y);
    }
    return s;
}

[[nodiscard]] enum geist_status cpu_neon_mul(struct geist_backend      *be,
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
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "gelu_tanh", GEIST_EW_XY, x, nullptr, nullptr, y, &v);
    if (s != GEIST_OK) {
        return s;
    }
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    gelu_accelerate(v.n, v.x, nullptr, nullptr, 1, v.y);
#else
    gelu_tanh_fp32(v.n, v.x, v.y);
#endif
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_gelu_tanh_mul(struct geist_backend      *be,
                                                       const struct geist_tensor *x,
                                                       const struct geist_tensor *z,
                                                       struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "gelu_tanh_mul", GEIST_EW_Z, x, z, nullptr, y, &v);
    if (s != GEIST_OK) {
        return s;
    }
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    gelu_accelerate(v.n, v.x, v.z, nullptr, 1, v.y);
#else
    gelu_tanh_mul_fp32(v.n, v.x, v.z, v.y);
#endif
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_gelu_tanh_mul_scaled(struct geist_backend      *be,
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
                be, GEIST_E_INVALID_ARG, "cpu_neon gelu_tanh_mul_scaled: null scale");
        return GEIST_E_INVALID_ARG;
    }
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    gelu_accelerate(v.n, v.x, v.z, scale, v.feat, v.y);
#else
    const float  kAlpha = 0.7978845608028654f;
    const float  kBeta  = 0.044715f;
    const size_t rows   = v.n / v.feat;
    for (size_t r = 0; r < rows; r++) {
        const size_t base = r * v.feat;
        for (size_t j = 0; j < v.feat; j++) {
            const size_t i     = base + j;
            const float  xi    = v.x[i];
            const float  inner = kAlpha * (xi + kBeta * xi * xi * xi);
            v.y[i]             = (0.5f * xi * (1.0f + tanhf(inner))) * v.z[i] * scale[j];
        }
    }
#endif
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_neon_relu_squared(struct geist_backend      *be,
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
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "silu", GEIST_EW_XY, x, nullptr, nullptr, y, &v);
    if (s == GEIST_OK) {
        silu_all(v.n, v.x, nullptr, v.y);
    }
    return s;
}

[[nodiscard]] enum geist_status cpu_neon_silu_mul(struct geist_backend      *be,
                                                  const struct geist_tensor *x,
                                                  const struct geist_tensor *z,
                                                  struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "silu_mul", GEIST_EW_Z, x, z, nullptr, y, &v);
    if (s == GEIST_OK) {
        silu_all(v.n, v.x, v.z, v.y);
    }
    return s;
}

[[nodiscard]] enum geist_status cpu_neon_rmsnorm(struct geist_backend      *be,
                                                 const struct geist_tensor *x,
                                                 const struct geist_tensor *w,
                                                 float                      eps,
                                                 struct geist_tensor       *y) {
    struct geist_host_ew    v;
    const enum geist_status s =
            geist_host_ew_views(be, "rmsnorm", GEIST_EW_W, x, nullptr, w, y, &v);
    if (s == GEIST_OK) {
        rmsnorm_fp32(v.n / v.feat, v.feat, v.x, v.w, eps, v.y);
    }
    return s;
}

/* The residual-norm fusions (#195): per row,
 *   y = (res + rmsnorm(x) * w) * scale        (w nullptr: res + x)
 *   z = rmsnorm(y) * z_w                      (rmsnorm_add_norm only)
 * on the same rmsnorm_fp32 and add_fp32 the separate prims run, and the
 * scale as scale_output's multiply, so every element gets the same
 * operations in the same order: the bytes match rmsnorm, add, the layer
 * scale and the next pre-norm run one after the other. Rows are split
 * across threads from RES_NORM_PAR_ROWS rows on: these four passes ran on
 * the main thread while the team waited, ~4 % of a Gemma 4 E2B prefill
 * chunk on the M1 Max; a decode token's row stays on the calling thread.
 * rmsnorm(x) * w goes through a row on the stack, so y may alias res or x;
 * RES_NORM_MAX_COLS bounds it, and the probe refuses wider rows. */
enum { RES_NORM_MAX_COLS = 8192, RES_NORM_PAR_ROWS = 8 };

struct res_norm_job {
    const float *res, *x, *w, *z_w;
    float       *y, *z;
    size_t       cols;
    float        eps, scale;
};

static void res_norm_rows(void *ctx, size_t r0, size_t r1) {
    const struct res_norm_job *j = ctx;
    float                      tmp[RES_NORM_MAX_COLS];
    for (size_t r = r0; r < r1; r++) {
        const size_t off = r * j->cols;
        const float *xr  = j->x + off;
        float       *yr  = j->y + off;
        if (j->w != nullptr) {
            rmsnorm_fp32(1, j->cols, xr, j->w, j->eps, tmp);
            xr = tmp;
        }
        add_fp32(j->cols, j->res + off, xr, yr);
        if (j->scale != 1.0f) {
            for (size_t i = 0; i < j->cols; i++) {
                yr[i] *= j->scale;
            }
        }
        if (j->z != nullptr) {
            rmsnorm_fp32(1, j->cols, yr, j->z_w, j->eps, j->z + off);
        }
    }
}

/* Validates the views (all F32 rows of one shape, w and z_w [cols]) and
 * runs the rows; z and z_w nullptr for rmsnorm_add. */
static enum geist_status res_norm_run(struct geist_backend      *be,
                                      const char                *op,
                                      float                      eps,
                                      float                      scale,
                                      const struct geist_tensor *res,
                                      const struct geist_tensor *x,
                                      const struct geist_tensor *w,
                                      const struct geist_tensor *z_w,
                                      struct geist_tensor       *y,
                                      struct geist_tensor       *z) {
    size_t       nr = 0, nx = 0, nw = 0, nzw = 0, ny = 0, nz = 0;
    const float *rp   = geist_host_tensor_f32(res, &nr);
    const float *xp   = geist_host_tensor_f32(x, &nx);
    const float *wp   = w != nullptr ? geist_host_tensor_f32(w, &nw) : nullptr;
    const float *zwp  = z_w != nullptr ? geist_host_tensor_f32(z_w, &nzw) : nullptr;
    float       *yp   = geist_host_tensor_f32(y, &ny);
    float       *zp   = z != nullptr ? geist_host_tensor_f32(z, &nz) : nullptr;
    const size_t cols = y != nullptr && y->ndim >= 1 && y->shape[y->ndim - 1] > 0
                                ? (size_t) y->shape[y->ndim - 1]
                                : 0;
    if (rp == nullptr || xp == nullptr || yp == nullptr || (w != nullptr && wp == nullptr) ||
        (z != nullptr && (zp == nullptr || zwp == nullptr)) || cols == 0 ||
        cols > RES_NORM_MAX_COLS || nr != ny || nx != ny || ny % cols != 0 ||
        (w != nullptr && nw != cols) || (z != nullptr && (nz != ny || nzw != cols))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon %s: bad inputs", op);
        return GEIST_E_INVALID_ARG;
    }
    struct res_norm_job job  = {.res   = rp,
                                .x     = xp,
                                .w     = wp,
                                .z_w   = zwp,
                                .y     = yp,
                                .z     = zp,
                                .cols  = cols,
                                .eps   = eps,
                                .scale = scale};
    const size_t        rows = ny / cols;
    if (rows < RES_NORM_PAR_ROWS) {
        res_norm_rows(&job, 0, rows);
    } else {
        geist_par_for(rows, res_norm_rows, &job);
    }
    return GEIST_OK;
}

[[nodiscard]] bool cpu_neon_res_norm_supported(size_t cols) {
    return cols > 0 && cols <= RES_NORM_MAX_COLS;
}

[[nodiscard]] enum geist_status cpu_neon_rmsnorm_add(struct geist_backend      *be,
                                                     const struct geist_tensor *res,
                                                     const struct geist_tensor *x,
                                                     const struct geist_tensor *w,
                                                     float                      eps,
                                                     struct geist_tensor       *y) {
    if (w == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon rmsnorm_add: null weight");
        return GEIST_E_INVALID_ARG;
    }
    return res_norm_run(be, "rmsnorm_add", eps, 1.0f, res, x, w, nullptr, y, nullptr);
}

[[nodiscard]] enum geist_status cpu_neon_rmsnorm_add_norm(struct geist_backend      *be,
                                                          float                      eps,
                                                          float                      scale,
                                                          const struct geist_tensor *res,
                                                          const struct geist_tensor *x,
                                                          const struct geist_tensor *w,
                                                          const struct geist_tensor *z_w,
                                                          struct geist_tensor       *y,
                                                          struct geist_tensor       *z) {
    if (z_w == nullptr || z == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon rmsnorm_add_norm: null z");
        return GEIST_E_INVALID_ARG;
    }
    return res_norm_run(be, "rmsnorm_add_norm", eps, scale, res, x, w, z_w, y, z);
}
