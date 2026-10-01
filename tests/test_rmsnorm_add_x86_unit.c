/*
 * test_rmsnorm_add_x86_unit — cpu_x86 rmsnorm and add.
 *
 *   1. rmsnorm against a double reference, x / sqrt(mean(x^2) + eps) * w,
 *      for widths with and without a vector tail (1, 7, 8, 9, 17, 128,
 *      5120, 5123) and 1 to 64 rows, so on the calling thread and on the
 *      team (from 16384 floats), in place as the layers call it and out of
 *      place. The factor is rounded to float once and two float multiplies
 *      follow: 3 ulp of the result at worst, gated at 4.
 *   2. A row of |x| near 1e20, whose squares overflow a float, normalized
 *      all the same (the sum is a double), and a row of zeros left zero.
 *   3. rmsnorm against cpu_scalar's, within the same 4 ulp: only the order
 *      of the double sum differs.
 *   4. add byte for byte against cpu_scalar's, lengths 1, 7, 8, 9, 16383,
 *      16384 and 100003, out of place and in place (y = a, y = b).
 *   5. The shapes cpu_scalar refuses, refused with cpu_scalar's status.
 *
 * SKIPs if cpu_x86 is not built.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr float  EPS     = 1e-6f;
static constexpr double ULP_REL = 5.96e-8; /* 2^-24 */

static uint32_t g_seed = 0x2545F491u;

static float frand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float) (g_seed >> 8) / 16777216.0f; /* [0, 1) */
}

/* An F32 [rows, cols] tensor over a fresh buffer holding `data`. */
static struct geist_tensor
upload(struct geist_backend *be, size_t rows, size_t cols, const float *data) {
    const size_t         n = rows * cols;
    struct geist_buffer *b = nullptr;
    be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &b);
    if (b != nullptr) {
        be->desc->vtbl->buffer_upload(b, n * sizeof(float), (const uint8_t *) data);
    }
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

static void download(struct geist_backend *be, size_t n, const struct geist_tensor *t, float *out) {
    be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t *) out, t->buffer);
}

static void drop(struct geist_backend *be, struct geist_tensor *t) {
    be->desc->vtbl->buffer_destroy(be, t->buffer);
}

/* y = rmsnorm(x) * w on `be`, in place or not; returns the status. */
static enum geist_status run_rmsnorm(struct geist_backend *be,
                                     size_t                rows,
                                     size_t                feat,
                                     const float          *x,
                                     const float          *w,
                                     bool                  in_place,
                                     float                *y) {
    struct geist_tensor tx = upload(be, rows, feat, x);
    struct geist_tensor tw = upload(be, 1, feat, w);
    tw.ndim                = 1;
    tw.shape[0]            = (int64_t) feat;
    tw.stride[0]           = 1;
    struct geist_tensor ty = in_place ? tx : upload(be, rows, feat, x);
    enum geist_status   s  = be->desc->prims->rmsnorm(be, &tx, &tw, EPS, &ty);
    download(be, rows * feat, &ty, y);
    drop(be, &tx);
    drop(be, &tw);
    if (!in_place) {
        drop(be, &ty);
    }
    return s;
}

/* Worst |y - ref| over the 4-ulp gate, ref = x / sqrt(mean(x^2) + eps) * w. */
static double
rmsnorm_err(size_t rows, size_t feat, const float *x, const float *w, const float *y) {
    double worst = 0.0;
    for (size_t r = 0; r < rows; r++) {
        const float *xr = x + r * feat;
        double       ss = 0.0;
        for (size_t i = 0; i < feat; i++) {
            ss += (double) xr[i] * (double) xr[i];
        }
        const double inv = 1.0 / sqrt(ss / (double) feat + (double) EPS);
        for (size_t i = 0; i < feat; i++) {
            const double ref = (double) xr[i] * inv * (double) w[i];
            const double tol = fmax(4.0 * ULP_REL * fabs(ref), 1e-37);
            worst            = fmax(worst, fabs((double) y[r * feat + i] - ref) / tol);
        }
    }
    return worst;
}

static int
check_rmsnorm(struct geist_backend *x86, struct geist_backend *sc, size_t rows, size_t feat) {
    const size_t n     = rows * feat;
    float       *x     = xmalloc(n * sizeof(float));
    float       *w     = xmalloc(feat * sizeof(float));
    float       *y     = xmalloc(n * sizeof(float));
    float       *ys    = xmalloc(n * sizeof(float));
    int          fails = 0;
    char         what[128];
    for (size_t i = 0; i < n; i++) {
        x[i] = (frand() * 2.0f - 1.0f) * 4.0f;
    }
    for (size_t i = 0; i < feat; i++) {
        w[i] = 0.5f + frand();
    }
    if (rows > 1) { /* a row of zeros, and one whose squares overflow a float */
        memset(x, 0, feat * sizeof(float));
        for (size_t i = 0; i < feat; i++) {
            x[feat + i] = (0.5f + frand()) * (i % 2 != 0 ? -1e20f : 1e20f);
        }
    }
    for (int in_place = 0; in_place <= 1; in_place++) {
        fails += geist_expect(run_rmsnorm(x86, rows, feat, x, w, in_place, y) == GEIST_OK,
                              "cpu_x86 rmsnorm ran");
        const double err = rmsnorm_err(rows, feat, x, w, y);
        snprintf(what,
                 sizeof what,
                 "rmsnorm %zux%zu%s within 4 ulp of the double reference",
                 rows,
                 feat,
                 in_place ? " in place" : "");
        fails += geist_expect(err <= 1.0, what);
        if (in_place == 0) {
            printf("  %3zu x %-5zu worst |y - ref| / 4 ulp = %.3f\n", rows, feat, err);
        }
    }
    if (rows > 1) {
        bool zero = true, finite = true;
        for (size_t i = 0; i < feat; i++) {
            zero   = zero && y[i] == 0.0f;
            finite = finite && fabsf(y[feat + i]) < 4.0f && y[feat + i] != 0.0f;
        }
        snprintf(what,
                 sizeof what,
                 "rmsnorm %zux%zu: zero row zero, 1e20 row normalized",
                 rows,
                 feat);
        fails += geist_expect(zero && finite, what);
    }
    fails += geist_expect(run_rmsnorm(sc, rows, feat, x, w, false, ys) == GEIST_OK,
                          "cpu_scalar rmsnorm ran");
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double tol = fmax(4.0 * ULP_REL * fmax(fabsf(y[i]), fabsf(ys[i])), 1e-37);
        worst            = fmax(worst, fabs((double) y[i] - (double) ys[i]) / tol);
    }
    snprintf(what, sizeof what, "rmsnorm %zux%zu within 4 ulp of cpu_scalar's", rows, feat);
    fails += geist_expect(worst <= 1.0, what);
    free(x);
    free(w);
    free(y);
    free(ys);
    return fails;
}

/* y = a + b on `be`; alias 0: y apart, 1: y = a, 2: y = b. */
static enum geist_status
run_add(struct geist_backend *be, size_t n, const float *a, const float *b, int alias, float *y) {
    struct geist_tensor ta = upload(be, 1, n, a);
    struct geist_tensor tb = upload(be, 1, n, b);
    struct geist_tensor ty = alias == 1 ? ta : alias == 2 ? tb : upload(be, 1, n, a);
    enum geist_status   s  = be->desc->prims->add(be, &ta, &tb, &ty);
    download(be, n, &ty, y);
    drop(be, &ta);
    drop(be, &tb);
    if (alias == 0) {
        drop(be, &ty);
    }
    return s;
}

static int check_add(struct geist_backend *x86, struct geist_backend *sc, size_t n) {
    float *a     = xmalloc(n * sizeof(float));
    float *b     = xmalloc(n * sizeof(float));
    float *y     = xmalloc(n * sizeof(float));
    float *ys    = xmalloc(n * sizeof(float));
    int    fails = 0;
    char   what[96];
    for (size_t i = 0; i < n; i++) {
        a[i] = (frand() * 2.0f - 1.0f) * 100.0f;
        b[i] = (frand() * 2.0f - 1.0f) * 0.01f;
    }
    fails += geist_expect(run_add(sc, n, a, b, 0, ys) == GEIST_OK, "cpu_scalar add ran");
    for (int alias = 0; alias <= 2; alias++) {
        fails += geist_expect(run_add(x86, n, a, b, alias, y) == GEIST_OK, "cpu_x86 add ran");
        snprintf(what, sizeof what, "add n=%zu alias=%d bit-identical to cpu_scalar's", n, alias);
        fails += geist_expect(memcmp(y, ys, n * sizeof(float)) == 0, what);
    }
    free(a);
    free(b);
    free(y);
    free(ys);
    return fails;
}

/* The shapes cpu_scalar refuses get cpu_scalar's status from cpu_x86. */
static int check_refusals(struct geist_backend *x86, struct geist_backend *sc) {
    float v[64] = {0};
    int   fails = 0;
    for (int k = 0; k < 2; k++) {
        struct geist_backend *be       = k == 0 ? sc : x86;
        struct geist_tensor   tx       = upload(be, 4, 8, v);
        struct geist_tensor   tw       = upload(be, 1, 7, v); /* 7 weights for 8 features */
        struct geist_tensor   ts       = upload(be, 1, 16, v);
        struct geist_tensor   th       = upload(be, 4, 8, v);
        th.dtype                       = GEIST_DTYPE_F16;
        const enum geist_status  r_w   = be->desc->prims->rmsnorm(be, &tx, &tw, EPS, &tx);
        const enum geist_status  r_y   = be->desc->prims->rmsnorm(be, &tx, &tx, EPS, &ts);
        const enum geist_status  a_n   = be->desc->prims->add(be, &tx, &ts, &tx);
        const enum geist_status  a_dt  = be->desc->prims->add(be, &tx, &th, &tx);
        const enum geist_status  a_nul = be->desc->prims->add(be, &tx, nullptr, &tx);
        static enum geist_status want[5];
        const enum geist_status  got[5] = {r_w, r_y, a_n, a_dt, a_nul};
        if (k == 0) {
            memcpy(want, got, sizeof want);
        } else {
            fails += geist_expect(memcmp(want, got, sizeof want) == 0 && want[0] != GEIST_OK &&
                                          want[1] != GEIST_OK && want[2] != GEIST_OK &&
                                          want[3] != GEIST_OK && want[4] != GEIST_OK,
                                  "bad shapes, dtypes and nullptr refused as cpu_scalar does");
        }
        drop(be, &tx);
        drop(be, &tw);
        drop(be, &ts);
        drop(be, &th);
    }
    return fails;
}

int main(void) {
    struct geist_backend *x86 = nullptr;
    struct geist_backend *sc  = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_x86", nullptr, nullptr, &x86) != GEIST_OK,
                  "cpu_x86 backend not compiled in");
    int fails = geist_expect(geist_backend_create("cpu_scalar", nullptr, nullptr, &sc) == GEIST_OK,
                             "cpu_scalar backend created");
    if (sc == nullptr) {
        geist_backend_destroy(x86);
        return GEIST_TEST_FAIL;
    }

    const size_t widths[] = {1, 7, 8, 9, 17, 128, 5120, 5123};
    const size_t rows[]   = {1, 3, 4, 64};
    for (size_t i = 0; i < sizeof widths / sizeof widths[0]; i++) {
        for (size_t j = 0; j < sizeof rows / sizeof rows[0]; j++) {
            fails += check_rmsnorm(x86, sc, rows[j], widths[i]);
        }
    }
    const size_t lengths[] = {1, 7, 8, 9, 16383, 16384, 100003};
    for (size_t k = 0; k < sizeof lengths / sizeof lengths[0]; k++) {
        fails += check_add(x86, sc, lengths[k]);
    }
    fails += check_refusals(x86, sc);

    geist_backend_destroy(sc);
    geist_backend_destroy(x86);
    if (fails == 0) {
        printf("PASS: cpu_x86 rmsnorm within 4 ulp of a double reference, add bit-identical "
               "to cpu_scalar\n");
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
