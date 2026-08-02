/*
 * test_backend_ops_unit — verifies the elementwise + rmsnorm ops in both
 * cpu_scalar and cpu_neon backends produce the same outputs.
 *
 * Phase B-4e foundation: with these ops in the backend vtable, lm.c's
 * forward pass can be migrated op-by-op through backend->vtbl->* instead
 * of direct calls into gemma4_kernels.c.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

/* Build a 1-row F32 DENSE tensor backed by an existing buffer. */
static struct geist_tensor make_tensor_1d(struct geist_buffer *buf, size_t n) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = 0,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 1,
            .shape  = {(int64_t) n},
            .stride = {1},
    };
}

/* Cross-reference a single op across the two backends. */
static int compare_op_outputs(const char  *op_name,
                              const float *out_a,
                              const float *out_b,
                              size_t       n,
                              float        rtol,
                              float        atol) {
    ptrdiff_t bad = geist_fp32_close_array(out_a, out_b, n, rtol, atol);
    if (bad >= 0) {
        fprintf(stderr,
                "FAIL %s: idx %td: scalar=%.6f neon=%.6f\n",
                op_name,
                bad,
                (double) out_a[bad],
                (double) out_b[bad]);
        return 1;
    }
    return 0;
}

/* Helper: build a backend, run an op of choice, get scalar host buffer back.
 * Returns 0 on success, -1 on error. */
static int
run_add(const char *backend_name, const float *a_in, const float *b_in, size_t n, float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    struct geist_buffer *ba = nullptr, *bb = nullptr, *by = nullptr;
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &ba);
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bb);
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    (void) be->desc->vtbl->buffer_upload(ba, n * sizeof(float), (const uint8_t *) a_in);
    (void) be->desc->vtbl->buffer_upload(bb, n * sizeof(float), (const uint8_t *) b_in);
    struct geist_tensor a = make_tensor_1d(ba, n);
    struct geist_tensor b = make_tensor_1d(bb, n);
    struct geist_tensor y = make_tensor_1d(by, n);
    enum geist_status   s = be->desc->prims->add(be, &a, &b, &y);
    if (s != GEIST_OK) {
        geist_backend_destroy(be);
        return -1;
    }
    (void) be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t *) out_host, by);
    (void) be->desc->vtbl->buffer_destroy(be, ba);
    (void) be->desc->vtbl->buffer_destroy(be, bb);
    (void) be->desc->vtbl->buffer_destroy(be, by);
    geist_backend_destroy(be);
    return 0;
}

static int run_gelu(const char *backend_name, const float *x_in, size_t n, float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    struct geist_buffer *bx = nullptr, *by = nullptr;
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bx);
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    (void) be->desc->vtbl->buffer_upload(bx, n * sizeof(float), (const uint8_t *) x_in);
    struct geist_tensor x = make_tensor_1d(bx, n);
    struct geist_tensor y = make_tensor_1d(by, n);
    enum geist_status   s = be->desc->prims->gelu_tanh(be, &x, &y);
    if (s != GEIST_OK) {
        geist_backend_destroy(be);
        return -1;
    }
    (void) be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t *) out_host, by);
    (void) be->desc->vtbl->buffer_destroy(be, bx);
    (void) be->desc->vtbl->buffer_destroy(be, by);
    geist_backend_destroy(be);
    return 0;
}

static int run_rmsnorm(const char  *backend_name,
                       const float *x_in,
                       const float *w_in,
                       size_t       n_rows,
                       size_t       feat,
                       float        eps,
                       float       *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    (void) be->desc->vtbl->buffer_create(
            be, n_rows * feat * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bx);
    (void) be->desc->vtbl->buffer_create(
            be, feat * sizeof(float), GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &bw);
    (void) be->desc->vtbl->buffer_create(
            be, n_rows * feat * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    (void) be->desc->vtbl->buffer_upload(bx, n_rows * feat * sizeof(float), (const uint8_t *) x_in);
    (void) be->desc->vtbl->buffer_upload(bw, feat * sizeof(float), (const uint8_t *) w_in);
    struct geist_tensor x = {
            .buffer = bx,
            .offset = 0,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {(int64_t) n_rows, (int64_t) feat},
            .stride = {(int64_t) feat, 1},
    };
    struct geist_tensor w = make_tensor_1d(bw, feat);
    struct geist_tensor y = x;
    y.buffer              = by;
    enum geist_status s   = be->desc->prims->rmsnorm(be, &x, &w, eps, &y);
    if (s != GEIST_OK) {
        geist_backend_destroy(be);
        return -1;
    }
    (void) be->desc->vtbl->buffer_download(n_rows * feat * sizeof(float), (uint8_t *) out_host, by);
    (void) be->desc->vtbl->buffer_destroy(be, bx);
    (void) be->desc->vtbl->buffer_destroy(be, bw);
    (void) be->desc->vtbl->buffer_destroy(be, by);
    geist_backend_destroy(be);
    return 0;
}

int main(void) {
    int          fails = 0;
    const size_t N     = 256;

    /* This is a cpu_scalar-vs-cpu_neon differential test; it only means
     * anything when both backends are compiled in. In a cpu_neon-less build
     * (BACKENDS without cpu_neon) skip rather than report bogus mismatches. */
    struct geist_backend *neon_probe = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_neon", nullptr, nullptr, &neon_probe) != GEIST_OK,
                  "cpu_neon backend not compiled in (scalar-only build)");
    geist_backend_destroy(neon_probe);

    /* Deterministic input. */
    float a[N], b[N], x[N], w[N];
    for (size_t i = 0; i < N; i++) {
        a[i] = (float) i * 0.01f - 1.2f;
        b[i] = (float) (i % 7) * 0.05f - 0.15f;
        x[i] = sinf((float) i * 0.31415f) * 2.0f;
        w[i] = 1.0f + (float) (i % 11) * 0.07f;
    }

    /* ---- add: scalar ≡ neon, bit-identical ---- */
    float ya_scalar[N], ya_neon[N];
    fails += check(run_add("cpu_scalar", a, b, N, ya_scalar) == 0, "scalar add ran");
    fails += check(run_add("cpu_neon", a, b, N, ya_neon) == 0, "neon add ran");
    fails += compare_op_outputs("add", ya_scalar, ya_neon, N, 0.0f, 0.0f);

    /* Spot-check correctness vs hand-computed. */
    for (size_t i = 0; i < 5; i++) {
        float expected = a[i] + b[i];
        fails += check(fabsf(ya_scalar[i] - expected) < 1e-6f, "add correctness");
    }

    /* ---- gelu_tanh: tight tolerance (same scalar kernel both sides) ---- */
    float yg_scalar[N], yg_neon[N];
    fails += check(run_gelu("cpu_scalar", x, N, yg_scalar) == 0, "scalar gelu ran");
    fails += check(run_gelu("cpu_neon", x, N, yg_neon) == 0, "neon gelu ran");
    fails += compare_op_outputs("gelu_tanh", yg_scalar, yg_neon, N, 1e-5f, 1e-6f);

    /* gelu(0) = 0; gelu(very-positive) ≈ x; gelu(very-negative) ≈ 0. */
    fails += check(fabsf(yg_scalar[0]) < 1e-5f || x[0] != 0.0f, "gelu(0) sanity");

    /* ---- rmsnorm: scalar vs neon, tolerate reduction order ---- */
    float yr_scalar[N * 4], yr_neon[N * 4];
    /* 4 rows × 64 feat = 256 elements; we want n_rows=4, feat=64. */
    const size_t n_rows = 4, feat = 64;
    /* Reuse x as 4×64 input. */
    float rms_x[256], rms_w[64];
    for (size_t i = 0; i < 256; i++)
        rms_x[i] = x[i];
    for (size_t i = 0; i < 64; i++)
        rms_w[i] = w[i];
    fails += check(run_rmsnorm("cpu_scalar", rms_x, rms_w, n_rows, feat, 1e-6f, yr_scalar) == 0,
                   "scalar rmsnorm ran");
    fails += check(run_rmsnorm("cpu_neon", rms_x, rms_w, n_rows, feat, 1e-6f, yr_neon) == 0,
                   "neon rmsnorm ran");
    fails += compare_op_outputs("rmsnorm", yr_scalar, yr_neon, n_rows * feat, 1e-4f, 1e-5f);

    if (fails == 0) {
        printf("PASS: add / gelu_tanh / rmsnorm — cpu_scalar ≡ cpu_neon\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
