/*
 * test_gelu_x86_unit — cpu_x86 gelu_tanh{,_mul} vs scalar reference.
 *
 * The cpu_x86 gelu computes tanh as 1 - 2/(e^2u+1) with a ±10 clamp so the
 * inner expf auto-vectorizes (libmvec). This checks it matches the exact
 * tanhf reference across a wide range incl. extremes, where the clamp and
 * the e^2u form are most likely to diverge. SKIPs if cpu_x86 is absent.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static float gelu_ref(float v) {
    const float K0 = 0.7978845608028654f, K1 = 0.044715f;
    const float u = K0 * (v + K1 * v * v * v);
    return 0.5f * v * (1.0f + tanhf(u));
}

int main(void) {
    struct geist_backend *be = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_x86", nullptr, nullptr, &be) != GEIST_OK,
                  "cpu_x86 backend not compiled in");

    constexpr size_t N = 512;
    float            x[N], z[N];
    for (size_t i = 0; i < N; i++) {
        /* Span [-40, 40] so the ±10 u-clamp region and saturation are hit. */
        x[i] = ((float) i / (float) (N - 1)) * 80.0f - 40.0f;
        z[i] = 0.5f + (float) (i % 5) * 0.25f;
    }

    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    struct geist_buffer             *bx = nullptr, *bz = nullptr, *by = nullptr;
    v->buffer_create(be, N * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bx);
    v->buffer_create(be, N * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bz);
    v->buffer_create(be, N * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    v->buffer_upload(bx, N * sizeof(float), (const uint8_t *) x);
    v->buffer_upload(bz, N * sizeof(float), (const uint8_t *) z);

    struct geist_tensor tx = {.buffer = bx,
                              .dtype  = GEIST_DTYPE_F32,
                              .layout = GEIST_LAYOUT_DENSE,
                              .ndim   = 1,
                              .shape  = {(int64_t) N},
                              .stride = {1}};
    struct geist_tensor tz = tx;
    tz.buffer              = bz;
    struct geist_tensor ty = tx;
    ty.buffer              = by;

    int   fails = 0;
    float out[N];

    /* gelu_tanh */
    if (v->gelu_tanh(be, &tx, &ty) != GEIST_OK) {
        printf("FAIL: gelu_tanh ran\n");
        fails++;
    }
    v->buffer_download(N * sizeof(float), (uint8_t *) out, by);
    double maxd = 0.0;
    for (size_t i = 0; i < N; i++) {
        const double d = fabs((double) out[i] - (double) gelu_ref(x[i]));
        if (d > maxd)
            maxd = d;
    }
    printf("[gelu_tanh] max |Δ vs tanhf ref| = %g\n", maxd);
    if (maxd > 1e-4)
        fails++;

    /* gelu_tanh_mul */
    if (v->gelu_tanh_mul(be, &tx, &tz, &ty) != GEIST_OK) {
        printf("FAIL: gelu_tanh_mul ran\n");
        fails++;
    }
    v->buffer_download(N * sizeof(float), (uint8_t *) out, by);
    maxd = 0.0;
    for (size_t i = 0; i < N; i++) {
        const double d = fabs((double) out[i] - (double) (gelu_ref(x[i]) * z[i]));
        if (d > maxd)
            maxd = d;
    }
    printf("[gelu_tanh_mul] max |Δ vs tanhf ref| = %g\n", maxd);
    if (maxd > 1e-4)
        fails++;

    v->buffer_destroy(be, bx);
    v->buffer_destroy(be, bz);
    v->buffer_destroy(be, by);
    geist_backend_destroy(be);

    if (fails == 0)
        printf("PASS: cpu_x86 gelu ≡ scalar reference\n");
    return fails == 0 ? 0 : 1;
}
