/*
 * test_silu_x86_unit — cpu_x86 silu and silu_mul.
 *
 *   1. silu against a double reference, x / (1 + e^-x), over [-120, 120]
 *      and the edges (+-0, the -87 exp floor, NaN). The Cephes exp is 1.26
 *      ulp at worst, and under -ffast-math gcc divides with RCPPS and one
 *      Newton step: 4.25 ulp at worst on this data, so 8 ulp of the result
 *      (or 1e-30 absolute, where the floor makes a sub-1e-35 negative tail
 *      another tiny number) is the gate. A wrong sign, mask or floor is off
 *      by orders of magnitude more.
 *   2. silu_mul byte for byte against silu then prims->mul: the fused
 *      table's contract, which the exec plan relies on when it swaps one
 *      for the other.
 *   3. Lengths 1, 7, 8, 9, 1023, 1025 and 100003 (the masked tail, the
 *      4 KB work items and several threads), in place (y = x) as the FFN
 *      calls it, and the probe answering yes for SILU_MUL.
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

static uint32_t g_seed = 0x2545F491u;

static float frand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float) (g_seed >> 8) / 16777216.0f; /* [0, 1) */
}

static double silu_ref(float v) {
    return (double) v / (1.0 + exp(-(double) v));
}

/* An F32 1-D tensor over a fresh buffer holding n floats of `data`. */
static struct geist_tensor upload(struct geist_backend *be, size_t n, const float *data) {
    struct geist_buffer *b = nullptr;
    be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &b);
    if (b != nullptr) {
        be->desc->vtbl->buffer_upload(b, n * sizeof(float), (const uint8_t *) data);
    }
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {(int64_t) n},
                                  .stride = {1}};
}

static void download(struct geist_backend *be, size_t n, const struct geist_tensor *t, float *out) {
    be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t *) out, t->buffer);
}

static int check_length(struct geist_backend *be, size_t n) {
    const struct geist_backend_primitives *prims = be->desc->prims;
    const struct geist_backend_fused      *fused = geist_backend_fused_tbl(be);
    float                                 *x     = xmalloc(n * sizeof(float));
    float                                 *z     = xmalloc(n * sizeof(float));
    float                                 *y     = xmalloc(n * sizeof(float));
    float                                 *y2    = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        x[i] = (frand() * 2.0f - 1.0f) * 120.0f;
        z[i] = (frand() * 2.0f - 1.0f) * 3.0f;
    }
    /* the edges, where they fit */
    const float edges[] = {0.0f, -0.0f, -87.0f, -87.5f, 87.5f, 17.0f, -17.0f, 1e-30f, -1e-30f};
    for (size_t i = 0; i < sizeof edges / sizeof edges[0] && i < n; i++) {
        x[i] = edges[i];
    }
    int  fails = 0;
    char what[128];

    /* 1: silu vs the double reference */
    struct geist_tensor tx = upload(be, n, x);
    struct geist_tensor ty = upload(be, n, x);
    fails += geist_expect(prims->silu(be, &tx, &ty) == GEIST_OK, "silu ran");
    download(be, n, &ty, y);
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double ref = silu_ref(x[i]);
        const double tol = fmax(8.0 * 5.96e-8 * fabs(ref), 1e-30);
        worst            = fmax(worst, fabs((double) y[i] - ref) / tol);
    }
    snprintf(what, sizeof what, "n=%zu: silu within 8 ulp of the double reference", n);
    fails += geist_expect(worst <= 1.0, what);

    /* 2: silu_mul == silu then mul, byte for byte; in place like the FFN */
    struct geist_tensor tz = upload(be, n, z);
    fails += geist_expect(prims->mul(be, &ty, &tz, &ty) == GEIST_OK, "mul ran");
    download(be, n, &ty, y);
    struct geist_tensor tg = upload(be, n, x);
    fails += geist_expect(fused->silu_mul(be, &tg, &tz, &tg) == GEIST_OK, "silu_mul ran");
    download(be, n, &tg, y2);
    snprintf(what, sizeof what, "n=%zu: silu_mul bit-identical to silu then mul", n);
    fails += geist_expect(memcmp(y, y2, n * sizeof(float)) == 0, what);

    printf("  n=%-6zu worst |silu - ref| / 8 ulp = %.3f\n", n, worst);
    be->desc->vtbl->buffer_destroy(be, tx.buffer);
    be->desc->vtbl->buffer_destroy(be, ty.buffer);
    be->desc->vtbl->buffer_destroy(be, tz.buffer);
    be->desc->vtbl->buffer_destroy(be, tg.buffer);
    free(x);
    free(z);
    free(y);
    free(y2);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_x86", nullptr, nullptr, &be) != GEIST_OK,
                  "cpu_x86 backend not compiled in");
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    int                               fails = 0;

    const struct geist_fusion_query q = {
            .op = GEIST_FUSED_SILU_MUL, .m = 64, .d_model = 5120, .inter = 17408};
    fails += geist_expect(fused->silu_mul != nullptr && fused->supported != nullptr &&
                                  fused->supported(be, &q),
                          "cpu_x86 binds the fused SiLU x mul");

    const size_t lengths[] = {1, 7, 8, 9, 1023, 1025, 100003};
    for (size_t k = 0; k < sizeof lengths / sizeof lengths[0]; k++) {
        fails += check_length(be, lengths[k]);
    }

    /* NaN in, NaN out, by bit pattern: this file builds with -ffast-math,
     * under which isnan() may fold to false. */
    const uint32_t qnan = 0x7FC00000u;
    float          nan_in, nan_out = 0.0f;
    memcpy(&nan_in, &qnan, sizeof nan_in);
    struct geist_tensor tn = upload(be, 1, &nan_in);
    fails += geist_expect(be->desc->prims->silu(be, &tn, &tn) == GEIST_OK, "silu ran on NaN");
    download(be, 1, &tn, &nan_out);
    uint32_t out_bits = 0;
    memcpy(&out_bits, &nan_out, sizeof out_bits);
    fails += geist_expect((out_bits & 0x7F800000u) == 0x7F800000u && (out_bits & 0x007FFFFFu) != 0,
                          "silu(NaN) is NaN");
    be->desc->vtbl->buffer_destroy(be, tn.buffer);

    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS: cpu_x86 silu within 8 ulp, silu_mul bit-identical to silu then mul\n");
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
