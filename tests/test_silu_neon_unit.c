/*
 * test_silu_neon_unit — cpu_neon silu and silu_mul (#16).
 *
 *   1. silu byte for byte against cpu_scalar's: cpu_neon splits it across
 *      threads in blocks, and every element must still get the same
 *      operations.
 *   2. silu_mul byte for byte against silu then prims->mul: the fused
 *      table's contract, which the exec plan relies on when it swaps one
 *      for the other.
 *   3. Lengths 1, 4095, 4097, 16383 (the last on the calling thread),
 *      16384, 16385 and 524288 (one Llama 3.2 3B prefill chunk), in place
 *      (y = x) as the FFN calls it, and the probe answering yes for
 *      SILU_MUL.
 *
 * SKIPs if cpu_neon or cpu_scalar is not built.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_seed = 0x2545F491u;

static float frand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float) (g_seed >> 8) / 16777216.0f; /* [0, 1) */
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

/* silu (and with z, silu then mul) of x on `be`, in place, into out. */
static int
run_pair(struct geist_backend *be, size_t n, const float *x, const float *z, float *out) {
    const struct geist_backend_primitives *prims = be->desc->prims;
    struct geist_tensor                    tg    = upload(be, n, x);
    int fails = geist_expect(prims->silu(be, &tg, &tg) == GEIST_OK, "silu ran");
    if (z != nullptr) {
        struct geist_tensor tz = upload(be, n, z);
        fails += geist_expect(prims->mul(be, &tg, &tz, &tg) == GEIST_OK, "mul ran");
        be->desc->vtbl->buffer_destroy(be, tz.buffer);
    }
    download(be, n, &tg, out);
    be->desc->vtbl->buffer_destroy(be, tg.buffer);
    return fails;
}

static int check_length(struct geist_backend *neon, struct geist_backend *scalar, size_t n) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(neon);
    float                            *x     = xmalloc(n * sizeof(float));
    float                            *z     = xmalloc(n * sizeof(float));
    float                            *ref   = xmalloc(n * sizeof(float));
    float                            *y     = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        x[i] = (frand() * 2.0f - 1.0f) * 120.0f;
        z[i] = (frand() * 2.0f - 1.0f) * 3.0f;
    }
    const float edges[] = {0.0f, -0.0f, -87.0f, -88.5f, 88.5f, 17.0f, -17.0f, 1e-30f, -1e-30f};
    for (size_t i = 0; i < sizeof edges / sizeof edges[0] && i < n; i++) {
        x[i] = edges[i];
    }
    int  fails = 0;
    char what[128];

    /* 1: silu vs cpu_scalar */
    fails += run_pair(scalar, n, x, nullptr, ref);
    fails += run_pair(neon, n, x, nullptr, y);
    snprintf(what, sizeof what, "n=%zu: silu bit-identical to cpu_scalar", n);
    fails += geist_expect(memcmp(ref, y, n * sizeof(float)) == 0, what);

    /* 2: silu_mul vs silu then mul, in place like the FFN */
    fails += run_pair(neon, n, x, z, ref);
    struct geist_tensor tg = upload(neon, n, x);
    struct geist_tensor tz = upload(neon, n, z);
    fails += geist_expect(fused->silu_mul(neon, &tg, &tz, &tg) == GEIST_OK, "silu_mul ran");
    download(neon, n, &tg, y);
    snprintf(what, sizeof what, "n=%zu: silu_mul bit-identical to silu then mul", n);
    fails += geist_expect(memcmp(ref, y, n * sizeof(float)) == 0, what);

    neon->desc->vtbl->buffer_destroy(neon, tg.buffer);
    neon->desc->vtbl->buffer_destroy(neon, tz.buffer);
    free(x);
    free(z);
    free(ref);
    free(y);
    return fails;
}

int main(void) {
    struct geist_backend *neon = nullptr, *scalar = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_neon", nullptr, nullptr, &neon) != GEIST_OK,
                  "cpu_neon backend not compiled in");
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &scalar) != GEIST_OK) {
        geist_backend_destroy(neon);
        GEIST_SKIP_IF(true, "cpu_scalar backend not compiled in");
    }
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(neon);
    int                               fails = 0;

    const struct geist_fusion_query q = {
            .op = GEIST_FUSED_SILU_MUL, .m = 64, .d_model = 3072, .inter = 8192};
    fails += geist_expect(fused->silu_mul != nullptr && fused->supported != nullptr &&
                                  fused->supported(neon, &q),
                          "cpu_neon binds the fused SiLU x mul");

    const size_t lengths[] = {1, 4095, 4097, 16383, 16384, 16385, 524288};
    for (size_t k = 0; k < sizeof lengths / sizeof lengths[0]; k++) {
        fails += check_length(neon, scalar, lengths[k]);
    }

    geist_backend_destroy(scalar);
    geist_backend_destroy(neon);
    if (fails == 0) {
        printf("PASS: cpu_neon silu bit-identical to cpu_scalar, silu_mul to silu then mul\n");
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
