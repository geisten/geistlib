/* Native FP8 must match an independent E4M3 rounding oracle, not merely
 * stay close to the unquantized result. Exercises K tails, output tails,
 * zero blocks, offset views and a wide activation dynamic range. */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef GEIST_BACKEND_METAL
#include "../src/backends/metal/metal_internal.h"

static float round_fp8_block(float amax, float value) {
    if (amax == 0.0f || value == 0.0f) {
        return 0.0f;
    }
    /* Independent arithmetic oracle: E4M3 normals have three fraction
     * bits; subnormals have spacing 2^-9 before the E8M0 block scale. */
    const int    scale    = (int) fmax(-126.0, ceil(log2((double) amax / 448.0)));
    const int    exponent = (int) floor(log2(fabs((double) value)));
    const double step     = ldexp(1.0, exponent - 3 > scale - 9 ? exponent - 3 : scale - 9);
    return (float) (nearbyint((double) value / step) * step);
}

static int run_case(struct geist_backend *be, size_t n_in, size_t rows, int power) {
    enum { N_OUT = 67, X_PAD = 0, Y_PAD = 0, OFFSET = 3 };
    const size_t                     xs = n_in + X_PAD, ys = N_OUT + Y_PAD;
    const size_t                     nx = OFFSET + rows * xs, ny = OFFSET + rows * ys + 5;
    const size_t                     wbytes = N_OUT * (n_in / 32) * 18;
    const struct geist_backend_vtbl *v      = be->desc->vtbl;
    struct geist_buffer             *bx = nullptr, *bw = nullptr, *by = nullptr;
    int                              fails = 0;
    if (v->buffer_create(be, nx * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &bx) != GEIST_OK ||
        v->buffer_create(be, wbytes, GEIST_BUFFER_WEIGHT, 0, &bw) != GEIST_OK ||
        v->buffer_create(be, ny * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &by) != GEIST_OK) {
        fails = geist_expect(false, "FP8 buffer allocation");
        goto done;
    }
    float   *x = v->buffer_map(bx), *y = v->buffer_map(by);
    uint8_t *w = v->buffer_map(bw);
    if (x == nullptr || y == nullptr || w == nullptr) {
        fails = geist_expect(false, "FP8 buffer map");
        goto done;
    }
    for (size_t i = 0; i < nx; i++)
        x[i] = -999.0f;
    for (size_t i = 0; i < ny; i++)
        y[i] = -999.0f;
    for (size_t r = 0; r < rows; r++) {
        for (size_t k = 0; k < n_in; k++) {
            /* First block is zero, others span exact dyadic values. */
            x[OFFSET + r * xs + k] =
                    k < 32 ? 0.0f
                           : ldexpf((float) ((int) ((r * 13 + k * 17) % 127) - 63) / 64.0f, power);
        }
    }
    for (size_t r = 0; r < N_OUT; r++) {
        for (size_t b = 0; b < n_in / 32; b++) {
            uint8_t *block = w + (r * (n_in / 32) + b) * 18;
            block[0]       = 0;
            block[1]       = 0x3c; /* scale = half(1) */
            for (size_t k = 0; k < 16; k++) {
                block[2 + k] = (uint8_t) (((r + b + k) % 16) | (((r * 3 + b + k * 7) % 16) << 4));
            }
        }
    }
    struct geist_weight weight = {.raw        = w,
                                  .raw_nbytes = wbytes,
                                  .n_in       = (int32_t) n_in,
                                  .n_out      = N_OUT,
                                  .dtype      = GEIST_DTYPE_Q4_0};
    fails |= geist_expect(v->resolve_weight(be, &weight) == GEIST_OK, "FP8 weight resolve");
    const struct geist_tensor tx = {.buffer = bx,
                                    .dtype  = GEIST_DTYPE_F32,
                                    .ndim   = 2,
                                    .shape  = {rows, n_in},
                                    .stride = {xs, 1},
                                    .offset = OFFSET * sizeof(float)};
    const struct geist_tensor tw = {.buffer = bw,
                                    .dtype  = GEIST_DTYPE_Q4_0,
                                    .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                                    .ndim   = 2,
                                    .shape  = {N_OUT, n_in},
                                    .stride = {n_in, 1}};
    struct geist_tensor       ty = {.buffer = by,
                                    .dtype  = GEIST_DTYPE_F32,
                                    .ndim   = 2,
                                    .shape  = {rows, N_OUT},
                                    .stride = {ys, 1},
                                    .offset = OFFSET * sizeof(float)};
    fails |= geist_expect(geist_backend_fused_tbl(be)->linear_t(be, &tx, &weight, &tw, rows, &ty) ==
                                  GEIST_OK,
                          "FP8 offset linear");
    /* Mapping drains the pending GPU batch before host reads. */
    y = v->buffer_map(by);
    fails |= geist_expect(y != nullptr, "FP8 result map");
    if (y == nullptr)
        goto done;
    double error2 = 0.0, ref2 = 0.0;
    for (size_t r = 0; r < rows; r++) {
        for (size_t o = 0; o < N_OUT; o++) {
            double ref = 0.0;
            for (size_t b = 0; b < n_in / 32; b++) {
                const uint8_t *block = w + (o * (n_in / 32) + b) * 18;
                float          amax  = 0.0f;
                for (size_t k = 0; k < 32; k++)
                    amax = fmaxf(amax, fabsf(x[OFFSET + r * xs + b * 32 + k]));
                for (size_t k = 0; k < 32; k++) {
                    /* All integer Q4_0 weights here are exact in E4M3. */
                    const int q = ((block[2 + k % 16] >> (k < 16 ? 0 : 4)) & 15) - 8;
                    ref += (double) q * round_fp8_block(amax, x[OFFSET + r * xs + b * 32 + k]);
                }
            }
            const double diff = (double) y[OFFSET + r * ys + o] - ref;
            fails |= geist_expect(isfinite(y[OFFSET + r * ys + o]), "FP8 finite output");
            error2 += diff * diff;
            ref2 += ref * ref;
        }
        for (size_t o = N_OUT; o < ys; o++) {
            fails |=
                    geist_expect(y[OFFSET + r * ys + o] == -999.0f, "FP8 output padding untouched");
        }
    }
    for (size_t i = 0; i < OFFSET; i++)
        fails |= geist_expect(y[i] == -999.0f, "FP8 output prefix untouched");
    for (size_t i = OFFSET + rows * ys; i < ny; i++) {
        fails |= geist_expect(y[i] == -999.0f, "FP8 output suffix untouched");
    }
    const double nrmse = sqrt(error2 / fmax(ref2, 1e-300));
    printf("FP8 oracle K=%zu rows=%zu power=%d NRMSE=%.9g\n", n_in, rows, power, nrmse);
    fails |= geist_expect(isfinite(nrmse) && nrmse < 1e-5, "native FP8 matches E4M3 oracle");
done:
    v->buffer_destroy(be, bx);
    v->buffer_destroy(be, bw);
    v->buffer_destroy(be, by);
    return fails;
}
#endif

int main(void) {
#ifndef GEIST_BACKEND_METAL
    return GEIST_TEST_SKIP;
#else
    if (setenv("GEIST_METAL_FP8", "1", 1) != 0 || setenv("GEIST_METAL_TENSOR", "1", 1) != 0)
        return GEIST_TEST_ERROR;
    struct geist_backend   *be     = nullptr;
    const enum geist_status status = geist_backend_create("metal", nullptr, nullptr, &be);
    if (status == GEIST_E_UNSUPPORTED || status == GEIST_E_NOT_FOUND)
        return GEIST_TEST_SKIP;
    if (status != GEIST_OK)
        return GEIST_TEST_FAIL;
    struct metal_state *st = be->state;
    if (!st->fp8_mm) {
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    int fails = 0;
    fails |= run_case(be, 32, 13, 0);
    fails |= run_case(be, 160, 129, -32);
    fails |= run_case(be, 256, 128, 32);
    /* A requested FP8 path must really compile; fallback is not coverage. */
    fails |= geist_expect(st->fp8_mm, "M6 native FP8 pipeline compiled");
    geist_backend_destroy(be);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
#endif
}
