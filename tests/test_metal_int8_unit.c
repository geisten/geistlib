/* Independent blockwise INT8 oracle against canonical CPU weight codecs.
 * Tests raw Q8_0/Q4_0/Q4_K scale semantics, affine Q4_K correction, ties,
 * zero/NaN inputs, tails, offsets, scratch reuse and exact/short bypass. */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>
#include "quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef GEIST_BACKEND_METAL
#include "../src/backends/metal/metal_internal.h"

static uint32_t random_state = 0x1731u;
static uint8_t  random_byte(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return (uint8_t) (random_state >> 24);
}

static void fill_weights(size_t bytes, uint8_t out[static bytes], enum geist_dtype dtype) {
    const size_t   block = dtype == GEIST_DTYPE_Q8_0 ? 34u : dtype == GEIST_DTYPE_Q4_0 ? 18u : 144u;
    const uint16_t scales[] = {0x3800, 0x3c00, 0x4000, 0x4400};
    for (size_t off = 0; off < bytes; off += block) {
        for (size_t i = 0; i < block; i++)
            out[off + i] = random_byte();
        const uint16_t d = scales[(off / block) % 4u] |
                           (dtype == GEIST_DTYPE_Q4_0 && off / block % 3u == 0u ? 0x8000u : 0u);
        memcpy(out + off, &d, sizeof d);
        if (dtype == GEIST_DTYPE_Q4_K) {
            const uint16_t dm = scales[(off / block + 1u) % 4u];
            memcpy(out + off + 2, &dm, sizeof dm);
            /* All packed six-bit scale/min fields, including high bits. */
        }
    }
}

static int run_case(struct geist_backend *be,
                    enum geist_dtype      dtype,
                    size_t                rows,
                    size_t                n_in,
                    size_t                n_out,
                    int                   power,
                    bool                  exact,
                    bool                  poison) {
    enum { OFF = 3, WOFF = 32, GUARD = 7 };
    const struct geist_backend_vtbl  *v      = be->desc->vtbl;
    const struct geist_backend_fused *f      = geist_backend_fused_tbl(be);
    struct metal_state               *st     = be->state;
    const bool                        native = !exact && rows >= 64 && n_in >= 256 && n_out >= 64;
    const size_t                      nx = rows * n_in, ny = rows * n_out;
    const size_t                      block = dtype == GEIST_DTYPE_Q4_K ? 256u : 32u;
    const size_t bb = dtype == GEIST_DTYPE_Q4_K ? 144u : dtype == GEIST_DTYPE_Q8_0 ? 34u : 18u;
    const size_t wb = n_out * (n_in / block) * bb;
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    float               *qx = malloc(nx * sizeof(float)), *wr = malloc(n_in * sizeof(float));
    float               *readback    = malloc((OFF + ny + GUARD) * sizeof(float));
    uint8_t             *weight_copy = malloc(wb);
    int                  fails       = 0;
    if (!qx || !wr || !readback || !weight_copy ||
        v->buffer_create(be, (OFF + nx + GUARD) * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &bx) !=
                GEIST_OK ||
        v->buffer_create(be, WOFF + wb + GUARD, GEIST_BUFFER_WEIGHT, 0, &bw) != GEIST_OK ||
        v->buffer_create(be, (OFF + ny + GUARD) * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &by) !=
                GEIST_OK) {
        fails = geist_expect(false, "INT8 fixture allocations");
        goto cleanup;
    }
    float   *x = v->buffer_map(bx), *y = v->buffer_map(by);
    uint8_t *w = v->buffer_map(bw);
    if (!x || !y || !w) {
        fails = geist_expect(false, "INT8 fixture maps");
        goto cleanup;
    }
    memset(w, 0xa5, WOFF + wb + GUARD);
    fill_weights(wb, w + WOFF, dtype);
    memcpy(weight_copy, w + WOFF, wb);
    for (size_t i = 0; i < OFF + nx + GUARD; i++)
        x[i] = -777.0f;
    for (size_t i = 0; i < OFF + ny + GUARD; i++)
        y[i] = -999.0f;
    for (size_t r = 0; r < rows; r++) {
        for (size_t k = 0; k < n_in; k++) {
            const float value = (k % 32u) == 31u
                                        ? 127.0f
                                        : (float) ((int) ((r * 17u + k * 23u) % 223u) - 111) * 0.5f;
            x[OFF + r * n_in + k] = r == rows - 1 ? 0.0f : ldexpf(value, power);
        }
        for (size_t b = 0; b < n_in / 32u; b++) {
            float amax = 0.0f;
            for (size_t k = 0; k < 32; k++)
                amax = fmaxf(amax, fabsf(x[OFF + r * n_in + b * 32u + k]));
            const float d = amax > 0 ? fmaxf(amax / 127.0f, 0x1p-126f) : 1.0f;
            for (size_t k = 0; k < 32; k++) {
                const float value          = x[OFF + r * n_in + b * 32u + k];
                qx[r * n_in + b * 32u + k] = native ? nearbyintf(value / d) * d : value;
            }
        }
    }
    if (poison)
        x[OFF + 5] = NAN;
    struct geist_weight weight = {.raw        = w + WOFF,
                                  .raw_nbytes = wb,
                                  .n_in       = (int32_t) n_in,
                                  .n_out      = (int32_t) n_out,
                                  .dtype      = (uint16_t) dtype,
                                  .flags      = exact ? GEIST_W_EXACT : 0};
    if (v->resolve_weight(be, &weight) != GEIST_OK) {
        fails = geist_expect(false, "INT8 weight resolve");
        goto cleanup;
    }
    const struct geist_tensor tx = {.buffer = bx,
                                    .offset = OFF * sizeof(float),
                                    .dtype  = GEIST_DTYPE_F32,
                                    .layout = GEIST_LAYOUT_DENSE,
                                    .ndim   = 2,
                                    .shape  = {(int64_t) rows, (int64_t) n_in},
                                    .stride = {(int64_t) n_in, 1}};
    const struct geist_tensor tw = {.buffer = bw,
                                    .offset = WOFF,
                                    .dtype  = dtype,
                                    .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                                    .ndim   = 2,
                                    .shape  = {(int64_t) n_out, (int64_t) n_in}};
    struct geist_tensor       ty = {.buffer = by,
                                    .offset = OFF * sizeof(float),
                                    .dtype  = GEIST_DTYPE_F32,
                                    .layout = GEIST_LAYOUT_DENSE,
                                    .ndim   = 2,
                                    .shape  = {(int64_t) rows, (int64_t) n_out},
                                    .stride = {(int64_t) n_out, 1}};
    const uint64_t calls = st->int8_gemm_dispatches, quant_calls = st->int8_quant_dispatches;
    if (f == nullptr || f->linear_t == nullptr ||
        f->linear_t(be, &tx, &weight, &tw, rows, &ty) != GEIST_OK) {
        fails = geist_expect(false, "INT8 linear dispatch");
        goto cleanup;
    }
    struct geist_buffer *scratch  = st->int8_x_buffer;
    const size_t         capacity = scratch ? scratch->bytes : 0;
    if (native) {
        fails |= geist_expect(scratch != nullptr && capacity >= nx + nx / 32u * 8u,
                              "INT8 workspace extent");
        if (f->linear_t(be, &tx, &weight, &tw, rows, &ty) != GEIST_OK) {
            fails = geist_expect(false, "INT8 repeated dispatch");
            goto cleanup;
        }
        fails |= geist_expect(st->int8_x_buffer == scratch && st->int8_x_buffer->bytes == capacity,
                              "INT8 steady-state scratch reuse");
    }
    fails |= geist_expect(st->int8_gemm_dispatches - calls == (native ? 2u : 0u),
                          "INT8 routing and bypass");
    fails |= geist_expect(st->int8_quant_dispatches - quant_calls == (native ? 2u : 0u),
                          "one activation quantization per matrix call");
    if (v->buffer_download((OFF + ny + GUARD) * sizeof(float), (uint8_t *) readback, by) !=
        GEIST_OK) {
        fails = geist_expect(false, "INT8 result download");
        goto cleanup;
    }
    double error2 = 0.0, reference2 = 0.0;
    for (size_t o = 0; o < n_out; o++) {
        if (!quant_dequant_row(dtype, n_in * n_out, o * n_in, n_in, weight_copy, wr)) {
            fails = geist_expect(false, "canonical weight decoder");
            goto cleanup;
        }
        for (size_t r = 0; r < rows; r++) {
            const float got = readback[OFF + r * n_out + o];
            if (poison && r == 0) {
                fails |= geist_expect(isnan(got), "non-finite activation propagation");
                continue;
            }
            double want = 0.0;
            for (size_t k = 0; k < n_in; k++)
                want += (double) qx[r * n_in + k] * wr[k];
            fails |= geist_expect(isfinite(got), "INT8 finite output");
            error2 += ((double) got - want) * ((double) got - want);
            reference2 += want * want;
        }
    }
    for (size_t i = 0; i < OFF; i++)
        fails |= geist_expect(readback[i] == -999.0f, "INT8 prefix guard");
    for (size_t i = OFF + ny; i < OFF + ny + GUARD; i++)
        fails |= geist_expect(readback[i] == -999.0f, "INT8 suffix guard");
    fails |= geist_expect(memcmp(w + WOFF, weight_copy, wb) == 0, "INT8 source weights unchanged");
    const double nrmse = sqrt(error2 / fmax(reference2, 1e-300));
    printf("INT8 oracle dtype=%d rows=%zu K=%zu N=%zu power=%d exact=%d NRMSE=%.9g\n",
           dtype,
           rows,
           n_in,
           n_out,
           power,
           exact,
           nrmse);
    fails |= geist_expect(isfinite(nrmse) && nrmse < 2e-6,
                          "INT8 agrees with independently quantized CPU oracle");
cleanup:
    v->buffer_destroy(be, bx);
    v->buffer_destroy(be, bw);
    v->buffer_destroy(be, by);
    free(qx);
    free(wr);
    free(readback);
    free(weight_copy);
    return fails;
}
#endif

int main(void) {
#ifndef GEIST_BACKEND_METAL
    return GEIST_TEST_SKIP;
#else
    if (setenv("GEIST_METAL_INT8", "1", 1) || setenv("GEIST_METAL_TENSOR", "1", 1))
        return GEIST_TEST_ERROR;
    struct geist_backend   *be     = nullptr;
    const enum geist_status result = geist_backend_create("metal", nullptr, nullptr, &be);
    if (result == GEIST_E_UNSUPPORTED || result == GEIST_E_NOT_FOUND)
        return GEIST_TEST_SKIP;
    if (result != GEIST_OK)
        return GEIST_TEST_FAIL;
    struct metal_state *st = be->state;
    if (!st->int8_mm) {
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    int fails = 0;
    fails |= run_case(be, GEIST_DTYPE_Q8_0, 64, 256, 65, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q8_0, 129, 544, 257, -20, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q8_0, 128, 1056, 67, 20, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q4_0, 129, 544, 67, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q4_K, 129, 768, 67, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q4_K, 64, 256, 65, -20, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q8_0, 64, 256, 65, 0, true, false);
    fails |= run_case(be, GEIST_DTYPE_Q8_0, 8, 256, 65, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q8_0, 1, 256, 65, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q8_0, 64, 256, 65, 0, false, true);
    fails |= geist_expect(st->int8_mm && st->int8_q4k_pipeline[2] != nullptr,
                          "M6 INT8 really compiled; fallback is not coverage");
    geist_backend_destroy(be);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
#endif
}
