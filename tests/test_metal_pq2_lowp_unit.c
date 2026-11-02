/* Independent PQ2 low-precision oracle against canonical CPU codecs.
 * Tests original PQ2 codes/scales, signed INT2 bias correction, ties,
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

#include "metal_pq2_oracle.h"

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
    const bool                        native = !exact && rows >= 64 && dtype == GEIST_DTYPE_PQ2_0;
    const size_t                      nx = rows * n_in, ny = rows * n_out;
    const size_t                      block = dtype == GEIST_DTYPE_PQ2_0 ? 128u : 32u;
    const size_t                      bb    = 34u;
    const size_t                      wb    = n_out * (n_in / block) * bb;
    struct geist_buffer              *bx = nullptr, *bw = nullptr, *by = nullptr;
    float   *qx = malloc(nx * sizeof(float)), *wr = malloc(n_in * sizeof(float));
    float   *readback    = malloc((OFF + ny + GUARD) * sizeof(float));
    uint8_t *weight_copy = malloc(wb);
    int      fails       = 0;
    if (!qx || !wr || !readback || !weight_copy ||
        v->buffer_create(be, (OFF + nx + GUARD) * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &bx) !=
                GEIST_OK ||
        v->buffer_create(be, WOFF + wb + GUARD, GEIST_BUFFER_WEIGHT, 0, &bw) != GEIST_OK ||
        v->buffer_create(be, (OFF + ny + GUARD) * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &by) !=
                GEIST_OK) {
        fails = geist_expect(false, "PQ2 fixture allocations");
        goto cleanup;
    }
    float   *x = v->buffer_map(bx), *y = v->buffer_map(by);
    uint8_t *w = v->buffer_map(bw);
    if (!x || !y || !w) {
        fails = geist_expect(false, "PQ2 fixture maps");
        goto cleanup;
    }
    memset(w, 0xa5, WOFF + wb + GUARD);
    fill_weights(wb, w + WOFF, dtype);
    memset(w + WOFF, 0, wb / n_out); /* Zero weight row and zero-scale handling. */
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
            x[OFF + r * n_in + k] =
                    r == rows - 1 ? 0.0f : ldexpf(value, power + (int) ((k / 128u) % 3u) - 1);
        }
        float rowmax = 0;
        if (native && st->pq2_lowp_mode >= 7u)
            for (size_t k = 0; k < n_in; k++)
                rowmax = fmaxf(rowmax, fabsf(x[OFF + r * n_in + k]));
        for (size_t b = 0; b < n_in / 128u; b++) {
            float amax = 0;
            for (size_t k = 0; k < 128u; k++)
                amax = fmaxf(amax, fabsf(x[OFF + r * n_in + b * 128u + k]));
            for (size_t k = 0; k < 128u; k++)
                qx[r * n_in + b * 128u + k] =
                        native ? round_block(st->pq2_lowp_mode,
                                             st->pq2_lowp_mode >= 7u ? rowmax : amax,
                                             x[OFF + r * n_in + b * 128u + k])
                               : x[OFF + r * n_in + b * 128u + k];
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
        fails = geist_expect(false, "PQ2 weight resolve");
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
    const uint64_t calls = st->pq2_lowp_mm_dispatches, quant_calls = st->pq2_lowp_quant_dispatches;
    if (f == nullptr || f->linear_t == nullptr ||
        f->linear_t(be, &tx, &weight, &tw, rows, &ty) != GEIST_OK) {
        fails = geist_expect(false, "PQ2 linear dispatch");
        goto cleanup;
    }
    struct geist_buffer *scratch  = st->pq2_lowp_x;
    const size_t         capacity = scratch ? scratch->bytes : 0;
    if (native && ((scratch == nullptr && st->pq2_lowp_mode != 5u) || !st->pq2_lowp_mode)) {
        fails = geist_expect(false, "PQ2 compilation/workspace failed on M6");
        goto cleanup;
    }
    if (native && st->pq2_lowp_mode != 5u) {
        const size_t needed = st->pq2_lowp_mode >= 7u
                                      ? (rows + n_out) * (n_in + 4u)
                                      : nx * (st->pq2_lowp_mode == 4u ? 2u : 1u) + nx / 128u * 8u;
        fails |= geist_expect(capacity >= needed, "PQ2 workspace extent");
        if (f->linear_t(be, &tx, &weight, &tw, rows, &ty) != GEIST_OK) {
            fails = geist_expect(false, "PQ2 repeated dispatch");
            goto cleanup;
        }
        fails |= geist_expect(st->pq2_lowp_x == scratch && st->pq2_lowp_x->bytes == capacity,
                              "PQ2 steady-state scratch reuse");
    }
    fails |= geist_expect(st->pq2_lowp_mm_dispatches - calls ==
                                  (native ? (st->pq2_lowp_mode == 5u ? 1u : 2u) : 0u),
                          "PQ2 routing and bypass");
    fails |= geist_expect(st->pq2_lowp_quant_dispatches - quant_calls ==
                                  (native && st->pq2_lowp_mode != 5u ? 2u : 0u),
                          "one activation quantization per matrix call");
    if (v->buffer_download((OFF + ny + GUARD) * sizeof(float), (uint8_t *) readback, by) !=
        GEIST_OK) {
        fails = geist_expect(false, "PQ2 result download");
        goto cleanup;
    }
    double error2 = 0.0, reference2 = 0.0;
    for (size_t o = 0; o < n_out; o++) {
        if (!quant_dequant_row(dtype, n_in * n_out, o * n_in, n_in, weight_copy, wr)) {
            fails = geist_expect(false, "canonical weight decoder");
            goto cleanup;
        }
        if (native && st->pq2_lowp_mode >= 7u) {
            float amax = 0;
            for (size_t k = 0; k < n_in; k++)
                amax = fmaxf(amax, fabsf(wr[k]));
            for (size_t k = 0; k < n_in; k++)
                wr[k] = round_block(st->pq2_lowp_mode, amax, wr[k]);
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
            fails |= geist_expect(isfinite(got), "PQ2 finite output");
            error2 += ((double) got - want) * ((double) got - want);
            reference2 += want * want;
        }
    }
    for (size_t i = 0; i < OFF; i++)
        fails |= geist_expect(readback[i] == -999.0f, "PQ2 prefix guard");
    for (size_t i = OFF + ny; i < OFF + ny + GUARD; i++)
        fails |= geist_expect(readback[i] == -999.0f, "PQ2 suffix guard");
    fails |= geist_expect(memcmp(w + WOFF, weight_copy, wb) == 0, "PQ2 source weights unchanged");
    const double nrmse = sqrt(error2 / fmax(reference2, 1e-300));
    printf("PQ2 oracle dtype=%d rows=%zu K=%zu N=%zu power=%d exact=%d NRMSE=%.9g\n",
           dtype,
           rows,
           n_in,
           n_out,
           power,
           exact,
           nrmse);
    fails |= geist_expect(isfinite(nrmse) && nrmse < 2e-6,
                          "PQ2 agrees with independently quantized CPU oracle");
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
    if (getenv("GEIST_METAL_PQ2_MODE") == nullptr && setenv("GEIST_METAL_PQ2_MODE", "int8", 1))
        return GEIST_TEST_ERROR;
    if (setenv("GEIST_METAL_TENSOR", "1", 1))
        return GEIST_TEST_ERROR;
    struct geist_backend   *be     = nullptr;
    const enum geist_status result = geist_backend_create("metal", nullptr, nullptr, &be);
    if (result == GEIST_E_UNSUPPORTED || result == GEIST_E_NOT_FOUND)
        return GEIST_TEST_SKIP;
    if (result != GEIST_OK)
        return GEIST_TEST_FAIL;
    struct metal_state *st = be->state;
    if (st->pq2_lowp_mode == 0) {
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    int fails = 0;
    fails |= run_case(be, GEIST_DTYPE_PQ2_0, 64, 128, 65, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_PQ2_0, 129, 384, 67, -20, false, false);
    fails |= run_case(be, GEIST_DTYPE_PQ2_0, 128, 1024, 257, 20, false, false);
    fails |= run_case(be, GEIST_DTYPE_PQ2_0, 64, 128, 65, 0, true, false);
    fails |= run_case(be, GEIST_DTYPE_PQ2_0, 8, 256, 65, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_PQ2_0, 1, 256, 65, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_Q8_0, 64, 256, 65, 0, false, false);
    fails |= run_case(be, GEIST_DTYPE_PQ2_0, 64, 256, 65, 0, false, true);
    fails |= geist_expect(st->pq2_lowp_mode != 0 && st->pq2_lowp_mm_pipeline[5] != nullptr,
                          "M6 PQ2 low precision compiled, not silently bypassed");
    geist_backend_destroy(be);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
#endif
}
