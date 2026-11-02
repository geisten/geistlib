/* Packed PQ2 matrix benchmark. Same source/shape for every candidate.
 * Usage: bench_metal_pq2 M N K [repeats [cycles]]. Run under host lock.
 * Measures actual fused linear (activation preparation + GEMM + wait),
 * with 64 discarded warmups and independent sampled CPU dot checks. */
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>
#include <geist_util.h>
#include "quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef GEIST_BACKEND_METAL
#include "../src/backends/metal/metal_internal.h"
#include "metal_pq2_oracle.h"
#endif

int main(int argc, char **argv) {
#ifndef GEIST_BACKEND_METAL
    (void) argc;
    (void) argv;
    return GEIST_TEST_SKIP;
#else
    size_t dims[5] = {128, 17408, 5120, 8, 3};
    if (argc < 4 || argc > 6)
        return GEIST_TEST_ERROR;
    for (int i = 1; i < argc; i++) {
        char *end                 = nullptr;
        errno                     = 0;
        unsigned long       z     = strtoul(argv[i], &end, 10);
        const unsigned long limit = i == 1 ? 2048ul : i <= 3 ? 65536ul : i == 4 ? 128ul : 32ul;
        if (errno || end == argv[i] || *end || z == 0 || z > limit)
            return GEIST_TEST_ERROR;
        dims[i - 1] = (size_t) z;
    }
    const size_t m = dims[0], n = dims[1], k = dims[2];
    size_t       nx, ny, nw, wb, xb, yb;
    if (k % 128u || ckd_mul(&nx, m, k) || ckd_mul(&ny, m, n) || ckd_mul(&nw, n, k) ||
        nx > 64u * 1024u * 1024u || ny > 64u * 1024u * 1024u || nw > 256u * 1024u * 1024u ||
        ckd_mul(&wb, nw / 128u, 34u) || ckd_mul(&xb, nx, 4u) || ckd_mul(&yb, ny, 4u))
        return GEIST_TEST_ERROR;
    struct geist_backend *be     = nullptr;
    enum geist_status     status = geist_backend_create("metal", nullptr, nullptr, &be);
    if (status == GEIST_E_UNSUPPORTED || status == GEIST_E_NOT_FOUND)
        return GEIST_TEST_SKIP;
    if (status != GEIST_OK)
        return GEIST_TEST_FAIL;
    struct metal_state               *st = be->state;
    const struct geist_backend_vtbl  *v  = be->desc->vtbl;
    const struct geist_backend_fused *f  = geist_backend_fused_tbl(be);
    struct geist_buffer              *bx = nullptr, *bw = nullptr, *by = nullptr;
    float                            *wr = malloc(k * sizeof(float));
    int                               rc = GEIST_TEST_FAIL;
    if (!wr || !f || !f->linear_t ||
        v->buffer_create(be, xb, GEIST_BUFFER_ACTIVATION, 0, &bx) != GEIST_OK ||
        v->buffer_create(be, wb, GEIST_BUFFER_WEIGHT, 0, &bw) != GEIST_OK ||
        v->buffer_create(be, yb, GEIST_BUFFER_ACTIVATION, 0, &by) != GEIST_OK)
        goto done;
    float   *x = v->buffer_map(bx), *y = v->buffer_map(by);
    uint8_t *w = v->buffer_map(bw);
    if (!x || !y || !w)
        goto done;
    uint32_t rng = 0x3751u;
    for (size_t i = 0; i < nx; i++) {
        rng            = rng * 1664525u + 1013904223u;
        const size_t b = (i % k) / 128u;
        const float  a = ldexpf(1.0f, (int) (b % 5u) - 2);
        x[i]           = b % 13u == 0 ? 0.0f : ((float) (rng >> 8) / 16777216.0f * 2.0f - 1.0f) * a;
        if (i % 128u == 127u && b % 7u == 0 && b % 13u != 0)
            x[i] = a * 16.0f;
    }
    fill_weights(wb, w, GEIST_DTYPE_PQ2_0);
    struct geist_weight weight = {.raw        = w,
                                  .raw_nbytes = wb,
                                  .n_in       = (int32_t) k,
                                  .n_out      = (int32_t) n,
                                  .dtype      = GEIST_DTYPE_PQ2_0};
    if (v->resolve_weight(be, &weight) != GEIST_OK)
        goto done;
    const struct geist_tensor tx        = {.buffer = bx,
                                           .dtype  = GEIST_DTYPE_F32,
                                           .layout = GEIST_LAYOUT_DENSE,
                                           .ndim   = 2,
                                           .shape  = {(int64_t) m, (int64_t) k},
                                           .stride = {(int64_t) k, 1}};
    const struct geist_tensor tw        = {.buffer = bw,
                                           .dtype  = GEIST_DTYPE_PQ2_0,
                                           .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                                           .ndim   = 2,
                                           .shape  = {(int64_t) n, (int64_t) k}};
    struct geist_tensor       ty        = {.buffer = by,
                                           .dtype  = GEIST_DTYPE_F32,
                                           .layout = GEIST_LAYOUT_DENSE,
                                           .ndim   = 2,
                                           .shape  = {(int64_t) m, (int64_t) n},
                                           .stride = {(int64_t) n, 1}};
    const unsigned            requested = st->pq2_lowp_mode;
    for (unsigned j = 0; j < 64; j++)
        if (f->linear_t(be, &tx, &weight, &tw, m, &ty) != GEIST_OK)
            goto done;
    if (requested && (st->pq2_lowp_mode != requested || st->pq2_lowp_mm_dispatches != 64u)) {
        fprintf(stderr, "PQ2 candidate silently fell back\n");
        goto done;
    }
    for (size_t cycle = 0; cycle < dims[4]; cycle++) {
        const uint64_t q0 = st->pq2_lowp_quant_dispatches, g0 = st->pq2_lowp_mm_dispatches;
        const uint64_t start = geist_now_ns();
        for (size_t j = 0; j < dims[3]; j++)
            if (f->linear_t(be, &tx, &weight, &tw, m, &ty) != GEIST_OK)
                goto done;
        const double us = (double) (geist_now_ns() - start) / (1000.0 * (double) dims[3]);
        if (requested && (st->pq2_lowp_quant_dispatches - q0 != (requested == 5u ? 0u : dims[3]) ||
                          st->pq2_lowp_mm_dispatches - g0 != dims[3]))
            goto done;
        y = v->buffer_map(by);
        if (!y)
            goto done;
        double err2 = 0, ref2 = 0;
        for (size_t sample = 0; sample < 16; sample++) {
            const size_t row = sample == 1 ? m - 1 : (sample * 73u) % m,
                         col = sample == 1 ? n - 1 : (sample * 137u) % n;
            if (!quant_dequant_row(GEIST_DTYPE_PQ2_0, nw, col * k, k, w, wr))
                goto done;
            float rowmax = 0;
            if (requested >= 7u) {
                for (size_t j = 0; j < k; j++)
                    rowmax = fmaxf(rowmax, fabsf(x[row * k + j]));
                float wa = 0;
                for (size_t j = 0; j < k; j++)
                    wa = fmaxf(wa, fabsf(wr[j]));
                for (size_t j = 0; j < k; j++)
                    wr[j] = round_block(requested, wa, wr[j]);
            }
            double ref = 0;
            for (size_t b = 0; b < k / 128u; b++) {
                float amax = 0;
                for (size_t j = 0; j < 128u; j++)
                    amax = fmaxf(amax, fabsf(x[row * k + b * 128u + j]));
                for (size_t j = 0; j < 128u; j++) {
                    const float raw = x[row * k + b * 128u + j];
                    const float value =
                            requested ? round_block(requested, requested >= 7u ? rowmax : amax, raw)
                                      : raw;
                    ref += (double) value * wr[b * 128u + j];
                }
            }
            const double got = y[row * n + col], error = got - ref;
            if (!isfinite(got))
                goto done;
            err2 += error * error;
            ref2 += ref * ref;
        }
        const double nrmse = sqrt(err2 / fmax(ref2, 1e-300));
        if (!isfinite(nrmse) || nrmse > (requested && requested != 5u ? 2e-5 : 1e-3)) {
            fprintf(stderr, "PQ2 matrix oracle failed mode=%u NRMSE=%g\n", requested, nrmse);
            goto done;
        }
        struct geist_backend_resources res = {0};
        const double                   mb  = geist_backend_resources_snapshot(be, &res) == GEIST_OK
                                                     ? (double) res.allocated_bytes / 1048576.0
                                                     : -1;
        printf("{\"mode\":%u,\"tile\":%u,\"simdgroups\":%u,\"m\":%zu,\"n\":%zu,\"k\":%zu,\"cycle\":"
               "%zu,\"repeats\":%zu,\"wall_us\":%.6f,\"tops\":%.6f,\"nrmse\":%.9g,\"backend_"
               "allocated_mb\":%.3f,\"scratch_bytes\":%zu,\"weight_bytes\":%zu,\"shared_acc\":%s}"
               "\n",
               requested,
               st->pq2_lowp_tile,
               st->pq2_lowp_simdgroups,
               m,
               n,
               k,
               cycle,
               dims[3],
               us,
               2.0 * (double) m * (double) n * (double) k / (us * 1e6),
               nrmse,
               mb,
               st->pq2_lowp_x ? st->pq2_lowp_x->bytes : 0,
               wb,
               st->pq2_lowp_shared_acc ? "true" : "false");
        fflush(stdout);
    }
    rc = GEIST_TEST_PASS;
done:
    free(wr);
    v->buffer_destroy(be, bx);
    v->buffer_destroy(be, bw);
    v->buffer_destroy(be, by);
    geist_backend_destroy(be);
    return rc;
#endif
}
