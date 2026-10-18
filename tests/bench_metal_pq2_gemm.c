/*
 * bench_metal_pq2_gemm — time one PQ2_0 prefill GEMM on Metal (#77).
 *
 * Shapes are Bonsai 2 27B's (qwen35 hybrid): the FFN gate/up and down
 * projections and the DeltaNet qkv projection, at m = 256 (the Metal
 * prefill chunk, preferred_m_max), 512 and 2048 rows. Each sample encodes
 * ITERS back-to-back GEMMs on backend buffers and drains the batch with a
 * 4-byte download; the median sample over REPS gives ms per GEMM and
 * TFLOP/s (2*m*n_in*n_out). Weights are random PQ2_0 blocks with d = 1:
 * the kernel does the same work for any codes.
 *
 * Usage: bench_metal_pq2_gemm [iters] [reps] [m ...] (default m: 256 512
 * 2048). SKIPs without a Metal device.
 * Timing: run it under the host's benchmark lock.
 */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { PQ2_BLOCK_ELEMS = 128, PQ2_BLOCK_BYTES = 34, MAX_REPS = 64 };

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

static void *dev_alloc(struct geist_backend *be, size_t bytes, struct geist_buffer **out) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v->buffer_create(be, bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, out) != GEIST_OK) {
        return nullptr;
    }
    return v->buffer_map(*out);
}

static int bench_shape(struct geist_backend *mt,
                       const char           *name,
                       size_t                n_in,
                       size_t                n_out,
                       size_t                m,
                       int                   iters,
                       int                   reps) {
    const struct geist_backend_vtbl *v       = mt->desc->vtbl;
    const size_t                     w_bytes = n_out * (n_in / PQ2_BLOCK_ELEMS) * PQ2_BLOCK_BYTES;
    struct geist_buffer             *bw = nullptr, *bx = nullptr, *by = nullptr;
    uint8_t                         *blob = dev_alloc(mt, w_bytes, &bw);
    float                           *x    = dev_alloc(mt, m * n_in * sizeof(float), &bx);
    float                           *y    = dev_alloc(mt, m * n_out * sizeof(float), &by);
    if (blob == nullptr || x == nullptr || y == nullptr) {
        fprintf(stderr, "%s: backend buffer alloc failed\n", name);
        return GEIST_TEST_ERROR;
    }
    uint32_t rng = 0x9e3779b9u;
    for (size_t b = 0; b < w_bytes; b += PQ2_BLOCK_BYTES) {
        blob[b]     = 0x00; /* d = fp16(1.0) */
        blob[b + 1] = 0x3C;
        for (size_t i = 2; i < PQ2_BLOCK_BYTES; i++) {
            rng         = rng * 1664525u + 1013904223u;
            blob[b + i] = (uint8_t) (rng >> 24);
        }
    }
    for (size_t i = 0; i < m * n_in; i++) {
        rng  = rng * 1664525u + 1013904223u;
        x[i] = ((float) (rng >> 24) - 127.5f) / 128.0f;
    }
    struct geist_weight w = {.raw        = blob,
                             .raw_nbytes = w_bytes,
                             .n_in       = (int32_t) n_in,
                             .n_out      = (int32_t) n_out,
                             .dtype      = (uint16_t) GEIST_DTYPE_PQ2_0};
    if (v->resolve_weight(mt, &w) != GEIST_OK || w.linear_mN == nullptr) {
        fprintf(stderr, "%s: resolve_weight failed\n", name);
        return GEIST_TEST_ERROR;
    }
    float  probe[1];
    double samples[MAX_REPS];
    for (int r = -1; r < reps; r++) { /* r == -1: warm-up */
        const double t0 = now_s();
        for (int i = 0; i < iters; i++) {
            w.linear_mN(m, x, &w, mt, y);
        }
        if (v->buffer_download(sizeof probe, (uint8_t *) probe, by) != GEIST_OK) {
            fprintf(stderr, "%s: download failed\n", name);
            return GEIST_TEST_ERROR;
        }
        if (r >= 0) {
            samples[r] = (now_s() - t0) / iters;
        }
    }
    qsort(samples, (size_t) reps, sizeof samples[0], cmp_double);
    const double t     = samples[reps / 2];
    const double flops = 2.0 * (double) m * (double) n_in * (double) n_out;
    printf("%-10s n_in=%-6zu n_out=%-6zu m=%-5zu %8.3f ms  %6.2f TFLOP/s  (min %.3f max %.3f)\n",
           name,
           n_in,
           n_out,
           m,
           t * 1e3,
           flops / t * 1e-12,
           samples[0] * 1e3,
           samples[reps - 1] * 1e3);
    v->buffer_destroy(mt, bw);
    v->buffer_destroy(mt, bx);
    v->buffer_destroy(mt, by);
    return GEIST_TEST_PASS;
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 20;
    const int reps  = argc > 2 ? atoi(argv[2]) : 7;
    if (iters < 1 || reps < 1 || reps > MAX_REPS) {
        fprintf(stderr, "usage: %s [iters >= 1] [reps 1..%d]\n", argv[0], MAX_REPS);
        return GEIST_TEST_ERROR;
    }
    struct geist_backend *mt = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &mt) != GEIST_OK || mt == nullptr) {
        GEIST_SKIP("metal backend unavailable (not built or no device)");
    }
    static const struct {
        const char *name;
        size_t      n_in, n_out;
    } shapes[] = {
            {"ffn_up", 5120, 17408},
            {"ffn_down", 17408, 5120},
            {"attn_qkv", 5120, 10240},
    };
    size_t ms[16] = {256, 512, 2048};
    size_t n_ms   = 3;
    if (argc > 3) {
        n_ms = 0;
        for (int a = 3; a < argc && n_ms < sizeof ms / sizeof ms[0]; a++) {
            const long m = atol(argv[a]);
            if (m < 1) {
                fprintf(stderr, "bad row count: %s\n", argv[a]);
                geist_backend_destroy(mt);
                return GEIST_TEST_ERROR;
            }
            ms[n_ms++] = (size_t) m;
        }
    }
    int rc = GEIST_TEST_PASS;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0] && rc == GEIST_TEST_PASS; s++) {
        for (size_t j = 0; j < n_ms && rc == GEIST_TEST_PASS; j++) {
            rc = bench_shape(
                    mt, shapes[s].name, shapes[s].n_in, shapes[s].n_out, ms[j], iters, reps);
        }
    }
    geist_backend_destroy(mt);
    return rc;
}
