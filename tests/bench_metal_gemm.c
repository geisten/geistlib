/*
 * bench_metal_gemm — time one quantized prefill GEMM on Metal (#590).
 *
 * bench_metal_pq2_gemm for the GGUF block formats: one weight of DTYPE
 * (q4_0, q8_0, q4_k, q6_k), n_in x n_out, applied to m rows through the
 * weight's linear_mN, the call the prefill makes. Each sample encodes ITERS
 * back-to-back GEMMs on backend buffers in one prefill batch, as the
 * engine's prefill submits them (GEIST_BENCH_NO_BATCH=1: one sequence per
 * GEMM, which adds a command buffer's fixed cost to each), and drains them
 * with a 4-byte download; the median sample over REPS gives ms per GEMM and
 * TFLOP/s
 * (2 * m * n_in * n_out). Weights are random blocks with scales in
 * [2^-6, 2^-5): the kernel does the same work for any codes. The last
 * column is an FNV-1a hash over the output's bytes, so two builds of a
 * kernel can be compared for identical results on the same shape; before it
 * the largest difference to cpu_scalar's result on a sample of rows, over
 * the largest magnitude there.
 *
 * Usage: bench_metal_gemm DTYPE n_in n_out [iters] [reps] [m ...]
 * (default 20 iters, 7 reps, m 256). SKIPs without a Metal device.
 * Timing: run it under the host's benchmark lock.
 */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_REPS = 64 };

struct format {
    const char *name;
    uint16_t    dtype;
    size_t      block_elems, block_bytes;
    size_t      d_off[2]; /* byte offsets of the fp16 scales; SIZE_MAX = none */
};

static const struct format formats[] = {
        {"q4_0", (uint16_t) GEIST_DTYPE_Q4_0, 32, 18, {0, SIZE_MAX}},
        {"q8_0", (uint16_t) GEIST_DTYPE_Q8_0, 32, 34, {0, SIZE_MAX}},
        {"q4_k", (uint16_t) GEIST_DTYPE_Q4_K, 256, 144, {0, 2}},
        {"q6_k", (uint16_t) GEIST_DTYPE_Q6_K, 256, 210, {208, SIZE_MAX}},
};

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

static uint64_t fnv1a(size_t n, const uint8_t bytes[static n]) {
    uint64_t h = 0xcbf29ce484222325u;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ bytes[i]) * 0x100000001b3u;
    }
    return h;
}

/* max |y - ref| / max |ref| over a sample of rows, ref from cpu_scalar's
 * one-row linear on the same blocks; -1 if the reference is unavailable. */
static double ref_error(struct geist_backend *cpu,
                        const struct format  *fmt,
                        size_t                n_in,
                        size_t                n_out,
                        size_t                m,
                        size_t                w_bytes,
                        const uint8_t        *blob,
                        const float          *x,
                        const float          *y) {
    struct geist_weight w   = {.raw        = blob,
                               .raw_nbytes = w_bytes,
                               .n_in       = (int32_t) n_in,
                               .n_out      = (int32_t) n_out,
                               .dtype      = fmt->dtype};
    float              *ref = malloc(n_out * sizeof(float));
    if (cpu == nullptr || ref == nullptr || cpu->desc->vtbl->resolve_weight(cpu, &w) != GEIST_OK ||
        w.linear_mN == nullptr) {
        free(ref);
        return -1.0;
    }
    const size_t rows[] = {0, 1, 15, 16, 31, 32, 63, 64, 127, 128, m / 2, m - 2, m - 1};
    double       err = 0.0, mag = 0.0;
    for (size_t r = 0; r < sizeof rows / sizeof rows[0]; r++) {
        if (rows[r] >= m) {
            continue;
        }
        w.linear_mN(1, x + rows[r] * n_in, &w, cpu, ref);
        for (size_t i = 0; i < n_out; i++) {
            const double d = fabs((double) y[rows[r] * n_out + i] - (double) ref[i]);
            err            = d > err ? d : err;
            mag            = fabs((double) ref[i]) > mag ? fabs((double) ref[i]) : mag;
        }
    }
    free(ref);
    return mag > 0.0 ? err / mag : -1.0;
}

static int bench_shape(struct geist_backend *mt,
                       struct geist_backend *cpu,
                       const struct format  *fmt,
                       size_t                n_in,
                       size_t                n_out,
                       size_t                m,
                       int                   iters,
                       int                   reps) {
    const struct geist_backend_vtbl *v       = mt->desc->vtbl;
    const size_t                     w_bytes = n_out * (n_in / fmt->block_elems) * fmt->block_bytes;
    const size_t                     y_bytes = m * n_out * sizeof(float);
    struct geist_buffer             *bw = nullptr, *bx = nullptr, *by = nullptr;
    uint8_t                         *blob = dev_alloc(mt, w_bytes, &bw);
    float                           *x    = dev_alloc(mt, m * n_in * sizeof(float), &bx);
    float                           *y    = dev_alloc(mt, y_bytes, &by);
    uint8_t                         *out  = malloc(y_bytes);
    if (blob == nullptr || x == nullptr || y == nullptr || out == nullptr) {
        fprintf(stderr, "%s: buffer alloc failed\n", fmt->name);
        return GEIST_TEST_ERROR;
    }
    uint32_t rng = 0x9e3779b9u;
    for (size_t b = 0; b < w_bytes; b += fmt->block_bytes) {
        for (size_t i = 0; i < fmt->block_bytes; i++) {
            rng         = rng * 1664525u + 1013904223u;
            blob[b + i] = (uint8_t) (rng >> 24);
        }
        for (size_t s = 0; s < 2; s++) {
            /* fp16 in [2^-6, 2^-5) with a random mantissa: finite outputs
             * for any codes, and products that round, so the hash moves
             * when a kernel sums in another order. */
            if (fmt->d_off[s] != SIZE_MAX) {
                blob[b + fmt->d_off[s] + 1] = 0x24 | (blob[b + fmt->d_off[s] + 1] & 0x03);
            }
        }
    }
    for (size_t i = 0; i < m * n_in; i++) {
        rng  = rng * 1664525u + 1013904223u;
        x[i] = ((float) (rng >> 8) - 8388607.5f) / 8388608.0f;
    }
    struct geist_weight w = {.raw        = blob,
                             .raw_nbytes = w_bytes,
                             .n_in       = (int32_t) n_in,
                             .n_out      = (int32_t) n_out,
                             .dtype      = fmt->dtype};
    if (v->resolve_weight(mt, &w) != GEIST_OK || w.linear_mN == nullptr) {
        fprintf(stderr, "%s: resolve_weight failed\n", fmt->name);
        return GEIST_TEST_ERROR;
    }
    /* GEIST_BENCH_NO_BATCH=1: every GEMM as a sequence of its own, with the
     * command buffer that costs. */
    const char *nb    = getenv("GEIST_BENCH_NO_BATCH");
    const bool  batch = nb == nullptr || nb[0] != '1';
    double      samples[MAX_REPS];
    for (int r = -1; r < reps; r++) { /* r == -1: warm-up */
        const double t0  = now_s();
        const int    tok = batch && v->parallel_region_begin != nullptr
                                   ? v->parallel_region_begin(mt, GEIST_REGION_PREFILL_BATCH)
                                   : 0;
        for (int i = 0; i < iters; i++) {
            w.linear_mN(m, x, &w, mt, y);
        }
        if (batch && v->parallel_region_end != nullptr) {
            v->parallel_region_end(mt, tok);
        }
        if (v->buffer_download(4, out, by) != GEIST_OK) {
            fprintf(stderr, "%s: download failed\n", fmt->name);
            return GEIST_TEST_ERROR;
        }
        if (r >= 0) {
            samples[r] = (now_s() - t0) / iters;
        }
    }
    if (v->buffer_download(y_bytes, out, by) != GEIST_OK) {
        fprintf(stderr, "%s: download failed\n", fmt->name);
        return GEIST_TEST_ERROR;
    }
    qsort(samples, (size_t) reps, sizeof samples[0], cmp_double);
    const double t     = samples[reps / 2];
    const double flops = 2.0 * (double) m * (double) n_in * (double) n_out;
    printf("%-5s n_in=%-6zu n_out=%-6zu m=%-5zu %8.3f ms  %6.2f TFLOP/s  (min %.3f max %.3f)  "
           "ref_err=%.2e y_fnv=%016llx\n",
           fmt->name,
           n_in,
           n_out,
           m,
           t * 1e3,
           flops / t * 1e-12,
           samples[0] * 1e3,
           samples[reps - 1] * 1e3,
           ref_error(cpu, fmt, n_in, n_out, m, w_bytes, blob, x, (const float *) out),
           (unsigned long long) fnv1a(y_bytes, out));
    fflush(stdout);
    free(out);
    v->buffer_destroy(mt, bw);
    v->buffer_destroy(mt, bx);
    v->buffer_destroy(mt, by);
    return GEIST_TEST_PASS;
}

int main(int argc, char **argv) {
    const struct format *fmt = nullptr;
    for (size_t i = 0; argc > 1 && i < sizeof formats / sizeof formats[0]; i++) {
        if (strcmp(argv[1], formats[i].name) == 0) {
            fmt = &formats[i];
        }
    }
    const long n_in  = argc > 2 ? atol(argv[2]) : 0;
    const long n_out = argc > 3 ? atol(argv[3]) : 0;
    const int  iters = argc > 4 ? atoi(argv[4]) : 20;
    const int  reps  = argc > 5 ? atoi(argv[5]) : 7;
    if (fmt == nullptr || n_in < 1 || n_out < 1 || (size_t) n_in % fmt->block_elems != 0 ||
        iters < 1 || reps < 1 || reps > MAX_REPS) {
        fprintf(stderr,
                "usage: %s q4_0|q8_0|q4_k|q6_k n_in n_out [iters >= 1] [reps 1..%d] [m ...]\n"
                "n_in must be a multiple of the format's block (32, K-quants 256)\n",
                argv[0],
                MAX_REPS);
        return GEIST_TEST_ERROR;
    }
    struct geist_backend *mt = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &mt) != GEIST_OK || mt == nullptr) {
        GEIST_SKIP("metal backend unavailable (not built or no device)");
    }
    struct geist_backend *cpu = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu) != GEIST_OK) {
        cpu = nullptr;
    }
    int rc = GEIST_TEST_PASS;
    for (int a = 6; a < argc || a == 6; a++) {
        const long m = a < argc ? atol(argv[a]) : 256;
        if (m < 1) {
            fprintf(stderr, "bad row count: %s\n", argv[a]);
            rc = GEIST_TEST_ERROR;
        }
        if (rc != GEIST_TEST_PASS) {
            break;
        }
        rc = bench_shape(mt, cpu, fmt, (size_t) n_in, (size_t) n_out, (size_t) m, iters, reps);
    }
    if (cpu != nullptr) {
        geist_backend_destroy(cpu);
    }
    geist_backend_destroy(mt);
    return rc;
}
