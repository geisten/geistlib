/*
 * bench_linear_mN — one GGUF weight's linear on one backend, per m.
 *
 * The model-level profile (GEIST_PROFILE_FORWARD) says which stage costs
 * most; this says how fast one kernel runs on one real weight, so it can be
 * set beside llama.cpp's `test-backend-ops perf -o MUL_MAT` for the same
 * shape (#16, #17). It resolves the tensor through the backend's own
 * resolve_weight (the kernel the model would run), then times linear_mN
 * (m > 1) or linear_m1 (m == 1) on random activations.
 *
 * Usage: bench_linear_mN <model.gguf> <tensor> [m ...]   (default m: 1 128 512)
 * GEIST_BENCH_BACKEND names the backend (default "auto"). Prints, per m,
 * ms per call and GFLOP/s (2 * m * n_in * n_out per call).
 */
#define _POSIX_C_SOURCE 200809L

#include "gguf_reader.h"
#include "heap.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static int geist_dtype_of(gguf_dtype_t t) {
    switch (t) {
    case GGUF_TYPE_F32:
        return GEIST_DTYPE_F32;
    case GGUF_TYPE_Q4_0:
        return GEIST_DTYPE_Q4_0;
    case GGUF_TYPE_Q8_0:
        return GEIST_DTYPE_Q8_0;
    case GGUF_TYPE_Q4_K:
        return GEIST_DTYPE_Q4_K;
    case GGUF_TYPE_Q5_K:
        return GEIST_DTYPE_Q5_K;
    case GGUF_TYPE_Q6_K:
        return GEIST_DTYPE_Q6_K;
    default:
        return -1;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { /* run without arguments by `make bench`: nothing to time */
        printf("SKIP: usage: %s <model.gguf> <tensor> [m ...]\n", argv[0]);
        return GEIST_TEST_SKIP;
    }
    const char      *err = nullptr;
    struct gguf_ctx *g   = gguf_open(argv[1], &err);
    if (g == nullptr) {
        fprintf(stderr, "gguf_open: %s\n", err != nullptr ? err : "?");
        return GEIST_TEST_ERROR;
    }
    const struct gguf_tensor_t *t  = gguf_get_tensor(g, argv[2]);
    const int                   dt = t != nullptr ? geist_dtype_of(t->dtype) : -1;
    if (t == nullptr || t->n_dims != 2 || dt < 0) {
        fprintf(stderr, "%s: no 2-D tensor of a supported dtype\n", argv[2]);
        return GEIST_TEST_ERROR;
    }
    const char           *bname = getenv("GEIST_BENCH_BACKEND");
    struct geist_backend *be    = nullptr;
    if (geist_backend_create(bname != nullptr ? bname : "auto", nullptr, nullptr, &be) !=
        GEIST_OK) {
        fprintf(stderr, "backend: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_weight w = {.raw        = t->data,
                             .raw_nbytes = t->nbytes,
                             .n_in       = (int32_t) t->dims[0],
                             .n_out      = (int32_t) t->dims[1],
                             .dtype      = (uint16_t) dt};
    if (be->desc->vtbl->resolve_weight(be, &w) != GEIST_OK || w.linear_m1 == nullptr ||
        w.linear_mN == nullptr) {
        fprintf(stderr, "resolve_weight failed\n");
        return GEIST_TEST_ERROR;
    }
    const size_t n_in = (size_t) w.n_in, n_out = (size_t) w.n_out;
    size_t       ms[16] = {1, 128, 512};
    size_t       n_m    = 3;
    if (argc > 3) {
        n_m = 0;
        for (int i = 3; i < argc && n_m < 16; i++) {
            ms[n_m++] = (size_t) strtoul(argv[i], nullptr, 10);
        }
    }
    printf("%s %s [%zu -> %zu] on %s\n",
           argv[2],
           gguf_dtype_name(t->dtype),
           n_in,
           n_out,
           be->desc->name);
    int fails = 0;
    for (size_t k = 0; k < n_m; k++) {
        const size_t m = ms[k];
        float       *x = malloc(m * n_in * sizeof *x);
        float       *y = malloc(m * n_out * sizeof *y);
        if (x == nullptr || y == nullptr || m == 0) {
            fails++;
            free(x);
            free(y);
            continue;
        }
        srand(1);
        for (size_t i = 0; i < m * n_in; i++) {
            x[i] = (float) rand() / (float) RAND_MAX - 0.5f;
        }
        /* One warm call sizes the run to ~0.5 s. */
        double t0 = now_s();
        if (m == 1) {
            w.linear_m1(x, &w, be, y);
        } else {
            w.linear_mN(m, x, &w, be, y);
        }
        const double warm  = now_s() - t0;
        const int    calls = warm > 0.0 ? (int) (0.5 / warm) + 1 : 1;
        t0                 = now_s();
        for (int c = 0; c < calls; c++) {
            if (m == 1) {
                w.linear_m1(x, &w, be, y);
            } else {
                w.linear_mN(m, x, &w, be, y);
            }
        }
        const double el  = now_s() - t0;
        const double per = el / (double) calls;
        printf("  m=%-5zu %9.3f ms/call  %8.1f GFLOP/s\n",
               m,
               1e3 * per,
               2.0 * (double) m * (double) n_in * (double) n_out / per * 1e-9);
        free(x);
        free(y);
    }
    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
        void *aux = (void *) w.aux_fp32;
        safe_free(&aux);
    }
    geist_backend_destroy(be);
    gguf_close(g);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
