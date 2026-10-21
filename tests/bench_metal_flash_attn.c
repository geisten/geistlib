/*
 * bench_metal_flash_attn — time one prefill flash-attention call on Metal
 * (#91).
 *
 * Shapes are the attention layers of Bonsai 2 27B (qwen35: 24 query heads,
 * 4 KV heads, head_dim 256) and Gemma 4 E2B (8 query heads, 1 KV head;
 * sliding layers head_dim 256 with a 512 window, global layers head_dim
 * 512). A chunk of m query rows (default 320, the prefill chunk) sits at
 * the end of a cache of kv rows (default 1024 2048 4096 8192), causal, the
 * way a prompt's last chunk sees it. Each sample encodes ITERS
 * back-to-back prims->attention calls on an F16 cache (an F32 cache with
 * "f32" as the third argument) in one prefill batch, as the engine's
 * prefill submits them, and drains them with a 4-byte download; the median
 * sample over REPS gives ms per call.
 *
 * TFLOP/s counts the two matrix products only, over the (query, key)
 * pairs the mask lets through: 4 * head_dim * q_heads * sum_i n_i, where
 * n_i is the number of keys query row i attends (causal, clipped to the
 * window). Without a mask this is the usual 4*m*kv*hd*heads.
 *
 * GEIST_FA_M=<m> sets the query rows instead (1: the decode kernels, #17);
 * a Llama 3.2 3B shape (24/8 heads, head_dim 128) runs last.
 *
 * Usage: bench_metal_flash_attn [iters] [reps] [f16|f32] [kv ...].
 * SKIPs without a Metal device. Timing: run it under the host's benchmark
 * lock.
 */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include "quant.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_REPS = 64, M_ROWS = 320 };

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

struct shape {
    const char *name;
    size_t      qh, kvh, hd, sw;
};

static int bench_shape(struct geist_backend *be,
                       const struct shape   *sh,
                       size_t                m,
                       size_t                kv,
                       bool                  f32kv,
                       int                   iters,
                       int                   reps) {
    const struct geist_backend_vtbl       *v  = be->desc->vtbl;
    const struct geist_backend_primitives *pr = be->desc->prims;
    const size_t                           qe = m * sh->qh * sh->hd, ke = kv * sh->kvh * sh->hd;
    const size_t                           kv_elem = f32kv ? sizeof(float) : sizeof(uint16_t);
    struct geist_buffer *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr;
    float               *q  = dev_alloc(be, qe * sizeof(float), &bq);
    void                *k  = dev_alloc(be, ke * kv_elem, &bk);
    void                *vv = dev_alloc(be, ke * kv_elem, &bv);
    float               *o  = dev_alloc(be, qe * sizeof(float), &bo);
    if (q == nullptr || k == nullptr || vv == nullptr || o == nullptr) {
        fprintf(stderr, "%s: backend buffer alloc failed\n", sh->name);
        return GEIST_TEST_ERROR;
    }
    uint32_t rng = 0x9e3779b9u;
    for (size_t i = 0; i < qe; i++) {
        rng  = rng * 1664525u + 1013904223u;
        q[i] = ((float) (rng >> 24) - 127.5f) / 512.0f;
    }
    for (size_t i = 0; i < ke; i++) {
        rng           = rng * 1664525u + 1013904223u;
        const float a = ((float) (rng >> 24) - 127.5f) / 128.0f;
        rng           = rng * 1664525u + 1013904223u;
        const float b = ((float) (rng >> 24) - 127.5f) / 128.0f;
        if (f32kv) {
            ((float *) k)[i]  = a;
            ((float *) vv)[i] = b;
        } else {
            ((uint16_t *) k)[i]  = fp32_to_fp16_rne(a);
            ((uint16_t *) vv)[i] = fp32_to_fp16_rne(b);
        }
    }
    const enum geist_dtype kdt = f32kv ? GEIST_DTYPE_F32 : GEIST_DTYPE_F16;
    struct geist_tensor    tq  = {.buffer = bq,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 3,
                                  .shape  = {(int64_t) m, (int64_t) sh->qh, (int64_t) sh->hd},
                                  .stride = {(int64_t) (sh->qh * sh->hd), (int64_t) sh->hd, 1}};
    struct geist_tensor    tk  = {.buffer = bk,
                                  .dtype  = kdt,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 3,
                                  .shape  = {(int64_t) kv, (int64_t) sh->kvh, (int64_t) sh->hd},
                                  .stride = {(int64_t) (sh->kvh * sh->hd), (int64_t) sh->hd, 1}};
    struct geist_tensor    tv  = tk;
    tv.buffer                  = bv;
    struct geist_tensor to     = tq;
    to.buffer                  = bo;
    const size_t q_off         = kv - m;
    double       pairs         = 0.0;
    for (size_t i = 0; i < m; i++) {
        const size_t n = q_off + i + 1;
        pairs += (double) (sh->sw > 0 && n > sh->sw ? sh->sw : n);
    }
    float  probe[1];
    double samples[MAX_REPS];
    for (int r = -1; r < reps; r++) { /* r == -1: warm-up */
        const double t0  = now_s();
        const int    tok = v->parallel_region_begin != nullptr
                                   ? v->parallel_region_begin(be, GEIST_REGION_PREFILL_BATCH)
                                   : 0;
        for (int i = 0; i < iters; i++) {
            if (pr->attention(be, &tq, &tk, &tv, q_off, sh->sw, &to) != GEIST_OK) {
                fprintf(stderr, "%s: attention failed\n", sh->name);
                return GEIST_TEST_ERROR;
            }
        }
        if (v->parallel_region_end != nullptr) {
            v->parallel_region_end(be, tok);
        }
        if (v->buffer_download(sizeof probe, (uint8_t *) probe, bo) != GEIST_OK) {
            fprintf(stderr, "%s: download failed\n", sh->name);
            return GEIST_TEST_ERROR;
        }
        if (r >= 0) {
            samples[r] = (now_s() - t0) / iters;
        }
    }
    qsort(samples, (size_t) reps, sizeof samples[0], cmp_double);
    const double t     = samples[reps / 2];
    const double flops = 4.0 * (double) sh->hd * (double) sh->qh * pairs;
    printf("%-11s %s m=%-4zu kv=%-5zu heads=%zu/%zu hd=%-3zu sw=%-3zu %8.3f ms  %5.2f TFLOP/s  "
           "(min %.3f max %.3f)\n",
           sh->name,
           f32kv ? "f32kv" : "f16kv",
           m,
           kv,
           sh->qh,
           sh->kvh,
           sh->hd,
           sh->sw,
           t * 1e3,
           flops / t * 1e-12,
           samples[0] * 1e3,
           samples[reps - 1] * 1e3);
    fflush(stdout);
    v->buffer_destroy(be, bq);
    v->buffer_destroy(be, bk);
    v->buffer_destroy(be, bv);
    v->buffer_destroy(be, bo);
    return GEIST_TEST_PASS;
}

int main(int argc, char **argv) {
    const int  iters = argc > 1 ? atoi(argv[1]) : 8;
    const int  reps  = argc > 2 ? atoi(argv[2]) : 7;
    const bool f32kv = argc > 3 && strcmp(argv[3], "f32") == 0;
    if (iters < 1 || reps < 1 || reps > MAX_REPS) {
        fprintf(stderr,
                "usage: %s [iters >= 1] [reps 1..%d] [f16|f32] [kv ...]\n",
                argv[0],
                MAX_REPS);
        return GEIST_TEST_ERROR;
    }
    const char  *m_env       = getenv("GEIST_FA_M");
    const size_t m_rows      = m_env != nullptr && atol(m_env) > 0 ? (size_t) atol(m_env) : M_ROWS;
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        GEIST_SKIP("metal backend unavailable (not built or no device)");
    }
    static const struct shape shapes[] = {
            {"bonsai", 24, 4, 256, 0},
            {"gemma.swa", 8, 1, 256, 512},
            {"gemma.glob", 8, 1, 512, 0},
            {"llama3.3b", 24, 8, 128, 0},
    };
    size_t kvs[16] = {1024, 2048, 4096, 8192};
    size_t n_kv    = 4;
    if (argc > 4) {
        n_kv = 0;
        for (int a = 4; a < argc && n_kv < sizeof kvs / sizeof kvs[0]; a++) {
            const long kv = atol(argv[a]);
            if (kv < (long) m_rows) {
                fprintf(stderr, "bad kv length (< %zu): %s\n", m_rows, argv[a]);
                geist_backend_destroy(be);
                return GEIST_TEST_ERROR;
            }
            kvs[n_kv++] = (size_t) kv;
        }
    }
    int rc = GEIST_TEST_PASS;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0] && rc == GEIST_TEST_PASS; s++) {
        for (size_t j = 0; j < n_kv && rc == GEIST_TEST_PASS; j++) {
            rc = bench_shape(be, &shapes[s], m_rows, kvs[j], f32kv, iters, reps);
        }
    }
    geist_backend_destroy(be);
    return rc;
}
