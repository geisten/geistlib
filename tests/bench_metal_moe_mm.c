/*
 * bench_metal_moe_mm — time Metal's grouped expert GEMM (#494).
 *
 * One call of fused->linear_t_ids (moe_mm_<T>, from 9 rows) or of
 * fused->moe_ffn_ids (moe_mm_glu_<T> + moe_mm_<T>) on random experts, for
 * the shapes llama.cpp's `test-backend-ops perf -o MUL_MAT_ID` times
 * (Qwen3-30B-A3B: 128 experts, 8 used, 768 x 2048; gpt-oss-20b: 32 experts,
 * 4 used, 2880 x 2880; x one row per slot, as its b = false) and the
 * OLMoE-1B-7B projections (64 experts, 8 used, gate / up 1024 x 2048 with x
 * one row per token, down 2048 x 1024 per slot, and the whole FFN). Each
 * token routes to k distinct experts drawn uniformly (llama.cpp's test
 * draws a shuffled permutation per token).
 *
 * Each sample encodes ITERS back-to-back calls in one prefill batch and
 * drains them with a 4-byte download; the median over REPS gives ms per
 * call. TFLOP/s = 2 * m * k * n_out * n_in per projection, llama.cpp's
 * count.
 *
 * Usage: bench_metal_moe_mm [iters] [reps] [m] [filter]. filter: a
 * substring of the row name ("q30b", "gptoss", "olmoe", "Q4_K", ...).
 * SKIPs without a Metal device. Timing: run it under the host's benchmark
 * lock.
 */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_REPS = 64, MAX_EXP = 128 };

struct qfmt {
    enum geist_dtype dtype;
    size_t           elems, bytes;
    bool             trailing_d; /* Q6_K: d is the trailing f16 */
    const char      *name;
};

static const struct qfmt Q4K = {GEIST_DTYPE_Q4_K, 256, 144, false, "Q4_K"};
static const struct qfmt Q6K = {GEIST_DTYPE_Q6_K, 256, 210, true, "Q6_K"};
static const struct qfmt Q80 = {GEIST_DTYPE_Q8_0, 32, 34, false, "Q8_0"};
static const struct qfmt MX4 = {GEIST_DTYPE_MXFP4, 32, 17, false, "MXFP4"};
static const struct qfmt Q50 = {GEIST_DTYPE_Q5_0, 32, 22, false, "Q5_0"};

static uint32_t rng = 0x13579bdfu;
static uint32_t rng_u32(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

static struct geist_tensor mat(struct geist_buffer *b, size_t rows, size_t cols) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

/* n experts of rows x cols of q: random blocks with a scale that keeps a
 * weight ~0.1 (half-staged activations stay finite). */
struct experts {
    uint8_t             *blob;
    struct geist_weight  w0;
    struct geist_buffer *buf;
    struct geist_tensor  t;
};

static int make_experts(struct geist_backend *be,
                        const struct qfmt    *q,
                        size_t                n,
                        size_t                rows,
                        size_t                cols,
                        struct experts       *ex) {
    const size_t rb = (cols / q->elems) * q->bytes, wb = rows * rb;
    ex->blob = malloc(n * wb + 16384);
    if (ex->blob == nullptr)
        return GEIST_TEST_ERROR;
    uint8_t *al = ex->blob + (16384 - ((uintptr_t) ex->blob % 16384)) % 16384;
    for (size_t b = 0; b < n * rows * (cols / q->elems); b++) {
        uint8_t *blk = al + b * q->bytes;
        for (size_t i = 0; i < q->bytes; i++)
            blk[i] = (uint8_t) (rng_u32() >> 24);
        if (q->dtype == GEIST_DTYPE_MXFP4) {
            blk[0] = (uint8_t) (120 + (rng_u32() >> 24) % 4);
            continue;
        }
        const uint8_t dh = q->dtype == GEIST_DTYPE_Q4_K   ? 0x14
                           : q->dtype == GEIST_DTYPE_Q6_K ? 0x0C
                           : q->dtype == GEIST_DTYPE_Q8_0 ? 0x1C
                                                          : 0x24;
        uint8_t      *d  = q->trailing_d ? blk + q->bytes - 2 : blk;
        d[0]             = 0x00;
        d[1]             = dh;
        if (q->dtype == GEIST_DTYPE_Q4_K) {
            blk[2] = 0x00;
            blk[3] = dh;
        }
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    ex->w0                             = (struct geist_weight) {.raw        = al,
                                                                .raw_nbytes = wb,
                                                                .n_in       = (int64_t) cols,
                                                                .n_out      = (int64_t) rows,
                                                                .dtype      = (uint16_t) q->dtype};
    if (v->resolve_weight(be, &ex->w0) != GEIST_OK ||
        v->buffer_create_aliased(be, al, n * wb, GEIST_BUFFER_WEIGHT, &ex->buf) != GEIST_OK)
        return GEIST_TEST_ERROR;
    ex->t = (struct geist_tensor) {.buffer = ex->buf,
                                   .dtype  = q->dtype,
                                   .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                                   .ndim   = 2,
                                   .shape  = {(int64_t) (n * rows), (int64_t) cols},
                                   .stride = {(int64_t) cols, 1}};
    return GEIST_TEST_PASS;
}

static void free_experts(struct geist_backend *be, struct experts *ex) {
    if (ex->buf != nullptr)
        be->desc->vtbl->buffer_destroy(be, ex->buf);
    free(ex->blob);
    *ex = (struct experts) {0};
}

/* k distinct experts per token, uniform. */
static void make_ids(size_t n, size_t m, size_t k, uint32_t *ids) {
    for (size_t t = 0; t < m; t++) {
        bool used[MAX_EXP] = {false};
        for (size_t j = 0; j < k; j++) {
            uint32_t e;
            do
                e = rng_u32() % (uint32_t) n;
            while (used[e]);
            used[e]        = true;
            ids[t * k + j] = e;
        }
    }
}

/* GEIST_BENCH_IDS=<file>: a model's routing instead, records of (n, m, k)
 * uint32 then m * k uint32 ids; the GEIST_BENCH_IDS_REC-th (default 0)
 * record of this n, m and k. false: none. */
static bool load_ids(size_t n, size_t m, size_t k, uint32_t *ids) {
    const char *path = getenv("GEIST_BENCH_IDS");
    const char *rec  = getenv("GEIST_BENCH_IDS_REC");
    long        want = rec != nullptr ? atol(rec) : 0;
    FILE       *f    = path != nullptr ? fopen(path, "rb") : nullptr;
    bool        ok   = false;
    uint32_t    hd[3];
    while (f != nullptr && !ok && fread(hd, 4, 3, f) == 3) {
        const size_t nk = (size_t) hd[1] * hd[2];
        if (hd[0] == n && hd[1] == m && hd[2] == k && want-- == 0)
            ok = fread(ids, 4, nk, f) == nk;
        else if (fseek(f, (long) (nk * 4), SEEK_CUR) != 0)
            break;
    }
    if (f != nullptr)
        fclose(f);
    return ok;
}

/* A row: one projection (ffn == false) or gate + up + down (ffn). */
struct row {
    const char        *name;
    const struct qfmt *q;
    const struct qfmt *qd; /* ffn: down's dtype */
    size_t             n, k, n_out, n_in;
    bool               per_slot, ffn;
};

static int bench_row(struct geist_backend *be, const struct row *r, size_t m, int iters, int reps) {
    const struct geist_backend_vtbl  *v    = be->desc->vtbl;
    const struct geist_backend_fused *f    = be->desc->fused;
    const size_t                      nk   = m * r->k;
    struct experts                    e[3] = {0};
    struct geist_buffer              *bx = nullptr, *by = nullptr, *bg = nullptr, *bi = nullptr;
    int                               rc = GEIST_TEST_ERROR;
    float                            *xs = malloc((r->per_slot ? nk : m) * r->n_in * sizeof *xs);
    uint32_t                         *id = malloc(nk * sizeof *id);
    if (xs == nullptr || id == nullptr)
        goto out;
    /* ffn: gate / up n_out x n_in, down n_in x n_out. */
    if (make_experts(be, r->q, r->n, r->n_out, r->n_in, &e[0]) != GEIST_TEST_PASS ||
        (r->ffn && (make_experts(be, r->q, r->n, r->n_out, r->n_in, &e[1]) != GEIST_TEST_PASS ||
                    make_experts(be, r->qd, r->n, r->n_in, r->n_out, &e[2]) != GEIST_TEST_PASS)))
        goto out;
    const size_t xr = r->per_slot ? nk : m, yc = r->ffn ? r->n_in : r->n_out;
    for (size_t i = 0; i < xr * r->n_in; i++)
        xs[i] = ((float) (rng_u32() >> 24) - 127.5f) / 64.0f;
    if (!load_ids(r->n, m, r->k, id))
        make_ids(r->n, m, r->k, id);
    if (v->buffer_create(be, xr * r->n_in * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bx) !=
                GEIST_OK ||
        v->buffer_create(be, nk * yc * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &by) != GEIST_OK ||
        v->buffer_create(be, nk * r->n_out * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bg) !=
                GEIST_OK ||
        v->buffer_create(be, nk * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) != GEIST_OK ||
        v->buffer_upload(bx, xr * r->n_in * 4, (const uint8_t *) xs) != GEIST_OK ||
        v->buffer_upload(bi, nk * 4, (const uint8_t *) id) != GEIST_OK)
        goto out;
    struct geist_tensor tx = mat(bx, xr, r->n_in), ty = mat(by, nk, yc), tg = mat(bg, nk, r->n_out);
    const struct geist_moe_ffn_args fa = {.n       = r->n,
                                          .m       = m,
                                          .k       = r->k,
                                          .w0      = {&e[0].w0, &e[1].w0, &e[2].w0},
                                          .experts = {&e[0].t, &e[1].t, &e[2].t},
                                          .ids     = bi,
                                          .x       = &tx,
                                          .g       = &tg,
                                          .y       = &ty};
    float                           probe[1];
    double                          samples[MAX_REPS];
    for (int s = -2; s < reps; s++) { /* two warm-ups */
        const double t0  = now_s();
        const int    tok = v->parallel_region_begin != nullptr
                                   ? v->parallel_region_begin(be, GEIST_REGION_PREFILL_BATCH)
                                   : 0;
        for (int i = 0; i < iters; i++) {
            const enum geist_status st = r->ffn ? f->moe_ffn_ids(be, &fa)
                                                : f->linear_t_ids(be,
                                                                  r->n,
                                                                  m,
                                                                  r->k,
                                                                  r->per_slot,
                                                                  &e[0].w0,
                                                                  &e[0].t,
                                                                  bi,
                                                                  &tx,
                                                                  &ty);
            if (st != GEIST_OK) {
                fprintf(stderr, "%s %s: call failed (%d)\n", r->name, r->q->name, (int) st);
                goto out;
            }
        }
        if (v->parallel_region_end != nullptr)
            v->parallel_region_end(be, tok);
        if (v->buffer_download(sizeof probe, (uint8_t *) probe, by) != GEIST_OK)
            goto out;
        if (s >= 0)
            samples[s] = (now_s() - t0) / iters;
    }
    qsort(samples, (size_t) reps, sizeof samples[0], cmp_double);
    const double t = samples[reps / 2];
    const double flops =
            2.0 * (double) nk * (double) r->n_out * (double) r->n_in * (r->ffn ? 3 : 1);
    printf("%-14s %-5s %-5s n=%-3zu k=%zu m=%-4zu %4zux%-4zu %-5s %8.3f us  %6.2f TFLOP/s  "
           "(min %.3f max %.3f)\n",
           r->name,
           r->q->name,
           r->ffn ? r->qd->name : "",
           r->n,
           r->k,
           m,
           r->n_out,
           r->n_in,
           r->ffn ? "ffn" : (r->per_slot ? "slot" : "token"),
           t * 1e6,
           flops / t * 1e-12,
           samples[0] * 1e6,
           samples[reps - 1] * 1e6);
    fflush(stdout);
    rc = GEIST_TEST_PASS;
out:
    if (rc != GEIST_TEST_PASS)
        fprintf(stderr, "%s %s: setup failed\n", r->name, r->q->name);
    for (int i = 0; i < 3; i++)
        free_experts(be, &e[i]);
    if (bx != nullptr)
        v->buffer_destroy(be, bx);
    if (by != nullptr)
        v->buffer_destroy(be, by);
    if (bg != nullptr)
        v->buffer_destroy(be, bg);
    if (bi != nullptr)
        v->buffer_destroy(be, bi);
    free(xs);
    free(id);
    return rc;
}

int main(int argc, char **argv) {
    const int    iters  = argc > 1 ? atoi(argv[1]) : 8;
    const int    reps   = argc > 2 ? atoi(argv[2]) : 9;
    const long   m_arg  = argc > 3 ? atol(argv[3]) : 512;
    const char  *filter = argc > 4 ? argv[4] : nullptr;
    const size_t m      = m_arg > 0 ? (size_t) m_arg : 0;
    if (iters < 1 || reps < 1 || reps > MAX_REPS || m < 9) {
        fprintf(stderr,
                "usage: %s [iters >= 1] [reps 1..%d] [m >= 9] [filter]\n",
                argv[0],
                MAX_REPS);
        return GEIST_TEST_ERROR;
    }
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        GEIST_SKIP("metal backend unavailable (not built or no device)");
    }
    if (be->desc->fused == nullptr || be->desc->fused->linear_t_ids == nullptr ||
        be->desc->fused->moe_ffn_ids == nullptr) {
        geist_backend_destroy(be);
        GEIST_SKIP("no grouped expert GEMM");
    }
    /* llama.cpp's test_mul_mat_id(type, F32, n, k, b, n_out, m, n_in): b
     * (one x row per token) is gate / up, !b down and Qwen3-30B's case. */
    static const struct row shapes[] = {
            {"olmoe.gate", nullptr, nullptr, 64, 8, 1024, 2048, false, false},
            {"olmoe.down", nullptr, nullptr, 64, 8, 2048, 1024, true, false},
            {"q30b", nullptr, nullptr, 128, 8, 768, 2048, true, false},
            {"gptoss.gate", nullptr, nullptr, 32, 4, 2880, 2880, false, false},
            {"gptoss.down", nullptr, nullptr, 32, 4, 2880, 2880, true, false},
            {"olmoe.ffn", nullptr, nullptr, 64, 8, 1024, 2048, false, true},
            {"gptoss.ffn", nullptr, nullptr, 32, 4, 2880, 2880, false, true},
    };
    static const struct qfmt *const qs[] = {&Q4K, &Q6K, &Q80, &Q50, &MX4};
    int                             rc   = GEIST_TEST_PASS;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0] && rc == GEIST_TEST_PASS; i++)
        for (size_t j = 0; j < sizeof qs / sizeof qs[0] && rc == GEIST_TEST_PASS; j++) {
            struct row r = shapes[i];
            r.q          = qs[j];
            r.qd         = qs[j];
            char key[64];
            snprintf(key, sizeof key, "%s %s", r.name, r.q->name);
            if ((filter != nullptr && strstr(key, filter) == nullptr) || r.n_in % r.q->elems != 0 ||
                (r.ffn && r.n_out % r.q->elems != 0))
                continue; /* gpt-oss's 2880: no whole K-quant blocks */
            rc = bench_row(be, &r, m, iters, reps);
        }
    geist_backend_destroy(be);
    return rc;
}
