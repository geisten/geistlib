/*
 * test_linear_fallback_unit — a linear kernel that cannot take its fast
 * path still computes y.
 *
 * A linear kernel returns void (geist_weight.h): its caller takes y as
 * written. The fast CPU kernels had paths that returned without computing
 * it: zeros when their activation scratch could not be had, nothing (or a
 * null dereference) when the thread's workspace could not, nothing for
 * more rows than their stack accumulators hold (GEIST_QUANT_M_CAP). They
 * compute y with geist_linear_ref there, which needs neither.
 *
 * For every CPU backend in the build and every dtype it binds, on a weight
 * of 12 rows and one of 32, both 512 wide:
 *   - without heap: linear_m1, linear_mN (m = 9) and, where the two
 *     weights share them, the pair kernels run with every heap allocation
 *     failing (heap_fail_allocations), on a fresh thread — no workspace
 *     yet — and on one that ran them on 256-wide weights first, so every
 *     buffer it has must grow. Each row of each y must be, bit for bit, one
 *     of the answers a kernel may give without scratch: its own (it needed
 *     none), linear_m1's for that row (an mN kernel that falls back row by
 *     row), the single kernel's (a pair that splits) or geist_linear_ref's.
 *   - m = GEIST_QUANT_M_CAP + 1, heap working: each row of linear_mN's y
 *     within 10 % (L2) of the reference's. A kernel that takes any m gives
 *     its own answer there, x quantized, not the reference's bits: up to
 *     2 % off on these weights (Q4_1). A row of zeros is 100 % off.
 * Zeros are none of these, and neither is a y the kernel never wrote:
 * every y starts as NaN.
 *
 * cpu_neon runs again with its opt-in paths switched on, which default
 * off on this host class: predecoded and tiled Q4_K prefill, dequant +
 * SGEMM prefill from m = 4 in tiles of 8 rows, the Q6_K x8 GEMV, TL1, the
 * dequant trampolines in place of the native mN kernels, and the row
 * kernels in place of the default-on x8 GEMVs.
 */
#define _POSIX_C_SOURCE 200809L /* setenv, unsetenv */

#include "test_helpers.h"
#include "weight_aux.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "heap.h"
#include "linear_ref.h"
#include "quant.h"
#include "quant_fixtures.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

constexpr size_t N_IN   = 512;
constexpr size_t N_WARM = 256;
constexpr size_t M      = 9; /* an 8-row tile and a tail */
constexpr size_t M_BIG  = GEIST_QUANT_M_CAP + 1;

static const size_t N_OUT[2] = {12, 32};

constexpr double BIG_M_TOL = 0.10;

struct env_kv {
    const char *name, *value;
};

/* cpu_neon's opt-in kernels, a pass for each set that can run together:
 * at m >= the SGEMM threshold the tiled Q4_K kernels never run. */
static const struct env_kv PASS_TILED[] = {
        {"GEIST_Q4K_PREDECODE", "1"},
        {"GEIST_Q4K_MTILE_PREFILL", "1"},
        {"GEIST_Q4K_BLOCK_Q8_PREFILL", "1"},
        {"GEIST_Q6K_X8_GEMV", "1"},
        {"GEIST_TL1", "1"},
        {"GEIST_Q5K_NATIVE_MN", "0"},
        {"GEIST_Q8_0_NATIVE_MN", "0"},
        {"GEIST_Q4_01_NATIVE_MN", "0"},
        {"GEIST_IQ4XS_NATIVE_MN", "0"},
        {"GEIST_TQ2_0_NATIVE_MN", "0"},
        {"GEIST_Q4_0_X8_GEMV", "0"},
        {"GEIST_PQ2_0_X8_GEMV", "0"},
};
static const struct env_kv PASS_NTILED[] = {
        {"GEIST_Q4K_PREDECODE", "1"},
        {"GEIST_Q4K_MTILE_PREFILL", "1"},
        {"GEIST_Q4K_NTILE_PREFILL", "1"},
};
static const struct env_kv PASS_SGEMM[] = {
        {"GEIST_Q4K_SGEMM_PREFILL", "1"},
        {"GEIST_Q6K_SGEMM_PREFILL", "1"},
        {"GEIST_QK_SGEMM_THRESHOLD", "4"},
        {"GEIST_QK_SGEMM_TILE_ROWS", "8"},
};

static const struct {
    const char          *name;
    size_t               n;
    const struct env_kv *kv;
} PASSES[] = {
        {"default", 0, nullptr},
        {"tiled", sizeof PASS_TILED / sizeof PASS_TILED[0], PASS_TILED},
        {"n-tiled", sizeof PASS_NTILED / sizeof PASS_NTILED[0], PASS_NTILED},
        {"sgemm", sizeof PASS_SGEMM / sizeof PASS_SGEMM[0], PASS_SGEMM},
};

/* Two weights of one dtype and width, as a layer's gate and up are. */
struct wpair {
    struct geist_weight w[2];
    uint8_t            *raw[2];
};

/* One weight's y from each kernel, each [M, n]: m1 (linear_m1 of every
 * row of x where a test needs them all, else of row 0), mN, and the pair
 * kernels' share, p1 (row 0) and pN. */
struct ys {
    float *m1, *mN, *p1, *pN;
};

static bool has_pair_m1(const struct wpair *p) {
    return p->w[0].linear_pair_m1 != nullptr && p->w[0].linear_pair_m1 == p->w[1].linear_pair_m1;
}

static bool has_pair_mN(const struct wpair *p) {
    return p->w[0].linear_pair_mN != nullptr && p->w[0].linear_pair_mN == p->w[1].linear_pair_mN;
}

static void wpair_free(struct wpair *p) {
    for (size_t i = 0; i < 2; i++) {
        weight_aux_free(&p->w[i]);
        free(p->raw[i]);
        p->raw[i] = nullptr;
    }
}

/* 1: resolved; 0: the backend has no kernel for the dtype; -1: OOM. */
static int wpair_make(struct geist_backend *be, size_t d, size_t n_in, struct wpair *p) {
    *p = (struct wpair) {0};
    for (size_t i = 0; i < 2; i++) {
        size_t bytes = 0;
        p->raw[i]    = make_weight(d, n_in, N_OUT[i], &bytes);
        if (p->raw[i] == nullptr) {
            wpair_free(p);
            return -1;
        }
        p->w[i] = (struct geist_weight) {.raw        = p->raw[i],
                                         .raw_nbytes = bytes,
                                         .n_in       = (int32_t) n_in,
                                         .n_out      = (int32_t) N_OUT[i],
                                         .dtype      = (uint16_t) DTYPES[d].dt};
        if (be->desc->vtbl->resolve_weight(be, &p->w[i]) != GEIST_OK) {
            wpair_free(p);
            return 0;
        }
    }
    return 1;
}

static void
run_kernels(struct geist_backend *be, const struct wpair *p, const float *x, struct ys y[2]) {
    for (size_t i = 0; i < 2; i++) {
        p->w[i].linear_m1(x, &p->w[i], be, y[i].m1);
        p->w[i].linear_mN(M, x, &p->w[i], be, y[i].mN);
    }
    if (has_pair_m1(p)) {
        p->w[0].linear_pair_m1(x, &p->w[0], &p->w[1], be, y[0].p1, y[1].p1);
    }
    if (has_pair_mN(p)) {
        p->w[0].linear_pair_mN(M, x, &p->w[0], &p->w[1], be, y[0].pN, y[1].pN);
    }
}

struct starve {
    struct geist_backend *be;
    const struct wpair   *warm; /* run first, heap working; nullptr: none */
    const float          *xw;   /* [M, N_WARM] */
    struct ys            *yw;   /* its y */
    const struct wpair   *t;
    const float          *x; /* [M, N_IN] */
    struct ys            *y;
};

static void *run_starved(void *arg) {
    struct starve *s = arg;
    if (s->warm != nullptr) {
        run_kernels(s->be, s->warm, s->xw, s->yw);
    }
    heap_fail_allocations(true);
    run_kernels(s->be, s->t, s->x, s->y);
    heap_fail_allocations(false);
    return nullptr;
}

/* Every row of y [rows, n] is, bit for bit, that row of one of the nc
 * candidates. */
static bool
rows_match(size_t rows, size_t n, size_t nc, const float *y, const float *const c[static nc]) {
    for (size_t i = 0; i < rows; i++) {
        bool hit = false;
        for (size_t k = 0; k < nc && !hit; k++) {
            hit = memcmp(y + i * n, c[k] + i * n, n * sizeof *y) == 0;
        }
        if (!hit) {
            return false;
        }
    }
    return true;
}

/* The largest, over the rows, of |y_i - r_i| / |r_i| (L2); +inf for a row
 * that is not finite. */
static double worst_rel(size_t rows, size_t n, const float *y, const float *r) {
    double worst = 0.0;
    for (size_t i = 0; i < rows; i++) {
        double e = 0.0, s = 0.0;
        for (size_t j = 0; j < n; j++) {
            const double yv = y[i * n + j], rv = r[i * n + j];
            if (!isfinite(yv)) {
                return INFINITY;
            }
            e += (yv - rv) * (yv - rv);
            s += rv * rv;
        }
        const double rel = s > 0.0 ? sqrt(e / s) : sqrt(e);
        worst            = rel > worst ? rel : worst;
    }
    return worst;
}

static void fill_nan(size_t n, float *y) {
    for (size_t i = 0; i < n; i++) {
        y[i] = NAN;
    }
}

/* Lay out a ys for n outputs from *cur onward. */
static struct ys ys_at(size_t n, float **cur) {
    const struct ys y = {
            .m1 = *cur, .mN = *cur + M * n, .p1 = *cur + 2 * M * n, .pN = *cur + 3 * M * n};
    *cur += 4 * M * n;
    return y;
}

static int check_starved(const char         *tag,
                         const char         *how,
                         const struct wpair *t,
                         const struct ys     norm[2],
                         const float *const  ref[2],
                         const struct ys     got[2]) {
    int  fails = 0;
    char msg[224];
    for (size_t i = 0; i < 2; i++) {
        const size_t       n    = N_OUT[i];
        const float *const c1[] = {norm[i].m1, ref[i]};
        snprintf(msg,
                 sizeof msg,
                 "%s %zux%zu, %s: linear_m1 gives its own y or the reference's",
                 tag,
                 n,
                 N_IN,
                 how);
        fails += geist_expect(rows_match(1, n, 2, got[i].m1, c1), msg);
        const float *const cN[] = {norm[i].mN, norm[i].m1, ref[i]};
        snprintf(msg,
                 sizeof msg,
                 "%s %zux%zu, %s: linear_mN gives its own, m1's or the reference's",
                 tag,
                 n,
                 N_IN,
                 how);
        fails += geist_expect(rows_match(M, n, 3, got[i].mN, cN), msg);
        if (has_pair_m1(t)) {
            const float *const p1[] = {norm[i].p1, norm[i].m1, ref[i]};
            snprintf(msg,
                     sizeof msg,
                     "%s %zux%zu, %s: linear_pair_m1 gives its own, m1's or the reference's",
                     tag,
                     n,
                     N_IN,
                     how);
            fails += geist_expect(rows_match(1, n, 3, got[i].p1, p1), msg);
        }
        if (has_pair_mN(t)) {
            const float *const pN[] = {norm[i].pN, norm[i].mN, norm[i].m1, ref[i]};
            snprintf(msg,
                     sizeof msg,
                     "%s %zux%zu, %s: linear_pair_mN gives its own, mN's, m1's or the reference's",
                     tag,
                     n,
                     N_IN,
                     how);
            fails += geist_expect(rows_match(M, n, 4, got[i].pN, pN), msg);
        }
    }
    return fails;
}

static int run_dtype(const char *backend, const char *pass, struct geist_backend *be, size_t d) {
    char tag[96];
    snprintf(tag, sizeof tag, "%s (%s) %s", backend, pass, DTYPES[d].name);
    struct wpair t, warm;
    const int    rt = wpair_make(be, d, N_IN, &t);
    if (rt <= 0) {
        if (rt < 0) {
            fprintf(stderr, "FAIL: %s: test allocation\n", tag);
        }
        return rt < 0; /* 0: no kernel for this dtype here */
    }
    if (wpair_make(be, d, N_WARM, &warm) != 1) {
        fprintf(stderr, "FAIL: %s: a %zu-wide weight\n", tag, N_WARM);
        wpair_free(&t);
        return 1;
    }
    const size_t nsum = N_OUT[0] + N_OUT[1];
    /* x [M_BIG, N_IN] and xw [M, N_WARM]; per weight: the answers with the
     * heap (ys), the reference [M_BIG, n], two starved runs (ys each), the
     * warm run's y (ys) and the m > cap run [M_BIG, n]. */
    const size_t floats = M_BIG * N_IN + M * N_WARM + nsum * (4 * 4 * M + 2 * M_BIG);
    float       *buf    = malloc(floats * sizeof *buf);
    if (buf == nullptr) {
        fprintf(stderr, "FAIL: %s: test allocation\n", tag);
        wpair_free(&t);
        wpair_free(&warm);
        return 1;
    }
    fill_nan(floats, buf);
    float *cur = buf, *x = cur, *xw = cur + M_BIG * N_IN;
    cur += M_BIG * N_IN + M * N_WARM;
    struct ys norm[2], fresh[2], warmed[2], yw[2];
    float    *ref[2], *big[2];
    for (size_t i = 0; i < 2; i++) {
        norm[i]   = ys_at(N_OUT[i], &cur);
        fresh[i]  = ys_at(N_OUT[i], &cur);
        warmed[i] = ys_at(N_OUT[i], &cur);
        yw[i]     = ys_at(N_OUT[i], &cur);
        ref[i]    = cur;
        big[i]    = cur + M_BIG * N_OUT[i];
        cur += 2 * M_BIG * N_OUT[i];
    }
    for (size_t i = 0; i < M_BIG * N_IN; i++) {
        x[i] = next_f();
    }
    for (size_t i = 0; i < M * N_WARM; i++) {
        xw[i] = next_f();
    }

    int fails = 0;
    run_kernels(be, &t, x, norm);
    for (size_t i = 0; i < 2; i++) {
        for (size_t r = 1; r < M; r++) { /* norm.m1: linear_m1 of every row */
            t.w[i].linear_m1(x + r * N_IN, &t.w[i], be, norm[i].m1 + r * N_OUT[i]);
        }
        geist_linear_ref(M_BIG, x, &t.w[i], ref[i]);
    }

    struct starve s[2] = {
            {.be = be, .warm = nullptr, .t = &t, .x = x, .y = fresh},
            {.be = be, .warm = &warm, .xw = xw, .yw = yw, .t = &t, .x = x, .y = warmed},
    };
    const char *const how[2] = {"no heap, fresh thread", "no heap, every buffer to grow"};
    for (size_t k = 0; k < 2; k++) {
        pthread_t th;
        if (pthread_create(&th, nullptr, run_starved, &s[k]) != 0 ||
            pthread_join(th, nullptr) != 0) {
            fprintf(stderr, "FAIL: %s: thread\n", tag);
            fails++;
            continue;
        }
        fails += check_starved(tag, how[k], &t, norm, (const float *const *) ref, s[k].y);
    }

    /* More rows than the kernels' stack accumulators hold. */
    char msg[224];
    for (size_t i = 0; i < 2; i++) {
        t.w[i].linear_mN(M_BIG, x, &t.w[i], be, big[i]);
        const double e = worst_rel(M_BIG, N_OUT[i], big[i], ref[i]);
        snprintf(msg,
                 sizeof msg,
                 "%s %zux%zu: linear_mN at m = %zu within 10 %% of the reference (worst %.3g)",
                 tag,
                 N_OUT[i],
                 N_IN,
                 M_BIG,
                 e);
        fails += geist_expect(e <= BIG_M_TOL, msg);
    }
    if (has_pair_mN(&t)) {
        fill_nan(M_BIG * N_OUT[0], big[0]);
        fill_nan(M_BIG * N_OUT[1], big[1]);
        t.w[0].linear_pair_mN(M_BIG, x, &t.w[0], &t.w[1], be, big[0], big[1]);
        for (size_t i = 0; i < 2; i++) {
            const double e = worst_rel(M_BIG, N_OUT[i], big[i], ref[i]);
            snprintf(msg,
                     sizeof msg,
                     "%s %zux%zu: linear_pair_mN at m = %zu within 10 %% of the reference "
                     "(worst %.3g)",
                     tag,
                     N_OUT[i],
                     N_IN,
                     M_BIG,
                     e);
            fails += geist_expect(e <= BIG_M_TOL, msg);
        }
    }

    wpair_free(&t);
    wpair_free(&warm);
    free(buf);
    return fails;
}

int main(void) {
    int fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        /* The opt-in passes switch cpu_neon's kernels; the other backends
         * read none of these variables. */
        const size_t passes =
                strcmp(BACKENDS[b], "cpu_neon") == 0 ? sizeof PASSES / sizeof PASSES[0] : 1;
        for (size_t p = 0; p < passes; p++) {
            for (size_t k = 0; k < PASSES[p].n; k++) {
                setenv(PASSES[p].kv[k].name, PASSES[p].kv[k].value, 1);
            }
            struct geist_backend *be = nullptr;
            const bool ok = geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) == GEIST_OK &&
                            be != nullptr;
            for (size_t k = 0; k < PASSES[p].n; k++) {
                unsetenv(PASSES[p].kv[k].name);
            }
            if (!ok) {
                break; /* not in this build */
            }
            ran++;
            for (size_t d = 0; d < N_DTYPES; d++) {
                fails += run_dtype(BACKENDS[b], PASSES[p].name, be, d);
            }
            geist_backend_destroy(be);
        }
    }
    if (ran == 0) {
        printf("SKIP: no CPU backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: every CPU linear kernel computes y without its scratch and past its row cap\n");
    return GEIST_TEST_PASS;
}
