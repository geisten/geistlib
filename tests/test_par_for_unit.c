/*
 * test_par_for_unit — geist_par_for (src/base/par.h, #618) on whichever
 * backend this build has: the OpenMP team, or (built without OpenMP, as
 * MODE=tsan is) the pthread pool.
 *
 *   1. n == 0 calls nothing;
 *   2. one thread, or n == 1: a single fn(ctx, 0, n) on the caller;
 *   3. n below the thread count, and a large n: every index exactly once,
 *      in at most min(n, threads) contiguous non-empty ranges whose sizes
 *      differ by at most one, each on its own thread;
 *   4. a geist_par_for inside a body runs serially on that body's thread;
 *   5. two pthreads calling it at once, each with its own thread count,
 *      both get every index exactly once (the TSan leg runs this test).
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "par.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

constexpr size_t MAX_RANGES = 64;

/* What one geist_par_for saw: per-index visit counts and every range. */
struct visits {
    size_t                 n;
    _Atomic unsigned char *count;
    _Atomic size_t         n_ranges;
    size_t                 begin[MAX_RANGES];
    size_t                 end[MAX_RANGES];
    pthread_t              thread[MAX_RANGES];
    _Atomic bool           overflow;
};

static void record(void *ctx, size_t begin, size_t end) {
    struct visits *v = ctx;
    const size_t   k = atomic_fetch_add(&v->n_ranges, 1);
    if (k < MAX_RANGES) {
        v->begin[k]  = begin;
        v->end[k]    = end;
        v->thread[k] = pthread_self();
    } else {
        atomic_store(&v->overflow, true);
    }
    for (size_t i = begin; i < end && i < v->n; i++) {
        atomic_fetch_add(&v->count[i], 1);
    }
}

static int cmp_size(const void *a, const void *b) {
    const size_t x = *(const size_t *) a;
    const size_t y = *(const size_t *) b;
    return (x > y) - (x < y);
}

/* Runs geist_par_for(n) at `threads` and checks case 3's contract. Returns
 * the number of failed checks. */
static int check_cover(size_t n, size_t threads, const char *what) {
    geist_par_set_max_threads(threads);
    struct visits *v = xmalloc(sizeof *v);
    *v               = (struct visits) {.n = n};
    v->count         = xmalloc(n > 0 ? n : 1);
    for (size_t i = 0; i < n; i++) {
        atomic_init(&v->count[i], 0);
    }
    geist_par_for(n, record, v);

    int          fails = 0;
    const size_t k     = atomic_load(&v->n_ranges);
    const size_t cap   = n < threads ? n : threads;
    size_t       bad   = 0;
    for (size_t i = 0; i < n; i++) {
        bad += atomic_load(&v->count[i]) != 1;
    }
    if (bad != 0 || k > cap || (n > 0 && k == 0) || atomic_load(&v->overflow)) {
        fprintf(stderr,
                "FAIL: %s (n=%zu, threads=%zu): %zu indices not visited once, %zu ranges "
                "(cap %zu)\n",
                what,
                n,
                threads,
                bad,
                k,
                cap);
        fails++;
    }
    /* Contiguous, non-empty, balanced: sorted by begin they tile [0, n). */
    size_t order[MAX_RANGES];
    for (size_t i = 0; i < k && i < MAX_RANGES; i++) {
        order[i] = v->begin[i] * MAX_RANGES + i; /* begin-major, then slot */
    }
    qsort(order, k < MAX_RANGES ? k : MAX_RANGES, sizeof order[0], cmp_size);
    size_t at   = 0;
    size_t lo   = SIZE_MAX;
    size_t hi   = 0;
    bool   tile = true;
    for (size_t j = 0; j < k && j < MAX_RANGES; j++) {
        const size_t i    = order[j] % MAX_RANGES;
        const size_t size = v->end[i] - v->begin[i];
        tile              = tile && v->begin[i] == at && v->end[i] > v->begin[i];
        at                = v->end[i];
        lo                = size < lo ? size : lo;
        hi                = size > hi ? size : hi;
    }
    if (n > 0 && (!tile || at != n || hi - lo > 1)) {
        fprintf(stderr, "FAIL: %s (n=%zu): ranges do not tile [0, n) evenly\n", what, n);
        fails++;
    }
#if defined(_OPENMP) || !defined(__APPLE__)
    /* One range per thread (GCD may run several on one worker). */
    for (size_t a = 0; a < k && a < MAX_RANGES; a++) {
        for (size_t b = a + 1; b < k && b < MAX_RANGES; b++) {
            if (pthread_equal(v->thread[a], v->thread[b])) {
                fprintf(stderr, "FAIL: %s (n=%zu): two ranges on one thread\n", what, n);
                fails++;
                a = b = MAX_RANGES;
            }
        }
    }
#endif
    free((void *) v->count);
    free(v);
    return fails;
}

static void never(void *ctx, size_t begin, size_t end) {
    (void) begin;
    (void) end;
    atomic_store((_Atomic bool *) ctx, true);
}

/* ---- 4. nesting ---------------------------------------------------------- */

struct nest {
    _Atomic size_t inner_visits;
    _Atomic size_t inner_calls;
    _Atomic bool   foreign; /* an inner range ran off its outer body's thread */
};

struct inner_ctx {
    struct nest *nest;
    pthread_t    outer;
};

static void inner(void *ctx, size_t begin, size_t end) {
    const struct inner_ctx *c = ctx;
    if (!pthread_equal(pthread_self(), c->outer)) {
        atomic_store(&c->nest->foreign, true);
    }
    atomic_fetch_add(&c->nest->inner_calls, 1);
    atomic_fetch_add(&c->nest->inner_visits, end - begin);
}

constexpr size_t NEST_INNER = 1000;

static void outer(void *ctx, size_t begin, size_t end) {
    struct inner_ctx c = {.nest = ctx, .outer = pthread_self()};
    for (size_t i = begin; i < end; i++) {
        geist_par_for(NEST_INNER, inner, &c);
    }
}

static int check_nested(void) {
    geist_par_set_max_threads(4);
    struct nest  n       = {0};
    const size_t outer_n = 8;
    geist_par_for(outer_n, outer, &n);
    const size_t visits = atomic_load(&n.inner_visits);
    const size_t calls  = atomic_load(&n.inner_calls);
    if (visits != outer_n * NEST_INNER || calls != outer_n || atomic_load(&n.foreign)) {
        fprintf(stderr,
                "FAIL: nested: %zu inner visits (want %zu), %zu inner ranges (want %zu, one "
                "serial range per outer index)%s\n",
                visits,
                outer_n * NEST_INNER,
                calls,
                outer_n,
                atomic_load(&n.foreign) ? ", some off the outer thread" : "");
        return 1;
    }
    return 0;
}

/* ---- 5. concurrent callers ---------------------------------------------- */

struct caller {
    size_t threads;
    int    fails;
};

static void *call_many(void *arg) {
    struct caller *c = arg;
    for (int it = 0; it < 300 && c->fails == 0; it++) {
        c->fails += check_cover(1000 + (size_t) it, c->threads, "concurrent");
    }
    return nullptr;
}

static int check_concurrent(void) {
    struct caller c[2] = {{.threads = 3}, {.threads = 4}};
    pthread_t     t[2];
    for (int i = 0; i < 2; i++) {
        if (pthread_create(&t[i], nullptr, call_many, &c[i]) != 0) {
            fprintf(stderr, "ERROR: pthread_create\n");
            exit(GEIST_TEST_ERROR);
        }
    }
    for (int i = 0; i < 2; i++) {
        pthread_join(t[i], nullptr);
    }
    return c[0].fails + c[1].fails;
}

int main(void) {
    int fails = 0;

    /* 1. */
    _Atomic bool called = false;
    geist_par_set_max_threads(4);
    geist_par_for(0, never, &called);
    fails += geist_expect(!atomic_load(&called), "n == 0 called fn");

    /* 2. and 3. */
    fails += check_cover(1, 4, "n == 1");
    fails += check_cover(1000, 1, "one thread");
    fails += check_cover(3, 8, "n < threads");
    fails += check_cover(7, 7, "n == threads");
    fails += check_cover((size_t) 1 << 20, 4, "large n");
    fails += check_cover((size_t) 1 << 20, 16, "large n, 16 threads");
    fails += check_cover(100003, 5, "uneven split");

    /* The count is the calling thread's and sticks. */
    geist_par_set_max_threads(3);
    geist_par_set_max_threads(0); /* ignored */
    fails += geist_expect(geist_par_max_threads() == 3, "geist_par_max_threads after set(3)");

    /* 4. and 5. */
    fails += check_nested();
    fails += check_concurrent();

    if (fails != 0) {
        fprintf(stderr, "FAIL: %d check(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: geist_par_for (%s)\n",
#if defined(_OPENMP)
           "OpenMP"
#elif defined(__APPLE__)
           "GCD"
#else
           "pthread pool"
#endif
    );
    return GEIST_TEST_PASS;
}
