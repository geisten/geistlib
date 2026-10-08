#define _POSIX_C_SOURCE 200809L /* sysconf, clock_gettime, sched_yield */
/*
 * src/base/par.c — geist_par_for on OpenMP, GCD or a pthread pool. See par.h.
 *
 * Layer: ENGINE.
 */
#include "par.h"

#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#else
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#else
#include "omp_idle.h"

#include <pthread.h>
#include <sched.h>
#include <time.h>
#endif
#endif

/* Range t of T over [0, n): n / T items, one more for t < n % T. */
static inline void par_range(size_t n, size_t t, size_t T, size_t *begin, size_t *end) {
    const size_t q = n / T;
    const size_t r = n % T;
    *begin         = t * q + (t < r ? t : r);
    *end           = *begin + q + (t < r ? 1 : 0);
}

#if defined(_OPENMP)

void geist_par_for(size_t n, geist_par_fn fn, void *ctx) {
    if (n == 0) {
        return;
    }
    if (n == 1 || omp_in_parallel() || omp_get_max_threads() <= 1) {
        fn(ctx, 0, n);
        return;
    }
#pragma omp parallel
    {
        const size_t T = (size_t) omp_get_num_threads();
        const size_t t = (size_t) omp_get_thread_num();
        if (t < n) {
            size_t b;
            size_t e;
            par_range(n, t, T < n ? T : n, &b, &e);
            fn(ctx, b, e);
        }
    }
}

size_t geist_par_max_threads(void) {
    const int n = omp_get_max_threads();
    return n > 0 ? (size_t) n : 1;
}

void geist_par_set_max_threads(size_t n) {
    if (n > 0) {
        omp_set_num_threads(n > (size_t) INT32_MAX ? INT32_MAX : (int) n);
    }
}

#else /* !_OPENMP */

/* This thread's count (0: the default) and whether it runs a body. */
static _Thread_local size_t tl_max_threads = 0;
static _Thread_local bool   tl_inside      = false;

/* OMP_NUM_THREADS's leading count (so one variable sizes both builds),
 * else the online CPUs. Racing first calls compute the same value. */
static size_t default_threads(void) {
    static _Atomic size_t cached = 0;
    size_t                n      = atomic_load_explicit(&cached, memory_order_relaxed);
    if (n == 0) {
        const char *e = getenv("OMP_NUM_THREADS");
        const long  v = e != nullptr ? strtol(e, nullptr, 10) : 0;
        if (v > 0) {
            n = (size_t) v;
        } else {
            const long c = sysconf(_SC_NPROCESSORS_ONLN);
            n            = c > 0 ? (size_t) c : 1;
        }
        atomic_store_explicit(&cached, n, memory_order_relaxed);
    }
    return n;
}

size_t geist_par_max_threads(void) {
    return tl_max_threads > 0 ? tl_max_threads : default_threads();
}

void geist_par_set_max_threads(size_t n) {
    if (n > 0) {
        tl_max_threads = n;
    }
}

#if defined(__APPLE__)

/* ponytail: untested here (no mac in the loop); kept to the one call. */
struct gcd_job {
    size_t       n;
    size_t       T;
    geist_par_fn fn;
    void        *ctx;
};

static void gcd_range(void *p, size_t t) {
    const struct gcd_job *job = p;
    size_t                b;
    size_t                e;
    par_range(job->n, t, job->T, &b, &e);
    const bool outer = tl_inside;
    tl_inside        = true;
    job->fn(job->ctx, b, e);
    tl_inside = outer;
}

void geist_par_for(size_t n, geist_par_fn fn, void *ctx) {
    if (n == 0) {
        return;
    }
    const size_t max = geist_par_max_threads();
    const size_t T   = n < max ? n : max;
    if (T <= 1 || tl_inside) {
        fn(ctx, 0, n);
        return;
    }
    struct gcd_job job = {.n = n, .T = T, .fn = fn, .ctx = ctx};
    dispatch_apply_f(T, DISPATCH_APPLY_AUTO, &job, gcd_range);
}

#else /* pthread pool */

/* Workers beyond this are not created; T is capped to it. */
constexpr size_t POOL_MAX = 256;
/* The published word is (generation << 16) | T. */
constexpr unsigned WORD_T_BITS = 16;
static_assert(POOL_MAX < (1u << WORD_T_BITS), "T must fit the word's low bits");

/* One cache line (or more) per worker: the caller polls `done` and
 * `sleeping` of each, so a worker finishing touches only its own line. */
struct par_worker {
    alignas(64) _Atomic uint64_t done; /* the word of the last range finished */
    atomic_bool     sleeping;
    uint64_t        start_word; /* the word when created; set before pthread_create */
    pthread_mutex_t mu;
    pthread_cond_t  cv;
};

static struct {
    /* Serializes callers; guards everything below that is not atomic. */
    pthread_mutex_t lock;
    size_t          n_workers; /* workers 1..n_workers exist */
    uint64_t        gen;
    long long       spin_ns; /* idle spin before a worker sleeps */
    /* The task, written under `lock` before `word` publishes it, read by
     * workers 1..T-1 until each releases `done`. */
    size_t       n;
    geist_par_fn fn;
    void        *ctx;
    alignas(64) _Atomic uint64_t word;   /* its own line: every worker polls it */
    struct par_worker workers[POOL_MAX]; /* [0] unused: the caller is range 0 */
} g_pool = {.lock = PTHREAD_MUTEX_INITIALIZER};

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Waits for the word to move past `seen`: spins until the worker has been
 * idle (no range of its own) for spin_ns since `idle_since`, then sleeps on
 * its condition variable. Only a call that gives the worker a range wakes
 * it, so a worker above the decode team does not spin on every word. The
 * seq_cst store of `sleeping` before the re-check pairs with the caller's
 * seq_cst store of `word` before it loads `sleeping`: one of the two sees
 * the other. */
static uint64_t
wait_word(struct par_worker *me, uint64_t seen, long long spin_ns, long long idle_since) {
    for (unsigned i = 1;; i++) {
        const uint64_t w = atomic_load_explicit(&g_pool.word, memory_order_acquire);
        if (w != seen) {
            return w;
        }
        cpu_relax();
        if ((i & 255u) == 0 && now_ns() - idle_since > spin_ns) {
            break;
        }
    }
    pthread_mutex_lock(&me->mu);
    atomic_store(&me->sleeping, true);
    uint64_t w;
    while ((w = atomic_load(&g_pool.word)) == seen) {
        pthread_cond_wait(&me->cv, &me->mu);
    }
    atomic_store(&me->sleeping, false);
    pthread_mutex_unlock(&me->mu);
    return w;
}

static void *worker_main(void *arg) {
    const size_t       id      = (size_t) (uintptr_t) arg;
    struct par_worker *me      = &g_pool.workers[id];
    const long long    spin_ns = g_pool.spin_ns;
    uint64_t           seen    = me->start_word;
    long long          idle    = now_ns();
    tl_inside                  = true; /* a body's own geist_par_for runs serially */
    for (;;) {
        seen           = wait_word(me, seen, spin_ns, idle);
        const size_t T = (size_t) (seen & ((1u << WORD_T_BITS) - 1));
        if (id < T) {
            size_t b;
            size_t e;
            par_range(g_pool.n, id, T, &b, &e);
            g_pool.fn(g_pool.ctx, b, e);
            atomic_store_explicit(&me->done, seen, memory_order_release);
            idle = now_ns();
        }
    }
    return nullptr;
}

/* Grows the pool to T - 1 workers; returns the T it can serve. Under lock. */
static size_t ensure_workers(size_t T) {
    if (g_pool.n_workers == 0) {
        g_pool.spin_ns = (long long) geist_omp_idle_spin_ms(nullptr) * 1000000LL;
    }
    while (g_pool.n_workers + 1 < T) {
        const size_t       id = g_pool.n_workers + 1;
        struct par_worker *w  = &g_pool.workers[id];
        pthread_mutex_init(&w->mu, nullptr);
        pthread_cond_init(&w->cv, nullptr);
        atomic_init(&w->sleeping, false);
        atomic_init(&w->done, 0);
        w->start_word = atomic_load_explicit(&g_pool.word, memory_order_relaxed);
        pthread_t tid;
        if (pthread_create(&tid, nullptr, worker_main, (void *) (uintptr_t) id) != 0) {
            pthread_cond_destroy(&w->cv);
            pthread_mutex_destroy(&w->mu);
            break;
        }
        pthread_detach(tid);
        g_pool.n_workers = id;
    }
    return T < g_pool.n_workers + 1 ? T : g_pool.n_workers + 1;
}

void geist_par_for(size_t n, geist_par_fn fn, void *ctx) {
    if (n == 0) {
        return;
    }
    size_t T = geist_par_max_threads();
    T        = T < n ? T : n;
    T        = T < POOL_MAX ? T : POOL_MAX;
    if (T <= 1 || tl_inside) {
        fn(ctx, 0, n);
        return;
    }
    pthread_mutex_lock(&g_pool.lock);
    T = ensure_workers(T);
    if (T <= 1) {
        pthread_mutex_unlock(&g_pool.lock);
        fn(ctx, 0, n);
        return;
    }
    g_pool.n   = n;
    g_pool.fn  = fn;
    g_pool.ctx = ctx;
    g_pool.gen++;
    const uint64_t word = (g_pool.gen << WORD_T_BITS) | T;
    atomic_store(&g_pool.word, word); /* publishes the task */
    for (size_t i = 1; i < T; i++) {
        struct par_worker *w = &g_pool.workers[i];
        if (atomic_load(&w->sleeping)) {
            pthread_mutex_lock(&w->mu);
            pthread_cond_signal(&w->cv);
            pthread_mutex_unlock(&w->mu);
        }
    }
    size_t b;
    size_t e;
    par_range(n, 0, T, &b, &e);
    tl_inside = true;
    fn(ctx, b, e);
    tl_inside = false;
    /* Spin for the stragglers; yield once that takes long (oversubscribed
     * host), so a descheduled worker gets the core back. */
    unsigned spins = 0;
    for (size_t i = 1; i < T; i++) {
        while (atomic_load_explicit(&g_pool.workers[i].done, memory_order_acquire) != word) {
            if (++spins < (1u << 14)) {
                cpu_relax();
            } else {
                sched_yield();
            }
        }
    }
    pthread_mutex_unlock(&g_pool.lock);
}

#endif /* __APPLE__ */
#endif /* _OPENMP */
