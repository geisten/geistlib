/*
 * src/backends/cpu_neon/parallel.c — spin-pool parallel_for impl.
 *
 * Layer: BACKEND (cpu_neon). See parallel.h for the rationale.
 *
 * Design:
 *   - One global static pool, lazily initialized.
 *   - Workers spin on `atomic task_epoch`. Master bumps epoch to
 *     dispatch a task; workers see the new epoch, run their chunk,
 *     atomically decrement `pending_workers`. Master spins waiting
 *     for `pending_workers == 0`.
 *   - Static chunking: worker w handles `i ∈ [w * chunk, (w+1) * chunk)`
 *     where `chunk = (n + N - 1) / N`. The last worker may get fewer.
 *   - Master runs chunk 0 inline (so we have N total threads active
 *     when GEIST_THREADS=N: 1 master + (N-1) spinning workers).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "parallel.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#define GEIST_PP_MAX_THREADS 16

struct geist_pp_state {
    /* Workers spin until task_epoch differs from their last-seen. */
    _Atomic uint64_t task_epoch;
    /* Snapshot of the current task; written by master, read by workers
     * AFTER they observe a fresh epoch (acts as a release/acquire on
     * the epoch atomic). */
    size_t           n_total;
    geist_pp_body_fn body_fn;
    void            *ctx;
    /* Decremented by each worker on completion; master spins on this. */
    _Atomic int    pending_workers;
    _Atomic size_t next_index;
    size_t         chunk_size;
    int            dynamic_chunks;
    /* Total active threads (master + workers). */
    int         n_threads;
    pthread_t   workers[GEIST_PP_MAX_THREADS];
    _Atomic int shutdown;
    /* Worker arg storage (so threads see stable indices). */
    int worker_ids[GEIST_PP_MAX_THREADS];
    /* Hybrid wait fallback: after `spin_budget` rounds of pure spin
     * with no new task, workers fall through to cond_wait on `cv`.
     * Master broadcasts on cv after publishing a task — but only when
     * `sleeping > 0` (lock-free guard so the hot path skips the syscall
     * cost). For always-busy decode loops `sleeping` stays at 0 and
     * workers never leave the spin loop. */
    pthread_mutex_t wait_mutex;
    pthread_cond_t  wait_cv;
    _Atomic int     sleeping;
};

static struct geist_pp_state g_state = {
        .task_epoch      = 0,
        .pending_workers = 0,
        .n_threads       = 0,
        .shutdown        = 0,
        .wait_mutex      = PTHREAD_MUTEX_INITIALIZER,
        .wait_cv         = PTHREAD_COND_INITIALIZER,
        .sleeping        = 0,
};
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

/* Spin rounds before falling back to cond_wait. Tuned so that any
 * back-to-back parallel_for sequence (typical decode loop) stays in
 * fast-spin mode; only seconds-scale idle (e.g. session paused for
 * user input) hits the sleep path. ~1e7 rounds ≈ 5-10 ms on a busy
 * core, well above the worst-case inter-region gap in geist. */
#define GEIST_PP_SPIN_BUDGET (10ULL * 1000 * 1000)

/* Run the chunk for worker index `w` of `N`. */
static inline void do_chunk(int w, int N, size_t n_total, geist_pp_body_fn fn, void *ctx) {
    const size_t chunk = (n_total + (size_t) N - 1) / (size_t) N;
    const size_t lo    = (size_t) w * chunk;
    if (lo >= n_total)
        return;
    const size_t hi = (lo + chunk > n_total) ? n_total : lo + chunk;
    for (size_t i = lo; i < hi; i++)
        fn(i, ctx);
}

static void do_dynamic_chunks(size_t n_total, size_t grain, geist_pp_body_fn fn, void *ctx);

static void *worker_main(void *arg) {
    const int w         = *(int *) arg;
    uint64_t  last_seen = 0;
    for (;;) {
        uint64_t epoch;
        uint64_t spin = 0;
        for (;;) {
            epoch = atomic_load_explicit(&g_state.task_epoch, memory_order_acquire);
            if (epoch != last_seen)
                break;
            if (atomic_load_explicit(&g_state.shutdown, memory_order_relaxed))
                return NULL;
            if (++spin >= GEIST_PP_SPIN_BUDGET) {
                /* Long idle: fall through to cond_wait so we stop
                 * burning a core. Increment `sleeping` so the master
                 * knows to broadcast; recheck epoch under the mutex
                 * to avoid lost-wakeup races. */
                atomic_fetch_add_explicit(&g_state.sleeping, 1, memory_order_release);
                pthread_mutex_lock(&g_state.wait_mutex);
                while (atomic_load_explicit(&g_state.task_epoch, memory_order_acquire) ==
                               last_seen &&
                       !atomic_load_explicit(&g_state.shutdown, memory_order_relaxed)) {
                    pthread_cond_wait(&g_state.wait_cv, &g_state.wait_mutex);
                }
                pthread_mutex_unlock(&g_state.wait_mutex);
                atomic_fetch_sub_explicit(&g_state.sleeping, 1, memory_order_release);
                spin = 0;
            }
        }
        last_seen = epoch;
        if (g_state.dynamic_chunks) {
            do_dynamic_chunks(g_state.n_total, g_state.chunk_size, g_state.body_fn, g_state.ctx);
        } else {
            do_chunk(w, g_state.n_threads, g_state.n_total, g_state.body_fn, g_state.ctx);
        }
        atomic_fetch_sub_explicit(&g_state.pending_workers, 1, memory_order_release);
    }
}

static int detect_thread_count(void) {
    const char *env = getenv("GEIST_THREADS");
    if (!env)
        env = getenv("OMP_NUM_THREADS");
    if (env) {
        const int n = atoi(env);
        if (n > 0)
            return n > GEIST_PP_MAX_THREADS ? GEIST_PP_MAX_THREADS : n;
    }
    /* Fallback: physical cores. */
#if defined(__APPLE__)
    int    ncpu = 0;
    size_t sz   = sizeof(ncpu);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &ncpu, &sz, NULL, 0) == 0 && ncpu > 0) {
        return ncpu > GEIST_PP_MAX_THREADS ? GEIST_PP_MAX_THREADS : ncpu;
    }
#endif
    long n = 4;
#if defined(_SC_NPROCESSORS_ONLN)
    n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1)
        n = 4;
#endif
    if (n > GEIST_PP_MAX_THREADS)
        n = GEIST_PP_MAX_THREADS;
    return (int) n;
}

static void init_pool(void) {
    g_state.n_threads = detect_thread_count();
    if (g_state.n_threads <= 1)
        return; /* no workers; master does all */
    for (int w = 1; w < g_state.n_threads; w++) {
        g_state.worker_ids[w] = w;
        if (pthread_create(&g_state.workers[w], NULL, worker_main, &g_state.worker_ids[w]) != 0) {
            fprintf(stderr, "geist_pp: pthread_create(%d) failed\n", w);
            /* Reduce thread count to whatever we got. */
            g_state.n_threads = w;
            return;
        }
    }
}

void geist_pp_parallel_for(size_t n, geist_pp_body_fn body_fn, void *ctx) {
    pthread_once(&g_init_once, init_pool);
    const int N = g_state.n_threads > 0 ? g_state.n_threads : 1;
    if (N == 1 || n <= 1) {
        for (size_t i = 0; i < n; i++)
            body_fn(i, ctx);
        return;
    }
    /* Publish task. */
    g_state.n_total        = n;
    g_state.body_fn        = body_fn;
    g_state.ctx            = ctx;
    g_state.dynamic_chunks = 0;
    atomic_store_explicit(&g_state.pending_workers, N - 1, memory_order_relaxed);
    /* Bumping the epoch releases the task fields above to workers. */
    atomic_fetch_add_explicit(&g_state.task_epoch, 1, memory_order_release);
    /* Wake any sleeping workers. Skip the syscall when none are
     * sleeping (the always-busy hot path: workers are still spinning
     * and `sleeping == 0`). */
    if (atomic_load_explicit(&g_state.sleeping, memory_order_acquire) > 0) {
        pthread_mutex_lock(&g_state.wait_mutex);
        pthread_cond_broadcast(&g_state.wait_cv);
        pthread_mutex_unlock(&g_state.wait_mutex);
    }
    /* Master does chunk 0 inline. */
    do_chunk(0, N, n, body_fn, ctx);
    /* Wait for all workers. */
    while (atomic_load_explicit(&g_state.pending_workers, memory_order_acquire) != 0) {
        /* spin */
    }
}

static void do_dynamic_chunks(size_t n_total, size_t grain, geist_pp_body_fn fn, void *ctx) {
    if (grain == 0) {
        grain = 1;
    }
    for (;;) {
        const size_t lo =
                atomic_fetch_add_explicit(&g_state.next_index, grain, memory_order_relaxed);
        if (lo >= n_total) {
            return;
        }
        size_t hi = lo + grain;
        if (hi > n_total) {
            hi = n_total;
        }
        for (size_t i = lo; i < hi; i++) {
            fn(i, ctx);
        }
    }
}

void geist_pp_parallel_for_grain(size_t n, size_t grain, geist_pp_body_fn body_fn, void *ctx) {
    pthread_once(&g_init_once, init_pool);
    const int N = g_state.n_threads > 0 ? g_state.n_threads : 1;
    if (N == 1 || n <= 1) {
        for (size_t i = 0; i < n; i++)
            body_fn(i, ctx);
        return;
    }
    if (grain == 0) {
        grain = 1;
    }
    g_state.n_total        = n;
    g_state.body_fn        = body_fn;
    g_state.ctx            = ctx;
    g_state.chunk_size     = grain;
    g_state.dynamic_chunks = 1;
    atomic_store_explicit(&g_state.next_index, 0, memory_order_relaxed);
    atomic_store_explicit(&g_state.pending_workers, N - 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_state.task_epoch, 1, memory_order_release);
    if (atomic_load_explicit(&g_state.sleeping, memory_order_acquire) > 0) {
        pthread_mutex_lock(&g_state.wait_mutex);
        pthread_cond_broadcast(&g_state.wait_cv);
        pthread_mutex_unlock(&g_state.wait_mutex);
    }
    do_dynamic_chunks(n, grain, body_fn, ctx);
    while (atomic_load_explicit(&g_state.pending_workers, memory_order_acquire) != 0) {
        /* spin */
    }
}
