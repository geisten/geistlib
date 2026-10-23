/*
 * src/archs/transformer/moe_pool.c - helper threads for a decode step's
 * cold experts (#161). Contract: moe_pool.h.
 */
#define _POSIX_C_SOURCE 200809L /* clock_gettime, sched_yield */
#include "moe_pool.h"

#include "heap.h"
#include "omp_idle.h"
#include "par.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define cpu_relax() _mm_pause()
#elif defined(__aarch64__)
#define cpu_relax() __asm__ __volatile__("yield")
#else
#define cpu_relax() ((void) 0)
#endif

struct moe_pool {
    size_t          n_threads;
    pthread_t      *threads;
    pthread_mutex_t run_mu; /* one job at a time */
    pthread_mutex_t mu;     /* sleeping helpers */
    pthread_cond_t  cv;
    long long       spin_ns;
    _Atomic bool    stop;
    _Atomic int     sleepers;
    /* (generation << 32) | the next unclaimed task: a claim is a CAS on
     * it, so a helper of an earlier generation can never take a task. */
    _Atomic uint64_t claim;
    _Atomic size_t   n_tasks; /* of the current generation */
    _Atomic size_t   done;    /* tasks of it finished */
    uint64_t         gen;     /* the caller's, under run_mu */
    moe_pool_fn      fn;      /* read only after a successful claim */
    void            *ctx;
};

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Claims and runs tasks of generation `gen` until none is left. The job's
 * fields are read only after a claim succeeded: the job is then unfinished
 * (this task is not done), so its caller is still waiting and they hold. */
static void run_tasks(struct moe_pool *p, uint64_t gen) {
    for (;;) {
        uint64_t c = atomic_load_explicit(&p->claim, memory_order_acquire);
        if ((c >> 32) != gen ||
            (c & 0xffffffffu) >= atomic_load_explicit(&p->n_tasks, memory_order_relaxed))
            return;
        if (!atomic_compare_exchange_weak_explicit(
                    &p->claim, &c, c + 1, memory_order_acq_rel, memory_order_relaxed))
            continue;
        p->fn(p->ctx, (size_t) (c & 0xffffffffu));
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
}

static void *helper_main(void *arg) {
    struct moe_pool *p = arg;
    geist_par_set_inline(true); /* the kernels a task calls stay on this thread */
    uint64_t  seen = atomic_load_explicit(&p->claim, memory_order_acquire) >> 32;
    long long idle = now_ns();
    for (unsigned i = 1; !atomic_load_explicit(&p->stop, memory_order_relaxed); i++) {
        const uint64_t g = atomic_load_explicit(&p->claim, memory_order_acquire) >> 32;
        if (g != seen) {
            run_tasks(p, g);
            seen = g;
            idle = now_ns();
            continue;
        }
        cpu_relax();
        if ((i & 255u) != 0 || now_ns() - idle < p->spin_ns)
            continue;
        /* Idle past the bound: sleep until the next job. The seq_cst
         * increment before the re-check pairs with the caller's seq_cst
         * store of claim before it loads sleepers: one sees the other. */
        pthread_mutex_lock(&p->mu);
        atomic_fetch_add(&p->sleepers, 1);
        while ((atomic_load(&p->claim) >> 32) == seen && !atomic_load(&p->stop))
            pthread_cond_wait(&p->cv, &p->mu);
        atomic_fetch_sub(&p->sleepers, 1);
        pthread_mutex_unlock(&p->mu);
    }
    return nullptr;
}

struct moe_pool *moe_pool_create(size_t n_helpers) {
    if (n_helpers == 0)
        return nullptr;
    struct moe_pool *p = heap_calloc_aligned(1, sizeof *p, alignof(struct moe_pool));
    if (p == nullptr)
        return nullptr;
    p->threads = heap_calloc_array_aligned(pthread_t, n_helpers);
    if (p->threads == nullptr) {
        void *v = p;
        safe_free(&v);
        return nullptr;
    }
    pthread_mutex_init(&p->run_mu, nullptr);
    pthread_mutex_init(&p->mu, nullptr);
    pthread_cond_init(&p->cv, nullptr);
    p->spin_ns = (long long) geist_omp_idle_spin_ms(nullptr) * 1000000LL;
    atomic_init(&p->stop, false);
    atomic_init(&p->sleepers, 0);
    atomic_init(&p->claim, 0);
    atomic_init(&p->n_tasks, 0);
    atomic_init(&p->done, 0);
    for (size_t i = 0; i < n_helpers; i++) {
        if (pthread_create(&p->threads[i], nullptr, helper_main, p) != 0)
            break;
        p->n_threads++;
    }
    if (p->n_threads == 0) {
        moe_pool_destroy(p);
        return nullptr;
    }
    return p;
}

void moe_pool_destroy(struct moe_pool *p) {
    if (p == nullptr)
        return;
    pthread_mutex_lock(&p->mu);
    atomic_store(&p->stop, true);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (size_t i = 0; i < p->n_threads; i++)
        pthread_join(p->threads[i], nullptr);
    pthread_cond_destroy(&p->cv);
    pthread_mutex_destroy(&p->mu);
    pthread_mutex_destroy(&p->run_mu);
    void *t = p->threads, *v = p;
    safe_free(&t);
    safe_free(&v);
}

bool moe_pool_run(struct moe_pool *p, size_t n, moe_pool_fn fn, void *ctx) {
    if (p == nullptr || n > 0xffffffffu || pthread_mutex_trylock(&p->run_mu) != 0)
        return false;
    if (n == 0) {
        pthread_mutex_unlock(&p->run_mu);
        return true;
    }
    p->fn  = fn;
    p->ctx = ctx;
    atomic_store_explicit(&p->n_tasks, n, memory_order_relaxed);
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    p->gen = (p->gen + 1) & 0xffffffffu;
    atomic_store(&p->claim, p->gen << 32); /* publishes the job (seq_cst) */
    if (atomic_load(&p->sleepers) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
    geist_par_set_inline(true);
    run_tasks(p, p->gen);
    geist_par_set_inline(false);
    /* Only the tasks a helper claimed are left: wait for those. */
    for (unsigned spins = 0; atomic_load_explicit(&p->done, memory_order_acquire) < n;) {
        if (++spins < (1u << 14))
            cpu_relax();
        else
            sched_yield();
    }
    pthread_mutex_unlock(&p->run_mu);
    return true;
}
