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
    /* The job's generation, its task count and the next unclaimed task in
     * one word (CLAIM_*): a claim is a CAS on it, so a helper can take a
     * task only of the job the word names, and only one below its count. A
     * count of its own, read beside the word, could already be the next
     * job's while the word still names this one: the helper took task n. */
    _Atomic uint64_t claim;
    _Atomic size_t   done; /* tasks of the current job finished */
    uint64_t         gen;  /* the caller's, under run_mu */
    /* The job: read only after a successful claim, when it is the claimed
     * generation's (that job is unfinished, so its caller has not moved
     * on); atomics so the read pairs with the caller's store directly. */
    _Atomic(moe_pool_fn) fn;
    _Atomic(void *)      ctx;
};

/* claim: generation in bits 63..40, task count 39..20, next task 19..0. */
constexpr uint64_t CLAIM_IDX_BITS = 20;
constexpr uint64_t CLAIM_MAX      = (1u << CLAIM_IDX_BITS) - 1;

static inline uint64_t claim_gen(uint64_t c) {
    return c >> (2 * CLAIM_IDX_BITS);
}

static inline uint64_t claim_n(uint64_t c) {
    return (c >> CLAIM_IDX_BITS) & CLAIM_MAX;
}

static inline uint64_t claim_idx(uint64_t c) {
    return c & CLAIM_MAX;
}

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
        if (claim_gen(c) != gen || claim_idx(c) >= claim_n(c))
            return;
        if (!atomic_compare_exchange_weak_explicit(
                    &p->claim, &c, c + 1, memory_order_acq_rel, memory_order_relaxed))
            continue;
        const moe_pool_fn fn = atomic_load_explicit(&p->fn, memory_order_acquire);
        fn(atomic_load_explicit(&p->ctx, memory_order_acquire), (size_t) claim_idx(c));
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
}

static void *helper_main(void *arg) {
    struct moe_pool *p = arg;
    geist_par_set_inline(true); /* the kernels a task calls stay on this thread */
    uint64_t  seen = claim_gen(atomic_load_explicit(&p->claim, memory_order_acquire));
    long long idle = now_ns();
    for (unsigned i = 1; !atomic_load_explicit(&p->stop, memory_order_relaxed); i++) {
        const uint64_t g = claim_gen(atomic_load_explicit(&p->claim, memory_order_acquire));
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
        while (claim_gen(atomic_load(&p->claim)) == seen && !atomic_load(&p->stop))
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
    atomic_init(&p->done, 0);
    atomic_init(&p->fn, nullptr);
    atomic_init(&p->ctx, nullptr);
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
    if (p == nullptr || n > CLAIM_MAX || pthread_mutex_trylock(&p->run_mu) != 0)
        return false;
    if (n == 0) {
        pthread_mutex_unlock(&p->run_mu);
        return true;
    }
    atomic_store_explicit(&p->fn, fn, memory_order_release);
    atomic_store_explicit(&p->ctx, ctx, memory_order_release);
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    p->gen = (p->gen + 1) & ((UINT64_C(1) << (64 - 2 * CLAIM_IDX_BITS)) - 1);
    /* publishes the job (seq_cst) */
    atomic_store(&p->claim, p->gen << (2 * CLAIM_IDX_BITS) | (uint64_t) n << CLAIM_IDX_BITS);
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
