/*
 * src/base/par.h — one parallel-for for the CPU kernels, with or without
 * OpenMP (#618).
 *
 * Layer: ENGINE.
 *
 * Without OpenMP every `#pragma omp parallel for` compiles to a serial loop,
 * and iOS, a mac without libomp and MSVC have no OpenMP. Kernels call
 * geist_par_for instead; this file picks the threads:
 *
 *   OpenMP                 one `omp parallel` region (what the pragma did)
 *   Apple without OpenMP   GCD dispatch_apply_f
 *   otherwise              a lazily created pthread pool; idle workers spin
 *                          for the idle bound of omp_idle.h, then sleep
 *
 * Loops whose items cost unequal time (short GEMV rows, dequant tiles,
 * attention heads) use the schedule(dynamic) helpers below, which hand out
 * chunks from an atomic counter.
 *
 * Every backend splits [0, n) the same way: into T = min(n, the thread
 * count) contiguous ranges, range t covering n / T items plus one more for
 * t < n % T (OpenMP's schedule(static) split). Which thread gets which range
 * is unspecified; a body whose ranges write disjoint outputs gives the same
 * bytes whatever T is.
 */
#ifndef GEIST_INTERNAL_PAR_H
#define GEIST_INTERNAL_PAR_H

#include <stdatomic.h>
#include <stddef.h>

/* Processes items [begin, end) of the loop; begin < end. */
typedef void (*geist_par_fn)(void *ctx, size_t begin, size_t end);

/* Runs `fn` over [0, n) on up to geist_par_max_threads() threads, the
 * calling thread one of them, and returns when every range is done.
 * n == 0 calls nothing. A call from inside a geist_par_for body (or an
 * OpenMP parallel region) runs fn(ctx, 0, n) on the calling thread.
 * Any thread may call it; concurrent callers of the pthread pool take
 * turns. */
void geist_par_for(size_t n, geist_par_fn fn, void *ctx);

/* The calling thread's thread count for geist_par_for (>= 1):
 * omp_get_max_threads() with OpenMP; else the last
 * geist_par_set_max_threads of this thread, defaulting to OMP_NUM_THREADS
 * if set, else the online CPUs. */
[[nodiscard]] size_t geist_par_max_threads(void);

/* Sets the calling thread's count (omp_set_num_threads with OpenMP).
 * n == 0 is ignored. */
void geist_par_set_max_threads(size_t n);

/* ---- Items of unequal cost: OpenMP's schedule(dynamic) ---------------- */

/* A body's scratch slot, below geist_par_max_threads(): each range of one
 * call claims the next index from `next`, which the caller zeroes before
 * the call. A call runs at most that many ranges, each once. */
static inline size_t geist_par_slot(atomic_size_t *next) {
    return atomic_fetch_add_explicit(next, 1, memory_order_relaxed);
}

/* Pull form, for a body that keeps its per-item loop: geist_par_for_pull
 * zeroes *next and runs fn on min(n_items, threads) ranges, ignoring the
 * range bounds; fn takes items one at a time, in order, with
 * `for (size_t i; geist_par_next(next, n_items, &i);)` until none is left.
 * A body whose items write disjoint outputs gives the same bytes as a
 * static split. */
static inline bool geist_par_next(atomic_size_t *next, size_t n_items, size_t *i) {
    *i = atomic_fetch_add_explicit(next, 1, memory_order_relaxed);
    return *i < n_items;
}

static inline void
geist_par_for_pull(size_t n_items, atomic_size_t *next, geist_par_fn fn, void *ctx) {
    const size_t threads = geist_par_max_threads();
    atomic_init(next, 0);
    geist_par_for(n_items < threads ? n_items : threads, fn, ctx);
}

/* Range form: runs fn over [0, n) in chunks of `grain` items (0 is taken
 * as 1) handed out in order, on up to geist_par_max_threads() threads;
 * returns when all are done. One chunk, one thread, or a call from inside
 * a geist_par_for body runs fn(ctx, 0, n) on the calling thread. */
struct geist_par_dyn {
    geist_par_fn  fn;
    void         *ctx;
    size_t        n, grain;
    atomic_size_t next; /* the first item of the next chunk */
};

static inline void geist_par_dyn_range(void *p, size_t, size_t) {
    struct geist_par_dyn *d     = p;
    const size_t          n     = d->n;
    const size_t          grain = d->grain;
    for (size_t lo; (lo = atomic_fetch_add_explicit(&d->next, grain, memory_order_relaxed)) < n;) {
        d->fn(d->ctx, lo, n - lo < grain ? n : lo + grain);
    }
}

static inline void geist_par_for_dynamic(size_t n, size_t grain, geist_par_fn fn, void *ctx) {
    if (n == 0) {
        return;
    }
    if (grain == 0) {
        grain = 1;
    }
    const size_t chunks  = n / grain + (n % grain != 0);
    const size_t threads = geist_par_max_threads();
    if (chunks <= 1 || threads <= 1) {
        fn(ctx, 0, n);
        return;
    }
    struct geist_par_dyn d = {.fn = fn, .ctx = ctx, .n = n, .grain = grain};
    atomic_init(&d.next, 0);
    geist_par_for(chunks < threads ? chunks : threads, geist_par_dyn_range, &d);
}

#endif /* GEIST_INTERNAL_PAR_H */
