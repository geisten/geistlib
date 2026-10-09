/*
 * src/backends/cpu_neon/parallel.h — OpenMP's schedule(dynamic, grain) on
 * geist_par_for (#618).
 *
 * Layer: BACKEND (cpu_neon, internal).
 *
 * The cpu_neon kernels run their loops on geist_par_for (src/base/par.h),
 * which splits [0, n) into schedule(static) ranges. Loops whose items cost
 * unequal time (short GEMV rows, dequant tiles, attention heads) used
 * schedule(dynamic, grain); cpu_neon_par_for_dynamic keeps that: it runs
 * one range per thread, and each takes `grain` items at a time, in order,
 * from a shared atomic counter until none is left, as attention_driver.h
 * does for cpu_x86. A body whose items write disjoint outputs gives the
 * same bytes either way.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_PARALLEL_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_PARALLEL_H

#include "par.h"

#include <stdatomic.h>
#include <stddef.h>

/* One cpu_neon_par_for_dynamic call. */
struct cpu_neon_par_dyn {
    geist_par_fn  fn;
    void         *ctx;
    size_t        n, grain;
    atomic_size_t next; /* the first item of the next chunk */
};

static inline void cpu_neon_par_dyn_range(void *p, size_t, size_t) {
    struct cpu_neon_par_dyn *d     = p;
    const size_t             n     = d->n;
    const size_t             grain = d->grain;
    for (size_t lo; (lo = atomic_fetch_add_explicit(&d->next, grain, memory_order_relaxed)) < n;) {
        d->fn(d->ctx, lo, n - lo < grain ? n : lo + grain);
    }
}

/* Runs fn over [0, n) in chunks of `grain` items (0 is taken as 1) handed
 * out in order, on up to geist_par_max_threads() threads; returns when all
 * are done. One chunk, one thread, or a call from inside a geist_par_for
 * body runs fn(ctx, 0, n) on the calling thread. */
static inline void cpu_neon_par_for_dynamic(size_t n, size_t grain, geist_par_fn fn, void *ctx) {
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
    struct cpu_neon_par_dyn d = {.fn = fn, .ctx = ctx, .n = n, .grain = grain};
    atomic_init(&d.next, 0);
    geist_par_for(chunks < threads ? chunks : threads, cpu_neon_par_dyn_range, &d);
}

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_PARALLEL_H */
