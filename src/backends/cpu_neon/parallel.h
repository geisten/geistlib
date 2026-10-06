/*
 * src/backends/cpu_neon/parallel.h — minimal spin-pool parallel_for.
 *
 * Layer: BACKEND (cpu_neon, internal).
 *
 * Opt-in (GEIST_PP=1) alternative to `#pragma omp parallel for` for the
 * per-row decode kernels, with lower per-call dispatch cost: workers spin
 * on an atomic epoch counter, and the master publishes a task with one
 * `atomic_fetch_add`. OpenMP spawn-and-join costs ~30-50 μs per region,
 * ~11 % of decode time on a Pi 5.
 *
 * API contract:
 *   - One global pool, lazily initialized on first parallel_for.
 *   - Pool size from GEIST_THREADS, else OMP_NUM_THREADS, else the
 *     performance-core count (Apple) or online CPUs; capped at 16.
 *   - parallel_for splits `[0, n)` into `n_threads` contiguous chunks
 *     and dispatches one chunk per worker. parallel_for_grain uses a
 *     dynamic atomic chunk cursor, useful for kernels whose row cost is
 *     not uniform.
 *   - body_fn is called for each i ∈ [0, n) exactly once.
 *   - body_fn must be lock-free (workers run concurrently).
 *
 * Thread safety: NOT re-entrant. Nested geist_pp_parallel_for from
 * within a body_fn will deadlock. (No nesting in current callers.)
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_PARALLEL_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_PARALLEL_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_neon/parallel.h is internal to the backend layer."
#endif

#include <stdbool.h>
#include <stddef.h>

typedef void (*geist_pp_body_fn)(size_t i, void *ctx);

/* Run `body_fn(i, ctx)` for each `i` in `[0, n)`. Returns when all
 * iterations are complete. Master participates as worker 0; workers
 * 1..N-1 are spun up on first call. */
void geist_pp_parallel_for(size_t n, geist_pp_body_fn body_fn, void *ctx);

/* Dynamic-chunk variant. `grain` is the number of contiguous iterations
 * each worker claims at a time; 0 is treated as 1. This costs one
 * atomic fetch_add per chunk but gives better load balance for kernels
 * with uneven row/tile cost. */
void geist_pp_parallel_for_grain(size_t n, size_t grain, geist_pp_body_fn body_fn, void *ctx);

/* Whether GEIST_PP=1 routes the row kernels to this pool instead of
 * OpenMP. Read once, then cached. */
bool geist_pp_enabled(void);

/* Run `body_fn(i, ctx)` for each `i` in `[0, n)` the way the per-row
 * decode kernels dispatch: on this pool under GEIST_PP=1; else as an
 * `omp for` work-share when already inside a team (no team spawn); else
 * as an `omp parallel for`, serial without OpenMP. Static schedule in
 * every case. */
void cpu_neon_parallel_rows(size_t n, geist_pp_body_fn body_fn, void *ctx);

/* Number of threads the pool uses (>= 1). Cheap to call. */
size_t geist_pp_thread_count(void);

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_PARALLEL_H */
