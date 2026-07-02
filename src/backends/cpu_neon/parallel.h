/*
 * src/backends/cpu_neon/parallel.h — minimal spin-pool parallel_for.
 *
 * Layer: BACKEND (cpu_neon, internal).
 *
 * Replaces `#pragma omp parallel for schedule(static)` in hot-path
 * matmul kernels with a custom pthread pool. The win comes from
 * lower per-call dispatch overhead:
 *
 *   OpenMP (libgomp / libomp) `parallel for` regions on a decode token:
 *     - 210 matmuls × ~30-50 μs per spawn-and-join even with active wait
 *     - Profile on Pi 5: 11% of decode time in __gomp_* (≈ 9 ms / 89 ms)
 *
 *   Custom pool: workers spin on an atomic epoch counter; the master
 *   sets task fields and bumps the epoch with one `atomic_fetch_add`.
 *   Each parallel_for is one atomic write + spin-wait for ack. ggml
 *   uses the same pattern and gets ~4% in their thread-coord slice.
 *
 * API contract:
 *   - One global pool, lazily initialized on first parallel_for.
 *   - Pool size from GEIST_THREADS env (default: omp_get_max_threads()
 *     if libomp present, else hw_concurrency capped at 8).
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

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_PARALLEL_H */
