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
 * Every backend splits [0, n) the same way: into T = min(n, the thread
 * count) contiguous ranges, range t covering n / T items plus one more for
 * t < n % T (OpenMP's schedule(static) split). Which thread gets which range
 * is unspecified; a body whose ranges write disjoint outputs gives the same
 * bytes whatever T is.
 */
#ifndef GEIST_INTERNAL_PAR_H
#define GEIST_INTERNAL_PAR_H

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

#endif /* GEIST_INTERNAL_PAR_H */
