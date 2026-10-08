/*
 * src/backends/cpu_x86/threads.c — cpu_x86's thread count per phase.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * The ambient OpenMP team is one thread per logical CPU. Decode, an M=1
 * GEMV streaming weights from DRAM, gets contention rather than bandwidth
 * from SMT siblings (benchmark/results/THREAD-SWEEP-AMD9950X.md, #504).
 *
 * Policy, as in cpu_neon:
 *   - decode: one thread per physical core when the probe sees SMT; only
 *     ever lowers the team;
 *   - prefill: the ambient team (compute-bound, the sweep shows no gain
 *     from fewer threads);
 *   - an explicit OMP_NUM_THREADS is the user's choice and stays;
 *   - GEIST_DECODE_THREADS / GEIST_PREFILL_THREADS override either phase
 *     (documented in docs/QUICKSTART.md for every CPU backend).
 *
 * Decode is not pinned to one L3 domain: 13-15 % slower on the 9950X
 * (X86.md).
 *
 * The count is geist_par's (par.h): the OpenMP team with OpenMP, the pool's
 * per-thread count without.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "threads.h"

#include "hw_probe.h"
#include "par.h"
#include "parse.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>

int cpu_x86_decode_threads(int env_decode, bool omp_threads_set, size_t logical, size_t physical) {
    if (env_decode >= 0) {
        return env_decode;
    }
    if (omp_threads_set || physical == 0 || physical >= logical || physical > (size_t) INT_MAX) {
        return 0;
    }
    return (int) physical;
}

/* -1 when unset or empty; 0 when not a positive count; else the count. */
static int env_count(const char *name) {
    const char *e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return -1;
    }
    long v;
    return geist_parse_long(e, &v) && v > 0 && v <= INT_MAX ? (int) v : 0;
}

/* Team size for each phase, 0 = ambient. Computed once, on the first
 * region of the process (a prefill, in practice, whose run time hides the
 * sysfs reads of the topology probe): neither the environment nor the
 * topology changes under a running engine. Racing first calls compute the
 * same values, so plain atomic stores suffice. */
static int region_threads(enum geist_parallel_region region) {
    static _Atomic int prefill = INT_MIN;
    static _Atomic int decode  = INT_MIN;
    if (atomic_load_explicit(&decode, memory_order_acquire) == INT_MIN) {
        struct geist_hw_probe hw;
        geist_hw_probe_fill(&hw);
        const char *omp = getenv("OMP_NUM_THREADS");
        const int   p   = env_count("GEIST_PREFILL_THREADS");
        atomic_store_explicit(&prefill, p > 0 ? p : 0, memory_order_relaxed);
        atomic_store_explicit(&decode,
                              cpu_x86_decode_threads(env_count("GEIST_DECODE_THREADS"),
                                                     omp != nullptr && omp[0] != '\0',
                                                     hw.logical_cores,
                                                     hw.physical_cores),
                              memory_order_release);
    }
    return region == GEIST_REGION_DECODE_STEP
                   ? atomic_load_explicit(&decode, memory_order_relaxed)
                   : atomic_load_explicit(&prefill, memory_order_relaxed);
}

int cpu_x86_parallel_region_begin(struct geist_backend *be, enum geist_parallel_region region) {
    (void) be;
    const int target = region_threads(region);
    if (target <= 0) {
        return 0;
    }
    const size_t max  = geist_par_max_threads();
    const int    prev = max > (size_t) INT_MAX ? INT_MAX : (int) max;
    /* Prefill moves either way; decode only caps down — never adds
     * threads to a memory-bound GEMV. */
    const bool apply = region == GEIST_REGION_DECODE_STEP ? target < prev : target != prev;
    if (!apply) {
        return 0;
    }
    geist_par_set_max_threads((size_t) target);
    return prev; /* > 0: restore to this in _end */
}

void cpu_x86_parallel_region_end(struct geist_backend *be, int token) {
    (void) be;
    if (token > 0) {
        geist_par_set_max_threads((size_t) token);
    }
}
