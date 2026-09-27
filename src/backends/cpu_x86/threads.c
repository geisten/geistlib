/*
 * src/backends/cpu_x86/threads.c — cpu_x86's OpenMP team size per phase.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Without these hooks cpu_x86 ran every phase at the ambient OpenMP team,
 * which is one thread per LOGICAL CPU unless OMP_NUM_THREADS says
 * otherwise. On an SMT host that doubles the decode team, and decode — an
 * M=1 GEMV streaming the weights from DRAM — gets contention, not
 * bandwidth, from the sibling threads. Ryzen 9 9950X, 16 cores / 32
 * threads (benchmark/results/THREAD-SWEEP-AMD9950X.md): decode 49.5 t/s at
 * 16 threads, 45.9 at 32; prefill 493.7 vs 500.0, i.e. flat. #504 saw
 * decode at 1.41 s with the default 32 threads against 0.88 s at 16.
 *
 * So, as cpu_neon does for its phases:
 *   - decode: one thread per physical core when the probe sees SMT; only
 *     ever lowers the team;
 *   - prefill: the ambient team (compute-bound, the sweep shows no gain
 *     from fewer threads);
 *   - an explicit OMP_NUM_THREADS is the user's choice and stays;
 *   - GEIST_DECODE_THREADS / GEIST_PREFILL_THREADS override either phase
 *     (documented in docs/QUICKSTART.md for every CPU backend).
 *
 * Not done: pinning decode to one L3 domain. X86.md measured it 13-15 %
 * slower on the 9950X (half the cores, the same shared memory controller).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "threads.h"

#include "hw_probe.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

int cpu_x86_decode_threads(int env_decode, bool omp_threads_set, size_t logical, size_t physical) {
    if (env_decode >= 0) {
        return env_decode;
    }
    if (omp_threads_set || physical == 0 || physical >= logical || physical > (size_t) INT_MAX) {
        return 0;
    }
    return (int) physical;
}

#if defined(_OPENMP)
/* -1 when unset or empty; 0 when not a positive count; else the count. */
static int env_count(const char *name) {
    const char *e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return -1;
    }
    const int v = atoi(e);
    return v > 0 ? v : 0;
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
    const int prev = omp_get_max_threads();
    /* Prefill moves either way; decode only caps down — never adds
     * threads to a memory-bound GEMV. */
    const bool apply = region == GEIST_REGION_DECODE_STEP ? target < prev : target != prev;
    if (!apply) {
        return 0;
    }
    omp_set_num_threads(target);
    return prev; /* > 0: restore to this in _end */
}

void cpu_x86_parallel_region_end(struct geist_backend *be, int token) {
    (void) be;
    if (token > 0) {
        omp_set_num_threads(token);
    }
}
#endif /* _OPENMP */
