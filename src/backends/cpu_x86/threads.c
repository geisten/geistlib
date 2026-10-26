/*
 * src/backends/cpu_x86/threads.c — cpu_x86's thread count per phase.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * The ambient OpenMP team is one thread per logical CPU. Decode, an M=1
 * GEMV streaming weights from DRAM, gets contention rather than bandwidth
 * from SMT siblings (benchmark/results/THREAD-SWEEP-AMD9950X.md, #504).
 *
 * Policy:
 *   - decode: one thread per physical core when the probe sees SMT; only
 *     ever lowers the team;
 *   - prefill: also one thread per physical core with SMT. The tiled int8
 *     GEMM alone gains 20 % from the SMT siblings, but the rest of a
 *     prefill (DeltaNet, norms, attention, the barriers between them)
 *     loses more: 9950X pp512 at 32 threads against 16, after #18:
 *     Qwen3.5 4B -9 %, Qwen3.8 27B -4 %, Gemma 4 E2B -20 %. (The 2026-09
 *     sweep, before the tiled kernels, saw no difference.)
 *   - an explicit OMP_NUM_THREADS is the user's choice and stays;
 *   - GEIST_DECODE_THREADS / GEIST_PREFILL_THREADS override either phase
 *     (documented in docs/QUICKSTART.md for every CPU backend);
 *   - geist_backend_opts.max_threads caps both phases: an application that
 *     shares the machine (geist-runtime#93) gets at most that many threads,
 *     in every region.
 *
 * Decode is not pinned to one L3 domain: 13-15 % slower on the 9950X
 * (X86.md).
 *
 * The count is geist_par's (par.h): the OpenMP team with OpenMP, the pool's
 * per-thread count without.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "threads.h"

#include "backend_state.h"

#include "hw_probe.h"
#include "par.h"
#include "env.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>

int cpu_x86_phase_threads(int env, bool omp_threads_set, size_t logical, size_t physical) {
    if (env >= 0) {
        return env;
    }
    if (omp_threads_set || physical == 0 || physical >= logical || physical > (size_t) INT_MAX) {
        return 0;
    }
    return (int) physical;
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
        const char *omp     = getenv("OMP_NUM_THREADS");
        const bool  omp_set = omp != nullptr && omp[0] != '\0';
        atomic_store_explicit(&prefill,
                              cpu_x86_phase_threads(geist_env_count("GEIST_PREFILL_THREADS"),
                                                    omp_set,
                                                    hw.logical_cores,
                                                    hw.physical_cores),
                              memory_order_relaxed);
        atomic_store_explicit(&decode,
                              cpu_x86_phase_threads(geist_env_count("GEIST_DECODE_THREADS"),
                                                    omp_set,
                                                    hw.logical_cores,
                                                    hw.physical_cores),
                              memory_order_release);
    }
    return region == GEIST_REGION_DECODE_STEP
                   ? atomic_load_explicit(&decode, memory_order_relaxed)
                   : atomic_load_explicit(&prefill, memory_order_relaxed);
}

int cpu_x86_parallel_region_begin(struct geist_backend *be, enum geist_parallel_region region) {
    const struct cpu_x86_state *st     = be != nullptr ? be->state : nullptr;
    const int                   cap    = st != nullptr ? st->max_threads : 0;
    int                         target = region_threads(region);
    if (cap > 0 && (target <= 0 || target > cap)) {
        target = cap;
    }
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
