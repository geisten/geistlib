/*
 * src/backends/common/cpu_threads.h — the part of the CPU backends'
 * parallel-region hooks that does not depend on their thread policy.
 *
 * Layer: BACKEND (shared by cpu_neon and cpu_x86). Each backend decides
 * how many threads a phase wants; this applies the decision through
 * geist_par_set_max_threads (par.h: the OpenMP team with OpenMP, the
 * pool's count without) and restores the previous count afterwards.
 */
#pragma once

#include <geist_backend.h>

/* parallel_region_begin's body once the backend has its `target` for
 * `region` (0 or less: leave the team alone). `cap` is
 * geist_backend_opts.max_threads (0: none): an application sharing the
 * machine (geist-runtime#93) gets at most that many threads in every
 * region, the ambient one included. Prefill moves the count either way;
 * decode only lowers it, never adding threads to a memory-bound GEMV.
 * Returns the count to restore in _end, or 0 when nothing changed. */
[[nodiscard]] int geist_cpu_region_begin(enum geist_parallel_region region, int target, int cap);

/* The vtable's parallel_region_end for a backend whose _begin returns
 * geist_cpu_region_begin's token. */
void geist_cpu_region_end(struct geist_backend *be, int token);
