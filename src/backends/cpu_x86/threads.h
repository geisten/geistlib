/*
 * src/backends/cpu_x86/threads.h — cpu_x86's OpenMP team size per phase.
 *
 * Layer: BACKEND (cpu_x86, internal). See threads.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_THREADS_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_THREADS_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/threads.h is internal to the backend layer."
#endif

#include <geist_backend.h>

#include <stddef.h>

/* Decode team size, or 0 to keep the ambient one. Pure — every input is an
 * argument — so the policy is unit-tested without the host it describes.
 *
 *   env_decode       GEIST_DECODE_THREADS: -1 unset, 0 "leave the team
 *                    alone", n > 0 a size (the hook only ever lowers it)
 *   omp_threads_set  OMP_NUM_THREADS is set: the user sized the team
 *   logical,
 *   physical         hw_probe's counts; physical 0 = unknown
 *
 * With neither variable set: one thread per physical core when the probe
 * sees SMT (physical < logical). Decode streams the weights from DRAM, and
 * an SMT sibling adds contention, not bandwidth. */
[[nodiscard]] int
cpu_x86_decode_threads(int env_decode, bool omp_threads_set, size_t logical, size_t physical);

/* The vtable's parallel_region_begin / _end. Defined in OpenMP builds only;
 * backend.c installs them there and leaves both slots null otherwise. */
int  cpu_x86_parallel_region_begin(struct geist_backend *be, enum geist_parallel_region region);
void cpu_x86_parallel_region_end(struct geist_backend *be, int token);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_THREADS_H */
