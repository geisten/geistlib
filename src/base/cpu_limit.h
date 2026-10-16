/*
 * src/base/cpu_limit.h — the CPUs this process may run on (#27).
 *
 * Layer: ENGINE.
 *
 * sysconf(_SC_NPROCESSORS_ONLN) counts the host's online CPUs. Inside a
 * container with `docker run --cpus=4` (a cgroup CPU quota) or under
 * `taskset` / `--cpuset-cpus` (an affinity mask) that overbooks the
 * process: CFS throttles the cgroup every period and every barrier waits
 * on a throttled thread. geist_par caps its default team at this limit
 * (par.c); an explicit OMP_NUM_THREADS still decides.
 */
#ifndef GEIST_BASE_CPU_LIMIT_H
#define GEIST_BASE_CPU_LIMIT_H

#include <stddef.h>

/* The CPUs a cgroup v2 `cpu.max` grants ("<quota> <period>" or
 * "max <period>"), rounded up; 0 for no limit or unparseable text. */
[[nodiscard]] size_t geist_cpu_limit_cgroup_v2(const char *cpu_max);

/* The same for cgroup v1's cpu.cfs_quota_us / cpu.cfs_period_us; quota
 * <= 0 (-1 = unlimited) or period <= 0 gives 0. */
[[nodiscard]] size_t geist_cpu_limit_cfs(long quota_us, long period_us);

/* The smaller of the affinity mask's CPU count and the cgroup quota, read
 * once per process; 0 when neither limits (and on non-Linux hosts). */
[[nodiscard]] size_t geist_cpu_limit(void);

#endif /* GEIST_BASE_CPU_LIMIT_H */
