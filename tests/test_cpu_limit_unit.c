/*
 * test_cpu_limit_unit — the CPU limit of a container or cpuset (#27).
 *
 * 1. cgroup v2 cpu.max text and cgroup v1 cfs values to a CPU count:
 *    unlimited, whole and fractional quotas, garbage.
 * 2. geist_cpu_limit() on this host: 0 (no limit) or at most the online
 *    CPUs, and geist_par_max_threads() never above a nonzero limit unless
 *    OMP_NUM_THREADS is set.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1 /* _SC_NPROCESSORS_ONLN under strict POSIX */

#include "cpu_limit.h"
#include "par.h"
#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    int fails = 0;
    static const struct {
        const char *text;
        size_t      want;
    } V2[] = {
            {"max 100000\n", 0},
            {"400000 100000\n", 4},
            {"150000 100000", 2},
            {"100000 100000", 1},
            {"50000 100000", 1},
            {"", 0},
            {"garbage", 0},
            {"400000", 0},
    };
    for (size_t i = 0; i < sizeof V2 / sizeof V2[0]; i++) {
        const size_t got = geist_cpu_limit_cgroup_v2(V2[i].text);
        if (got != V2[i].want) {
            fprintf(stderr, "FAIL: cpu.max \"%s\": %zu, want %zu\n", V2[i].text, got, V2[i].want);
            fails++;
        }
    }
    if (geist_cpu_limit_cgroup_v2(nullptr) != 0 || geist_cpu_limit_cfs(-1, 100000) != 0 ||
        geist_cpu_limit_cfs(250000, 100000) != 3 || geist_cpu_limit_cfs(100000, 0) != 0) {
        fprintf(stderr, "FAIL: cfs / null cases\n");
        fails++;
    }
    const size_t limit  = geist_cpu_limit();
    const long   online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0 && limit > (size_t) online) {
        fprintf(stderr, "FAIL: limit %zu above %ld online CPUs\n", limit, online);
        fails++;
    }
    const char *omp = getenv("OMP_NUM_THREADS");
    if (limit > 0 && (omp == nullptr || omp[0] == '\0') && geist_par_max_threads() > limit) {
        fprintf(stderr, "FAIL: team %zu above the limit %zu\n", geist_par_max_threads(), limit);
        fails++;
    }
    printf("%s: limit %zu, team %zu\n", fails ? "FAIL" : "PASS", limit, geist_par_max_threads());
    return fails ? GEIST_TEST_FAIL : 0;
}
