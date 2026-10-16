#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* sched_getaffinity, CPU_COUNT */
#endif
/*
 * src/base/cpu_limit.c — see cpu_limit.h.
 *
 * Layer: ENGINE.
 */
#include "cpu_limit.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sched.h>
#endif

static size_t ceil_div(long a, long b) {
    return (size_t) ((a + b - 1) / b);
}

size_t geist_cpu_limit_cfs(long quota_us, long period_us) {
    if (quota_us <= 0 || period_us <= 0) {
        return 0;
    }
    return ceil_div(quota_us, period_us);
}

size_t geist_cpu_limit_cgroup_v2(const char *cpu_max) {
    if (cpu_max == nullptr || strncmp(cpu_max, "max", 3) == 0) {
        return 0;
    }
    char      *end    = nullptr;
    const long quota  = strtol(cpu_max, &end, 10);
    char      *end2   = nullptr;
    const long period = end != cpu_max ? strtol(end, &end2, 10) : 0;
    if (end == cpu_max || end2 == end) {
        return 0;
    }
    return geist_cpu_limit_cfs(quota, period);
}

#if defined(__linux__)
/* The first line of a small file, or false. */
static bool read_line(const char *path, char *buf, size_t n) {
    FILE *f = fopen(path, "r");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fgets(buf, (int) n, f) != nullptr;
    fclose(f);
    return ok;
}

static long read_long(const char *path) {
    char buf[64];
    return read_line(path, buf, sizeof buf) ? strtol(buf, nullptr, 10) : 0;
}

/* cgroup v2: this process's group from /proc/self/cgroup ("0::<path>"),
 * its cpu.max, else the root's (a container usually sees its own group as
 * the root). v1: the cfs files of the cpu controller. */
static size_t quota_limit(void) {
    char  line[512];
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (f != nullptr) {
        while (fgets(line, sizeof line, f) != nullptr) {
            if (strncmp(line, "0::", 3) == 0) {
                line[strcspn(line, "\n")] = '\0';
                char path[600];
                char max[64];
                snprintf(path, sizeof path, "/sys/fs/cgroup%s/cpu.max", line + 3);
                if (read_line(path, max, sizeof max)) {
                    fclose(f);
                    return geist_cpu_limit_cgroup_v2(max);
                }
            }
        }
        fclose(f);
    }
    char max[64];
    if (read_line("/sys/fs/cgroup/cpu.max", max, sizeof max)) {
        return geist_cpu_limit_cgroup_v2(max);
    }
    const long q = read_long("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
    const long p = read_long("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
    return geist_cpu_limit_cfs(q, p);
}

static size_t affinity_limit(void) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) {
        return 0;
    }
    const int n = CPU_COUNT(&set);
    return n > 0 ? (size_t) n : 0;
}
#endif

size_t geist_cpu_limit(void) {
    static _Atomic long cached = -1;
    long                v      = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        size_t n = 0;
#if defined(__linux__)
        const size_t a = affinity_limit();
        const size_t q = quota_limit();
        n              = a == 0 ? q : (q == 0 || a < q ? a : q);
#endif
        v = (long) n;
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return (size_t) v;
}
