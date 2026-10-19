/*
 * test_mem_available_unit — MemAvailable from /proc/meminfo (#78: the
 * offload default needs room for the twins' host copy).
 *
 * 1. geist_meminfo_available on meminfo text: the line found among others,
 *    missing, malformed, a unit other than kB.
 * 2. geist_mem_available on this host: 0 (unknown) off Linux, nonzero on it.
 */
#include "hw_probe.h"
#include "test_helpers.h"

#include <stdio.h>

int main(void) {
    int fails = 0;
    static const struct {
        const char *text;
        size_t      want;
    } M[] = {
            {"MemTotal:       61640000 kB\nMemFree:  1000 kB\nMemAvailable:   56628224 kB\n",
             (size_t) 56628224 * 1024},
            {"MemAvailable: 1 kB", 1024},
            {"MemTotal: 5 kB\n", 0},
            {"MemAvailable: lots\n", 0},
            {"MemAvailable: 7 MB\n", 0},
            {"", 0},
    };
    for (size_t i = 0; i < sizeof M / sizeof M[0]; i++) {
        const size_t got = geist_meminfo_available(M[i].text);
        if (got != M[i].want) {
            fprintf(stderr, "FAIL: case %zu: %zu, want %zu\n", i, got, M[i].want);
            fails++;
        }
    }
    if (geist_meminfo_available(nullptr) != 0) {
        fprintf(stderr, "FAIL: null text\n");
        fails++;
    }
    const size_t avail = geist_mem_available();
#if defined(__linux__)
    if (avail == 0) {
        fprintf(stderr, "FAIL: no MemAvailable on Linux\n");
        fails++;
    }
#else
    if (avail != 0) {
        fprintf(stderr, "FAIL: %zu bytes off Linux, want 0 (unknown)\n", avail);
        fails++;
    }
#endif
    printf("%s: available %zu MiB\n", fails ? "FAIL" : "PASS", avail >> 20);
    return fails ? GEIST_TEST_FAIL : 0;
}
