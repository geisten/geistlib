/*
 * test_hw_core_split_unit — Apple perf-level topology to a P/E split (#513).
 *
 * 1. M1-M4 layout (Performance + Efficiency): P = perflevel0, as before.
 * 2. M6 layout (Super + Performance + Efficiency): P = Super + Performance,
 *    not perflevel0 alone (the 2-thread pool of #513).
 * 3. Efficiency found by name, not index; one level only; missing names
 *    fall back to perflevel0 = P, perflevel1 = E.
 * 4. geist_hw_perf_cores on this host: positive on Apple silicon, 0 off Apple.
 */
#include "hw_probe.h"
#include "test_helpers.h"

#include <stdio.h>

int main(void) {
    int fails = 0;
    static const struct {
        const char               *what;
        size_t                    n;
        struct geist_hw_perflevel l[3];
        int                       perf, eff;
    } C[] = {
            {"M1 Max", 2, {{"Performance", 8}, {"Efficiency", 2}}, 8, 2},
            {"M4", 2, {{"Performance", 4}, {"Efficiency", 6}}, 4, 6},
            {"M6", 3, {{"Super", 2}, {"Performance", 4}, {"Efficiency", 6}}, 6, 6},
            {"E first", 2, {{"Efficiency", 4}, {"Performance", 4}}, 4, 4},
            {"one level", 1, {{"Performance", 8}}, 8, 0},
            {"unnamed", 3, {{"", 2}, {"", 4}, {"", 6}}, 2, 10},
            {"negative count", 2, {{"Performance", -1}, {"Efficiency", 2}}, 0, 2},
    };
    for (size_t i = 0; i < sizeof C / sizeof C[0]; i++) {
        const struct geist_hw_core_split got = geist_hw_core_split(C[i].n, C[i].l);
        if (got.perf != C[i].perf || got.eff != C[i].eff) {
            fprintf(stderr,
                    "FAIL: %s: %dP+%dE, want %dP+%dE\n",
                    C[i].what,
                    got.perf,
                    got.eff,
                    C[i].perf,
                    C[i].eff);
            fails++;
        }
    }
    const int pc = geist_hw_perf_cores();
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    if (pc <= 0) {
        fprintf(stderr, "FAIL: %d performance cores on Apple silicon\n", pc);
        fails++;
    }
#elif !defined(__APPLE__)
    if (pc != 0) {
        fprintf(stderr, "FAIL: %d performance cores off Apple, want 0\n", pc);
        fails++;
    }
#endif
    printf("%s: %d performance cores on this host\n", fails ? "FAIL" : "PASS", pc);
    return fails ? GEIST_TEST_FAIL : 0;
}
