/*
 * test_x86_thread_policy_unit — cpu_x86's thread count per phase.
 *
 * 1. The pure decode policy (cpu_x86_decode_threads) over the topologies it
 *    has to get right: an SMT desktop (16 cores / 32 threads), a 4C/8T
 *    handheld, 4-way SMT, no SMT (typical CI VMs), an unknown topology, an
 *    explicit OMP_NUM_THREADS, and the GEIST_DECODE_THREADS override.
 * 2. The hooks as the arch layer calls them, through the backend's vtable:
 *    GEIST_DECODE_THREADS / GEIST_PREFILL_THREADS are set before the first
 *    region (they are read once per process), then the count inside each
 *    region (and with OpenMP the team) and its restoration afterwards are
 *    checked — decode only ever lowers it, prefill moves it either way.
 *    Without OpenMP the hooks size geist_par_for's pool (#618).
 * 3. caps.manages_host_threads mirrors the hooks' presence.
 */
#define _POSIX_C_SOURCE 200809L /* setenv */

#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>

#if !defined(GEIST_BACKEND_CPU_X86)
int main(void) {
    printf("SKIP: needs cpu_x86 in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#define GEIST_INTERNAL_BACKEND_LAYER
#include "par.h"
#include "src/backends/cpu_x86/threads.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

static int check_policy(void) {
    static const struct {
        const char *what;
        int         env;
        bool        omp;
        size_t      logical;
        size_t      physical;
        int         want;
    } CASES[] = {
            {"SMT desktop, 16C/32T", -1, false, 32, 16, 16},
            {"SMT handheld, 4C/8T", -1, false, 8, 4, 4},
            {"4-way SMT, 16C/64T", -1, false, 64, 16, 16},
            {"no SMT, 4C/4T", -1, false, 4, 4, 0},
            {"physical cores unknown", -1, false, 32, 0, 0},
            {"logical cores unknown", -1, false, 0, 16, 0},
            {"OMP_NUM_THREADS set", -1, true, 32, 16, 0},
            {"GEIST_DECODE_THREADS=6", 6, false, 32, 16, 6},
            {"GEIST_DECODE_THREADS=6 and OMP_NUM_THREADS", 6, true, 32, 16, 6},
            {"GEIST_DECODE_THREADS=0 (leave the team)", 0, false, 32, 16, 0},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) {
        const int got = cpu_x86_decode_threads(
                CASES[i].env, CASES[i].omp, CASES[i].logical, CASES[i].physical);
        if (got != CASES[i].want) {
            fprintf(stderr,
                    "FAIL: %s: decode team %d, want %d\n",
                    CASES[i].what,
                    got,
                    CASES[i].want);
            fails++;
        }
    }
    return fails;
}

#if defined(_OPENMP)
/* The team a parallel region actually gets. */
static int team_size(void) {
    int n = 0;
#pragma omp parallel
    {
#pragma omp single
        n = omp_get_num_threads();
    }
    return n;
}
#endif

/* Enter `region` with `ambient` threads configured; expect `inside` inside
 * and `ambient` again after. */
static int check_region(const struct geist_backend_vtbl *v,
                        struct geist_backend            *be,
                        enum geist_parallel_region       region,
                        const char                      *name,
                        int                              ambient,
                        int                              inside) {
    geist_par_set_max_threads((size_t) ambient);
    const int tok = v->parallel_region_begin(be, region);
    const int got = (int) geist_par_max_threads();
#if defined(_OPENMP)
    const int team = team_size();
#else
    const int team = got;
#endif
    v->parallel_region_end(be, tok);
    const int after = (int) geist_par_max_threads();
    if (got != inside || team != inside || after != ambient) {
        fprintf(stderr,
                "FAIL: %s from %d threads: %d configured / %d in the team (want %d), %d after "
                "(want %d)\n",
                name,
                ambient,
                got,
                team,
                inside,
                after,
                ambient);
        return 1;
    }
    return 0;
}

static int check_hooks(void) {
    setenv("GEIST_DECODE_THREADS", "2", 1);
    setenv("GEIST_PREFILL_THREADS", "3", 1);
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_x86", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_x86 backend did not register on this host\n");
        return -1;
    }
    const struct geist_backend_vtbl *v     = be->desc->vtbl;
    const bool                       hooks = v->parallel_region_begin != nullptr;
    int                              fails = 0;
    if (hooks != (v->parallel_region_end != nullptr) ||
        hooks != be->desc->caps.manages_host_threads) {
        fprintf(stderr, "FAIL: region hooks and caps.manages_host_threads disagree\n");
        fails++;
    }
    if (!hooks) {
        fprintf(stderr, "FAIL: cpu_x86 without region hooks\n");
        fails++;
    } else {
        fails += check_region(v, be, GEIST_REGION_DECODE_STEP, "decode", 4, 2);
        fails += check_region(v, be, GEIST_REGION_DECODE_STEP, "decode", 1, 1); /* never raises */
        fails += check_region(v, be, GEIST_REGION_PREFILL_BATCH, "prefill", 4, 3);
        fails += check_region(v, be, GEIST_REGION_PREFILL_BATCH, "prefill", 2, 3);
    }
    geist_backend_destroy(be);
    return fails;
}

int main(void) {
    int       fails = check_policy();
    const int hooks = check_hooks();
    if (hooks < 0) {
        return fails != 0 ? GEIST_TEST_FAIL : GEIST_TEST_SKIP;
    }
    fails += hooks;
    if (fails != 0) {
        fprintf(stderr, "FAIL: %d check(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: decode team policy and region hooks\n");
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_X86 */
