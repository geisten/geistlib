/*
 * test_build_isa_unit — geist_hw_build_isa_missing names each instruction-
 * set feature the build's target flags let the compiler use, once a host
 * lacks it, and nothing on a host that has them all. geist_backend_create
 * refuses such a host with that name instead of letting it die with SIGILL
 * in a decode: mk/target-linux.mk builds aarch64 for ARMv8.2 with dotprod
 * and fp16, which a Cortex-A53 or A72 does not have.
 *
 * The misses are this host's probe with every feature the build uses set
 * and then one cleared at a time. On this host itself, backend creation
 * has to agree with the check: it succeeds where nothing is missing and
 * refuses with the missing feature's name where something is (a build for
 * ARMv8.2 run on an emulated Cortex-A72).
 */
#include "test_helpers.h"

#include "hw_probe.h"

#include <geist.h>

#include <stddef.h>
#include <stdio.h>
#include <string.h>

static const struct {
    size_t      bit; /* offset of the probe's bool */
    const char *name;
} USED[] = {
#if defined(__ARM_FEATURE_DOTPROD)
        {offsetof(struct geist_hw_probe, has_dotprod), "dotprod"},
#endif
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        {offsetof(struct geist_hw_probe, has_fp16), "fp16"},
#endif
#if defined(__x86_64__) && defined(__AVX2__)
        {offsetof(struct geist_hw_probe, has_avx2), "AVX2"},
#endif
#if defined(__x86_64__) && defined(__FMA__)
        {offsetof(struct geist_hw_probe, has_fma), "FMA"},
#endif
#if defined(__x86_64__) && defined(__BMI2__)
        {offsetof(struct geist_hw_probe, has_bmi2), "BMI2"},
#endif
#if defined(__x86_64__) && defined(__F16C__)
        {offsetof(struct geist_hw_probe, has_fp16), "F16C"},
#endif
        {0, nullptr},
};

int main(void) {
    struct geist_hw_probe host;
    geist_hw_probe_fill(&host);
    int fails = 0;

    /* One feature missing at a time, from a host that has all of them. */
    struct geist_hw_probe full = host;
    for (size_t i = 0; USED[i].name != nullptr; i++) {
        *(bool *) ((char *) &full + USED[i].bit) = true;
    }
    fails += geist_expect(geist_hw_build_isa_missing(&full) == nullptr,
                          "a host with every feature this build uses is accepted");
    for (size_t i = 0; USED[i].name != nullptr; i++) {
        struct geist_hw_probe hw               = full;
        *(bool *) ((char *) &hw + USED[i].bit) = false;
        const char *miss                       = geist_hw_build_isa_missing(&hw);
        char        what[96];
        snprintf(what, sizeof what, "a host without %s is refused, naming it", USED[i].name);
        fails += geist_expect(miss != nullptr && strstr(miss, USED[i].name) != nullptr, what);
        printf("  without %-8s -> %s\n", USED[i].name, miss != nullptr ? miss : "(accepted)");
    }

    /* This host: backend creation agrees with the check. */
    const char             *lacks = geist_hw_build_isa_missing(&host);
    struct geist_backend   *be    = nullptr;
    const enum geist_status cs    = geist_backend_create("auto", nullptr, nullptr, &be);
    if (lacks == nullptr) {
        fails += geist_expect(cs == GEIST_OK, "a backend is created on this host");
    } else {
        printf("  this host lacks %s\n", lacks);
        fails += geist_expect(cs == GEIST_E_UNSUPPORTED &&
                                      strstr(geist_last_create_error(), lacks) != nullptr,
                              "backend creation refuses this host, naming what it lacks");
    }
    geist_backend_destroy(be);
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: a host is refused exactly when it lacks a feature this build uses\n");
    return GEIST_TEST_PASS;
}
