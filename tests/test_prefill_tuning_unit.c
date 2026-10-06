/*
 * test_prefill_tuning_unit — default + delta + override resolution of the
 * per-model prefill knobs (prefill_tuning.h), and how the blocktime lands
 * in KMP_BLOCKTIME (omp_idle.h): idle default, model value, explicit
 * environment.
 */
#define _POSIX_C_SOURCE 200809L /* setenv, unsetenv */
#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/prefill_tuning.h"
#include "src/base/omp_idle.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const char *g_m_max, *g_bt, *g_idle;
static const char *fake_env(const char *n) {
    if (strcmp(n, "GEIST_M_MAX") == 0)
        return g_m_max;
    if (strcmp(n, "GEIST_PREFILL_BLOCKTIME_MS") == 0)
        return g_bt;
    if (strcmp(n, "GEIST_IDLE_SPIN_MS") == 0)
        return g_idle;
    return nullptr;
}

static bool blocktime_is(const char *want) {
    const char *v = getenv("KMP_BLOCKTIME");
    return want == nullptr ? v == nullptr : (v != nullptr && strcmp(v, want) == 0);
}

int main(void) {
    int                                 fails = 0;
    const size_t                        GIB   = (size_t) 1 << 30;
    struct transformer_prefill_resolved r;

    /* Bonsai class: qwen35 at 7.2 GiB gets default + 64, the idle bound. */
    r = transformer_prefill_resolve("qwen35", 7 * GIB, 64, 512, fake_env);
    fails += geist_expect(r.m_max == 128 && r.prefill_blocktime_ms == GEIST_IDLE_SPIN_MS &&
                                  !r.m_max_from_env,
                          "qwen35 27B: 64 + 64 = 128, idle bound");
    /* A small qwen35 is not in that row: default stays. */
    r = transformer_prefill_resolve("qwen35", 800u << 20, 64, 512, fake_env);
    fails += geist_expect(r.m_max == 64, "qwen35 0.8B: default 64");
    /* Unknown family: no delta, the idle bound. */
    r = transformer_prefill_resolve("gemma4", 3 * GIB, 64, 512, fake_env);
    fails += geist_expect(r.m_max == 64 && r.prefill_blocktime_ms == GEIST_IDLE_SPIN_MS,
                          "gemma4: default chunk, idle bound");
    g_idle = "50";
    r      = transformer_prefill_resolve("gemma4", 3 * GIB, 64, 512, fake_env);
    fails += geist_expect(r.prefill_blocktime_ms == 50, "GEIST_IDLE_SPIN_MS sets the idle bound");
    g_idle = nullptr;
    /* The backend cap bounds the delta. */
    r = transformer_prefill_resolve("qwen35", 7 * GIB, 64, 96, fake_env);
    fails += geist_expect(r.m_max == 96, "delta clamped to caps.max_m");
    /* Overrides win over the table, absolutely. */
    g_m_max = "48";
    g_bt    = "200";
    r       = transformer_prefill_resolve("qwen35", 7 * GIB, 64, 512, fake_env);
    fails += geist_expect(r.m_max == 48 && r.m_max_from_env && r.prefill_blocktime_ms == 200,
                          "GEIST_M_MAX / GEIST_PREFILL_BLOCKTIME_MS override");
    g_m_max = nullptr;
    g_bt    = "-1";
    r       = transformer_prefill_resolve("llama", 1 * GIB, 64, 0, fake_env);
    fails += geist_expect(r.prefill_blocktime_ms == GEIST_IDLE_SPIN_MS,
                          "a negative override is ignored: the idle bound stays");
    g_bt = nullptr;

    /* KMP_BLOCKTIME. An explicit value survives the idle default and any
     * model; one set here is replaced by the first model, and only it. */
    setenv("KMP_BLOCKTIME", "77", 1);
    geist_omp_idle_default();
    geist_omp_blocktime_apply(0);
    fails += geist_expect(blocktime_is("77"), "an explicit KMP_BLOCKTIME is never overwritten");
    unsetenv("KMP_BLOCKTIME");
    geist_omp_idle_default();
    fails += geist_expect(blocktime_is("200"), "idle default: KMP_BLOCKTIME=200");
    r = (struct transformer_prefill_resolved) {.m_max = 64, .prefill_blocktime_ms = 0};
    transformer_prefill_apply_blocktime(&r);
    fails += geist_expect(blocktime_is("0"), "the first model's value replaces the default");
    geist_omp_blocktime_apply(300);
    fails += geist_expect(blocktime_is("0"), "a second model changes nothing");

    if (fails == 0)
        printf("PASS: prefill tuning resolves default + delta + override; blocktime lands as "
               "specified\n");
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
