/*
 * test_prefill_tuning_unit — default + delta + override resolution of the
 * per-model prefill knobs (prefill_tuning.h), hermetic.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/prefill_tuning.h"
#include "test_helpers.h"

#include <stdint.h>
#include <string.h>

static const char *g_m_max, *g_bt;
static const char *fake_env(const char *n) {
    if (strcmp(n, "GEIST_M_MAX") == 0)
        return g_m_max;
    if (strcmp(n, "GEIST_PREFILL_BLOCKTIME_MS") == 0)
        return g_bt;
    return nullptr;
}

int main(void) {
    int                                 fails = 0;
    const size_t                        GIB   = (size_t) 1 << 30;
    struct transformer_prefill_resolved r;

    /* Bonsai class: qwen35 at 7.2 GiB gets default + 64, blocktime 0. */
    r = transformer_prefill_resolve("qwen35", 7 * GIB, 64, 512, fake_env);
    fails += geist_expect(r.m_max == 128 && r.prefill_blocktime_ms == 0 && !r.m_max_from_env,
                          "qwen35 27B: 64 + 64 = 128, blocktime 0");
    /* A small qwen35 is not in that row: default stays. */
    r = transformer_prefill_resolve("qwen35", 800u << 20, 64, 512, fake_env);
    fails += geist_expect(r.m_max == 64, "qwen35 0.8B: default 64");
    /* Unknown family: the "*" row, delta 0. */
    r = transformer_prefill_resolve("gemma4", 3 * GIB, 64, 512, fake_env);
    fails += geist_expect(r.m_max == 64 && r.prefill_blocktime_ms == -1, "gemma4: default");
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
    fails += geist_expect(r.prefill_blocktime_ms == -1, "-1 leaves the runtime's policy alone");

    if (fails == 0)
        printf("PASS: prefill tuning resolves default + delta + override\n");
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
