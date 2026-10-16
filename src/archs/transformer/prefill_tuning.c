/*
 * src/archs/transformer/prefill_tuning.c — per-model prefill knobs.
 *
 * Layer: ARCHITECTURE (transformer). See prefill_tuning.h for the model.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "prefill_tuning.h"

#include "omp_idle.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static const char *sys_getenv(const char *name) {
    return getenv(name);
}

static long env_long(const char *(*getenv_fn)(const char *), const char *name, bool *set) {
    const char *v = getenv_fn(name);
    *set          = v != nullptr && v[0] != '\0';
    return *set ? strtol(v, nullptr, 10) : 0;
}

struct transformer_prefill_resolved
transformer_prefill_resolve(const struct transformer_prefill_delta *delta,
                            size_t                                  weight_bytes,
                            size_t                                  base_m_max,
                            size_t                                  cap_m_max,
                            const char *(*getenv_fn)(const char *) ) {
    if (getenv_fn == nullptr) {
        getenv_fn = sys_getenv;
    }
    struct transformer_prefill_resolved out = {.m_max = base_m_max,
                                               .prefill_blocktime_ms =
                                                       geist_omp_idle_spin_ms(getenv_fn),
                                               .m_max_from_env = false};
    /* The family's delta (families/<name>.c; today only qwen35's). Blocktime
     * stays at the idle bound for every model: 0 on the qwen35 27B measured
     * prefill -9.6 % but decode +18.3 % (#651), and on a 0.8B it costs more
     * decode than it gains prefill. */
    if (delta != nullptr && delta->m_max_rows > 0 && weight_bytes >= delta->from_bytes) {
        out.m_max = base_m_max + delta->m_max_rows;
    }
    bool set;
    long v = env_long(getenv_fn, "GEIST_M_MAX", &set);
    if (set && v > 0) {
        out.m_max          = (size_t) v;
        out.m_max_from_env = true;
    }
    v = env_long(getenv_fn, "GEIST_PREFILL_BLOCKTIME_MS", &set);
    if (set && v >= 0 && v <= 1000000) {
        out.prefill_blocktime_ms = (int) v;
    }
    if (cap_m_max > 0 && out.m_max > cap_m_max) {
        out.m_max = cap_m_max;
    }
    return out;
}

void transformer_prefill_apply_blocktime(const struct transformer_prefill_resolved *r) {
    if (r != nullptr) {
        geist_omp_blocktime_apply(r->prefill_blocktime_ms);
    }
}
