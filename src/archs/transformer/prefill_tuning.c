#define _POSIX_C_SOURCE 200809L /* setenv */
/*
 * src/archs/transformer/prefill_tuning.c — per-model prefill knobs.
 *
 * Layer: ARCHITECTURE (transformer). See prefill_tuning.h for the model.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "prefill_tuning.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

constexpr size_t GIB = (size_t) 1 << 30;

static const char *sys_getenv(const char *name) {
    return getenv(name);
}

static long env_long(const char *(*getenv_fn)(const char *), const char *name, bool *set) {
    const char *v = getenv_fn(name);
    *set          = v != nullptr && v[0] != '\0';
    return *set ? strtol(v, nullptr, 10) : 0;
}

struct transformer_prefill_resolved
transformer_prefill_resolve(const char *family,
                            size_t      weight_bytes,
                            size_t      base_m_max,
                            size_t      cap_m_max,
                            const char *(*getenv_fn)(const char *) ) {
    if (getenv_fn == nullptr) {
        getenv_fn = sys_getenv;
    }
    struct transformer_prefill_resolved out = {
            .m_max = base_m_max, .prefill_blocktime_ms = -1, .m_max_from_env = false};
    /* qwen35 hybrids from 4 GiB up (Ternary-Bonsai-2-27B, Qwen3.5-27B):
     * 27B weights dequantized once per chunk are the prefill's cost, so
     * twice the chunk (256 is no better than 128). Everything else keeps
     * the platform default and the runtime's own spin policy — on a 0.8B,
     * blocktime 0 costs more decode than it gains prefill. */
    if (family != nullptr && strcmp(family, "qwen35") == 0 && weight_bytes >= 4 * GIB) {
        out.m_max                = base_m_max + 64;
        out.prefill_blocktime_ms = 0;
    }
    bool set;
    long v = env_long(getenv_fn, "GEIST_M_MAX", &set);
    if (set && v > 0) {
        out.m_max          = (size_t) v;
        out.m_max_from_env = true;
    }
    v = env_long(getenv_fn, "GEIST_PREFILL_BLOCKTIME_MS", &set);
    if (set && v >= -1) {
        out.prefill_blocktime_ms = (int) v;
    }
    if (cap_m_max > 0 && out.m_max > cap_m_max) {
        out.m_max = cap_m_max;
    }
    return out;
}

void transformer_prefill_apply_blocktime(const struct transformer_prefill_resolved *r) {
    if (r == nullptr || r->prefill_blocktime_ms < 0) {
        return;
    }
    char buf[16];
    snprintf(buf, sizeof buf, "%d", r->prefill_blocktime_ms);
    setenv("KMP_BLOCKTIME", buf, 0); /* overwrite=0: environment and first model win */
}
