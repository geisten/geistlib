#define _POSIX_C_SOURCE 200809L /* setenv */
/*
 * src/archs/transformer/prefill_tuning.c — per-model prefill knobs.
 *
 * Layer: ARCHITECTURE (transformer). See prefill_tuning.h for the model.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "prefill_tuning.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

constexpr size_t GIB = (size_t) 1 << 30;

/* The table. Deltas come from measurements named in the header; a model
 * class without a row gets default + 0. Keep specific rows first. */
static const struct transformer_prefill_tuning TABLE[] = {
        /* qwen35 hybrids from 4 GiB up (Ternary-Bonsai-2-27B, Qwen3.5-27B):
         * 27B weights dequantized once per chunk are the prefill's cost,
         * so twice the chunk; 256 measured no better than 128. */
        {.family               = "qwen35",
         .min_bytes            = 4 * GIB,
         .max_bytes            = SIZE_MAX,
         .m_max_delta          = 64,
         .prefill_blocktime_ms = 0},
        /* Everything else: the platform default and the runtime's own
         * spin policy — on a 0.8B, blocktime 0 costs 16 % decode for 8 %
         * prefill (M1 Max, quiet), so no blanket value yet. */
        {.family               = "*",
         .min_bytes            = 0,
         .max_bytes            = SIZE_MAX,
         .m_max_delta          = 0,
         .prefill_blocktime_ms = -1},
};

static const struct transformer_prefill_tuning *lookup(const char *family, size_t bytes) {
    for (size_t i = 0; i < sizeof TABLE / sizeof TABLE[0]; i++) {
        const struct transformer_prefill_tuning *r = &TABLE[i];
        const bool fam_ok = strcmp(r->family, "*") == 0 ||
                            (family != nullptr && strcmp(r->family, family) == 0);
        if (fam_ok && bytes >= r->min_bytes && bytes < r->max_bytes) {
            return r;
        }
    }
    return nullptr;
}

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
    const struct transformer_prefill_tuning *row = lookup(family, weight_bytes);
    if (row != nullptr) {
        const long m = (long) base_m_max + row->m_max_delta;
        out.m_max    = m < 8 ? 8 : (size_t) m;
        if (row->prefill_blocktime_ms >= 0) {
            out.prefill_blocktime_ms = row->prefill_blocktime_ms;
        }
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
