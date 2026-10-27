/*
 * src/archs/transformer/forward/profile.h - private forward profiler.
 */
#pragma once

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "forward/profile.h is a private architecture-layer header"
#endif

#include "clock.h" /* geist_now_ns: the profiler's clock */

#include <stdatomic.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Shared diagnostics: concurrent sessions accumulate into the same
 * per-stage counters (one file-scope sink per stage group, not per
 * session), so the counters and `registered` are atomic.
 *
 * Relaxed ordering throughout: these numbers are read once at exit and
 * nothing else is ordered against them, so the counters cost a plain
 * load/store on every target. Profiling stays free when disabled —
 * transformer_profile_add returns on t0 == 0 before touching them. */
struct transformer_forward_profile {
    const char        *title;
    const char *const *stage_names;
    size_t             stage_count;
    _Atomic uint64_t  *ns;
    _Atomic uint64_t  *calls;
    _Atomic bool       registered;
};

bool transformer_profile_enabled(struct transformer_forward_profile *profile);

/* Adds the time since t0 to `stage` and returns it (0 when t0 is 0, i.e.
 * profiling is off), so a caller can take a nested stage out of its own. */
uint64_t
transformer_profile_add(struct transformer_forward_profile *profile, size_t stage, uint64_t t0);

/* A file-scope profile `var` with its counters and the names of its `n`
 * stages: TRANSFORMER_PROFILE_DEFINE(g_x, "title", X_COUNT, "a", "b"); */
#define TRANSFORMER_PROFILE_DEFINE(var, title_, n, ...)                       \
    static _Atomic uint64_t                   var##_ns[n];                    \
    static _Atomic uint64_t                   var##_calls[n];                 \
    static const char *const                  var##_names[n] = {__VA_ARGS__}; \
    static struct transformer_forward_profile var            = {              \
            .title       = title_,                                            \
            .stage_names = var##_names,                                       \
            .stage_count = n,                                                 \
            .ns          = var##_ns,                                          \
            .calls       = var##_calls,                                       \
    }
