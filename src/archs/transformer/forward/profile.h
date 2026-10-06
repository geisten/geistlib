/*
 * src/archs/transformer/forward/profile.h - private forward profiler.
 */
#pragma once

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "forward/profile.h is a private architecture-layer header"
#endif

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

bool     transformer_profile_enabled(struct transformer_forward_profile *profile);
uint64_t transformer_profile_now_ns(void);
void     transformer_profile_add(struct transformer_forward_profile *profile,
                                 size_t                              stage,
                                 uint64_t                            t0);
