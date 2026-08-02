/*
 * src/archs/transformer/forward/profile.c - private forward profiler.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "profile.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { TRANSFORMER_PROFILE_MAX_SINKS = 16 };

/* Diagnostics only (env-gated); still made thread-safe so concurrent
 * sessions with profiling on don't race the cache/registry (TSan). */
static struct transformer_forward_profile *g_profiles[TRANSFORMER_PROFILE_MAX_SINKS];
static size_t                              g_profile_count;
static pthread_mutex_t                     g_profile_mu      = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int                         g_profile_enabled = -1;
static bool                                g_profile_atexit_registered;

static void transformer_profile_print_all(void) {
    for (size_t p = 0; p < g_profile_count; p++) {
        struct transformer_forward_profile *profile = g_profiles[p];
        if (profile == nullptr) {
            continue;
        }

        uint64_t total = 0;
        for (size_t i = 0; i < profile->stage_count; i++) {
            total += profile->ns[i];
        }
        if (total == 0) {
            continue;
        }

        fprintf(stderr, "%s profile:\n", profile->title);
        for (size_t i = 0; i < profile->stage_count; i++) {
            const double ms  = (double) profile->ns[i] / 1000000.0;
            const double pct = 100.0 * (double) profile->ns[i] / (double) total;
            fprintf(stderr,
                    "  %-10s %10.2f ms  %5.1f%%  (%llu calls)\n",
                    profile->stage_names[i],
                    ms,
                    pct,
                    (unsigned long long) profile->calls[i]);
        }
    }
}

static bool transformer_profile_env_enabled(void) {
    if (atomic_load(&g_profile_enabled) < 0) {
        pthread_mutex_lock(&g_profile_mu);
        if (atomic_load(&g_profile_enabled) < 0) {
            const char *env = getenv("GEIST_PROFILE_PREFILL");
            if (env == nullptr || env[0] == '\0') {
                env = getenv("GEIST_PROFILE_FORWARD");
            }
            const int on = (env != nullptr && env[0] == '1') ? 1 : 0;
            if (on && !g_profile_atexit_registered) {
                atexit(transformer_profile_print_all);
                g_profile_atexit_registered = true;
            }
            atomic_store(&g_profile_enabled, on);
        }
        pthread_mutex_unlock(&g_profile_mu);
    }
    return atomic_load(&g_profile_enabled) != 0;
}

static void transformer_profile_register(struct transformer_forward_profile *profile) {
    if (profile == nullptr || profile->registered) {
        return;
    }
    pthread_mutex_lock(&g_profile_mu);
    if (!profile->registered && g_profile_count < TRANSFORMER_PROFILE_MAX_SINKS) {
        g_profiles[g_profile_count++] = profile;
        profile->registered           = true;
    }
    pthread_mutex_unlock(&g_profile_mu);
}

bool transformer_profile_enabled(struct transformer_forward_profile *profile) {
    if (!transformer_profile_env_enabled()) {
        return false;
    }
    transformer_profile_register(profile);
    return true;
}

uint64_t transformer_profile_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

void transformer_profile_add(struct transformer_forward_profile *profile,
                             size_t                              stage,
                             uint64_t                            t0) {
    if (profile == nullptr || t0 == 0 || stage >= profile->stage_count) {
        return;
    }
    profile->ns[stage] += transformer_profile_now_ns() - t0;
    profile->calls[stage]++;
}
