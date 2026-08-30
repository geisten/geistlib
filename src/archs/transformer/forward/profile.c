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
static _Atomic size_t                      g_profile_count;
static pthread_mutex_t                     g_profile_mu      = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int                         g_profile_enabled = -1;
static bool                                g_profile_atexit_registered;

static void transformer_profile_print_all(void) {
    /* atexit: single-threaded by the time this runs, but the count was
     * published by other threads — acquire pairs with the release in
     * transformer_profile_register so the slots are visible too. */
    const size_t n = atomic_load_explicit(&g_profile_count, memory_order_acquire);
    for (size_t p = 0; p < n; p++) {
        struct transformer_forward_profile *profile = g_profiles[p];
        if (profile == nullptr) {
            continue;
        }

        uint64_t total = 0;
        for (size_t i = 0; i < profile->stage_count; i++) {
            total += atomic_load_explicit(&profile->ns[i], memory_order_relaxed);
        }
        if (total == 0) {
            continue;
        }

        fprintf(stderr, "%s profile:\n", profile->title);
        for (size_t i = 0; i < profile->stage_count; i++) {
            const uint64_t ns_i    = atomic_load_explicit(&profile->ns[i], memory_order_relaxed);
            const uint64_t calls_i = atomic_load_explicit(&profile->calls[i], memory_order_relaxed);
            fprintf(stderr,
                    "  %-10s %10.2f ms  %5.1f%%  (%llu calls)\n",
                    profile->stage_names[i],
                    (double) ns_i / 1000000.0,
                    100.0 * (double) ns_i / (double) total,
                    (unsigned long long) calls_i);
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
    /* The uncontended fast path reads `registered` without the mutex —
     * that is the point of it — so the flag has to be atomic. It used to
     * be a plain bool written under the lock and read outside it, which
     * is a race in the exact case the fast path exists for: every call
     * after the first, on every thread. */
    if (profile == nullptr || atomic_load_explicit(&profile->registered, memory_order_acquire)) {
        return;
    }
    pthread_mutex_lock(&g_profile_mu);
    const size_t n = atomic_load_explicit(&g_profile_count, memory_order_relaxed);
    if (!atomic_load_explicit(&profile->registered, memory_order_relaxed) &&
        n < TRANSFORMER_PROFILE_MAX_SINKS) {
        g_profiles[n] = profile;
        /* Publish the slot before the count: a reader that sees the new
         * count must see the pointer written into it. */
        atomic_store_explicit(&g_profile_count, n + 1, memory_order_release);
        atomic_store_explicit(&profile->registered, true, memory_order_release);
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
    atomic_fetch_add_explicit(
            &profile->ns[stage], transformer_profile_now_ns() - t0, memory_order_relaxed);
    atomic_fetch_add_explicit(&profile->calls[stage], 1, memory_order_relaxed);
}
