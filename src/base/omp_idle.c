#define _POSIX_C_SOURCE 200809L /* setenv */
/*
 * src/base/omp_idle.c — bounded idle spin for OpenMP workers. See omp_idle.h.
 *
 * Layer: ENGINE.
 */
#include "omp_idle.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

/* KMP_BLOCKTIME holds our idle default, not a user's or a model's value. */
static atomic_bool g_default_owned = false;

static const char *sys_getenv(const char *name) {
    return getenv(name);
}

int geist_omp_idle_spin_ms(const char *(*getenv_fn)(const char *) ) {
    const char *v = (getenv_fn != nullptr ? getenv_fn : sys_getenv)("GEIST_IDLE_SPIN_MS");
    if (v != nullptr && v[0] != '\0') {
        char      *end = nullptr;
        const long ms  = strtol(v, &end, 10);
        if (*end == '\0' && ms >= 0 && ms <= 1000000) {
            return (int) ms;
        }
    }
    return GEIST_IDLE_SPIN_MS;
}

static void set_blocktime(int ms, bool overwrite) {
    char buf[16];
    snprintf(buf, sizeof buf, "%d", ms);
    setenv("KMP_BLOCKTIME", buf, overwrite);
}

void geist_omp_idle_default(void) {
    if (getenv("KMP_BLOCKTIME") != nullptr) {
        return;
    }
    set_blocktime(geist_omp_idle_spin_ms(nullptr), false);
    atomic_store(&g_default_owned, true);
}

void geist_omp_blocktime_apply(int ms) {
    if (ms < 0) {
        return;
    }
    if (atomic_exchange(&g_default_owned, false)) {
        set_blocktime(ms, true);
    } else {
        set_blocktime(ms, false); /* unset: the model sets it; set: it wins */
    }
}
