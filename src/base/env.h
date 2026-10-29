//
// GEIST_* environment knobs: one rule for every on/off switch.
//
// A flag is off when it is unset, empty or exactly "0", and on for any other
// value ("1", "yes", "2", ...). A knob whose default is on passes dflt = true
// and is turned off with NAME=0 (or NAME=). Before this header the tree read
// flags five different ways ("[0] == '1'", "[0] != '0'", "set at all", ...),
// so NAME=yes enabled one knob and silently did nothing on the next.
//
// These read getenv on every call. A knob a hot path consults keeps its
// read-once cache (a static atomic) at the call site.
//
#pragma once

#include "parse.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>

/* `name` as an on/off switch: dflt when unset, false when empty or "0",
 * true otherwise. */
[[nodiscard]] static inline bool geist_env_flag(const char *name, bool dflt) {
    const char *v = getenv(name);
    if (v == nullptr) {
        return dflt;
    }
    return v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

/* `name` as a thread/item count: -1 when unset or empty, 0 when set to
 * anything but a whole number in [1, INT_MAX], else the number. */
[[nodiscard]] static inline int geist_env_count(const char *name) {
    const char *v = getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return -1;
    }
    long n;
    return geist_parse_long(v, &n) && n > 0 && n <= INT_MAX ? (int) n : 0;
}

/* GEIST_FAST_TANH (Apple): opt into vForce tanh/exp, ~1-2 ULP off libm,
 * which compounds past the strict parity gate. Read once per translation
 * unit; the gelu and softmax kernels call it per row. */
[[nodiscard]] static inline bool geist_fast_tanh_enabled(void) {
    static _Atomic int on = -1;
    int                v  = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        v = geist_env_flag("GEIST_FAST_TANH", false);
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v != 0;
}
