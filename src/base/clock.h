//
// One monotonic clock for timing (profilers, stats, spin deadlines).
// CLOCK_MONOTONIC: never for wall time or a pthread_cond_timedwait
// deadline, which count in CLOCK_REALTIME.
//
#pragma once

#include <stdint.h>
#include <time.h>

[[nodiscard]] static inline uint64_t geist_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}
