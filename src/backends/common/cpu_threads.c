/*
 * src/backends/common/cpu_threads.c — see cpu_threads.h.
 *
 * Layer: BACKEND.
 */
#include "cpu_threads.h"

#include "par.h"

#include <limits.h>
#include <stddef.h>

int geist_cpu_region_begin(const enum geist_parallel_region region, int target, const int cap) {
    if (cap > 0 && (target <= 0 || target > cap)) {
        target = cap;
    }
    if (target <= 0) {
        return 0;
    }
    const size_t max   = geist_par_max_threads();
    const int    prev  = max > (size_t) INT_MAX ? INT_MAX : (int) max;
    const bool   apply = region == GEIST_REGION_DECODE_STEP ? target < prev : target != prev;
    if (!apply) {
        return 0;
    }
    geist_par_set_max_threads((size_t) target);
    return prev; /* > 0: restore to this in _end */
}

void geist_cpu_region_end(struct geist_backend *be, const int token) {
    (void) be;
    if (token > 0) {
        geist_par_set_max_threads((size_t) token);
    }
}
