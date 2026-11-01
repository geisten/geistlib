/*
 * src/base/cleanup.h — scope-exit cleanup for cold code (#349, AGENT.md §3).
 *
 *     GEIST_CLEANUP(heap_freep) float *tmp = heap_alloc_array_aligned(float, n);
 *
 * runs heap_freep(&tmp) on every way out of tmp's scope, so an error path
 * returns instead of jumping to a label. Load, setup and error paths only:
 * scripts/check-cleanup-scope.sh refuses it under kernels/ and forward/.
 * A pointer that is handed out (stored in a struct, returned) takes no
 * cleanup; set it to nullptr first if it changes hands at the end.
 *
 * C23 has no defer; [[gnu::cleanup]] is the gcc/clang extension behind it.
 */
#pragma once

#include "heap.h"

#if !defined(__GNUC__) && !defined(__clang__)
#error "cleanup.h needs [[gnu::cleanup]] (gcc or clang); unsupported compiler."
#endif

#define GEIST_CLEANUP(fn) [[gnu::cleanup(fn)]]

/* For GEIST_CLEANUP: safe_free on a heap.h pointer (`pp` points to it). */
static inline void heap_freep(void *pp) {
    safe_free((void **) pp);
}
