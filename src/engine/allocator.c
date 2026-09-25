/*
 * src/engine/allocator.c — the default libc allocator, routed through the
 * project-wide heap.h interface.
 *
 * Layer: ENGINE.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include <geist.h>

#include "heap.h"

#include <stddef.h>

static void *libc_alloc(void *ctx, size_t bytes, size_t alignment) {
    (void) ctx;
    if (bytes == 0) {
        return nullptr;
    }
    /* heap.h enforces power-of-2 alignment >= 8 internally; route through. */
    return heap_alloc_aligned(bytes, alignment > 0 ? alignment : OPTIMAL_ALIGNMENT);
}

static void libc_free(void *ctx, void *ptr) {
    (void) ctx;
    /* safe_free tolerates null. It clears only the local `ptr` copy, not the
     * caller's pointer. */
    safe_free(&ptr);
}

const struct geist_allocator geist_libc_allocator = {
        .alloc    = libc_alloc,
        .free     = libc_free,
        .free_all = nullptr, /* libc has no free-all */
        .ctx      = nullptr,
};
