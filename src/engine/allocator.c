/*
 * src/engine/allocator.c — the default libc allocator, routed through the
 * project-wide heap.h interface (per AGENT.md).
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
    /* Route through heap.h's safe_free (the project's free interface, per
     * AGENT.md); it already tolerates null. Note safe_free nulls only our
     * local `ptr` copy — that has no effect on the caller's pointer, which
     * is theirs to clear. We don't pretend otherwise here. */
    safe_free(&ptr);
}

const struct geist_allocator geist_libc_allocator = {
        .alloc    = libc_alloc,
        .free     = libc_free,
        .free_all = nullptr, /* libc has no free-all */
        .ctx      = nullptr,
};
