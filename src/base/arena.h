/*
 * src/engine/arena.h — Bump-allocator scratch arena.
 *
 * Layer: ENGINE (internal). Inline helpers for malloc-free hot-path
 * scratch. Used as the per-session frame arena for transformer scratch
 * buffers (attention scores, dequant rows, logits row, etc.); reset to
 * zero at the start of each forward pass.
 *
 * Hot-path contract:
 *   - frame_arena_alloc is a pointer bump + bounds check.
 *   - frame_arena_reset is a single store.
 *   - No heap call. No syscall. No global state.
 *
 * Buffer ownership:
 *   - The arena's `base` storage is owned by the caller (typically a
 *     single heap_alloc_aligned at session_create time). The arena
 *     struct itself owns nothing — it just tracks offset + capacity.
 *   - Concurrent access requires external synchronization. A
 *     geist_session is single-threaded by design; the engine doesn't
 *     reuse arenas across threads.
 *
 * Sizing:
 *   - capacity is fixed at init time. Caller sums all per-forward
 *     scratch needs (max_seq * dims × few scratch_kinds) and passes
 *     that as capacity. frame_arena_alloc returns nullptr on overflow.
 *
 * Alignment:
 *   - align must be a power of two; ≥ 16 enforced internally for NEON
 *     loads. Pass 64 for cache-line alignment.
 */
#ifndef GEIST_INTERNAL_ARENA_H
#define GEIST_INTERNAL_ARENA_H

#ifndef GEIST_INTERNAL_ENGINE_LAYER
#error "src/engine/arena.h is internal to the engine layer."
#endif

#include <stddef.h>
#include <stdint.h>

struct frame_arena {
    void  *base;     /* not owned; caller-provided storage */
    size_t used;     /* bytes allocated from base so far */
    size_t capacity; /* total bytes available */
};

static inline void frame_arena_init(struct frame_arena *a, void *base, size_t capacity) {
    a->base     = base;
    a->capacity = capacity;
    a->used     = 0;
}

static inline void frame_arena_reset(struct frame_arena *a) {
    a->used = 0;
}

/* Returns a pointer to `bytes` of arena memory aligned to `align`
 * (rounded up to 16 if smaller). Returns nullptr if the arena is
 * exhausted; callers must check the result. */
[[nodiscard]] static inline void *
frame_arena_alloc(struct frame_arena *a, size_t bytes, size_t align) {
    if (align < 16)
        align = 16;
    const size_t mask    = align - 1;
    const size_t aligned = (a->used + mask) & ~mask;
    if (aligned + bytes > a->capacity || aligned + bytes < aligned) {
        return nullptr;
    }
    void *p = (char *) a->base + aligned;
    a->used = aligned + bytes;
    return p;
}

#endif /* GEIST_INTERNAL_ARENA_H */
