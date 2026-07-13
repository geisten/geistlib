//
// Created by germar on 09.03.25.
//
#include "heap.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif

// Ensure alignment is a power of 2 (e.g., 16, 32, 64)
uintptr_t aligned_size(const size_t size, const size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/* True iff x is a non-zero power of two. Allocation alignments must satisfy
 * this: the rounding mask ~(alignment-1) and aligned_alloc() are both
 * undefined otherwise. */
static bool size_is_pow2(const size_t x) {
    return x != 0u && (x & (x - 1u)) == 0u;
}

/* Round `size` up to a multiple of `alignment` (a power of two), reporting
 * size_t overflow instead of silently wrapping to a small value. A wrapped
 * result would under-allocate and hand back a buffer smaller than requested
 * (AGENT.md: "No silent truncation", correctness first). Returns false on
 * overflow; *out is left untouched. */
static bool checked_round_up(const size_t size, const size_t alignment, size_t *out) {
    if (size > SIZE_MAX - (alignment - 1u)) {
        return false;
    }
    *out = (size + alignment - 1u) & ~(alignment - 1u);
    return true;
}

/* Portable aligned allocation that pairs with plain free()/safe_free().
 *
 * aligned_alloc (C11) is present on every project target — macOS >= 10.15,
 * glibc, Raspberry Pi OS — and the engine builds with -std=c23, so that is
 * the path taken in practice. The posix_memalign fallback covers hosts that
 * ship POSIX but not C11 aligned_alloc; its memory also frees with free(),
 * so safe_free() stays valid. Windows/MSVC has neither and needs the
 * _aligned_malloc/_aligned_free pair (an incompatible free path), so it is
 * out of scope here (AGENT.md: portability behind clear boundaries) and
 * rejected at compile time rather than mis-freed at runtime.
 *
 * Preconditions (enforced by every caller): `alignment` is a power of two
 * >= OPTIMAL_ALIGNMENT (>= 8, hence a multiple of sizeof(void*)), and `size`
 * is already rounded to a multiple of `alignment`. */
static void *portable_aligned_alloc(const size_t alignment, const size_t size) {
#if defined(_MSC_VER)
#error "heap.c: MSVC needs the _aligned_malloc/_aligned_free pair; unsupported."
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    return aligned_alloc(alignment, size);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    void *p = nullptr;
    if (posix_memalign(&p, alignment, size) != 0) {
        return nullptr;
    }
    return p;
#else
#error "heap.c: no aligned allocation primitive available on this platform."
#endif
}

uintptr_t optimal_aligned_size(const size_t size) {
    return aligned_size(size, OPTIMAL_ALIGNMENT);
}

/**
 * Creates a memory arena with the specified size.
 *
 * The arena allocator is a simple and efficient way to manage memory for
 * short-lived allocations. Instead of calling malloc/free for each small
 * allocation, we allocate a large block upfront and distribute small chunks
 * from it.
 *
 * @param size Size of the arena in bytes
 * @return Initialized memory arena structure
 */
struct memory_arena create_memory_arena(const size_t size) {
    struct memory_arena arena = {};
    if (!try_create_memory_arena(&arena, size)) {
        /* A runtime/inference core must not exit() on a caller's behalf
         * (AGENT.md: outputs must remain well-defined on failure). Return a
         * null arena (memory == nullptr); the caller MUST check arena.memory
         * before use. arena_allocate_aligned() already rejects such arenas.
         * Use try_create_memory_arena() directly to handle failure inline. */
        fprintf(stderr, "create_memory_arena: failed to allocate %zu-byte arena\n", size);
    }
    return arena;
}

bool try_create_memory_arena(struct memory_arena *arena, const size_t size) {
    if (!arena) {
        return false;
    }
    /* Make sure the size is a multiple of the optimal alignment. Reject
     * sizes that would overflow when rounded up rather than wrapping to a
     * tiny allocation. */
    size_t aligned_size = 0u;
    if (!checked_round_up(size, OPTIMAL_ALIGNMENT, &aligned_size)) {
        arena->memory = nullptr;
        arena->size   = 0;
        arena->used   = 0;
        return false;
    }

    /* Allocate memory with optimal alignment for best performance */
    void *memory = portable_aligned_alloc(OPTIMAL_ALIGNMENT, aligned_size);
    if (!memory) {
        arena->memory = nullptr;
        arena->size   = 0;
        arena->used   = 0;
        return false;
    }

    /* Initialize the arena structure */
    arena->memory = memory;
    arena->size   = aligned_size;
    arena->used   = 0;
    return true;
}

/**
 * Allocates aligned memory from the arena.
 *
 * This function returns a pointer to a block of memory with the specified size
 * and alignment. The memory comes from the arena's pre-allocated block. This is
 * much faster than calling malloc for many small allocations.
 *
 * @param arena Pointer to the memory arena
 * @param size Number of bytes to allocate
 * @param alignment Required alignment (must be power of 2)
 * @return Pointer to allocated memory or nullptr if allocation fails
 */
void *arena_allocate_aligned(struct memory_arena *arena, size_t size, size_t alignment) {
    if (!arena || !arena->memory) {
        return nullptr;
    }

    /* Default to OPTIMAL_ALIGNMENT, then reject invalid alignments instead
     * of feeding a non-power-of-2 into the rounding mask / aligned_alloc. */
    if (alignment == 0u) {
        alignment = OPTIMAL_ALIGNMENT;
    }
    if (!size_is_pow2(alignment)) {
        return nullptr;
    }
    /* Verwende mindestens OPTIMAL_ALIGNMENT */
    if (alignment < OPTIMAL_ALIGNMENT) {
        alignment = OPTIMAL_ALIGNMENT;
    }

    /* Calculate aligned address */
    const uintptr_t current = (uintptr_t) arena->memory + arena->used;
    const uintptr_t aligned = aligned_size(current, alignment);
    const size_t    offset  = aligned - (uintptr_t) arena->memory;

    /* Round size up to a multiple of alignment for better data locality;
     * refuse on overflow rather than wrapping past the bounds check. */
    size_t aligned_size_value = 0u;
    if (!checked_round_up(size, alignment, &aligned_size_value)) {
        return nullptr;
    }

    /* Check if we have enough space. Compare without computing
     * offset + aligned_size_value (which could itself overflow size_t). */
    if (offset > arena->size || aligned_size_value > arena->size - offset) {
        fprintf(stderr,
                "Memory out of bounds (old: %zu + object size %zu > total: %zu)\n",
                offset,
                aligned_size_value,
                arena->size);
        return nullptr;
    }

    /* Update used space */
    arena->used = offset + aligned_size_value;

    /* Return aligned pointer */
    return (void *) aligned;
}

void *heap_alloc_aligned(const size_t size, size_t alignment) {
    size_t aligned = 0;

    if (size == 0u) {
        return nullptr;
    }
    if (alignment == 0u) {
        alignment = OPTIMAL_ALIGNMENT;
    }
    /* Validate the alignment is a power of two before using it as a mask or
     * passing it to aligned_alloc (UB otherwise). AGENT.md: express
     * invariants through explicit validation, not assertions. */
    if (!size_is_pow2(alignment)) {
        return nullptr;
    }
    if (alignment < OPTIMAL_ALIGNMENT) {
        alignment = OPTIMAL_ALIGNMENT;
    }
    /* aligned_alloc requires the size be a multiple of alignment; the
     * checked round-up enforces that and rejects size_t overflow. */
    if (!checked_round_up(size, alignment, &aligned)) {
        return nullptr;
    }
    void *p = portable_aligned_alloc(alignment, aligned);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    /* Big streaming allocations (backend weight repacks, lm_head sketch/Q8
     * blobs — hundreds of MB read every token) benefit from THP the same way
     * the GGUF mmap does (gguf_reader.c apply_mmap_advice): fewer TLB misses
     * in bandwidth-bound GEMVs. Advisory; same opt-out env. #102 Phase 0. */
    if (p != nullptr && aligned >= (2u << 20) && getenv("GEIST_NO_HUGEPAGE") == nullptr) {
        /* madvise needs a page-aligned address; the allocation is only
         * `alignment`-aligned. Advise the page-aligned sub-range. */
        const size_t    page  = 4096;
        const uintptr_t base  = (uintptr_t) p;
        const uintptr_t first = (base + page - 1) & ~(uintptr_t) (page - 1);
        const size_t    skip  = (size_t) (first - base);
        if (aligned > skip + page) {
            (void) madvise((void *) first, aligned - skip, MADV_HUGEPAGE);
        }
    }
#endif
    return p;
}

void *heap_calloc_aligned(const size_t count, const size_t size, const size_t alignment) {
    void *memory = nullptr;
    if (count == 0u || size == 0u) {
        return nullptr;
    }
    if (count > (SIZE_MAX / size)) {
        return nullptr;
    }
    memory = heap_alloc_aligned(count * size, alignment);
    if (!memory) {
        return nullptr;
    }
    memset(memory, 0, count * size);
    return memory;
}

/**
 * safe_free - a safer way to free dynamically allocated memory
 * @ptr: pointer to memory location
 *
 * Description: This safe_free() function takes care of freeing
 * dynamically allocated memory while ensuring the pointer
 * @ptr passed to it is not nullptr before trying to free it.
 * Also, after freeing the memory, it sets the pointer @ptr
 * to nullptr to avoid the issue of dangling pointers
 */
void safe_free(void **ptr) {
    if (ptr != nullptr && *ptr != nullptr) {
        free(*ptr);
        *ptr = nullptr;
    }
}

/**
 * Frees the entire memory arena.
 *
 * This function releases all memory allocated for the arena. After calling this
 * function, any pointers obtained from arena_allocate or arena_allocate_aligned
 * are invalid.
 *
 * @param arena Pointer to the memory arena
 */
void free_memory_arena(struct memory_arena *arena) {
    if (arena && arena->memory) {
        safe_free(&arena->memory);
        arena->memory = nullptr;
        arena->size   = 0;
        arena->used   = 0;
    }
}
