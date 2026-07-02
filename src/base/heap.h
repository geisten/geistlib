//
// Created by germar on 09.03.25.
//
#pragma once
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdalign.h>

/* Cache line size alignment for optimal CPU performance */
#define CACHE_LINE_SIZE 64 /* 64 bytes = 512 bits (typical cache line size) */

/* Define SIMD alignment based on available hardware capabilities */
#if defined(__AVX512F__)
#define SIMD_ALIGNMENT 64 /* 512 bits */
#elif defined(__AVX__) || defined(__AVX2__)
#define SIMD_ALIGNMENT 32 /* 256 bits */
#elif defined(__SSE__) || defined(__SSE2__) || defined(__NEON__)
#define SIMD_ALIGNMENT 16 /* 128 bits */
#else
#define SIMD_ALIGNMENT 8 /* Fallback */
#endif

/* Use the larger of cache line and SIMD alignment for optimal performance */
#define OPTIMAL_ALIGNMENT (CACHE_LINE_SIZE > SIMD_ALIGNMENT ? CACHE_LINE_SIZE : SIMD_ALIGNMENT)

static_assert((CACHE_LINE_SIZE & (CACHE_LINE_SIZE - 1)) == 0,
              "CACHE_LINE_SIZE must be a power of 2");
static_assert((SIMD_ALIGNMENT & (SIMD_ALIGNMENT - 1)) == 0, "SIMD_ALIGNMENT must be a power of 2");
static_assert(OPTIMAL_ALIGNMENT >= 8, "OPTIMAL_ALIGNMENT must be at least 8 bytes");

/* Memory arena for allocating temporary structures */
struct memory_arena {
    void  *memory;
    size_t size;
    size_t used;
};

/**
 * Creates a memory arena with the specified size.
 *
 * @param size Size of the arena in bytes
 * @return Initialized memory arena structure
 *
 * Example:
 *   struct memory_arena arena = create_memory_arena(1024 * 1024); // 1MB arena
 *   if (!arena.memory) { ... handle failure ... }
 *
 * Limitations:
 *   - On allocation failure the returned arena has memory == nullptr (and
 *     size/used == 0); the caller MUST check before use. This function does
 *     not exit() — prefer try_create_memory_arena() to handle failure inline.
 *   - The arena is not thread-safe, should only be used by one thread
 */
[[nodiscard]] struct memory_arena create_memory_arena(size_t size);
[[nodiscard]] bool                try_create_memory_arena(struct memory_arena *arena, size_t size);

/**
 * Allocates memory from the arena with specified alignment.
 *
 * @param arena Pointer to the memory arena
 * @param size Number of bytes to allocate
 * @param alignment Required alignment (must be power of 2)
 * @return Pointer to allocated memory or nullptr if allocation fails
 *
 * Example:
 *   float* buffer = arena_allocate_aligned(&arena, 10 * sizeof(float),
 * alignof(float));
 *
 * Limitations:
 *   - Does not support freeing individual allocations, only the entire arena
 *   - May return nullptr if there's not enough space in the arena
 */
void *arena_allocate_aligned(struct memory_arena *arena, size_t size, size_t alignment);
void *heap_alloc_aligned(size_t size, size_t alignment);
void *heap_calloc_aligned(size_t count, size_t size, size_t alignment);

uintptr_t aligned_size(size_t size, size_t alignment);

#define arena_allocate_array_aligned(_arena, _type, _num) \
    ((_type *) arena_allocate_aligned((_arena), (_num) * sizeof(_type), alignof(_type)))

#define heap_alloc_array_aligned(_type, _num) \
    ((_type *) heap_alloc_aligned((_num) * sizeof(_type), alignof(_type)))

#define heap_calloc_array_aligned(_type, _num) \
    ((_type *) heap_calloc_aligned((_num), sizeof(_type), alignof(_type)))

void safe_free(void **ptr);

/**
 * Frees the entire memory arena.
 *
 * @param arena Pointer to the memory arena
 *
 * Example:
 *   free_memory_arena(&arena);
 */
void free_memory_arena(struct memory_arena *arena);
