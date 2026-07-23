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

void *heap_alloc_aligned(size_t size, size_t alignment);
void *heap_calloc_aligned(size_t count, size_t size, size_t alignment);

#define heap_alloc_array_aligned(_type, _num) \
    ((_type *) heap_alloc_aligned((_num) * sizeof(_type), alignof(_type)))

#define heap_calloc_array_aligned(_type, _num) \
    ((_type *) heap_calloc_aligned((_num), sizeof(_type), alignof(_type)))

void safe_free(void **ptr);
