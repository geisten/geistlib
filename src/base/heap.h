/*
 * src/base/heap.h — aligned, overflow-checked allocation (AGENT.md §3).
 * Every allocation here frees with safe_free()/free().
 */
#pragma once
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdalign.h>

#define CACHE_LINE_SIZE 64

#if defined(__AVX512F__)
#define SIMD_ALIGNMENT 64
#elif defined(__AVX__) || defined(__AVX2__)
#define SIMD_ALIGNMENT 32
#elif defined(__SSE__) || defined(__SSE2__) || defined(__NEON__)
#define SIMD_ALIGNMENT 16
#else
#define SIMD_ALIGNMENT 8
#endif

/* Minimum alignment of every allocation: the larger of cache line and SIMD width. */
#define OPTIMAL_ALIGNMENT (CACHE_LINE_SIZE > SIMD_ALIGNMENT ? CACHE_LINE_SIZE : SIMD_ALIGNMENT)

static_assert((CACHE_LINE_SIZE & (CACHE_LINE_SIZE - 1)) == 0,
              "CACHE_LINE_SIZE must be a power of 2");
static_assert((SIMD_ALIGNMENT & (SIMD_ALIGNMENT - 1)) == 0, "SIMD_ALIGNMENT must be a power of 2");
static_assert(OPTIMAL_ALIGNMENT >= 8, "OPTIMAL_ALIGNMENT must be at least 8 bytes");

/* `alignment` 0 means OPTIMAL_ALIGNMENT; smaller powers of two are raised to
 * it, non-powers of two fail. Returns nullptr on a zero size, overflow or OOM.
 * On Linux, allocations >= 2 MiB are advised MADV_HUGEPAGE unless
 * GEIST_NO_HUGEPAGE is set. */
void *heap_alloc_aligned(size_t size, size_t alignment);
/* Zeroed count * size bytes; nullptr on overflow or a zero count/size. */
void *heap_calloc_aligned(size_t count, size_t size, size_t alignment);

/* Uninitialized count * size bytes. The product is computed here so the
 * overflow is refused instead of arriving wrapped. Returns nullptr on
 * overflow or on a zero count/size. */
void *heap_alloc_n_aligned(size_t count, size_t size, size_t alignment);

/* Successful heap_alloc_aligned calls since process start. Hot paths are
 * specified allocation-free (include/geist_weight.h, src/engine/sampler.h);
 * a bench or test asserts a zero delta across its measured loop instead of
 * trusting the comment. Monotonic, relaxed — a counter, not a barrier. */
[[nodiscard]] uint64_t heap_alloc_count(void);

/* Test hook: while `on`, every allocation in this header fails and returns
 * nullptr as if memory were exhausted — the only way a test reaches the
 * code that handles that. Process-wide and relaxed; nothing outside tests
 * sets it. */
void heap_fail_allocations(bool on);

/* Typed array allocation. Count and element size reach the overflow check
 * separately: model-controlled counts (tensor dims, image geometry, KV
 * capacities) must not wrap into a small allocation. */
#define heap_alloc_array_aligned(_type, _num) \
    ((_type *) heap_alloc_n_aligned((_num), sizeof(_type), alignof(_type)))

#define heap_calloc_array_aligned(_type, _num) \
    ((_type *) heap_calloc_aligned((_num), sizeof(_type), alignof(_type)))

/* free(*ptr) and set *ptr = nullptr; tolerates null ptr and null *ptr. */
void safe_free(void **ptr);

/* `size` bytes, OPTIMAL_ALIGNMENT-aligned, whose pages go back to the system
 * when freed: from 256 KiB up a mapping of their own, below that
 * heap_alloc_aligned. For model-lifetime buffers such as weight repacks,
 * which the malloc caches would otherwise keep after a model destroy (#733).
 * nullptr on a zero size or OOM. */
[[nodiscard]] void *heap_alloc_pages(size_t size);

/* Frees a heap_alloc_pages block of the same `size` and nulls *ptr;
 * tolerates null ptr and null *ptr. */
void heap_free_pages(void **ptr, size_t size);
