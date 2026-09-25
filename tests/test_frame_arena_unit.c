/*
 * test_frame_arena_unit — bump-allocator semantics + bounds.
 *
 * The frame_arena is the engine's scratch allocator for the hot path
 * (per-forward attention scores, dequant rows, logits row). The
 * hot-path allocation-free contract relies on this returning
 * pointers without touching the heap.
 *
 * Cases covered:
 *   - simple alloc + reset
 *   - alignment rounding (default 16, explicit higher)
 *   - exhaustion returns nullptr (not crash)
 *   - reset reuses the same pointer
 *   - overflow guard against size_t wrap
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "arena.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    /* 4 KB arena. */
    alignas(64) static unsigned char storage[4096];
    struct frame_arena               a;
    frame_arena_init(&a, storage, sizeof storage);

    /* Sequential allocs of 100 bytes, default-aligned. */
    void *p1 = frame_arena_alloc(&a, 100, 0);
    void *p2 = frame_arena_alloc(&a, 100, 0);
    void *p3 = frame_arena_alloc(&a, 100, 0);
    if (p1 == nullptr || p2 == nullptr || p3 == nullptr) {
        fprintf(stderr, "FAIL: small allocs returned null\n");
        return GEIST_TEST_FAIL;
    }
    /* Each alloc must be 16-aligned (min). */
    if (((uintptr_t) p1 & 15) || ((uintptr_t) p2 & 15) || ((uintptr_t) p3 & 15)) {
        fprintf(stderr, "FAIL: alignment <16\n");
        return GEIST_TEST_FAIL;
    }
    /* p2 must be at least 16 bytes (100 rounded up to next 16-multiple = 112) ahead. */
    if ((uintptr_t) p2 - (uintptr_t) p1 != 112) {
        fprintf(stderr,
                "FAIL: alignment-pad sizing: p2-p1=%zu, expected 112\n",
                (size_t) ((uintptr_t) p2 - (uintptr_t) p1));
        return GEIST_TEST_FAIL;
    }

    /* Cache-line align (64). */
    void *p4 = frame_arena_alloc(&a, 32, 64);
    if (p4 == nullptr) {
        fprintf(stderr, "FAIL: cacheline alloc\n");
        return GEIST_TEST_FAIL;
    }
    if ((uintptr_t) p4 & 63) {
        fprintf(stderr, "FAIL: 64-byte alignment not honored (p4=%p)\n", p4);
        return GEIST_TEST_FAIL;
    }

    /* Exhaustion: ask for more than remains; expect nullptr. */
    void *huge = frame_arena_alloc(&a, sizeof storage, 16);
    if (huge != nullptr) {
        fprintf(stderr, "FAIL: oversize alloc didn't return null\n");
        return GEIST_TEST_FAIL;
    }

    /* Reset: pointer reuse. */
    frame_arena_reset(&a);
    void *p1_again = frame_arena_alloc(&a, 100, 0);
    if (p1_again != p1) {
        fprintf(stderr, "FAIL: reset didn't reuse base (p1=%p, again=%p)\n", p1, p1_again);
        return GEIST_TEST_FAIL;
    }

    /* Overflow guard: huge alloc on near-full arena must return null. */
    frame_arena_reset(&a);
    (void) frame_arena_alloc(&a, sizeof storage - 32, 16);
    void *oversize = frame_arena_alloc(&a, SIZE_MAX / 2, 16);
    if (oversize != nullptr) {
        fprintf(stderr, "FAIL: SIZE_MAX/2 alloc wasn't refused\n");
        return GEIST_TEST_FAIL;
    }

    printf("PASS: frame_arena alloc / align / reset / exhaustion / overflow guard\n");
    return GEIST_TEST_PASS;
}
