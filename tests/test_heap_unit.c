/*
 * test_heap_unit — heap.h allocator safety contract.
 *
 * heap.c is the project-wide allocation interface (AGENT.md). These cases
 * lock down the safety guarantees that the engine and the GGUF loaders rely
 * on, in particular the ones added to stop silent integer-overflow
 * under-allocation:
 *
 *   - zero-size / empty inputs return nullptr (well-defined, no alloc)
 *   - minimal valid allocations succeed, are aligned, and are writable
 *   - invalid (non-power-of-2) alignment is refused, not fed to the mask
 *   - size_t-overflowing sizes are refused, not wrapped to a tiny buffer
 *   - heap_calloc_aligned refuses count*size overflow
 *   - safe_free tolerates null and nulls the caller's pointer
 *
 * Deterministic, allocation-bounded (the overflow cases never reach
 * aligned_alloc — the guards return before that), ASan/UBSan friendly.
 */
#include "test_helpers.h"

#include "heap.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            fprintf(stderr, "FAIL: %s\n", (msg)); \
            return GEIST_TEST_FAIL;               \
        }                                         \
    } while (0)

int main(void) {
    /* ---- heap_alloc_aligned -------------------------------------------- */

    /* Zero size: well-defined nullptr, no allocation. */
    CHECK(heap_alloc_aligned(0, 64) == nullptr, "alloc(0) should be null");

    /* Minimal valid: aligned to at least OPTIMAL_ALIGNMENT and writable. */
    void *p = heap_alloc_aligned(1, alignof(float));
    CHECK(p != nullptr, "alloc(1) should succeed");
    CHECK(((uintptr_t) p & (OPTIMAL_ALIGNMENT - 1)) == 0, "alloc not OPTIMAL_ALIGNMENT-aligned");
    memset(p, 0xAB, 1); /* writable; ASan would catch under-allocation */
    safe_free(&p);
    CHECK(p == nullptr, "safe_free should null the pointer");

    /* Explicit large power-of-2 alignment is honored. */
    void *pa = heap_alloc_aligned(32, 128);
    CHECK(pa != nullptr, "alloc with align=128 should succeed");
    CHECK(((uintptr_t) pa & 127) == 0, "align=128 not honored");
    safe_free(&pa);

    /* Non-power-of-2 alignment must be refused (would corrupt the mask /
     * is UB for aligned_alloc). */
    CHECK(heap_alloc_aligned(64, 24) == nullptr, "non-pow2 align must be null");
    CHECK(heap_alloc_aligned(64, 96) == nullptr, "non-pow2 align>OPTIMAL must be null");

    /* Overflow: rounding SIZE_MAX up to alignment would wrap to a tiny
     * value. Must be refused, not under-allocate. */
    CHECK(heap_alloc_aligned(SIZE_MAX, 64) == nullptr, "SIZE_MAX alloc must be refused");
    CHECK(heap_alloc_aligned(SIZE_MAX - 16, 64) == nullptr, "near-SIZE_MAX alloc must be refused");

    /* ---- heap_calloc_aligned ------------------------------------------- */

    CHECK(heap_calloc_aligned(0, 4, 64) == nullptr, "calloc(0,*) null");
    CHECK(heap_calloc_aligned(4, 0, 64) == nullptr, "calloc(*,0) null");

    /* count*size overflow must be refused. */
    CHECK(heap_calloc_aligned(SIZE_MAX, 2, 64) == nullptr, "calloc overflow must be refused");

    int32_t *zeros = heap_calloc_array_aligned(int32_t, 16);
    CHECK(zeros != nullptr, "calloc(16 int32) should succeed");
    for (int i = 0; i < 16; i++) {
        CHECK(zeros[i] == 0, "calloc memory not zeroed");
    }
    safe_free((void **) &zeros);

    /* safe_free null tolerance. */
    void *nullp = nullptr;
    safe_free(&nullp);
    safe_free(nullptr);

    printf("PASS: heap alloc/calloc — zero/overflow/alignment guards\n");
    return GEIST_TEST_PASS;
}
