/*
 * test_heap_unit — heap.h allocator safety contract.
 *
 * heap.c is the project-wide allocation interface. These cases
 * lock down the safety guarantees that the engine and the GGUF loaders rely
 * on, in particular against silent integer-overflow under-allocation:
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

#include <pthread.h>
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

/* heap_tl_grow on a thread that then exits: under LSan (MODE=asan) a
 * buffer not freed at thread exit is a leak (#100). */
static void *tl_grow_thread(void *arg) {
    static _Thread_local float *buf = nullptr;
    static _Thread_local size_t cap = 0;
    int                        *ok  = arg;
    *ok                             = heap_tl_grow_array(buf, cap, 100) && cap == 100;
    float *const   first            = buf;
    const uint64_t n                = heap_alloc_count();
    *ok = *ok && heap_tl_grow_array(buf, cap, 50) && buf == first && heap_alloc_count() == n;
    *ok = *ok && heap_tl_grow_array(buf, cap, 1000) && cap == 1000;
    heap_fail_allocations(true);
    *ok = *ok && !heap_tl_grow_array(buf, cap, 2000) && buf == nullptr && cap == 0;
    heap_fail_allocations(false);
    *ok = *ok && heap_tl_grow_array(buf, cap, 10);
    return nullptr;
}

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

    /* ---- heap_alloc_n_aligned + the array macro (issue #330) -----------
     * The count and the element size stay apart until the overflow check,
     * so a request for 2^62 int32s is refused, not wrapped into a 0- or
     * 16-byte allocation. */

    CHECK(heap_alloc_n_aligned(0, 4, 64) == nullptr, "alloc_n(0,*) null");
    CHECK(heap_alloc_n_aligned(4, 0, 64) == nullptr, "alloc_n(*,0) null");
    CHECK(heap_alloc_n_aligned(SIZE_MAX, 2, 64) == nullptr, "alloc_n overflow must be refused");
    CHECK(heap_alloc_n_aligned(SIZE_MAX / 2 + 1, 2, 64) == nullptr,
          "alloc_n exact-boundary overflow must be refused");
    /* SIZE_MAX * 1 does not overflow the multiply — it has to be refused
     * one step later, by the round-up inside heap_alloc_aligned. */
    CHECK(heap_alloc_n_aligned(SIZE_MAX, 1, 64) == nullptr, "alloc_n(SIZE_MAX,1) must be refused");

    /* Through the macro, with the element size the compiler supplies. */
    CHECK(heap_alloc_array_aligned(int32_t, SIZE_MAX) == nullptr,
          "array macro must refuse a count that overflows sizeof(int32_t)");
    CHECK(heap_alloc_array_aligned(int32_t, SIZE_MAX / 4 + 1) == nullptr,
          "array macro must refuse the exact overflow boundary");
    CHECK(heap_alloc_array_aligned(float, 0) == nullptr, "array macro with count 0 is null");

    /* And the valid case still allocates something writable of the right
     * size — a guard that refuses everything is not a guard. */
    int32_t *arr = heap_alloc_array_aligned(int32_t, 1024);
    CHECK(arr != nullptr, "array macro must still allocate a valid request");
    if (arr != nullptr) {
        memset(arr, 0x5A, 1024 * sizeof(int32_t)); /* ASan catches under-allocation */
        CHECK(((uintptr_t) arr & (alignof(int32_t) - 1)) == 0, "array macro alignment");
    }
    safe_free((void **) &arr);

    /* safe_free null tolerance. */
    void *nullp = nullptr;
    safe_free(&nullp);
    safe_free(nullptr);

    /* ---- heap_tl_grow (#100) ------------------------------------------- */
    for (int i = 0; i < 4; i++) {
        int       ok = 0;
        pthread_t th;
        CHECK(pthread_create(&th, nullptr, tl_grow_thread, &ok) == 0 &&
                      pthread_join(th, nullptr) == 0,
              "thread");
        CHECK(ok, "heap_tl_grow grows, keeps a big enough buffer, fails cleanly");
    }

    /* ---- heap_grow_array: keeps contents, leaves the buffer on OOM ----- */
    {
        uint32_t *g   = nullptr;
        size_t    cap = 0;
        CHECK(heap_grow_array(g, cap, 4) && g != nullptr && cap == 4, "grow from empty");
        for (uint32_t i = 0; i < 4; i++) {
            g[i] = 100 + i;
        }
        uint32_t *first = g;
        CHECK(heap_grow_array(g, cap, 3) && g == first && cap == 4, "big enough: no change");
        CHECK(heap_grow_array(g, cap, 8) && cap == 8 && g[0] == 100 && g[3] == 103,
              "grow keeps the contents");
        uint32_t *kept = g;
        heap_fail_allocations(true);
        const bool grew = heap_grow_array(g, cap, 16);
        heap_fail_allocations(false);
        CHECK(!grew && g == kept && cap == 8 && g[3] == 103, "OOM leaves buffer and cap");
        safe_free((void **) &g);
    }
    /* ---- heap_scratch_grow_array: drops the buffer on OOM -------------- */
    {
        float *s   = nullptr;
        size_t cap = 0;
        CHECK(heap_scratch_grow_array(s, cap, 8) && s != nullptr && cap == 8, "scratch grow");
        CHECK(heap_scratch_grow_array(s, cap, 4) && cap == 8, "scratch big enough");
        heap_fail_allocations(true);
        const bool grew = heap_scratch_grow_array(s, cap, 16);
        heap_fail_allocations(false);
        CHECK(!grew && s == nullptr && cap == 0, "scratch OOM: nullptr and 0, next call retries");
        CHECK(heap_scratch_grow_array(s, cap, 16) && s != nullptr && cap == 16, "scratch retry");
        CHECK(heap_grow_slow((void **) &s, &cap, 8, sizeof *s, 0, true) && cap == 16,
              "slow path called directly while big enough: no change");
        safe_free((void **) &s);
    }

    printf("PASS: heap alloc/calloc/alloc_n — zero/overflow/alignment guards, "
           "including the array macros at the SIZE_MAX boundary\n");
    return GEIST_TEST_PASS;
}
