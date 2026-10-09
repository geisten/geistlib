#include "heap.h"

#include "checked.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

/* See heap_alloc_count(). Relaxed: readers want a count, not ordering. */
static _Atomic uint64_t g_heap_allocs;

uint64_t heap_alloc_count(void) {
    return atomic_load_explicit(&g_heap_allocs, memory_order_relaxed);
}

/* See heap_fail_allocations(). */
static _Atomic bool g_heap_fail;

void heap_fail_allocations(bool on) {
    atomic_store_explicit(&g_heap_fail, on, memory_order_relaxed);
}

/* True iff x is a non-zero power of two. Allocation alignments must satisfy
 * this: the rounding mask ~(alignment-1) and aligned_alloc() are both
 * undefined otherwise. */
static bool size_is_pow2(const size_t x) {
    return x != 0u && (x & (x - 1u)) == 0u;
}

/* Portable aligned allocation that pairs with plain free()/safe_free().
 *
 * aligned_alloc (C11) is present on every project target — macOS >= 10.15,
 * glibc, Raspberry Pi OS — and the engine builds with -std=c23, so that is
 * the path taken in practice. The posix_memalign fallback covers hosts that
 * ship POSIX but not C11 aligned_alloc; its memory also frees with free(),
 * so safe_free() stays valid. Windows/MSVC has neither and needs the
 * _aligned_malloc/_aligned_free pair (an incompatible free path), so it is
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

void *heap_alloc_aligned(const size_t size, size_t alignment) {
    size_t aligned = 0;

    if (size == 0u || atomic_load_explicit(&g_heap_fail, memory_order_relaxed)) {
        return nullptr;
    }
    if (alignment == 0u) {
        alignment = OPTIMAL_ALIGNMENT;
    }
    if (!size_is_pow2(alignment)) {
        return nullptr;
    }
    if (alignment < OPTIMAL_ALIGNMENT) {
        alignment = OPTIMAL_ALIGNMENT;
    }
    /* aligned_alloc requires the size be a multiple of alignment; the
     * checked round-up enforces that and rejects size_t overflow. */
    if (geist_ckd_round_up_pow2(size, alignment, &aligned)) {
        return nullptr;
    }
    void *p = portable_aligned_alloc(alignment, aligned);
    if (p != nullptr) {
        atomic_fetch_add_explicit(&g_heap_allocs, 1u, memory_order_relaxed);
    }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    /* Big streaming allocations (weight repacks, lm_head blobs read every
     * token) get THP like the GGUF mmap (gguf_reader.c apply_mmap_advice):
     * fewer TLB misses in bandwidth-bound GEMVs. Advisory. */
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

void *heap_alloc_n_aligned(const size_t count, const size_t size, const size_t alignment) {
    size_t bytes = 0;
    if (count == 0u || size == 0u) {
        return nullptr;
    }
    if (ckd_mul(&bytes, count, size)) {
        return nullptr;
    }
    return heap_alloc_aligned(bytes, alignment);
}

void *heap_calloc_aligned(const size_t count, const size_t size, const size_t alignment) {
    void *memory = nullptr;
    if (count == 0u || size == 0u) {
        return nullptr;
    }
    size_t bytes = 0;
    if (ckd_mul(&bytes, count, size)) {
        return nullptr;
    }
    memory = heap_alloc_aligned(bytes, alignment);
    if (!memory) {
        return nullptr;
    }
    memset(memory, 0, bytes);
    return memory;
}

/* heap_alloc_large mappings, so safe_free can tell one from a malloc block
 * and hand it back with munmap (#733). A small array: a model holds a few
 * hundred, and only a page-aligned pointer is looked up. */
struct large_map {
    void  *p;
    size_t len;
};
static pthread_mutex_t   g_large_lock = PTHREAD_MUTEX_INITIALIZER;
static struct large_map *g_large;
static size_t            g_large_cap;
static _Atomic size_t    g_large_n;

static size_t page_bytes(void) {
    static _Atomic size_t page;
    size_t                v = atomic_load_explicit(&page, memory_order_relaxed);
    if (v == 0u) {
        const long sc = sysconf(_SC_PAGESIZE);
        v             = sc > 0 ? (size_t) sc : 4096u;
        atomic_store_explicit(&page, v, memory_order_relaxed);
    }
    return v;
}

static bool large_register(void *p, const size_t len) {
    pthread_mutex_lock(&g_large_lock);
    const size_t n  = atomic_load_explicit(&g_large_n, memory_order_relaxed);
    bool         ok = true;
    if (n == g_large_cap) {
        const size_t      cap  = g_large_cap != 0u ? 2u * g_large_cap : 256u;
        struct large_map *grow = realloc(g_large, cap * sizeof *grow);
        ok                     = grow != nullptr;
        if (ok) {
            g_large     = grow;
            g_large_cap = cap;
        }
    }
    if (ok) {
        g_large[n] = (struct large_map) {.p = p, .len = len};
        atomic_store_explicit(&g_large_n, n + 1u, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_large_lock);
    return ok;
}

/* Removes p from the registry; its mapping length, or 0 if p is not one. */
static size_t large_unregister(const void *p) {
    size_t len = 0;
    pthread_mutex_lock(&g_large_lock);
    const size_t n = atomic_load_explicit(&g_large_n, memory_order_relaxed);
    for (size_t i = 0; i < n; i++) {
        if (g_large[i].p == p) {
            len        = g_large[i].len;
            g_large[i] = g_large[n - 1u];
            atomic_store_explicit(&g_large_n, n - 1u, memory_order_relaxed);
            break;
        }
    }
    pthread_mutex_unlock(&g_large_lock);
    return len;
}

void *heap_alloc_large(const size_t size, const size_t alignment) {
    const size_t page = page_bytes();
    size_t       len  = 0;
    if (size < HEAP_LARGE_MIN || alignment > page || !size_is_pow2(page) ||
        geist_ckd_round_up_pow2(size, page, &len)) {
        return heap_alloc_aligned(size, alignment);
    }
    if ((alignment != 0u && !size_is_pow2(alignment)) ||
        atomic_load_explicit(&g_heap_fail, memory_order_relaxed)) {
        return nullptr;
    }
    void *p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    if (!large_register(p, len)) {
        (void) munmap(p, len);
        return nullptr;
    }
    atomic_fetch_add_explicit(&g_heap_allocs, 1u, memory_order_relaxed);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    /* THP as in heap_alloc_aligned; the mapping is page-aligned. */
    if (len >= (2u << 20) && getenv("GEIST_NO_HUGEPAGE") == nullptr) {
        (void) madvise(p, len, MADV_HUGEPAGE);
    }
#endif
    return p;
}

void safe_free(void **ptr) {
    if (ptr == nullptr || *ptr == nullptr) {
        return;
    }
    /* Mappings are page-aligned; anything else is a malloc block. */
    size_t len = 0;
    if (atomic_load_explicit(&g_large_n, memory_order_relaxed) != 0u &&
        ((uintptr_t) *ptr & (page_bytes() - 1u)) == 0u) {
        len = large_unregister(*ptr);
    }
    if (len != 0u) {
        (void) munmap(*ptr, len);
    } else {
        free(*ptr);
    }
    *ptr = nullptr;
}
