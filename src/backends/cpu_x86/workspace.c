/*
 * src/backends/cpu_x86/workspace.c — per-thread, backend-owned kernel
 * scratch. Port of the cpu_neon mechanism (workspace.c there) that fixed
 * the concurrent-session data race test_multi_session_parallel_int
 * caught: one workspace per calling thread, all of them owned and freed
 * by the backend.
 *
 * Allocation flows through heap.h per AGENT.md memory rule.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "backend_state.h"

#include "kernel_w8a8.h" /* W8A8_BLOCK_ELEMS — smallest sum_a granularity */

#include "heap.h"

#include <stdbool.h>
#include <string.h>

/* Global generation source so a recycled cpu_x86_state address can't
 * satisfy another state's TLS cache entry. */
static _Atomic uint64_t g_ws_generation = 1;

uint64_t cpu_x86_ws_next_generation(void) {
    return atomic_fetch_add(&g_ws_generation, 1);
}

static _Thread_local struct {
    uint64_t                  generation;
    struct cpu_x86_workspace *ws;
} tls_ws_cache;

static struct cpu_x86_workspace *ws_find_or_mint(struct cpu_x86_state *st) {
    if (tls_ws_cache.generation == st->ws_generation) {
        return tls_ws_cache.ws;
    }
    const pthread_t self = pthread_self();
    for (struct cpu_x86_ws_node *n = atomic_load(&st->ws_head); n != nullptr;
         n                         = atomic_load(&n->next)) {
        if (pthread_equal(n->tid, self)) {
            tls_ws_cache.generation = st->ws_generation;
            tls_ws_cache.ws         = &n->ws;
            return &n->ws;
        }
    }
    struct cpu_x86_ws_node *node =
            heap_alloc_aligned(sizeof(*node), alignof(struct cpu_x86_ws_node));
    if (node == nullptr) {
        return nullptr;
    }
    memset(node, 0, sizeof(*node));
    node->tid                        = self;
    struct cpu_x86_ws_node *expected = atomic_load(&st->ws_head);
    do {
        atomic_store(&node->next, expected);
    } while (!atomic_compare_exchange_weak(&st->ws_head, &expected, node));
    tls_ws_cache.generation = st->ws_generation;
    tls_ws_cache.ws         = &node->ws;
    return &node->ws;
}

struct cpu_x86_workspace *cpu_x86_ws_acquire(struct cpu_x86_state *st, size_t n_in) {
    struct cpu_x86_workspace *ws = ws_find_or_mint(st);
    if (ws == nullptr) {
        return nullptr;
    }
    if (n_in <= ws->scratch_cap) {
        return ws;
    }
    int8_t *new_acts = heap_alloc_aligned(n_in * sizeof(int8_t), OPTIMAL_ALIGNMENT);
    if (new_acts == nullptr) {
        return nullptr;
    }
    /* Size sum_a for the SMALLEST block granularity — W8A8 (16) — so the
     * buffer covers both Q4_K (W4A8, 32-elem blocks) and Q6_K (W8A8)
     * callers sharing this scratch. */
    const size_t n_blocks  = n_in / W8A8_BLOCK_ELEMS;
    int32_t     *new_sum_a = heap_alloc_aligned(n_blocks * sizeof(int32_t), OPTIMAL_ALIGNMENT);
    if (new_sum_a == nullptr) {
        safe_free((void **) &new_acts);
        return nullptr;
    }
    safe_free((void **) &ws->acts_scratch);
    safe_free((void **) &ws->sum_a_scratch);
    ws->acts_scratch  = new_acts;
    ws->sum_a_scratch = new_sum_a;
    ws->scratch_cap   = n_in;
    return ws;
}

/* Grow one high-water buffer to `need` bytes. Keeps the existing pointer
 * when it already covers the request — the steady state after the first
 * prefill chunk is "no allocation at all". */
static bool ws_grow_bytes(void **p, size_t *cap, size_t need) {
    if (need == 0 || *cap >= need) {
        return true;
    }
    void *fresh = heap_alloc_aligned(need, OPTIMAL_ALIGNMENT);
    if (fresh == nullptr) {
        return false;
    }
    safe_free(p);
    *p   = fresh;
    *cap = need;
    return true;
}

struct cpu_x86_workspace *cpu_x86_ws_acquire_mN(struct cpu_x86_state *st,
                                                size_t                acts_bytes,
                                                size_t                sum_a_bytes,
                                                size_t                scale_bytes,
                                                size_t                aux_bytes) {
    struct cpu_x86_workspace *ws = ws_find_or_mint(st);
    if (ws == nullptr) {
        return nullptr;
    }
    if (!ws_grow_bytes((void **) &ws->mN_acts, &ws->mN_acts_cap, acts_bytes) ||
        !ws_grow_bytes((void **) &ws->mN_sum_a, &ws->mN_sum_a_cap, sum_a_bytes) ||
        !ws_grow_bytes((void **) &ws->mN_scale, &ws->mN_scale_cap, scale_bytes) ||
        !ws_grow_bytes((void **) &ws->mN_aux, &ws->mN_aux_cap, aux_bytes)) {
        return nullptr;
    }
    return ws;
}

void cpu_x86_ws_destroy_all(struct cpu_x86_state *st) {
    struct cpu_x86_ws_node *n = atomic_exchange(&st->ws_head, nullptr);
    while (n != nullptr) {
        struct cpu_x86_ws_node *next = atomic_load(&n->next);
        safe_free((void **) &n->ws.acts_scratch);
        safe_free((void **) &n->ws.sum_a_scratch);
        safe_free((void **) &n->ws.mN_acts);
        safe_free((void **) &n->ws.mN_sum_a);
        safe_free((void **) &n->ws.mN_scale);
        safe_free((void **) &n->ws.mN_aux);
        void *p = n;
        safe_free(&p);
        n = next;
    }
}
