/*
 * src/backends/cpu_neon/workspace.c — per-thread, backend-owned kernel
 * scratch.
 *
 * History: file-scope `_Thread_local` caches leaked peak working-set
 * across model reloads (review #4 / V12), so the scratch moved onto the
 * backend — which then made it SHARED across concurrent sessions and a
 * data race (caught by test_multi_session_parallel_int). This version
 * keeps both properties: one workspace per calling thread (isolation),
 * all of them owned and freed by the backend (lifetime).
 *
 * Allocation flows through heap.h per AGENT.md memory rule. All grow
 * helpers preserve any existing pointer when no resize is needed,
 * matching the legacy TLS grow-on-demand behavior bit-for-bit.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "heap.h"

#include <string.h>

/* Global generation source so a recycled cpu_neon_state address can't
 * satisfy another state's TLS cache entry. */
static _Atomic uint64_t g_ws_generation = 1;

uint64_t cpu_neon_ws_next_generation(void) {
    return atomic_fetch_add(&g_ws_generation, 1);
}

static _Thread_local struct {
    uint64_t                   generation;
    struct cpu_neon_workspace *ws;
} tls_ws_cache;

struct cpu_neon_workspace *cpu_neon_ws(struct cpu_neon_state *st) {
    if (tls_ws_cache.generation == st->ws_generation) {
        return tls_ws_cache.ws;
    }
    const pthread_t self = pthread_self();
    for (struct cpu_neon_ws_node *n = atomic_load(&st->ws_head); n != nullptr;
         n                          = atomic_load(&n->next)) {
        if (pthread_equal(n->tid, self)) {
            tls_ws_cache.generation = st->ws_generation;
            tls_ws_cache.ws         = &n->ws;
            return &n->ws;
        }
    }
    struct cpu_neon_ws_node *node =
            heap_alloc_aligned(sizeof(*node), alignof(struct cpu_neon_ws_node));
    if (node == nullptr) {
        return nullptr;
    }
    memset(node, 0, sizeof(*node));
    node->tid                         = self;
    struct cpu_neon_ws_node *expected = atomic_load(&st->ws_head);
    do {
        atomic_store(&node->next, expected);
    } while (!atomic_compare_exchange_weak(&st->ws_head, &expected, node));
    tls_ws_cache.generation = st->ws_generation;
    tls_ws_cache.ws         = &node->ws;
    return &node->ws;
}

void cpu_neon_ws_destroy_all(struct cpu_neon_state *st) {
    struct cpu_neon_ws_node *n = atomic_exchange(&st->ws_head, nullptr);
    while (n != nullptr) {
        struct cpu_neon_ws_node *next = atomic_load(&n->next);
        cpu_neon_workspace_destroy(&n->ws);
        void *p = n;
        safe_free(&p);
        n = next;
    }
}

void cpu_neon_workspace_destroy(struct cpu_neon_workspace *ws) {
    if (ws == nullptr) {
        return;
    }
    safe_free((void **) &ws->m1_xq);
    ws->m1_xq_cap = 0;
    safe_free((void **) &ws->m1_bsum);
    ws->m1_bsum_cap = 0;
    safe_free((void **) &ws->mN_xq);
    ws->mN_xq_cap = 0;
    safe_free((void **) &ws->mN_sc);
    ws->mN_sc_cap = 0;
    safe_free((void **) &ws->mN_bsum);
    ws->mN_bsum_cap = 0;
    safe_free((void **) &ws->qk_mN_xq);
    ws->qk_mN_xq_cap = 0;
    safe_free((void **) &ws->qk_mN_sc);
    ws->qk_mN_sc_cap = 0;
    safe_free((void **) &ws->qk_mN_sum32);
    ws->qk_mN_sum32_cap = 0;
    safe_free((void **) &ws->dequant_w_fp32);
    ws->dequant_w_fp32_cap = 0;
    safe_free((void **) &ws->ffn_gate);
    ws->ffn_gate_cap = 0;
    safe_free((void **) &ws->ffn_up);
    ws->ffn_up_cap = 0;
    safe_free((void **) &ws->ffn_mid);
    ws->ffn_mid_cap = 0;
    safe_free((void **) &ws->ffn_mid_q8);
    ws->ffn_mid_q8_cap = 0;
    safe_free((void **) &ws->ffn_mid_sc);
    ws->ffn_mid_sc_cap = 0;
    safe_free((void **) &ws->elt_f32);
    ws->elt_f32_cap = 0;
}
