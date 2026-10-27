/*
 * src/backends/common/thread_ws.c — see thread_ws.h.
 *
 * Layer: BACKEND. Allocation flows through heap.h.
 */
#include "thread_ws.h"

#include "heap.h"

#include <string.h>

static _Atomic uint64_t g_ws_generation = 1;

uint64_t geist_thread_ws_next_generation(void) {
    return atomic_fetch_add(&g_ws_generation, 1);
}

void *geist_thread_ws_find_or_mint(struct geist_thread_ws_node *_Atomic *head,
                                   const uint64_t                        generation,
                                   const size_t                          ws_bytes,
                                   struct geist_thread_ws_cache         *cache) {
    const pthread_t self = pthread_self();
    for (struct geist_thread_ws_node *n = atomic_load(head); n != nullptr;
         n                              = atomic_load(&n->next)) {
        if (pthread_equal(n->tid, self)) {
            *cache = (struct geist_thread_ws_cache) {.generation = generation, .ws = n->ws};
            return n->ws;
        }
    }
    const size_t                 bytes = sizeof(struct geist_thread_ws_node) + ws_bytes;
    struct geist_thread_ws_node *node =
            heap_alloc_aligned(bytes, alignof(struct geist_thread_ws_node));
    if (node == nullptr) {
        return nullptr;
    }
    memset(node, 0, bytes);
    node->tid                             = self;
    struct geist_thread_ws_node *expected = atomic_load(head);
    do {
        atomic_store(&node->next, expected);
    } while (!atomic_compare_exchange_weak(head, &expected, node));
    *cache = (struct geist_thread_ws_cache) {.generation = generation, .ws = node->ws};
    return node->ws;
}

void geist_thread_ws_destroy_all(struct geist_thread_ws_node *_Atomic *head,
                                 void (*free_ws)(void *ws)) {
    struct geist_thread_ws_node *n = atomic_exchange(head, nullptr);
    while (n != nullptr) {
        struct geist_thread_ws_node *next = atomic_load(&n->next);
        free_ws(n->ws);
        void *p = n;
        safe_free(&p);
        n = next;
    }
}
