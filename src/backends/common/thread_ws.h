/*
 * src/backends/common/thread_ws.h — the per-thread workspace registry of
 * the CPU backends.
 *
 * Layer: BACKEND (shared by cpu_neon and cpu_x86). Concurrent sessions may
 * share one backend (geist_arch.h thread contract), so kernel scratch is
 * per calling thread: one node per (backend, thread), pushed onto the
 * backend's lock-free list on the thread's first call and freed with the
 * backend. A thread-local cache, keyed by a generation unique in the
 * process (a recycled backend address cannot match a stale entry), finds
 * the calling thread's node without walking the list.
 *
 * Each backend keeps its own `static _Thread_local struct
 * geist_thread_ws_cache` next to its accessor, so the hit path stays
 * inline in that translation unit.
 */
#pragma once

#include <pthread.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

struct geist_thread_ws_node {
    pthread_t tid;
    struct geist_thread_ws_node *_Atomic next;
    alignas(max_align_t) unsigned char ws[]; /* the backend's workspace, zeroed */
};

struct geist_thread_ws_cache {
    uint64_t generation;
    void    *ws;
};

/* A generation for a new backend state; never 0, never repeated. */
[[nodiscard]] uint64_t geist_thread_ws_next_generation(void);

/* The calling thread's workspace (ws_bytes) on the list at *head, minted
 * zeroed if it has none, and noted in *cache under `generation`. nullptr on
 * OOM of the node. */
[[nodiscard]] void *geist_thread_ws_find_or_mint(struct geist_thread_ws_node *_Atomic *head,
                                                 uint64_t                              generation,
                                                 size_t                                ws_bytes,
                                                 struct geist_thread_ws_cache         *cache);

/* geist_thread_ws_find_or_mint behind the cache hit. */
[[nodiscard]] static inline void *geist_thread_ws_get(struct geist_thread_ws_node *_Atomic *head,
                                                      uint64_t                      generation,
                                                      size_t                        ws_bytes,
                                                      struct geist_thread_ws_cache *cache) {
    if (cache->generation == generation) {
        return cache->ws;
    }
    return geist_thread_ws_find_or_mint(head, generation, ws_bytes, cache);
}

/* Detach every node from *head, hand each workspace to free_ws (which
 * frees what it points to, not the workspace itself), and free the nodes.
 * Called once, at backend destroy, when no kernel runs on it. */
void geist_thread_ws_destroy_all(struct geist_thread_ws_node *_Atomic *head,
                                 void (*free_ws)(void *ws));
