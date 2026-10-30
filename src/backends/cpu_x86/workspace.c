/*
 * src/backends/cpu_x86/workspace.c — per-thread, backend-owned kernel
 * scratch, as in cpu_neon: one workspace per calling thread
 * (common/thread_ws.h), so concurrent sessions on one backend do not share
 * buffers; the backend frees them all.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "backend_state.h"

#include "kernel_w8a8.h" /* W8A8_BLOCK_ELEMS — smallest sum_a granularity */

#include "checked.h"
#include "heap.h"

#include <stdbool.h>

static _Thread_local struct geist_thread_ws_cache tls_ws_cache;

static inline struct cpu_x86_workspace *ws_get(struct cpu_x86_state *st) {
    return geist_thread_ws_get(
            &st->ws_head, st->ws_generation, sizeof(struct cpu_x86_workspace), &tls_ws_cache);
}

/* `bytes` of scratch at *buf, contents dropped when it grows. */
static inline bool ws_grow(void **buf, size_t *cap, size_t bytes) {
    return heap_scratch_grow(buf, cap, bytes, 1, OPTIMAL_ALIGNMENT);
}

struct cpu_x86_workspace *cpu_x86_ws_acquire(struct cpu_x86_state *st, size_t n_in) {
    struct cpu_x86_workspace *ws = ws_get(st);
    /* sum_a is sized for the SMALLEST block granularity — W8A8 (16) — so
     * the buffer covers both Q4_K (W4A8, 32-elem blocks) and Q6_K (W8A8)
     * callers sharing this scratch. */
    if (ws == nullptr || !ws_grow((void **) &ws->acts_scratch, &ws->scratch_cap, n_in) ||
        !ws_grow((void **) &ws->sum_a_scratch,
                 &ws->sum_a_cap,
                 n_in / W8A8_BLOCK_ELEMS * sizeof(int32_t))) {
        return nullptr;
    }
    return ws;
}

/* High-water byte buffers: after the first prefill chunk, no allocation. */
struct cpu_x86_workspace *cpu_x86_ws_acquire_mN(struct cpu_x86_state *st,
                                                size_t                acts_bytes,
                                                size_t                sum_a_bytes,
                                                size_t                scale_bytes,
                                                size_t                aux_bytes) {
    struct cpu_x86_workspace *ws = ws_get(st);
    if (ws == nullptr || !ws_grow((void **) &ws->mN_acts, &ws->mN_acts_cap, acts_bytes) ||
        !ws_grow((void **) &ws->mN_sum_a, &ws->mN_sum_a_cap, sum_a_bytes) ||
        !ws_grow((void **) &ws->mN_scale, &ws->mN_scale_cap, scale_bytes) ||
        !ws_grow((void **) &ws->mN_aux, &ws->mN_aux_cap, aux_bytes)) {
        return nullptr;
    }
    return ws;
}

/* The calling thread's workspace with *buf (one of its fields) covering
 * `bytes`, zeroed when it grows. */
static struct cpu_x86_workspace *
acquire_zeroed(struct cpu_x86_workspace *ws, void **buf, size_t *cap, size_t bytes) {
    if (*cap < bytes) {
        void *fresh = heap_calloc_aligned(1, bytes, OPTIMAL_ALIGNMENT);
        if (fresh == nullptr) {
            return nullptr;
        }
        safe_free(buf);
        *buf = fresh;
        *cap = bytes;
    }
    return ws;
}

struct cpu_x86_workspace *cpu_x86_ws_acquire_tiled(struct cpu_x86_state *st, size_t bytes) {
    struct cpu_x86_workspace *ws = ws_get(st);
    return ws == nullptr ? nullptr : acquire_zeroed(ws, &ws->tiled, &ws->tiled_cap, bytes);
}

struct cpu_x86_workspace *cpu_x86_ws_acquire_tiled_acts(struct cpu_x86_state *st, size_t bytes) {
    struct cpu_x86_workspace *ws = ws_get(st);
    return ws == nullptr ? nullptr
                         : acquire_zeroed(ws, &ws->tiled_acts, &ws->tiled_acts_cap, bytes);
}

float *cpu_x86_ws_attn_part(struct cpu_x86_state *st, size_t floats) {
    struct cpu_x86_workspace *ws    = ws_get(st);
    size_t                    bytes = 0;
    if (ws == nullptr || ckd_mul(&bytes, floats, sizeof(float)) ||
        !ws_grow((void **) &ws->attn_part, &ws->attn_part_cap, bytes)) {
        return nullptr;
    }
    return ws->attn_part;
}

static void workspace_free(void *p) {
    struct cpu_x86_workspace *ws = p;
    safe_free((void **) &ws->acts_scratch);
    safe_free((void **) &ws->sum_a_scratch);
    safe_free((void **) &ws->mN_acts);
    safe_free(&ws->tiled);
    safe_free(&ws->tiled_acts);
    safe_free((void **) &ws->mN_sum_a);
    safe_free((void **) &ws->mN_scale);
    safe_free((void **) &ws->mN_aux);
    safe_free((void **) &ws->attn_part);
}

void cpu_x86_ws_destroy_all(struct cpu_x86_state *st) {
    geist_thread_ws_destroy_all(&st->ws_head, workspace_free);
}
