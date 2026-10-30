/*
 * src/backends/cpu_neon/workspace.c — per-thread, backend-owned kernel
 * scratch.
 *
 * One workspace per calling thread, so concurrent sessions do not race
 * (test_multi_session_parallel_int); all owned and freed by the backend,
 * so no scratch outlives a model reload. Allocation flows through heap.h.
 */

#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "heap.h"

static _Thread_local struct geist_thread_ws_cache tls_ws_cache;

struct cpu_neon_workspace *cpu_neon_ws(struct cpu_neon_state *st) {
    return geist_thread_ws_get(
            &st->ws_head, st->ws_generation, sizeof(struct cpu_neon_workspace), &tls_ws_cache);
}

static void workspace_free(void *p) {
    struct cpu_neon_workspace *ws = p;
    safe_free((void **) &ws->m1_xq);
    safe_free((void **) &ws->pq2_xp);
    safe_free((void **) &ws->pq2_tile);
    safe_free((void **) &ws->m1_bsum);
    safe_free((void **) &ws->mN_xq);
    safe_free((void **) &ws->mN_sc);
    safe_free((void **) &ws->mN_bsum);
    safe_free((void **) &ws->act_sum32);
    safe_free((void **) &ws->act_xq);
    safe_free((void **) &ws->act_scale);
    safe_free((void **) &ws->qk_mN_xq);
    safe_free((void **) &ws->qk_mN_sc);
    safe_free((void **) &ws->qk_mN_sum32);
    safe_free((void **) &ws->dequant_w_fp32);
    safe_free((void **) &ws->elt_f32);
    safe_free((void **) &ws->attn_part_out);
    safe_free((void **) &ws->attn_part_max);
    safe_free((void **) &ws->attn_part_sum);
}

void cpu_neon_ws_destroy_all(struct cpu_neon_state *st) {
    geist_thread_ws_destroy_all(&st->ws_head, workspace_free);
}
