/*
 * src/backends/cpu_neon/workspace.c — backend-owned kernel scratch.
 *
 * Replaces the file-scope `_Thread_local` caches that previously lived
 * through the OpenMP runtime's lifetime, leaking peak working-set
 * across model reloads. Each slot grows on demand from inside its
 * kernel; cpu_neon_workspace_destroy() at backend teardown frees
 * everything in one pass with no OMP barrier.
 *
 * Allocation flows through heap.h per AGENT.md memory rule. All grow
 * helpers preserve any existing pointer when no resize is needed,
 * matching the legacy TLS grow-on-demand behavior bit-for-bit.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "heap.h"

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
    safe_free((void **) &ws->attn_scores);
    ws->attn_scores_cap = 0;
}
