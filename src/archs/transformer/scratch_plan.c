/*
 * src/archs/transformer/scratch_plan.c - per-session scratch sizing.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "scratch_plan.h"
#include "arch_state.h"

#include <string.h>

bool transformer_scratch_caps_ok(const struct transformer_arch_state *st) {
    for (size_t i = 0; i < (size_t) st->n_layers; i++) {
        if (st->layers[i].head_dim > GEIST_SCRATCH_HEAD_DIM_MAX ||
            st->layers[i].intermediate > GEIST_SCRATCH_INTER_MAX) {
            return false;
        }
    }
    return true;
}

void transformer_scratch_plan_build(const struct transformer_arch_state *st,
                                    struct transformer_scratch_plan     *out) {

    memset(out, 0, sizeof(*out));
    const size_t F = sizeof(float);
    const size_t M = (st->sess != NULL && st->sess->m_max > 0) ? st->sess->m_max : st->m_max;
    const size_t head_dim_max = GEIST_SCRATCH_HEAD_DIM_MAX;
    const size_t q_out_max    = st->n_q_heads * head_dim_max;
    const size_t kv_out_max   = st->n_kv_heads * head_dim_max;
    const size_t inter_max    = GEIST_SCRATCH_INTER_MAX;

    out->hidden           = M * st->d_model * F;
    out->q_out            = M * q_out_max * F;
    out->kv_out           = M * kv_out_max * F;
    out->inter            = M * inter_max * F;
    out->ple_out          = M * st->ple_out * F;
    out->hidden_per       = M * st->hidden_per_layer * F;
    out->vocab            = M * st->vocab_size * F;
    out->pool_align_slack = 21u * 64u;
    out->pool_bytes       = out->hidden /*normed*/ + out->q_out /*q*/ + out->kv_out /*k*/ +
                            out->kv_out /*v*/ + out->q_out /*attn*/ +
                            out->hidden * 10 /*o, post_attn, h_post_attn, pre_ff,
                                               ffn_out, post_ff, h_post_ff,
                                               proj_ple, h_a, h_b*/
                      + out->inter * 2 /*gate, up*/ + out->hidden_per /*gate_ple*/ +
                      out->ple_out * 2 /*ple_lookup, per_layer_input*/ + out->vocab /*logits*/ +
                      out->pool_align_slack;
}
