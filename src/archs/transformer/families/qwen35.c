/*
 * src/archs/transformer/families/qwen35.c - the Qwen3.5 / 3.6 / 3.8 hybrid unit.
 *
 * Layer: ARCHITECTURE.
 *
 * Dense hybrids (incl. Ternary-Bonsai-2-27B): most layers mix tokens with a
 * gated DeltaNet (layer.c dispatches those to the DeltaNet block), every
 * few layers is full attention with per-head QK-norm, partial RoPE and an
 * output gate (q_proj emits query and gate; attn *= sigmoid(gate)); SwiGLU
 * FFN; optional prism Hadamard rotation; trailing MTP block(s).
 *
 *   h_post_attn = h_in + o_proj(gate(attention(rope(qk_norm(qkv(norm(h_in)))))))
 *   h_out       = blk_ffn_swiglu(h_post_attn)
 *
 * The output gate and the rotation are model features the blocks apply.
 *
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk (plus the DeltaNet block and
 * the MTP drafter, which only this family uses).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../rotation.h"
#include "../weight_load.h"
#include "../weight_load/internal.h"

#include "gguf_reader.h"
#include "heap.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ---- qwen35 (Qwen3.5/3.6/3.8 hybrid) ----------------------------------- */
/*
 * Hybrid token-mixer stack: with full_attention_interval N (default 4),
 * layer i is softmax attention iff (i+1) % N == 0, else gated DeltaNet
 * (recurrent conv + delta-rule state — see forward/layer_deltanet.c).
 * The attention layers are standard GQA with QK-norm, partial NEOX RoPE
 * (rope.dimension_count of head_dim, theta 1e7) and a sigmoid output
 * gate fed by a joint q+gate projection (has_attn_output_gate).
 *
 * block_count INCLUDES nextn_predict_layers trailing MTP block(s). They
 * are not part of the autoregressive stack: n_layers excludes them and
 * n_mtp_layers records them for separate loading/execution.
 *
 * GGUF norm weights arrive with the zero-centered (+1) already baked in
 * by the converter, so the standard rmsnorm path is correct as-is.
 */
static void
populate_q35(struct gguf_ctx *gguf, const char *arch, struct transformer_arch_state *st) {
    char key[96];
#define Q35_KEY(k) (snprintf(key, sizeof key, "%s.%s", arch, k), key)
    st->config.has_qk_norms         = true;
    st->config.has_attn_output_gate = true;
    st->config.ffn_activation       = GEIST_FFN_SWIGLU;

    family_populate_common(gguf, arch, nullptr, st);
    st->vocab_size = family_vocab_size(gguf, arch, nullptr); /* no <arch>.vocab_size key */
    uint32_t nextn = 0;
    gguf_get_meta_u32(gguf, Q35_KEY("nextn_predict_layers"), &nextn);
    if (nextn < st->n_layers) {
        st->n_mtp_layers = nextn;
        st->n_layers -= nextn;
    } else if (nextn != 0) {
        /* Make malformed geometry fail in state_create instead of treating
         * MTP blocks as autoregressive layers. */
        st->n_layers     = 0;
        st->n_mtp_layers = nextn;
    }
    /* DeltaNet geometry. GGUF key mapping (#281): state_size =
     * head_k (= head_v), group_count = n_k_heads, time_step_rank =
     * n_v_heads, inner_size = n_v_heads * head_v. */
    uint32_t u;
    if (gguf_get_meta_u32(gguf, Q35_KEY("ssm.group_count"), &u))
        st->config.dn_n_k_heads = u;
    if (gguf_get_meta_u32(gguf, Q35_KEY("ssm.time_step_rank"), &u))
        st->config.dn_n_v_heads = u;
    if (gguf_get_meta_u32(gguf, Q35_KEY("ssm.state_size"), &u))
        st->config.dn_head_k = u;
    if (gguf_get_meta_u32(gguf, Q35_KEY("ssm.inner_size"), &u) && st->config.dn_n_v_heads > 0)
        st->config.dn_head_v = u / st->config.dn_n_v_heads;
    if (gguf_get_meta_u32(gguf, Q35_KEY("ssm.conv_kernel"), &u))
        st->config.dn_conv_kernel = u;
#undef Q35_KEY
}

static void populate_qwen35(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    populate_q35(gguf, "qwen35", st);
}

/* `ff_key`: the FFN width's key, the dense FFN's or one expert's. */
static bool
populate_layers_q35(struct transformer_arch_state *st, const char *arch, const char *ff_key) {
    char key[96];
#define Q35_KEY(k) (snprintf(key, sizeof key, "%s.%s", arch, k), key)
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    uint32_t head_dim = 0, intermediate = 0, rot = 0, interval = 4;
    float    freq_base = 10000000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, Q35_KEY("attention.key_length"), &head_dim);
        gguf_get_meta_u32(g, Q35_KEY(ff_key), &intermediate);
        gguf_get_meta_u32(g, Q35_KEY("rope.dimension_count"), &rot);
        gguf_get_meta_u32(g, Q35_KEY("full_attention_interval"), &interval);
        gguf_get_meta_f32(g, Q35_KEY("rope.freq_base"), &freq_base);
    }
    if (head_dim == 0 || intermediate == 0 || interval == 0 || rot == 0 || rot > head_dim ||
        st->config.dn_n_k_heads == 0 || st->config.dn_n_v_heads == 0 || st->config.dn_head_k == 0 ||
        st->config.dn_head_v == 0 || st->config.dn_conv_kernel < 2) {
        return false;
    }
    /* Optional explicit bool array overrides the interval formula. */
    uint32_t       rec_vt = 0;
    uint64_t       rec_n  = 0;
    const uint8_t *rec    = nullptr;
    if (g != nullptr &&
        gguf_get_meta_array_info(g, Q35_KEY("attention.recurrent_layers"), &rec_vt, &rec_n, &rec)) {
        if (rec_vt != GGUF_META_VT_BOOL || rec_n < st->n_layers)
            rec = nullptr;
    }

    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        st->config.rope_partial_block       = true; /* ggml layout, see arch_config.h */
        const bool is_attn = rec != nullptr ? rec[i] == 0 : ((i + 1) % (size_t) interval == 0);
        family_uniform_layer(st, i, head_dim, intermediate, freq_base, (int) rot, L);
        L->mixer = is_attn ? GEIST_MIXER_ATTN : GEIST_MIXER_DELTANET;
    }
    for (size_t i = 0; i < st->n_mtp_layers; i++) {
        family_uniform_layer(st,
                             st->n_layers + i,
                             head_dim,
                             intermediate,
                             freq_base,
                             (int) rot,
                             &st->mtp_layers[i].block);
    }
    return true;
#undef Q35_KEY
}

static bool populate_layers_qwen35(struct transformer_arch_state *st) {
    return populate_layers_q35(st, "qwen35", "feed_forward_length");
}

/* ---- Tensors ---------------------------------------------------------- */

/* A gated-DeltaNet layer's own tensors (the dn_* group). attn_norm and the
 * dense FFN are the generic set's. */
[[nodiscard]] static enum geist_status wire_deltanet(struct transformer_arch_state    *st,
                                                     struct gguf_ctx                  *gguf,
                                                     struct transformer_layer_weights *L) {
    const size_t d         = st->d_model;
    const size_t n_v       = st->config.dn_n_v_heads;
    const size_t key_dim   = st->config.dn_n_k_heads * st->config.dn_head_k;
    const size_t value_dim = n_v * st->config.dn_head_v;
    const size_t conv_dim  = 2 * key_dim + value_dim;
    const struct {
        const char          *suffix;
        size_t               n_out, n_in; /* n_in 0: a norm-like vector of n_out */
        struct geist_tensor *view;
        struct geist_weight *w;
    } tensors[] = {
            {"attn_qkv.weight", conv_dim, d, &L->dn_qkv, &L->dn_qkv_w},
            {"attn_gate.weight", value_dim, d, &L->dn_z, &L->dn_z_w},
            {"ssm_conv1d.weight", conv_dim * st->config.dn_conv_kernel, 0, &L->dn_conv, nullptr},
            {"ssm_beta.weight", n_v, d, &L->dn_beta, &L->dn_beta_w},
            {"ssm_alpha.weight", n_v, d, &L->dn_alpha, &L->dn_alpha_w},
            {"ssm_a", n_v, 0, &L->dn_a, nullptr},
            {"ssm_dt.bias", n_v, 0, &L->dn_dt_bias, nullptr},
            {"ssm_norm.weight", st->config.dn_head_v, 0, &L->dn_norm, nullptr},
            {"ssm_out.weight", d, value_dim, &L->dn_out, &L->dn_out_w},
    };
    char name[64];
    for (size_t i = 0; i < sizeof tensors / sizeof tensors[0]; i++) {
        snprintf(name, sizeof name, "blk.%d.%s", L->layer_idx, tensors[i].suffix);
        enum geist_status s =
                tensors[i].n_in == 0
                        ? load_layer_norm(st, gguf, L, name, tensors[i].n_out, tensors[i].view)
                        : load_layer_matrix(st,
                                            gguf,
                                            L,
                                            name,
                                            tensors[i].n_out,
                                            tensors[i].n_in,
                                            false,
                                            tensors[i].view,
                                            tensors[i].w);
        if (s != GEIST_OK)
            return s;
    }
    return GEIST_OK;
}

/* The pre-FFN norm under its HF name, post_attention_norm (the qwen35
 * converter writes no ffn_norm; an attention layer takes ffn_norm where a
 * GGUF has one); then a DeltaNet layer's mixer, or an attention layer's
 * q_proj, which jointly produces query and gate: 2x rows, per head
 * [query(hd) | gate(hd)] (has_attn_output_gate). Also the MTP block's. */
[[nodiscard]] static enum geist_status qwen35_wire_layer(struct transformer_arch_state    *st,
                                                         struct gguf_ctx                  *gguf,
                                                         struct transformer_layer_weights *L) {
    const bool dn = L->mixer == GEIST_MIXER_DELTANET;
    char       name[64];
    snprintf(name, sizeof name, "blk.%d.ffn_norm.weight", L->layer_idx);
    if (dn || gguf_get_tensor(gguf, name) == nullptr) {
        snprintf(name, sizeof name, "blk.%d.post_attention_norm.weight", L->layer_idx);
    }
    enum geist_status s = load_layer_norm(st, gguf, L, name, st->d_model, &L->ffn_norm);
    if (s != GEIST_OK || dn) {
        return s == GEIST_OK ? wire_deltanet(st, gguf, L) : s;
    }
    snprintf(name, sizeof name, "blk.%d.attn_q.weight", L->layer_idx);
    return load_layer_matrix(
            st, gguf, L, name, 2 * L->q_out, st->d_model, false, &L->q_proj, &L->q_proj_w);
}

/* ---- Session state ---------------------------------------------------- */

/* Gated-DeltaNet recurrent state: backend buffers per DELTANET layer —
 * conv history (kernel-1 x conv_dim) + delta state S (n_v_heads x head_k x
 * head_v). Zero-initialized = empty sequence. */
[[nodiscard]] static enum geist_status alloc_deltanet_state(struct transformer_arch_session *sess,
                                                            size_t                           n_dn) {
    const struct transformer_arch_state *st        = sess->model;
    struct geist_backend                *be        = st->backend;
    const size_t                         key_dim   = st->config.dn_n_k_heads * st->config.dn_head_k;
    const size_t                         value_dim = st->config.dn_n_v_heads * st->config.dn_head_v;
    const size_t                         conv_dim  = 2 * key_dim + value_dim;
    const size_t                         conv_n    = (st->config.dn_conv_kernel - 1) * conv_dim;
    const size_t s_n    = st->config.dn_n_v_heads * st->config.dn_head_k * st->config.dn_head_v;
    sess->dn_conv_state = heap_alloc_aligned(st->n_layers * sizeof(*sess->dn_conv_state), 64);
    sess->dn_S          = heap_alloc_aligned(st->n_layers * sizeof(*sess->dn_S), 64);
    if (sess->dn_conv_state == nullptr || sess->dn_S == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "transformer: dn state alloc failed");
        return GEIST_E_OOM;
    }
    memset(sess->dn_conv_state, 0, st->n_layers * sizeof(*sess->dn_conv_state));
    memset(sess->dn_S, 0, st->n_layers * sizeof(*sess->dn_S));
    enum geist_status s;
    for (size_t li = 0; li < st->n_layers; li++) {
        if (st->layers[li].mixer != GEIST_MIXER_DELTANET)
            continue;
        s = be->desc->vtbl->buffer_create(be,
                                          conv_n * sizeof(float),
                                          GEIST_BUFFER_KV_CACHE,
                                          GEIST_MEMORY_AUTO,
                                          &sess->dn_conv_state[li]);
        if (s == GEIST_OK) {
            s = be->desc->vtbl->buffer_create(be,
                                              s_n * sizeof(float),
                                              GEIST_BUFFER_KV_CACHE,
                                              GEIST_MEMORY_AUTO,
                                              &sess->dn_S[li]);
        }
        if (s != GEIST_OK || sess->dn_conv_state[li] == nullptr || sess->dn_S[li] == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "transformer: dn state alloc failed");
            return GEIST_E_OOM;
        }
        s = transformer_dn_state_zero(be, sess->dn_conv_state[li], conv_n * sizeof(float));
        if (s == GEIST_OK) {
            s = transformer_dn_state_zero(be, sess->dn_S[li], s_n * sizeof(float));
        }
        if (s != GEIST_OK) {
            geist_backend_set_error(be, s, "transformer: dn state init failed");
            return s;
        }
    }
    /* Only the host mixer reads the state as dn_fresh says; a backend's
     * deltanet_mix gets buffers that hold the zeros. */
    if (geist_backend_fused_tbl(be)->deltanet_mix == nullptr) {
        sess->dn_fresh = heap_alloc_array_aligned(bool, st->n_layers);
        if (sess->dn_fresh == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "transformer: dn state alloc failed");
            return GEIST_E_OOM;
        }
        for (size_t li = 0; li < st->n_layers; li++)
            sess->dn_fresh[li] = sess->dn_S[li] != nullptr;
    }
    /* Speculative-decode transaction snapshot (#463): one backend buffer per
     * state kind, all n_dn layers concatenated, allocated once here rather
     * than lazily on the first verify. On-device (vtbl->buffer_copy) so a
     * snapshot/restore never visits host memory on batched-submit GPUs. */
    s = transformer_scratch_alloc(be, n_dn * conv_n * sizeof(float), &sess->dn_txn_conv_buf);
    if (s == GEIST_OK) {
        s = transformer_scratch_alloc(be, n_dn * s_n * sizeof(float), &sess->dn_txn_S_buf);
    }
    if (s != GEIST_OK) {
        geist_backend_set_error(be, s, "transformer: DeltaNet transaction alloc failed");
    }
    return s;
}

/* Scratch slabs of their own (not pool slots, see arch_state.h): the joint
 * q+gate projection and the saved gate of the attention layers, and the
 * DeltaNet projection outputs when the model has DeltaNet layers. */
[[nodiscard]] static enum geist_status alloc_qwen35_scratch(struct transformer_arch_session *sess,
                                                            size_t                           n_dn) {
    const struct transformer_arch_state *st        = sess->model;
    struct geist_backend                *be        = st->backend;
    const size_t                         m         = sess->m_max;
    const size_t                         key_dim   = st->config.dn_n_k_heads * st->config.dn_head_k;
    const size_t                         value_dim = st->config.dn_n_v_heads * st->config.dn_head_v;
    size_t                               q_out_max = 0;
    for (size_t li = 0; li < st->n_layers; li++)
        if (st->layers[li].q_out > q_out_max)
            q_out_max = st->layers[li].q_out;
    const struct {
        bool                  want;
        size_t                floats;
        struct geist_buffer **out;
    } bufs[] = {
            {true, m * 2 * q_out_max, &sess->qgate_joint},
            {true, m * q_out_max, &sess->qgate_gate},
            {n_dn > 0, m * (2 * key_dim + value_dim), &sess->dn_scratch_qkv},
            {n_dn > 0, m * value_dim, &sess->dn_scratch_z},
            {n_dn > 0, m * st->config.dn_n_v_heads, &sess->dn_scratch_b},
            {n_dn > 0, m * st->config.dn_n_v_heads, &sess->dn_scratch_a},
    };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (!bufs[i].want)
            continue;
        enum geist_status s =
                transformer_scratch_alloc(be, bufs[i].floats * sizeof(float), bufs[i].out);
        if (s != GEIST_OK)
            return s;
    }
    return GEIST_OK;
}

/* The MTP attention block is not part of the target stack and must not
 * alias its cache. One dense cache is modest compared with the 27B weights
 * and gives the draft head an unambiguous correctness baseline before
 * optional cache quantization is considered. */
[[nodiscard]] static enum geist_status alloc_mtp_state(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st = sess->model;
    struct geist_backend                *be = st->backend;
    const size_t kv_elems = sess->max_seq_len * st->n_kv_heads * st->mtp_layers[0].block.head_dim;
    const size_t hidden_bytes = sess->m_max * st->d_model * sizeof(float);
    enum geist_status s       = be->desc->vtbl->buffer_create(be,
                                                              kv_elems * sizeof(float),
                                                              GEIST_BUFFER_KV_CACHE,
                                                              GEIST_MEMORY_AUTO,
                                                              &sess->mtp_k_cache);
    if (s == GEIST_OK)
        s = be->desc->vtbl->buffer_create(be,
                                          kv_elems * sizeof(float),
                                          GEIST_BUFFER_KV_CACHE,
                                          GEIST_MEMORY_AUTO,
                                          &sess->mtp_v_cache);
    if (s == GEIST_OK)
        s = transformer_scratch_alloc(be, hidden_bytes, &sess->mtp_embed);
    if (s == GEIST_OK)
        s = transformer_scratch_alloc(be, hidden_bytes, &sess->mtp_hidden_norm);
    if (s == GEIST_OK)
        s = transformer_scratch_alloc(be, 2 * hidden_bytes, &sess->mtp_concat);
    if (s != GEIST_OK)
        return s;
    sess->mtp_pending_h      = heap_alloc_aligned(st->d_model * sizeof(float), 64);
    sess->mtp_target_raw     = heap_alloc_aligned(sess->m_max * st->d_model * sizeof(float), 64);
    sess->mtp_target_shifted = heap_alloc_aligned(sess->m_max * st->d_model * sizeof(float), 64);
    sess->mtp_txn_pending_h  = heap_alloc_aligned(st->d_model * sizeof(float), 64);
    if (sess->mtp_pending_h == nullptr || sess->mtp_target_raw == nullptr ||
        sess->mtp_target_shifted == nullptr || sess->mtp_txn_pending_h == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "transformer: MTP host state alloc failed");
        return GEIST_E_OOM;
    }
    memset(sess->mtp_pending_h, 0, st->d_model * sizeof(float));
    memset(sess->mtp_txn_pending_h, 0, st->d_model * sizeof(float));
    sess->mtp_kv_len = 0;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status qwen35_session_alloc(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st   = sess->model;
    size_t                               n_dn = 0;
    for (size_t li = 0; li < st->n_layers; li++)
        if (st->layers[li].mixer == GEIST_MIXER_DELTANET)
            n_dn++;
    enum geist_status s = n_dn > 0 ? alloc_deltanet_state(sess, n_dn) : GEIST_OK;
    if (s == GEIST_OK)
        s = alloc_qwen35_scratch(sess, n_dn);
    if (s == GEIST_OK && st->n_mtp_layers > 0)
        s = alloc_mtp_state(sess);
    return s;
}

/* Destroy every set buffer of a per-layer array and free the array. */
static void
free_layer_buffers(struct geist_backend *be, size_t n_layers, struct geist_buffer ***arr) {
    if (*arr == nullptr)
        return;
    for (size_t li = 0; li < n_layers; li++)
        transformer_buffer_release(be, &(*arr)[li]);
    void *p = *arr;
    safe_free(&p);
    *arr = nullptr;
}

static void qwen35_session_free(struct transformer_arch_session *sess) {
    struct geist_backend *be = sess->model->backend;
    free_layer_buffers(be, sess->model->n_layers, &sess->dn_conv_state);
    free_layer_buffers(be, sess->model->n_layers, &sess->dn_S);
    sess->dn_prefill_ws_floats = 0;
    void *host[]               = {
            sess->dn_prefill_ws,
            sess->dn_fresh,
            sess->dn_txn_ids,
            sess->mtp_pending_h,
            sess->mtp_target_raw,
            sess->mtp_target_shifted,
            sess->mtp_txn_pending_h,
    };
    for (size_t i = 0; i < sizeof host / sizeof host[0]; i++)
        safe_free(&host[i]);
    sess->dn_prefill_ws          = nullptr;
    sess->dn_fresh               = nullptr;
    sess->dn_txn_ids             = nullptr;
    sess->mtp_pending_h          = nullptr;
    sess->mtp_target_raw         = nullptr;
    sess->mtp_target_shifted     = nullptr;
    sess->mtp_txn_pending_h      = nullptr;
    struct geist_buffer **bufs[] = {
            &sess->qgate_joint,
            &sess->qgate_gate,
            &sess->dn_scratch_qkv,
            &sess->dn_scratch_z,
            &sess->dn_scratch_b,
            &sess->dn_scratch_a,
            &sess->dn_txn_conv_buf,
            &sess->dn_txn_S_buf,
            &sess->mtp_k_cache,
            &sess->mtp_v_cache,
            &sess->mtp_embed,
            &sess->mtp_hidden_norm,
            &sess->mtp_concat,
    };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        transformer_buffer_release(be, bufs[i]);
}

/* ---- Forward ---------------------------------------------------------- */

/* Each step runs only while s is GEIST_OK, so the first failure is what
 * the stage returns and nothing after it touches a buffer. Also the MTP
 * block's attention (forward/layer.c, with an empty plan). */
[[nodiscard]] static enum geist_status
qwen35_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const size_t                      d    = st->d_model;
    struct geist_buffer              *x    = sess->scratch_normed;

    enum geist_status s = blk_rmsnorm(ctx, d, ctx->h_in_buf, &L->attn_norm, x);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);
    if (s == GEIST_OK)
        s = blk_qkv(ctx, x); /* joint q+gate, split */
    if (s == GEIST_OK)
        s = blk_qk_norm(ctx, &L->q_norm, &L->k_norm);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = blk_o_proj(ctx); /* sigmoid gate, rotation */
    if (s == GEIST_OK)
        s = blk_add(ctx, d, ctx->h_in_buf, sess->scratch_o, sess->scratch_h_post_attn);
    return s;
}

/* ---- Registry entry --------------------------------------------------- */

const struct transformer_family transformer_family_qwen35 = {
        .name     = "qwen35",
        .rotation = true,
        /* 27B weights (Ternary-Bonsai-2-27B, Qwen3.5-27B) dequantized once
         * per chunk are the prefill's cost: twice the chunk from 4 GiB on
         * (256 is no better than 128). */
        .prefill_delta   = {.from_bytes = (size_t) 4 << 30, .m_max_rows = 64},
        .populate        = populate_qwen35,
        .populate_layers = populate_layers_qwen35,
        .wire_layer      = qwen35_wire_layer,
        .session_alloc   = qwen35_session_alloc,
        .session_free    = qwen35_session_free,
        .attention_block = qwen35_attention_block,
        .ffn_block       = blk_ffn_swiglu,
};

/* ---- qwen35moe (Qwen3.5/3.6 MoE hybrid, #169) ------------------------- */
/*
 * The qwen35 stack with blk_moe_ffn in place of the dense FFN: a softmax
 * router over expert_count experts of width expert_feed_forward_length,
 * the expert_used_count best renormalised (expert_weights_norm, default
 * true, as llama.cpp builds it), plus a shared expert every token runs,
 * weighted sigmoid(x . ffn_gate_inp_shexp). The shared expert must have
 * the routed experts' width: blk_moe_ffn runs it as one more expert.
 */
static void populate_qwen35moe(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    populate_q35(gguf, "qwen35moe", st);
    st->config.moe_norm_topk = true;
    uint32_t u;
    bool     b;
    if (gguf_get_meta_u32(gguf, "qwen35moe.expert_count", &u))
        st->config.moe_n_experts = u;
    if (gguf_get_meta_u32(gguf, "qwen35moe.expert_used_count", &u))
        st->config.moe_n_used = u;
    if (gguf_get_meta_bool(gguf, "qwen35moe.expert_weights_norm", &b))
        st->config.moe_norm_topk = b;
    /* A shared expert by its key or, without one, by its tensors: never
     * dropped silently (its width then is the experts', which the load's
     * shape check holds it to). */
    st->config.moe_n_shared =
            (gguf_get_meta_u32(gguf, "qwen35moe.expert_shared_feed_forward_length", &u) && u > 0) ||
            gguf_get_tensor(gguf, "blk.0.ffn_gate_inp_shexp.weight") != nullptr;
}

static bool populate_layers_qwen35moe(struct transformer_arch_state *st) {
    struct gguf_ctx                *g  = (struct gguf_ctx *) st->gguf;
    const struct geist_arch_config *c  = &st->config;
    uint32_t                        ff = 0, shared_ff = 0;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "qwen35moe.expert_feed_forward_length", &ff);
        if (!gguf_get_meta_u32(g, "qwen35moe.expert_shared_feed_forward_length", &shared_ff))
            shared_ff = ff;
    }
    if (c->moe_n_experts == 0 || c->moe_n_used == 0 || c->moe_n_used > c->moe_n_experts ||
        moe_slots(c) > GEIST_MOE_MAX_EXPERTS || (c->moe_n_shared > 0 && shared_ff != ff)) {
        return false;
    }
    return populate_layers_q35(st, "qwen35moe", "expert_feed_forward_length");
}

[[nodiscard]] static enum geist_status qwen35moe_wire_layer(struct transformer_arch_state    *st,
                                                            struct gguf_ctx                  *gguf,
                                                            struct transformer_layer_weights *L) {
    enum geist_status s = qwen35_wire_layer(st, gguf, L);
    return s == GEIST_OK ? moe_wire_layer(st, gguf, L, "qwen35moe") : s;
}

[[nodiscard]] static enum geist_status
qwen35moe_session_alloc(struct transformer_arch_session *sess) {
    enum geist_status s = qwen35_session_alloc(sess);
    return s == GEIST_OK ? moe_session_alloc(sess) : s;
}

static void qwen35moe_session_free(struct transformer_arch_session *sess) {
    moe_session_free(sess);
    qwen35_session_free(sess);
}

const struct transformer_family transformer_family_qwen35moe = {
        .name            = "qwen35moe",
        .rotation        = true,
        .populate        = populate_qwen35moe,
        .populate_layers = populate_layers_qwen35moe,
        .wire_layer      = qwen35moe_wire_layer,
        .session_alloc   = qwen35moe_session_alloc,
        .session_free    = qwen35moe_session_free,
        .attention_block = qwen35_attention_block,
        .ffn_block       = blk_moe_ffn,
        /* As qwen3moe: an MoE amplifies KV quantization error (#48, #53). */
        .kv_auto         = GEIST_KV_F16,
        .kv_int4_instead = GEIST_KV_F16,
};
