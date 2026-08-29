/*
 * src/archs/transformer/weight_load/layer_wiring.c — per-layer +
 * global tensor wiring.
 *
 * Layer: ARCHITECTURE.
 *
 * Extracted from weight_load.c during R5 of the C23/AGENT.md cleanup.
 * Contains:
 *
 *   layer_track_buf  / global_track_buf — owning-buf list helpers
 *   load_layer_norm  / load_layer_proj  / load_layer_scalar — per-tensor
 *                      load wrappers (all static).
 *   load_one_layer   — wire all tensors for one transformer block.
 *   load_globals     — wire embed/output/PLE tables + global norms.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../weight_load.h"

#include "gguf_dequant.h"
#include "gguf_reader.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int layer_track_buf(struct transformer_layer_weights *L, struct geist_buffer *buf) {
    if (L->n_bufs >= sizeof L->bufs / sizeof L->bufs[0]) {
        return -1;
    }
    L->bufs[L->n_bufs++] = buf;
    return 0;
}

static int global_track_buf(struct transformer_arch_state *st, struct geist_buffer *buf) {
    if (st->n_global_bufs >= sizeof st->global_bufs / sizeof st->global_bufs[0]) {
        return -1;
    }
    st->global_bufs[st->n_global_bufs++] = buf;
    return 0;
}

/* Load a per-layer norm (F32, expected_elems shape). Stores into *out_view
 * and tracks owning buffer in L->bufs. */
[[nodiscard]] static enum geist_status load_layer_norm(struct transformer_arch_state    *st,
                                                       struct gguf_ctx                  *gguf,
                                                       struct transformer_layer_weights *L,
                                                       const char                       *name,
                                                       size_t               expected_elems,
                                                       struct geist_tensor *out_view) {

    struct geist_backend       *be  = st->backend;
    const struct gguf_tensor_t *t   = nullptr;
    struct geist_buffer        *buf = nullptr;
    enum geist_status           s = load_tensor_to_buffer(st, gguf, name, expected_elems, &t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    if (t->dtype != GGUF_TYPE_F32) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: '%s' expected F32, got %s",
                                name,
                                gguf_dtype_name(t->dtype));
        return GEIST_E_FORMAT;
    }
    *out_view = make_view_1d(buf, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, (int64_t) expected_elems);
    if (layer_track_buf(L, buf) != 0) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(
                be, GEIST_E_INTERNAL, "transformer: layer buffer list overflow on '%s'", name);
        return GEIST_E_INTERNAL;
    }
    return GEIST_OK;
}

/* Load a per-layer 2D projection. dtype may be quantized or F32. */
[[nodiscard]] static enum geist_status
load_layer_proj(struct transformer_arch_state    *st,
                struct gguf_ctx                  *gguf,
                struct transformer_layer_weights *L,
                const char                       *name,
                size_t                            n_out,
                size_t                            n_in,
                struct geist_tensor              *out_view,
                struct geist_weight              *out_weight /* nullable */) {

    struct geist_backend       *be  = st->backend;
    const struct gguf_tensor_t *t   = nullptr;
    struct geist_buffer        *buf = nullptr;
    enum geist_status           s   = load_tensor_to_buffer(st, gguf, name, n_out * n_in, &t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    struct dtype_map_entry dm = map_gguf_dtype(t->dtype);
    if (!dm.supported) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "transformer: '%s' has unsupported dtype %s",
                                name,
                                gguf_dtype_name(t->dtype));
        return GEIST_E_UNSUPPORTED;
    }
    *out_view = make_view_2d(buf, dm.dtype, dm.layout, (int64_t) n_out, (int64_t) n_in);
    if (layer_track_buf(L, buf) != 0) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(
                be, GEIST_E_INTERNAL, "transformer: layer buffer list overflow on '%s'", name);
        return GEIST_E_INTERNAL;
    }
    /* P1.1.c (refactor v2): also populate the pre-resolved kernel
     * pointer table when caller asks for it. Falls back to legacy
     * v->linear() at the call site when resolve_weight returns
     * unsupported (e.g. Q5_K, F32 dense).
     *
     * Buffer's host pointer is exposed via buffer_map; struct geist_buffer
     * is opaque to the arch layer (the full definition lives in each
     * backend's internal.h). This is load-time, called once per weight,
     * so the buffer_map indirection cost is irrelevant. */
    if (out_weight != nullptr) {
        const struct geist_backend_vtbl *v    = be->desc->vtbl;
        void                            *host = v->buffer_map(buf);
        if (host == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "transformer: buffer_map(%s) returned null", name);
            return GEIST_E_BACKEND;
        }
        *out_weight = (struct geist_weight) {
                .raw        = (const uint8_t *) host + out_view->offset,
                .raw_nbytes = t->nbytes - (size_t) out_view->offset,
                .n_in       = (int32_t) n_in,
                .n_out      = (int32_t) n_out,
                .dtype      = (uint16_t) dm.dtype,
        };
        v->buffer_unmap(buf);
        if (v->resolve_weight != nullptr) {
            enum geist_status rs = v->resolve_weight(be, out_weight);
            if (rs != GEIST_OK && rs != GEIST_E_UNSUPPORTED) {
                /* Real failure — propagate. Unsupported is fine; caller
                 * checks linear_m1 nullness and falls back. */
                return rs;
            }
            /* On UNSUPPORTED, linear_m1 / linear_mN stay null — that's
             * the "use legacy" signal. */
        }
    }
    return GEIST_OK;
}

/* Read the layer's per-layer output scalar (1-element F32 tensor). The
 * scalar is loaded into the state struct directly, not as a buffer — it's
 * one float and gets folded into the per-layer output multiply. */
[[nodiscard]] static enum geist_status load_layer_scalar(struct geist_backend *be,
                                                         struct gguf_ctx      *gguf,
                                                         int                   layer_idx,
                                                         float                *out_scalar) {

    char name[64];
    snprintf(name, sizeof name, "blk.%d.layer_output_scale.weight", layer_idx);
    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, name);
    if (t == nullptr || t->dtype != GGUF_TYPE_F32 || gguf_tensor_elem_count(t) != 1) {
        geist_backend_set_error(be, GEIST_E_FORMAT, "transformer: '%s' missing or malformed", name);
        return GEIST_E_FORMAT;
    }
    *out_scalar = *(const float *) t->data;
    return GEIST_OK;
}

/* ---- One full layer ---------------------------------------------------- */

[[nodiscard]] enum geist_status load_one_layer(struct transformer_arch_state    *st,
                                               struct gguf_ctx                  *gguf,
                                               struct transformer_layer_weights *L) {

    struct geist_backend *be = st->backend;

    /* P1.5.c: geometry (is_full / is_kv_shared / head_dim / q_out /
     * kv_out / intermediate / sliding_window / rope_theta /
     * n_rotated_dims / layer_idx) is pre-filled by the family
     * populator's populate_layers hook in arch_family.c, before this
     * loader runs. The loader trusts those fields and just reads
     * tensors. */
    L->n_bufs = 0;

    char              path[64];
    enum geist_status s;

#define LP(suffix) snprintf(path, sizeof path, "blk.%d." suffix, L->layer_idx)

    /* ---- Gated-DeltaNet layer (#281): its own compact tensor set.
     * attn_norm + pre-FFN norm + FFN are shared with the attention
     * shape; everything between is the dn_* group. The qwen35
     * converter names the pre-FFN norm "post_attention_norm" (the HF
     * name) — there is no separate ffn_norm tensor. */
    if (L->mixer == GEIST_MIXER_DELTANET) {
        const size_t key_dim   = st->config.dn_n_k_heads * st->config.dn_head_k;
        const size_t value_dim = st->config.dn_n_v_heads * st->config.dn_head_v;
        const size_t conv_dim  = 2 * key_dim + value_dim;

        LP("attn_norm.weight");
        s = load_layer_norm(st, gguf, L, path, st->d_model, &L->attn_norm);
        if (s != GEIST_OK)
            return s;
        LP("post_attention_norm.weight");
        s = load_layer_norm(st, gguf, L, path, st->d_model, &L->ffn_norm);
        if (s != GEIST_OK)
            return s;

        LP("attn_qkv.weight");
        s = load_layer_proj(st, gguf, L, path, conv_dim, st->d_model, &L->dn_qkv, &L->dn_qkv_w);
        if (s != GEIST_OK)
            return s;
        LP("attn_gate.weight");
        s = load_layer_proj(st, gguf, L, path, value_dim, st->d_model, &L->dn_z, &L->dn_z_w);
        if (s != GEIST_OK)
            return s;
        LP("ssm_conv1d.weight");
        s = load_layer_norm(st, gguf, L, path, conv_dim * st->config.dn_conv_kernel, &L->dn_conv);
        if (s != GEIST_OK)
            return s;
        LP("ssm_beta.weight");
        s = load_layer_proj(st,
                            gguf,
                            L,
                            path,
                            st->config.dn_n_v_heads,
                            st->d_model,
                            &L->dn_beta,
                            &L->dn_beta_w);
        if (s != GEIST_OK)
            return s;
        LP("ssm_alpha.weight");
        s = load_layer_proj(st,
                            gguf,
                            L,
                            path,
                            st->config.dn_n_v_heads,
                            st->d_model,
                            &L->dn_alpha,
                            &L->dn_alpha_w);
        if (s != GEIST_OK)
            return s;
        LP("ssm_a");
        s = load_layer_norm(st, gguf, L, path, st->config.dn_n_v_heads, &L->dn_a);
        if (s != GEIST_OK)
            return s;
        LP("ssm_dt.bias");
        s = load_layer_norm(st, gguf, L, path, st->config.dn_n_v_heads, &L->dn_dt_bias);
        if (s != GEIST_OK)
            return s;
        LP("ssm_norm.weight");
        s = load_layer_norm(st, gguf, L, path, st->config.dn_head_v, &L->dn_norm);
        if (s != GEIST_OK)
            return s;
        LP("ssm_out.weight");
        s = load_layer_proj(st, gguf, L, path, st->d_model, value_dim, &L->dn_out, &L->dn_out_w);
        if (s != GEIST_OK)
            return s;

        LP("ffn_gate.weight");
        s = load_layer_proj(
                st, gguf, L, path, L->intermediate, st->d_model, &L->gate_proj, &L->gate_proj_w);
        if (s != GEIST_OK)
            return s;
        LP("ffn_up.weight");
        s = load_layer_proj(
                st, gguf, L, path, L->intermediate, st->d_model, &L->up_proj, &L->up_proj_w);
        if (s != GEIST_OK)
            return s;
        LP("ffn_down.weight");
        s = load_layer_proj(
                st, gguf, L, path, st->d_model, L->intermediate, &L->down_proj, &L->down_proj_w);
        if (s != GEIST_OK)
            return s;
        L->layer_scalar = 1.0f;
        return GEIST_OK;
    }

    /* Norms — all F32. Family-conditional gating (P1.5.d):
     * attn_q_norm, attn_k_norm, post_attention_norm, post_ffw_norm
     * are Gemma 3/4 family extras that Llama / Mistral don't have. */
    LP("attn_norm.weight");
    s = load_layer_norm(st, gguf, L, path, st->d_model, &L->attn_norm);
    if (s != GEIST_OK) {
        return s;
    }
    if (st->config.has_qk_norms) {
        LP("attn_q_norm.weight");
        s = load_layer_norm(st, gguf, L, path, L->head_dim, &L->q_norm);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->q_norm = (struct geist_tensor) {0};
    }
    if (st->config.has_gemma_attn_norms) {
        LP("post_attention_norm.weight");
        s = load_layer_norm(st, gguf, L, path, st->d_model, &L->post_attn_norm);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->post_attn_norm = (struct geist_tensor) {0};
    }
    LP("ffn_norm.weight");
    if (gguf_get_tensor(gguf, path) == nullptr && !st->config.has_gemma_attn_norms) {
        /* qwen35's converter keeps the HF name for the pre-FFN norm. */
        LP("post_attention_norm.weight");
    }
    s = load_layer_norm(st, gguf, L, path, st->d_model, &L->ffn_norm);
    if (s != GEIST_OK) {
        return s;
    }
    if (st->config.has_gemma_attn_norms) {
        LP("post_ffw_norm.weight");
        s = load_layer_norm(st, gguf, L, path, st->d_model, &L->post_ffw_norm);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->post_ffw_norm = (struct geist_tensor) {0};
    }
    /* P1.5.c: post_per_layer_norm is part of the PLE injection block —
     * only present in Gemma-family GGUFs. */
    if (st->config.has_ple) {
        LP("post_norm.weight");
        s = load_layer_norm(st, gguf, L, path, st->d_model, &L->post_per_layer_norm);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->post_per_layer_norm = (struct geist_tensor) {0};
    }

    /* Projections — quantized or F32. P1.1.d (refactor v2): every
     * per-layer projection now populates a geist_weight so the
     * forward hot path uses the pre-resolved kernel pointers. Sites
     * fall back to legacy v->linear() when the backend resolver
     * returns UNSUPPORTED (Q5_K, F32, etc.). */
    LP("attn_q.weight");
    /* qwen35: q_proj jointly produces query+gate — 2x rows, per-head
     * [query(hd) | gate(hd)] (#281). */
    const size_t q_rows = st->config.has_attn_output_gate ? 2 * L->q_out : L->q_out;
    s = load_layer_proj(st, gguf, L, path, q_rows, st->d_model, &L->q_proj, &L->q_proj_w);
    if (s != GEIST_OK) {
        return s;
    }
    LP("attn_output.weight");
    s = load_layer_proj(st, gguf, L, path, st->d_model, L->q_out, &L->o_proj, &L->o_proj_w);
    if (s != GEIST_OK) {
        return s;
    }

    /* BitNet SubLN: extra RMSNorm sitting between the attention output
     * and o_proj. Vector length is q_out (attn output is contracted
     * along q_out, not d_model, before o_proj). */
    if (st->config.has_sub_ln) {
        LP("attn_sub_norm.weight");
        s = load_layer_norm(st, gguf, L, path, L->q_out, &L->attn_sub_norm);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->attn_sub_norm = (struct geist_tensor) {0};
    }

    if (!L->is_kv_shared) {
        LP("attn_k.weight");
        s = load_layer_proj(st, gguf, L, path, L->kv_out, st->d_model, &L->k_proj, &L->k_proj_w);
        if (s != GEIST_OK) {
            return s;
        }
        LP("attn_v.weight");
        s = load_layer_proj(st, gguf, L, path, L->kv_out, st->d_model, &L->v_proj, &L->v_proj_w);
        if (s != GEIST_OK) {
            return s;
        }
        if (st->config.has_qk_norms) {
            LP("attn_k_norm.weight");
            s = load_layer_norm(st, gguf, L, path, L->head_dim, &L->k_norm);
            if (s != GEIST_OK) {
                return s;
            }
        } else {
            L->k_norm = (struct geist_tensor) {0};
        }
    } else {
        /* Zero out k/v/k_norm tensor structs so any accidental use is detectable
         * (buffer = nullptr → ops will fail loudly). */
        L->k_proj   = (struct geist_tensor) {0};
        L->v_proj   = (struct geist_tensor) {0};
        L->k_norm   = (struct geist_tensor) {0};
        L->k_proj_w = (struct geist_weight) {0};
        L->v_proj_w = (struct geist_weight) {0};
    }

    /* BitNet SQUARED_RELU FFN omits gate_proj entirely: the FFN is
     *   y = down_proj( relu(up_proj(x))² ).
     * GEGLU / SWIGLU families load gate_proj. */
    if (st->config.ffn_activation != GEIST_FFN_SQUARED_RELU) {
        LP("ffn_gate.weight");
        s = load_layer_proj(
                st, gguf, L, path, L->intermediate, st->d_model, &L->gate_proj, &L->gate_proj_w);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->gate_proj   = (struct geist_tensor) {0};
        L->gate_proj_w = (struct geist_weight) {0};
    }
    LP("ffn_up.weight");
    s = load_layer_proj(
            st, gguf, L, path, L->intermediate, st->d_model, &L->up_proj, &L->up_proj_w);
    if (s != GEIST_OK) {
        return s;
    }

    /* BitNet SubLN: extra RMSNorm between the FFN activation output and
     * down_proj. Vector length is intermediate (FFN inner dim). */
    if (st->config.has_sub_ln) {
        LP("ffn_sub_norm.weight");
        s = load_layer_norm(st, gguf, L, path, L->intermediate, &L->ffn_sub_norm);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->ffn_sub_norm = (struct geist_tensor) {0};
    }

    LP("ffn_down.weight");
    s = load_layer_proj(
            st, gguf, L, path, st->d_model, L->intermediate, &L->down_proj, &L->down_proj_w);
    if (s != GEIST_OK) {
        return s;
    }

    /* P1.5.c: per-layer PLE projections (inp_gate, proj) + layer
     * scalar are Gemma-family-only. Skip for !has_ple families;
     * default scalar to 1.0 so the forward "*= layer_scalar" is a
     * no-op. */
    if (st->config.has_ple) {
        LP("inp_gate.weight");
        s = load_layer_proj(st,
                            gguf,
                            L,
                            path,
                            st->hidden_per_layer,
                            st->d_model,
                            &L->per_layer_gate,
                            &L->per_layer_gate_w);
        if (s != GEIST_OK) {
            return s;
        }
        LP("proj.weight");
        s = load_layer_proj(st,
                            gguf,
                            L,
                            path,
                            st->d_model,
                            st->hidden_per_layer,
                            &L->per_layer_proj,
                            &L->per_layer_proj_w);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->per_layer_gate   = (struct geist_tensor) {0};
        L->per_layer_proj   = (struct geist_tensor) {0};
        L->per_layer_gate_w = (struct geist_weight) {0};
        L->per_layer_proj_w = (struct geist_weight) {0};
    }

#undef LP

    if (st->config.has_ple) {
        s = load_layer_scalar(be, gguf, L->layer_idx, &L->layer_scalar);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        L->layer_scalar = 1.0f;
    }

    return GEIST_OK;
}

/* ---- One Qwen3.5 MTP layer ------------------------------------------- */

[[nodiscard]] enum geist_status load_mtp_layer(struct transformer_arch_state        *st,
                                               struct gguf_ctx                      *gguf,
                                               struct transformer_mtp_layer_weights *M) {
    if (st == nullptr || gguf == nullptr || M == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = load_one_layer(st, gguf, &M->block);
    if (s != GEIST_OK) {
        return s;
    }

    char path[96];
#define MP(suffix) snprintf(path, sizeof path, "blk.%d.nextn." suffix, M->block.layer_idx)

    MP("eh_proj.weight");
    s = load_layer_proj(
            st, gguf, &M->block, path, st->d_model, 2 * st->d_model, &M->eh_proj, &M->eh_proj_w);
    if (s != GEIST_OK)
        return s;
    MP("enorm.weight");
    s = load_layer_norm(st, gguf, &M->block, path, st->d_model, &M->enorm);
    if (s != GEIST_OK)
        return s;
    MP("hnorm.weight");
    s = load_layer_norm(st, gguf, &M->block, path, st->d_model, &M->hnorm);
    if (s != GEIST_OK)
        return s;
    MP("shared_head_norm.weight");
    s = load_layer_norm(st, gguf, &M->block, path, st->d_model, &M->shared_head_norm);

#undef MP
    return s;
}

/* ---- Global weights ---------------------------------------------------- */

[[nodiscard]] enum geist_status
load_globals(struct geist_backend *be, struct gguf_ctx *gguf, struct transformer_arch_state *st) {

    const struct gguf_tensor_t *t   = nullptr;
    struct geist_buffer        *buf = nullptr;

    /* token_embd: [VOCAB, HIDDEN], any supported dtype. */
    enum geist_status s = load_tensor_to_buffer(
            st, gguf, "token_embd.weight", (size_t) st->vocab_size * st->d_model, &t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    struct dtype_map_entry dm = map_gguf_dtype(t->dtype);
    if (!dm.supported) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "transformer: token_embd dtype %s unsupported",
                                gguf_dtype_name(t->dtype));
        return GEIST_E_UNSUPPORTED;
    }
    st->embed_table =
            make_view_2d(buf, dm.dtype, dm.layout, (int64_t) st->vocab_size, (int64_t) st->d_model);
    if (global_track_buf(st, buf) != 0) {
        be->desc->vtbl->buffer_destroy(be, buf);
        return GEIST_E_INTERNAL;
    }
    /* P1.1.d (refactor v2): pre-resolve lm_head kernel pointers. Most
     * Llama-family BitNets tie lm_head to token_embd; in that case
     * embed_table_w just wraps the same buffer. P3.6: some 1bitLLM
     * variants (HF1BitLLM/Llama3-8B-1.58) ship a SEPARATE `output.weight`
     * tensor (Q6_K) alongside a Q4_K token_embd — different shapes-as-
     * stored, different roles. Try to load a standalone output.weight
     * first; fall back to tied if it isn't present.
     *
     * Unsupported dtypes (Q5_K bartowski variants) leave linear_m1 null
     * and the callers fall back to v->linear. */
    {
        const struct geist_backend_vtbl *v = be->desc->vtbl;

        const struct gguf_tensor_t *t_out   = nullptr;
        struct geist_buffer        *buf_out = nullptr;
        bool                        untied  = false;
        if (load_tensor_to_buffer(st,
                                  gguf,
                                  "output.weight",
                                  (size_t) st->vocab_size * st->d_model,
                                  &t_out,
                                  &buf_out) == GEIST_OK) {
            struct dtype_map_entry dmo = map_gguf_dtype(t_out->dtype);
            if (dmo.supported) {
                if (global_track_buf(st, buf_out) != 0) {
                    be->desc->vtbl->buffer_destroy(be, buf_out);
                    return GEIST_E_INTERNAL;
                }
                void *host_out = v->buffer_map(buf_out);
                if (host_out == nullptr) {
                    geist_backend_set_error(be,
                                            GEIST_E_BACKEND,
                                            "transformer: buffer_map(output.weight) returned null");
                    return GEIST_E_BACKEND;
                }
                st->embed_table_w = (struct geist_weight) {
                        .raw        = host_out,
                        .raw_nbytes = t_out->nbytes,
                        .n_in       = (int32_t) st->d_model,
                        .n_out      = (int32_t) st->vocab_size,
                        .dtype      = (uint16_t) dmo.dtype,
                };
                st->output_table = make_view_2d(buf_out,
                                                dmo.dtype,
                                                dmo.layout,
                                                (int64_t) st->vocab_size,
                                                (int64_t) st->d_model);
                v->buffer_unmap(buf_out);
                untied = true;
            } else {
                /* dtype unsupported — drop the buffer, fall back to tied. */
                be->desc->vtbl->buffer_destroy(be, buf_out);
            }
        }

        if (!untied) {
            void *host = v->buffer_map(buf);
            if (host == nullptr) {
                geist_backend_set_error(
                        be, GEIST_E_BACKEND, "transformer: buffer_map(token_embd) returned null");
                return GEIST_E_BACKEND;
            }
            st->embed_table_w = (struct geist_weight) {
                    .raw        = host,
                    .raw_nbytes = t->nbytes,
                    .n_in       = (int32_t) st->d_model,
                    .n_out      = (int32_t) st->vocab_size,
                    .dtype      = (uint16_t) dm.dtype,
            };
            st->output_table = st->embed_table;
            v->buffer_unmap(buf);
        }
        if (v->resolve_weight != nullptr) {
            enum geist_status rs = v->resolve_weight(be, &st->embed_table_w);
            if (rs != GEIST_OK && rs != GEIST_E_UNSUPPORTED) {
                return rs;
            }
        }
    }

    /* P1.5.c: PLE-only globals (per_layer_token_embd,
     * per_layer_model_proj, per_layer_proj_norm) skipped for
     * !has_ple families. Zero-init the tensor views so any accidental
     * access faults loudly at the buffer dereference. */
    if (!st->config.has_ple) {
        st->ple_table       = (struct geist_tensor) {0};
        st->model_proj      = (struct geist_tensor) {0};
        st->model_proj_norm = (struct geist_tensor) {0};
        st->model_proj_w    = (struct geist_weight) {0};
        goto load_output_norm;
    }

    /* per_layer_token_embd: [VOCAB, PLE_OUT]. */
    s = load_tensor_to_buffer(st,
                              gguf,
                              "per_layer_token_embd.weight",
                              (size_t) st->vocab_size * st->ple_out,
                              &t,
                              &buf);
    if (s != GEIST_OK) {
        return s;
    }
    dm = map_gguf_dtype(t->dtype);
    if (!dm.supported) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "transformer: per_layer_token_embd dtype %s unsupported",
                                gguf_dtype_name(t->dtype));
        return GEIST_E_UNSUPPORTED;
    }
    st->ple_table =
            make_view_2d(buf, dm.dtype, dm.layout, (int64_t) st->vocab_size, (int64_t) st->ple_out);
    if (global_track_buf(st, buf) != 0) {
        be->desc->vtbl->buffer_destroy(be, buf);
        return GEIST_E_INTERNAL;
    }

    /* per_layer_model_proj: [PLE_OUT, HIDDEN]. The Q3_K_M GGUF stores this
     * as F16, which neither cpu_scalar nor cpu_neon linear() supports today
     * (they handle F32 DENSE and Q3/4/5/6_K + Q8_0 BLOCK_QUANTIZED only).
     * Mirror lm.c's approach: dequantize at load time to F32 once. Cost is
     * ~52 MB resident, ~negligible. Other quantized globals (token_embd,
     * per_layer_token_embd) stay native — embedding_lookup will handle
     * those via row dequant in sub-step 3. */
    t = gguf_get_tensor(gguf, "per_layer_model_proj.weight");
    if (t == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_NOT_FOUND, "transformer: per_layer_model_proj.weight not found");
        return GEIST_E_NOT_FOUND;
    }
    if (gguf_tensor_elem_count(t) != (size_t) st->ple_out * st->d_model) {
        geist_backend_set_error(
                be, GEIST_E_FORMAT, "transformer: per_layer_model_proj element count mismatch");
        return GEIST_E_FORMAT;
    }
    {
        float *fp32 = gguf_dequant_to_fp32(t);
        if (fp32 == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_FORMAT,
                                    "transformer: per_layer_model_proj dequant failed (dtype %s)",
                                    gguf_dtype_name(t->dtype));
            return GEIST_E_FORMAT;
        }
        /* per_layer_model_proj is always heap-resident: GGUF stores it
         * as F16; we dequant to F32 (no mmap-alias possible because
         * the F32 form isn't a slice of the file). Two storage paths:
         *
         *   β mode  → bump-allocate from arena, memcpy in, then free
         *             the dequant scratch.
         *   mmap    → backend allocates its own buffer (the legacy
         *             path); we buffer_upload into it. arena is
         *             nullptr in mmap mode so we can't use it. */
        const size_t bytes = (size_t) st->ple_out * st->d_model * sizeof(float);
        if (st->weight_arena != nullptr) {
            void *arena_ptr = arena_alloc(st, bytes, 64);
            if (arena_ptr == nullptr) {
                void *p = fp32;
                safe_free(&p);
                geist_backend_set_error(
                        be,
                        GEIST_E_OOM,
                        "transformer: arena exhausted at per_layer_model_proj fp32");
                return GEIST_E_OOM;
            }
            memcpy(arena_ptr, fp32, bytes);
            void *p = fp32;
            safe_free(&p);
            s = be->desc->vtbl->buffer_create_aliased(
                    be, arena_ptr, bytes, GEIST_BUFFER_WEIGHT, &buf);
            if (s != GEIST_OK) {
                return s;
            }
        } else {
            s = be->desc->vtbl->buffer_create(
                    be, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &buf);
            if (s != GEIST_OK) {
                void *p = fp32;
                safe_free(&p);
                return s;
            }
            s       = be->desc->vtbl->buffer_upload(buf, bytes, (const uint8_t *) fp32);
            void *p = fp32;
            safe_free(&p);
            if (s != GEIST_OK) {
                be->desc->vtbl->buffer_destroy(be, buf);
                return s;
            }
        }
    }
    st->model_proj = make_view_2d(
            buf, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, (int64_t) st->ple_out, (int64_t) st->d_model);
    if (global_track_buf(st, buf) != 0) {
        be->desc->vtbl->buffer_destroy(be, buf);
        return GEIST_E_INTERNAL;
    }
    /* P1.1.e: pre-resolve model_proj (F32 dense → cblas trampolines). */
    {
        const struct geist_backend_vtbl *v    = be->desc->vtbl;
        void                            *host = v->buffer_map(buf);
        if (host == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "transformer: buffer_map(model_proj) returned null");
            return GEIST_E_BACKEND;
        }
        st->model_proj_w = (struct geist_weight) {
                .raw        = host,
                .raw_nbytes = (size_t) st->ple_out * st->d_model * sizeof(float),
                .n_in       = (int32_t) st->d_model,
                .n_out      = (int32_t) st->ple_out,
                .dtype      = (uint16_t) GEIST_DTYPE_F32,
        };
        v->buffer_unmap(buf);
        if (v->resolve_weight != nullptr) {
            enum geist_status rs = v->resolve_weight(be, &st->model_proj_w);
            if (rs != GEIST_OK && rs != GEIST_E_UNSUPPORTED) {
                return rs;
            }
        }
    }

    /* per_layer_proj_norm: [HIDDEN_PER_LAYER], F32. */
    s = load_tensor_to_buffer(
            st, gguf, "per_layer_proj_norm.weight", (size_t) st->hidden_per_layer, &t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    if (t->dtype != GGUF_TYPE_F32) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(be, GEIST_E_FORMAT, "transformer: per_layer_proj_norm must be F32");
        return GEIST_E_FORMAT;
    }
    st->model_proj_norm =
            make_view_1d(buf, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, (int64_t) st->hidden_per_layer);
    if (global_track_buf(st, buf) != 0) {
        be->desc->vtbl->buffer_destroy(be, buf);
        return GEIST_E_INTERNAL;
    }

load_output_norm:
    /* output_norm: [HIDDEN], F32. */
    s = load_tensor_to_buffer(st, gguf, "output_norm.weight", (size_t) st->d_model, &t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    if (t->dtype != GGUF_TYPE_F32) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(be, GEIST_E_FORMAT, "transformer: output_norm must be F32");
        return GEIST_E_FORMAT;
    }
    st->output_norm = make_view_1d(buf, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, (int64_t) st->d_model);
    if (global_track_buf(st, buf) != 0) {
        be->desc->vtbl->buffer_destroy(be, buf);
        return GEIST_E_INTERNAL;
    }

    return GEIST_OK;
}
