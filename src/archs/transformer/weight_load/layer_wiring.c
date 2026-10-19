/*
 * src/archs/transformer/weight_load/layer_wiring.c — per-layer +
 * global tensor wiring.
 *
 * Layer: ARCHITECTURE.
 * Contains:
 *
 *   layer_track_buf  / global_track_buf — owning-buf list helpers
 *   load_layer_norm  / load_layer_proj / load_layer_matrix /
 *   load_layer_experts — per-tensor load wrappers.
 *   load_layer_norm_set, load_sandwich_norms, load_projection_input_norms —
 *                      norm groups the family units' wire_layer hooks load.
 *   load_one_layer   — the tensors every family shares, after the family's.
 *   load_mtp_layer   — one Qwen3.5 MTP block.
 *   load_globals     — wire embed/output/PLE tables + global norms.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_family.h"
#include "../weight_load.h"

#include "gguf_dequant.h"
#include "gguf_reader.h"
#include "checked.h"
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
[[nodiscard]] enum geist_status load_layer_norm(struct transformer_arch_state    *st,
                                                struct gguf_ctx                  *gguf,
                                                struct transformer_layer_weights *L,
                                                const char                       *name,
                                                size_t                            expected_elems,
                                                struct geist_tensor              *out_view) {

    struct geist_backend *be  = st->backend;
    struct geist_buffer  *buf = nullptr;
    enum geist_status     s   = load_norm_to_f32_buffer(st, gguf, name, expected_elems, &buf);
    if (s != GEIST_OK) {
        return s;
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

/* Load a per-layer 2D projection. dtype may be quantized or F32.
 * rope_il_head_dim > 0 reorders the rows from interleaved to half-split
 * RoPE pairs (llama attn_q / attn_k, #464). */
[[nodiscard]] static enum geist_status
load_layer_proj_rope_il(struct transformer_arch_state    *st,
                        struct gguf_ctx                  *gguf,
                        struct transformer_layer_weights *L,
                        const char                       *name,
                        size_t                            n_out,
                        size_t                            n_in,
                        size_t                            rope_il_head_dim,
                        uint16_t                          weight_flags,
                        struct geist_tensor              *out_view,
                        struct geist_weight              *out_weight /* nullable */) {

    if (st->weights_on_host) {
        weight_flags |= (uint16_t) GEIST_W_HOST;
    }
    struct geist_backend       *be  = st->backend;
    const struct gguf_tensor_t *t   = nullptr;
    struct geist_buffer        *buf = nullptr;
    /* A matrix behind a geist_weight is resolved below; one without is
     * bound as it is. */
    const enum weight_storage storage = out_weight != nullptr ? WEIGHT_LINEAR : WEIGHT_BIND;
    enum geist_status         s       = load_tensor_to_buffer_rope_il(
            st, gguf, name, n_out * n_in, rope_il_head_dim, storage, &t, &buf);
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
    /* Also populate the pre-resolved kernel pointers when the caller asks
     * for them. struct geist_buffer is opaque to the arch layer, so the
     * host pointer comes from buffer_map (load time, once per weight). */
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
                .flags      = weight_flags,
        };
        v->buffer_unmap(buf);
        if (v->resolve_weight != nullptr) {
            be->calibration.locked = true; /* kernel binding begins — calibration frozen */
            enum geist_status rs   = v->resolve_weight(be, out_weight);
            if (rs != GEIST_OK && rs != GEIST_E_UNSUPPORTED) {
                /* Real failure — propagate. Unsupported is fine; caller
                 * checks linear_m1 nullness and falls back. */
                return rs;
            }
            /* The backend holds its own device copy of a matrix that
             * skipped the arena (#468), or a host repack its kernels read
             * instead of the source (#729): the GGUF pages behind it need
             * not stay resident. A fallback that still reads them faults
             * them back in from the file. */
            const bool own_copy =
                    weight_off_arena(be, storage) ||
                    (storage == WEIGHT_LINEAR && (out_weight->flags & GEIST_W_RAW_COLD) != 0);
            if (rs == GEIST_OK && own_copy) {
                gguf_release_range(gguf, out_weight->raw, out_weight->raw_nbytes);
            }
            /* A backend without a half-precision dense linear (vulkan) gets
             * a SMALL tensor widened to F32 once, the same way norm gammas
             * are (e.g. BF16 DeltaNet alpha/beta projections). Widening
             * doubles the resident bytes, so a model's main matrices stay
             * out of it. Past the cap nothing can run the matrix, so the
             * load fails here, naming the tensor, instead of at the first
             * prefill (#564). The half-precision copy stays tracked. */
            constexpr size_t widen_max_elems = 4u << 20; /* 16 MB as F32 */
            const bool       half = dm.dtype == GEIST_DTYPE_F16 || dm.dtype == GEIST_DTYPE_BF16;
            if (rs == GEIST_E_UNSUPPORTED && half && n_out * n_in > widen_max_elems) {
                geist_backend_set_error(be,
                                        GEIST_E_UNSUPPORTED,
                                        "transformer: %s has no %s linear for '%s' (%zux%zu); "
                                        "use a quantized GGUF or a CPU backend",
                                        be->desc->name,
                                        gguf_dtype_name(t->dtype),
                                        name,
                                        n_out,
                                        n_in);
                return GEIST_E_UNSUPPORTED;
            }
            if (rs == GEIST_E_UNSUPPORTED && half) {
                struct geist_buffer *buf32 = nullptr;
                s                          = load_f32_buffer_rope_il(
                        st, gguf, name, n_out * n_in, rope_il_head_dim, WEIGHT_LINEAR, &buf32);
                if (s != GEIST_OK) {
                    return s;
                }
                if (layer_track_buf(L, buf32) != 0) {
                    be->desc->vtbl->buffer_destroy(be, buf32);
                    geist_backend_set_error(be,
                                            GEIST_E_INTERNAL,
                                            "transformer: layer buffer list overflow on '%s'",
                                            name);
                    return GEIST_E_INTERNAL;
                }
                *out_view    = make_view_2d(buf32,
                                            GEIST_DTYPE_F32,
                                            GEIST_LAYOUT_DENSE,
                                            (int64_t) n_out,
                                            (int64_t) n_in);
                void *host32 = v->buffer_map(buf32);
                if (host32 == nullptr) {
                    geist_backend_set_error(
                            be, GEIST_E_BACKEND, "transformer: buffer_map(%s) returned null", name);
                    return GEIST_E_BACKEND;
                }
                *out_weight = (struct geist_weight) {
                        .raw        = (const uint8_t *) host32 + out_view->offset,
                        .raw_nbytes = n_out * n_in * sizeof(float),
                        .n_in       = (int32_t) n_in,
                        .n_out      = (int32_t) n_out,
                        .dtype      = (uint16_t) GEIST_DTYPE_F32,
                        .flags      = weight_flags,
                };
                v->buffer_unmap(buf32);
                rs = v->resolve_weight(be, out_weight);
                if (rs != GEIST_OK && rs != GEIST_E_UNSUPPORTED) {
                    return rs;
                }
            }
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
load_layer_proj(struct transformer_arch_state    *st,
                struct gguf_ctx                  *gguf,
                struct transformer_layer_weights *L,
                const char                       *name,
                size_t                            n_out,
                size_t                            n_in,
                struct geist_tensor              *out_view,
                struct geist_weight              *out_weight /* nullable */) {
    return load_layer_proj_rope_il(st, gguf, L, name, n_out, n_in, 0, 0, out_view, out_weight);
}

enum geist_status load_layer_matrix(struct transformer_arch_state    *st,
                                    struct gguf_ctx                  *gguf,
                                    struct transformer_layer_weights *L,
                                    const char                       *name,
                                    size_t                            n_out,
                                    size_t                            n_in,
                                    bool                              exact,
                                    struct geist_tensor              *out_view,
                                    struct geist_weight              *out_weight) {
    return load_layer_proj_rope_il(st,
                                   gguf,
                                   L,
                                   name,
                                   n_out,
                                   n_in,
                                   0,
                                   exact ? (uint16_t) GEIST_W_EXACT : 0,
                                   out_view,
                                   out_weight);
}

enum geist_status load_layer_experts(struct transformer_arch_state    *st,
                                     struct gguf_ctx                  *gguf,
                                     struct transformer_layer_weights *L,
                                     const char                       *name,
                                     size_t                            n_experts,
                                     size_t                            n_out,
                                     size_t                            n_in,
                                     struct geist_tensor             **out_views,
                                     struct geist_weight             **out_weights) {
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    *out_views                          = nullptr;
    *out_weights                        = nullptr;

    size_t per = 0, total = 0;
    if (n_experts == 0 || ckd_mul(&per, n_out, n_in) || ckd_mul(&total, per, n_experts)) {
        geist_backend_set_error(be, GEIST_E_FORMAT, "transformer: '%s' has no valid shape", name);
        return GEIST_E_FORMAT;
    }
    const struct gguf_tensor_t *t   = nullptr;
    struct geist_buffer        *buf = nullptr;
    enum geist_status s = load_tensor_to_buffer(st, gguf, name, total, WEIGHT_LINEAR, &t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    if (layer_track_buf(L, buf) != 0) {
        v->buffer_destroy(be, buf);
        geist_backend_set_error(
                be, GEIST_E_INTERNAL, "transformer: layer buffer list overflow on '%s'", name);
        return GEIST_E_INTERNAL;
    }
    const struct dtype_map_entry dm = map_gguf_dtype(t->dtype);
    if (!dm.supported || t->nbytes % n_experts != 0) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "transformer: experts '%s' (%s) cannot be split into %zu",
                                name,
                                gguf_dtype_name(t->dtype),
                                n_experts);
        return GEIST_E_UNSUPPORTED;
    }
    /* Every quantized format packs whole rows, so expert e's [n_out, n_in]
     * matrix is the e-th equal slice of the tensor's bytes. */
    const size_t         eb      = t->nbytes / n_experts;
    struct geist_tensor *views   = heap_calloc_array_aligned(struct geist_tensor, n_experts);
    struct geist_weight *weights = heap_calloc_array_aligned(struct geist_weight, n_experts);
    *out_views                   = views;
    *out_weights                 = weights;
    L->moe_n_experts             = n_experts; /* what the layer teardown walks */
    if (views == nullptr || weights == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "transformer: no memory for experts '%s'", name);
        return GEIST_E_OOM;
    }
    const uint8_t *host = (const uint8_t *) v->buffer_map(buf);
    if (host == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "transformer: buffer_map(%s) returned null", name);
        return GEIST_E_BACKEND;
    }
    for (size_t e = 0; e < n_experts && s == GEIST_OK; e++) {
        views[e] = make_view_2d(buf, dm.dtype, dm.layout, (int64_t) n_out, (int64_t) n_in);
        views[e].offset += (int64_t) (e * eb);
        weights[e] = (struct geist_weight) {
                .raw        = host + e * eb,
                .raw_nbytes = eb,
                .n_in       = (int32_t) n_in,
                .n_out      = (int32_t) n_out,
                .dtype      = (uint16_t) dm.dtype,
        };
        if (v->resolve_weight == nullptr) {
            continue;
        }
        be->calibration.locked = true;
        s                      = v->resolve_weight(be, &weights[e]);
        if (s == GEIST_E_UNSUPPORTED) {
            geist_backend_set_error(be,
                                    GEIST_E_UNSUPPORTED,
                                    "transformer: %s has no %s linear for experts '%s'",
                                    be->desc->name,
                                    gguf_dtype_name(t->dtype),
                                    name);
        } else if (s == GEIST_OK && (weights[e].flags & GEIST_W_RAW_COLD) != 0) {
            /* A host repack the kernels read instead of the source (#729). */
            gguf_release_range(gguf, weights[e].raw, eb);
        }
    }
    v->buffer_unmap(buf);
    return s;
}

/* ---- One full layer ---------------------------------------------------- */

enum geist_status load_layer_norm_set(struct transformer_arch_state    *st,
                                      struct gguf_ctx                  *gguf,
                                      struct transformer_layer_weights *L,
                                      size_t                            n,
                                      const struct layer_norm_spec      specs[static n]) {
    char name[96];
    for (size_t i = 0; i < n; i++) {
        snprintf(name, sizeof name, "blk.%d.%s", L->layer_idx, specs[i].suffix);
        enum geist_status s = load_layer_norm(st, gguf, L, name, specs[i].n_elems, specs[i].out);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}

enum geist_status load_sandwich_norms(struct transformer_arch_state    *st,
                                      struct gguf_ctx                  *gguf,
                                      struct transformer_layer_weights *L) {
    const struct layer_norm_spec norms[] = {
            {"post_attention_norm.weight", st->d_model, &L->post_attn_norm},
            {"post_ffw_norm.weight", st->d_model, &L->post_ffw_norm},
    };
    return load_layer_norm_set(st, gguf, L, sizeof norms / sizeof norms[0], norms);
}

enum geist_status load_projection_input_norms(struct transformer_arch_state    *st,
                                              struct gguf_ctx                  *gguf,
                                              struct transformer_layer_weights *L) {
    const size_t                 d       = st->d_model;
    const struct layer_norm_spec norms[] = {
            /* o_proj's and down_proj's inputs have the shape and position of
             * BitNet's SubLN gammas, so they load into those slots. */
            {"attn_output_norm_in.weight", L->q_out, &L->attn_sub_norm},
            {"attn_q_norm_in.weight", d, &L->q_norm_in},
            {"attn_k_norm_in.weight", d, &L->k_norm_in},
            {"attn_v_norm_in.weight", d, &L->v_norm_in},
            {"ffn_gate_norm_in.weight", d, &L->gate_norm_in},
            {"ffn_up_norm_in.weight", d, &L->up_norm_in},
            {"ffn_down_norm_in.weight", L->intermediate, &L->ffn_sub_norm},
    };
    return load_layer_norm_set(st, gguf, L, sizeof norms / sizeof norms[0], norms);
}

/* "blk.<layer>.<suffix>" into name[64]. */
static const char *
layer_tensor(char name[static 64], const struct transformer_layer_weights *L, const char *suffix) {
    snprintf(name, 64, "blk.%d.%s", L->layer_idx, suffix);
    return name;
}

/* The attention tensors of a softmax-attention layer: the per-head Q/K
 * norms where the model has them, q_proj unless the family loaded its own,
 * o_proj, and k/v unless the layer reads another layer's cache (their
 * slots then stay empty, so any accidental use fails loudly). Llama GGUFs
 * store q/k rows in interleaved RoPE pair order; they are reordered once
 * here so the forward pass rotates half-split pairs like every other family
 * (#464). */
[[nodiscard]] static enum geist_status load_attention(struct transformer_arch_state    *st,
                                                      struct gguf_ctx                  *gguf,
                                                      struct transformer_layer_weights *L) {
    const size_t      d          = st->d_model;
    const size_t      rope_il_hd = st->config.rope_interleaved ? L->head_dim : 0;
    const bool        qk_norms   = st->config.has_qk_norms;
    char              name[64];
    enum geist_status s = GEIST_OK;
    if (qk_norms) {
        s = load_layer_norm(
                st, gguf, L, layer_tensor(name, L, "attn_q_norm.weight"), L->head_dim, &L->q_norm);
    }
    if (s == GEIST_OK && L->q_proj.buffer == nullptr) {
        s = load_layer_proj_rope_il(st,
                                    gguf,
                                    L,
                                    layer_tensor(name, L, "attn_q.weight"),
                                    L->q_out,
                                    d,
                                    rope_il_hd,
                                    0,
                                    &L->q_proj,
                                    &L->q_proj_w);
    }
    if (s == GEIST_OK) {
        s = load_layer_proj(st,
                            gguf,
                            L,
                            layer_tensor(name, L, "attn_output.weight"),
                            d,
                            L->q_out,
                            &L->o_proj,
                            &L->o_proj_w);
    }
    if (s != GEIST_OK || L->is_kv_shared) {
        return s;
    }
    s = load_layer_proj_rope_il(st,
                                gguf,
                                L,
                                layer_tensor(name, L, "attn_k.weight"),
                                L->kv_out,
                                d,
                                rope_il_hd,
                                0,
                                &L->k_proj,
                                &L->k_proj_w);
    if (s == GEIST_OK) {
        s = load_layer_proj(st,
                            gguf,
                            L,
                            layer_tensor(name, L, "attn_v.weight"),
                            L->kv_out,
                            d,
                            &L->v_proj,
                            &L->v_proj_w);
    }
    if (s == GEIST_OK && qk_norms) {
        s = load_layer_norm(
                st, gguf, L, layer_tensor(name, L, "attn_k_norm.weight"), L->head_dim, &L->k_norm);
    }
    return s;
}

/* The dense FFN: gate (except under a squared-ReLU FFN, which has none:
 * y = down_proj(relu(up_proj(x))^2)), up and down. */
[[nodiscard]] static enum geist_status load_dense_ffn(struct transformer_arch_state    *st,
                                                      struct gguf_ctx                  *gguf,
                                                      struct transformer_layer_weights *L) {
    const size_t      d = st->d_model;
    char              name[64];
    enum geist_status s = GEIST_OK;
    if (st->config.ffn_activation != GEIST_FFN_SQUARED_RELU) {
        s = load_layer_proj(st,
                            gguf,
                            L,
                            layer_tensor(name, L, "ffn_gate.weight"),
                            L->intermediate,
                            d,
                            &L->gate_proj,
                            &L->gate_proj_w);
    }
    if (s == GEIST_OK) {
        s = load_layer_proj(st,
                            gguf,
                            L,
                            layer_tensor(name, L, "ffn_up.weight"),
                            L->intermediate,
                            d,
                            &L->up_proj,
                            &L->up_proj_w);
    }
    if (s == GEIST_OK) {
        s = load_layer_proj(st,
                            gguf,
                            L,
                            layer_tensor(name, L, "ffn_down.weight"),
                            d,
                            L->intermediate,
                            &L->down_proj,
                            &L->down_proj_w);
    }
    return s;
}

/* Geometry (mixer / is_full / is_kv_shared / head_dim / q_out / kv_out /
 * intermediate / ... / layer_idx) is pre-filled by the family's
 * populate_layers hook; this loader trusts those fields and reads tensors:
 * the family's own first (wire_layer), then the set every family shares,
 * skipping a slot the family filled. Slots a layer does not have stay as
 * state_create zeroed them. Every projection gets a geist_weight so the
 * forward hot path uses the pre-resolved kernel pointers. */
[[nodiscard]] enum geist_status load_one_layer(struct transformer_arch_state    *st,
                                               struct gguf_ctx                  *gguf,
                                               struct transformer_layer_weights *L) {
    L->n_bufs       = 0;
    L->layer_scalar = 1.0f; /* the forward's "*= layer_scalar" is a no-op unless a family sets it */

    char              name[64];
    enum geist_status s = GEIST_OK;
    if (st->family != nullptr && st->family->wire_layer != nullptr) {
        s = st->family->wire_layer(st, gguf, L);
    }
    if (s == GEIST_OK) {
        s = load_layer_norm(
                st, gguf, L, layer_tensor(name, L, "attn_norm.weight"), st->d_model, &L->attn_norm);
    }
    if (s == GEIST_OK && L->mixer == GEIST_MIXER_ATTN) {
        s = load_attention(st, gguf, L);
    }
    if (s == GEIST_OK && L->ffn_norm.buffer == nullptr) {
        s = load_layer_norm(
                st, gguf, L, layer_tensor(name, L, "ffn_norm.weight"), st->d_model, &L->ffn_norm);
    }
    /* A mixture-of-experts layer has no dense FFN: its family's wire_layer
     * loads the router and the experts (load_layer_experts). */
    if (s == GEIST_OK && st->config.moe_n_experts == 0) {
        s = load_dense_ffn(st, gguf, L);
    }
    return s;
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

/* Track a global buffer, destroying it when the list is full. */
[[nodiscard]] static enum geist_status
global_keep(struct geist_backend *be, struct transformer_arch_state *st, struct geist_buffer *buf) {
    if (global_track_buf(st, buf) != 0) {
        be->desc->vtbl->buffer_destroy(be, buf);
        return GEIST_E_INTERNAL;
    }
    return GEIST_OK;
}

/* A [vocab, cols] table in any supported dtype: token_embd or
 * per_layer_token_embd. */
[[nodiscard]] static enum geist_status load_table(struct geist_backend          *be,
                                                  struct gguf_ctx               *gguf,
                                                  struct transformer_arch_state *st,
                                                  const char                    *name,
                                                  size_t                         cols,
                                                  enum weight_storage            storage,
                                                  const struct gguf_tensor_t   **out_t,
                                                  struct geist_tensor           *out_view) {
    struct geist_buffer *buf = nullptr;
    enum geist_status    s   = load_tensor_to_buffer(
            st, gguf, name, (size_t) st->vocab_size * cols, storage, out_t, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    const struct dtype_map_entry dm = map_gguf_dtype((*out_t)->dtype);
    if (!dm.supported) {
        be->desc->vtbl->buffer_destroy(be, buf);
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "transformer: %.*s dtype %s unsupported",
                                (int) (strlen(name) - strlen(".weight")),
                                name,
                                gguf_dtype_name((*out_t)->dtype));
        return GEIST_E_UNSUPPORTED;
    }
    *out_view = make_view_2d(buf, dm.dtype, dm.layout, (int64_t) st->vocab_size, (int64_t) cols);
    return global_keep(be, st, buf);
}

/* Wrap `buf` (whose host bytes start at its offset 0) in a geist_weight of
 * [n_out, n_in] and resolve its kernel. */
[[nodiscard]] static enum geist_status resolve_global(struct geist_backend *be,
                                                      struct geist_buffer  *buf,
                                                      const char           *what,
                                                      size_t                raw_nbytes,
                                                      size_t                n_out,
                                                      size_t                n_in,
                                                      enum geist_dtype      dtype,
                                                      struct geist_weight  *out) {
    const struct geist_backend_vtbl *v    = be->desc->vtbl;
    void                            *host = v->buffer_map(buf);
    if (host == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "transformer: buffer_map(%s) returned null", what);
        return GEIST_E_BACKEND;
    }
    *out = (struct geist_weight) {
            .raw        = host,
            .raw_nbytes = raw_nbytes,
            .n_in       = (int32_t) n_in,
            .n_out      = (int32_t) n_out,
            .dtype      = (uint16_t) dtype,
    };
    v->buffer_unmap(buf);
    if (v->resolve_weight != nullptr) {
        be->calibration.locked = true;
        enum geist_status rs   = v->resolve_weight(be, out);
        if (rs != GEIST_OK && rs != GEIST_E_UNSUPPORTED) {
            return rs;
        }
    }
    return GEIST_OK;
}

/* The lm_head: a standalone `output.weight` (e.g. HF1BitLLM/Llama3-8B-1.58:
 * Q6_K beside a Q4_K token_embd) when present in a supported dtype and
 * shape; otherwise it is tied and embed_table_w wraps the token_embd buffer
 * (t_embd). */
[[nodiscard]] static enum geist_status load_lm_head(struct geist_backend          *be,
                                                    struct gguf_ctx               *gguf,
                                                    struct transformer_arch_state *st,
                                                    bool                           tied,
                                                    const struct gguf_tensor_t    *t_embd) {
    if (tied) {
        st->output_table = st->embed_table;
        return resolve_global(be,
                              st->embed_table.buffer,
                              "token_embd",
                              t_embd->nbytes,
                              st->vocab_size,
                              st->d_model,
                              st->embed_table.dtype,
                              &st->embed_table_w);
    }
    const struct gguf_tensor_t *t   = nullptr;
    struct geist_buffer        *buf = nullptr;
    enum geist_status           s   = load_tensor_to_buffer(st,
                                                            gguf,
                                                            "output.weight",
                                                            (size_t) st->vocab_size * st->d_model,
                                                            WEIGHT_LINEAR,
                                                            &t,
                                                            &buf);
    if (s == GEIST_OK) {
        s = global_keep(be, st, buf);
    }
    if (s != GEIST_OK) {
        return s;
    }
    const struct dtype_map_entry dm = map_gguf_dtype(t->dtype); /* supported: !tied */
    st->output_table =
            make_view_2d(buf, dm.dtype, dm.layout, (int64_t) st->vocab_size, (int64_t) st->d_model);
    return resolve_global(be,
                          buf,
                          "output.weight",
                          t->nbytes,
                          st->vocab_size,
                          st->d_model,
                          dm.dtype,
                          &st->embed_table_w);
}

/* per_layer_model_proj: [PLE_OUT, HIDDEN], stored F16 in the GGUF;
 * dequantized to F32 once at load (~52 MB resident) so every backend can
 * run it. Other quantized globals (token_embd, per_layer_token_embd) stay
 * native and are dequantized per row at lookup.
 *
 * The F32 form is not a slice of the file, so it cannot alias the mmap.
 * Three storage paths:
 *
 *   device copy → keep the dequant result as host memory of the state's
 *                 own and alias it; resolve_weight uploads it. In the arena
 *                 it took ~110 MB of Vulkan's BAR window on Gemma 4 E4B and
 *                 pushed the scratch pool to system memory (#658).
 *   β mode      → bump-allocate from arena, memcpy in, then free the
 *                 dequant scratch.
 *   mmap        → no arena; the backend allocates its own buffer and we
 *                 buffer_upload into it. */
[[nodiscard]] static enum geist_status load_ple_model_proj(struct geist_backend          *be,
                                                           struct gguf_ctx               *gguf,
                                                           struct transformer_arch_state *st,
                                                           struct geist_buffer          **out_buf) {
    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, "per_layer_model_proj.weight");
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
    float *fp32 = gguf_dequant_to_fp32(t);
    if (fp32 == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: per_layer_model_proj dequant failed (dtype %s)",
                                gguf_dtype_name(t->dtype));
        return GEIST_E_FORMAT;
    }
    const size_t      bytes = (size_t) st->ple_out * st->d_model * sizeof(float);
    void             *p     = fp32;
    enum geist_status s;
    if (weight_off_arena(be, WEIGHT_LINEAR)) {
        s = keep_host_weight(st, fp32); /* fp32 is freed with the state, or already was */
        if (s == GEIST_OK) {
            s = be->desc->vtbl->buffer_create_aliased(
                    be, fp32, bytes, GEIST_BUFFER_WEIGHT, out_buf);
        }
        return s;
    }
    if (st->weight_arena != nullptr) {
        void *arena_ptr = arena_alloc(st, bytes, 64);
        if (arena_ptr == nullptr) {
            safe_free(&p);
            geist_backend_set_error(
                    be, GEIST_E_OOM, "transformer: arena exhausted at per_layer_model_proj fp32");
            return GEIST_E_OOM;
        }
        memcpy(arena_ptr, fp32, bytes);
        safe_free(&p);
        return be->desc->vtbl->buffer_create_aliased(
                be, arena_ptr, bytes, GEIST_BUFFER_WEIGHT, out_buf);
    }
    s = be->desc->vtbl->buffer_create(be, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, out_buf);
    if (s != GEIST_OK) {
        safe_free(&p);
        return s;
    }
    s = be->desc->vtbl->buffer_upload(*out_buf, bytes, (const uint8_t *) fp32);
    safe_free(&p);
    if (s != GEIST_OK) {
        be->desc->vtbl->buffer_destroy(be, *out_buf);
    }
    return s;
}

/* The per-layer-embedding globals: per_layer_token_embd [VOCAB, PLE_OUT],
 * per_layer_model_proj (F32, resolved) and per_layer_proj_norm
 * [HIDDEN_PER_LAYER] (F32). */
[[nodiscard]] static enum geist_status load_ple_globals(struct geist_backend          *be,
                                                        struct gguf_ctx               *gguf,
                                                        struct transformer_arch_state *st) {
    const struct gguf_tensor_t *t = nullptr;
    enum geist_status           s = load_table(be,
                                               gguf,
                                               st,
                                               "per_layer_token_embd.weight",
                                               st->ple_out,
                                               WEIGHT_LOOKUP,
                                               &t,
                                               &st->ple_table);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_buffer *buf = nullptr;
    s                        = load_ple_model_proj(be, gguf, st, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    st->model_proj = make_view_2d(
            buf, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, (int64_t) st->ple_out, (int64_t) st->d_model);
    s = global_keep(be, st, buf);
    if (s == GEIST_OK) {
        s = resolve_global(be,
                           buf,
                           "model_proj",
                           (size_t) st->ple_out * st->d_model * sizeof(float),
                           st->ple_out,
                           st->d_model,
                           GEIST_DTYPE_F32,
                           &st->model_proj_w);
    }
    if (s != GEIST_OK) {
        return s;
    }

    s = load_tensor_to_buffer(st,
                              gguf,
                              "per_layer_proj_norm.weight",
                              (size_t) st->hidden_per_layer,
                              WEIGHT_BIND,
                              &t,
                              &buf);
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
    return global_keep(be, st, buf);
}

/* token_embd [VOCAB, HIDDEN] (the lm_head too when tied; beside a usable
 * output.weight only a lookup table), the lm_head, the PLE globals where
 * the model has them, and output_norm [HIDDEN] (F32). Global buffers are
 * tracked in this order. */
[[nodiscard]] enum geist_status
load_globals(struct geist_backend *be, struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    const struct gguf_tensor_t *t_lm = gguf_get_tensor(gguf, "output.weight");
    const bool tied = t_lm == nullptr || !map_gguf_dtype(t_lm->dtype).supported ||
                      gguf_tensor_elem_count(t_lm) != (size_t) st->vocab_size * st->d_model;
    const struct gguf_tensor_t *t_embd = nullptr;
    enum geist_status           s      = load_table(be,
                                                    gguf,
                                                    st,
                                                    "token_embd.weight",
                                                    st->d_model,
                                                    tied ? WEIGHT_LINEAR : WEIGHT_LOOKUP,
                                                    &t_embd,
                                                    &st->embed_table);
    if (s == GEIST_OK) {
        s = load_lm_head(be, gguf, st, tied, t_embd);
    }
    /* Without PLE the PLE views stay as state_create zeroed them. */
    if (s == GEIST_OK && st->config.has_ple) {
        s = load_ple_globals(be, gguf, st);
    }
    if (s != GEIST_OK) {
        return s;
    }
    struct geist_buffer *buf = nullptr;
    s = load_norm_to_f32_buffer(st, gguf, "output_norm.weight", (size_t) st->d_model, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    st->output_norm = make_view_1d(buf, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, (int64_t) st->d_model);
    return global_keep(be, st, buf);
}
