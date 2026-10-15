/*
 * src/archs/transformer/arch_family.c — family-populator implementations
 * and the `general.architecture` → family registry.
 *
 * Layer: ARCHITECTURE.
 *
 * Registry: gemma4, gemma3, llama, qwen3, qwen35, bitnet-b1.58, bitnet — a
 * static array searched linearly.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch_family.h"
#include "arch_state.h"

#include "gguf_reader.h"

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ---- BitNet embedding metadata ----------------------------------------- */
/*
 * The two facts that turn a stock qwen3 / gemma3 GGUF into an embedding
 * model: an RMSNorm on every projection's input, and a pooling. Both
 * populators read them, and there are two spellings in the wild.
 *
 *   tools/convert_bitnet_embedding.py writes
 *       bitnet.embedding.projection_input_norms  (bool)
 *       bitnet.embedding.pooling                 ("mean" | "last_token")
 *
 *   Microsoft's own published GGUFs (bitnet-embeddings-*-i2_s.gguf on the
 *   Hugging Face model pages) write NEITHER. They carry the per-projection
 *   norms as tensors only, and the pooling as gguf-py's standard numeric
 *   `{arch}.pooling_type`.
 *
 * Both spellings are read here; neither weakens the guarantee that the file
 * has to actually contain what the metadata promises.
 */
void populate_bitnet_embedding_meta(struct gguf_ctx               *gguf,
                                    const char                    *arch,
                                    struct transformer_arch_state *st) {
    /* Projection-input norms. The TENSOR is the evidence — metadata must
     * not promise norms the file lacks, so the key is not required. An
     * explicit `false` still wins, so a converter can opt out. */
    const bool have_norm_tensors = gguf_get_tensor(gguf, "blk.0.attn_q_norm_in.weight") != nullptr;
    bool       proj_norms        = true;
    (void) gguf_get_meta_bool(gguf, "bitnet.embedding.projection_input_norms", &proj_norms);
    st->config.has_projection_input_norms = have_norm_tensors && proj_norms;

    /* Pooling. Its absence is what makes a stock GGUF generative, so a
     * missing key in BOTH spellings means NONE, not a failure. A key that
     * is present but names a pooling this build cannot perform leaves
     * pooling_unsupported set, which populate_layers refuses — better a
     * clear load failure than a vector pooled the wrong way. */
    size_t      plen = 0;
    const char *pstr = gguf_get_meta_string(gguf, "bitnet.embedding.pooling", &plen);
    if (pstr != nullptr) {
        if (!geist_pooling_select(plen, pstr, &st->config.pooling)) {
            st->config.pooling             = GEIST_POOLING_NONE;
            st->config.pooling_unsupported = true;
        }
        return;
    }
    char key[96];
    snprintf(key, sizeof key, "%s.pooling_type", arch);
    uint32_t pooling_type = 0;
    if (!gguf_get_meta_u32(gguf, key, &pooling_type)) {
        st->config.pooling = GEIST_POOLING_NONE; /* a generative model */
        return;
    }
    if (!geist_pooling_from_gguf_type(pooling_type, &st->config.pooling)) {
        st->config.pooling             = GEIST_POOLING_NONE;
        st->config.pooling_unsupported = true;
    }
}

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
static void populate_qwen35(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    st->config.has_qk_norms         = true;
    st->config.has_attn_output_gate = true;
    st->config.ffn_activation       = GEIST_FFN_SWIGLU;

    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "qwen35.block_count", &u))
        st->n_layers = u;
    uint32_t nextn = 0;
    gguf_get_meta_u32(gguf, "qwen35.nextn_predict_layers", &nextn);
    if (nextn < st->n_layers) {
        st->n_mtp_layers = nextn;
        st->n_layers -= nextn;
    } else if (nextn != 0) {
        /* Make malformed geometry fail in state_create instead of treating
         * MTP blocks as autoregressive layers. */
        st->n_layers     = 0;
        st->n_mtp_layers = nextn;
    }
    if (gguf_get_meta_u32(gguf, "qwen35.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "qwen35.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen35.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_f32(gguf, "qwen35.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    { /* vocab from the tokenizer table (no qwen35.vocab_size key) */
        uint32_t       elem_vt = 0;
        uint64_t       count   = 0;
        const uint8_t *payload = nullptr;
        if (gguf_get_meta_array_info(gguf, "tokenizer.ggml.tokens", &elem_vt, &count, &payload))
            st->vocab_size = (size_t) count;
    }
    /* DeltaNet geometry. GGUF key mapping (#281): state_size =
     * head_k (= head_v), group_count = n_k_heads, time_step_rank =
     * n_v_heads, inner_size = n_v_heads * head_v. */
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.group_count", &u))
        st->config.dn_n_k_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.time_step_rank", &u))
        st->config.dn_n_v_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.state_size", &u))
        st->config.dn_head_k = u;
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.inner_size", &u) && st->config.dn_n_v_heads > 0)
        st->config.dn_head_v = u / st->config.dn_n_v_heads;
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.conv_kernel", &u))
        st->config.dn_conv_kernel = u;
    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static bool populate_layers_qwen35(struct transformer_arch_state *st) {
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    uint32_t head_dim = 0, intermediate = 0, rot = 0, interval = 4;
    float    freq_base = 10000000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "qwen35.attention.key_length", &head_dim);
        gguf_get_meta_u32(g, "qwen35.feed_forward_length", &intermediate);
        gguf_get_meta_u32(g, "qwen35.rope.dimension_count", &rot);
        gguf_get_meta_u32(g, "qwen35.full_attention_interval", &interval);
        gguf_get_meta_f32(g, "qwen35.rope.freq_base", &freq_base);
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
        gguf_get_meta_array_info(g, "qwen35.attention.recurrent_layers", &rec_vt, &rec_n, &rec)) {
        if (rec_vt != GGUF_META_VT_BOOL || rec_n < st->n_layers)
            rec = nullptr;
    }

    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        st->config.rope_partial_block       = true; /* ggml layout, see arch_config.h */
        const bool is_attn = rec != nullptr ? rec[i] == 0 : ((i + 1) % (size_t) interval == 0);
        L->layer_idx       = (int) i;
        L->mixer           = is_attn ? GEIST_MIXER_ATTN : GEIST_MIXER_DELTANET;
        L->is_full         = true;
        L->is_kv_shared    = false;
        L->head_dim        = head_dim;
        L->q_out           = st->n_q_heads * head_dim;
        L->kv_out          = st->n_kv_heads * head_dim;
        L->intermediate    = intermediate;
        L->sliding_window  = 0;
        L->rope_theta      = freq_base;
        L->n_rotated_dims  = (int) rot;
    }
    for (size_t i = 0; i < st->n_mtp_layers; i++) {
        struct transformer_layer_weights *L = &st->mtp_layers[i].block;
        L->layer_idx                        = (int) (st->n_layers + i);
        L->mixer                            = GEIST_MIXER_ATTN;
        L->is_full                          = true;
        L->is_kv_shared                     = false;
        L->head_dim                         = head_dim;
        L->q_out                            = st->n_q_heads * head_dim;
        L->kv_out                           = st->n_kv_heads * head_dim;
        L->intermediate                     = intermediate;
        L->sliding_window                   = 0;
        L->rope_theta                       = freq_base;
        L->n_rotated_dims                   = (int) rot;
    }
    return true;
}

/* ---- Registry --------------------------------------------------------- */

static const struct transformer_family FAMILY_QWEN35 = {
        .name            = "qwen35",
        .populate        = populate_qwen35,
        .populate_layers = populate_layers_qwen35,
};

static const struct transformer_family *const REGISTRY[] = {
        &transformer_family_gemma4, /* families/gemma4.c */
        &transformer_family_gemma3, /* families/gemma3.c */
        &transformer_family_llama,  /* families/llama.c */
        &transformer_family_qwen3,  /* families/qwen3.c */
        &FAMILY_QWEN35,
        &transformer_family_bitnet_b158, /* families/bitnet.c */
        &transformer_family_bitnet,
};
static const size_t REGISTRY_N = sizeof REGISTRY / sizeof REGISTRY[0];

/* Engine-facing match list: the exact `general.architecture` values this
 * architecture accepts, NULL-terminated. Referenced by desc_transformer in
 * src/engine/arch_registry.c — the engine gate rejects any GGUF whose arch
 * string is not listed here, before weights are touched. Must mirror
 * REGISTRY above (count enforced below; a string mismatch fails closed at
 * transformer_family_select, never loads the wrong family). */
const char *const geist_arch_transformer_gguf_names[] = {
        "gemma4",
        "gemma3",
        "llama",
        "qwen3",
        "qwen35",
        "bitnet-b1.58",
        "bitnet",
        nullptr,
};
static_assert(sizeof geist_arch_transformer_gguf_names /
                                      sizeof geist_arch_transformer_gguf_names[0] -
                              1 ==
                      sizeof REGISTRY / sizeof REGISTRY[0],
              "gguf name list must mirror REGISTRY");

const struct transformer_family *transformer_family_select(struct gguf_ctx *gguf) {
    size_t      arch_len = 0;
    const char *arch     = gguf_get_meta_string(gguf, "general.architecture", &arch_len);
    if (arch != nullptr) {
        for (size_t i = 0; i < REGISTRY_N; i++) {
            const struct transformer_family *f    = REGISTRY[i];
            const size_t                     flen = strlen(f->name);
            if (flen == arch_len && memcmp(arch, f->name, flen) == 0) {
                return f;
            }
        }
    }
    /* Fail closed: no silent Gemma-4 fallback. The engine gate should
     * have rejected the GGUF already; reaching this means a caller
     * bypassed it or the name lists diverged. */
    return nullptr;
}
