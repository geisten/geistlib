/*
 * src/archs/transformer/arch_family.c — family-populator implementations
 * and the `general.architecture` → family registry.
 *
 * Layer: ARCHITECTURE.
 *
 * P1.5: only the gemma4 family is wired today; Llama / Mistral
 * populators are placeholders for follow-up phases. The registry is
 * a static array iterated linearly (one entry today, growing to ~5
 * later) — no hash table needed.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch_family.h"
#include "arch_state.h"

#include "gguf_reader.h"

#include <stddef.h>
#include <string.h>

/* ---- Gemma 4 (current default) ---------------------------------------- */

/* Gemma 4 attention pattern: 4 sliding-window layers + 1 full-attention
 * layer, repeated 7 times = 35 layers. Layers >= 15 share K/V with
 * layer 13 (sliding source) or 14 (full source) — see
 * config.kv_sliding_src / kv_full_src. */
static const bool GEMMA4_LAYER_IS_FULL[] = {
        false, false, false, false, true,  false, false, false, false, true,  false, false,
        false, false, true,  false, false, false, false, true,  false, false, false, false,
        true,  false, false, false, false, true,  false, false, false, false, true,
};

static void populate_gemma4(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "gemma4.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "gemma4.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "gemma4.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "gemma4.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "gemma4.embedding_length_per_layer_input", &u))
        st->hidden_per_layer = u;
    st->ple_out = st->n_layers * st->hidden_per_layer;
    if (gguf_get_meta_f32(gguf, "gemma4.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    if (gguf_get_meta_f32(gguf, "gemma4.final_logit_softcapping", &f))
        st->config.logit_softcap = f;
    /* PLE flag + scales + KV-shared layer indices are family-constant,
     * not metadata-driven. state_create installs them in the default
     * config block before calling this populator. */
}

static void populate_layers_gemma4(struct transformer_arch_state *st) {
    /* Layers 0..14 are "owners" (compute + store K/V); layers 15..34
     * alias the source layer's cache per kv_*_src config. Loader uses
     * the is_kv_shared bit to skip K/V projection tensors and the
     * runtime uses kv_*_src to pick which earlier layer's cache to
     * read. Hardcoded threshold (15) matches Gemma 4 E2B; future
     * variants will need a meta key. */
    const int kv_share_threshold = 15;
    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        L->is_full        = (i < sizeof GEMMA4_LAYER_IS_FULL / sizeof GEMMA4_LAYER_IS_FULL[0])
                                    ? GEMMA4_LAYER_IS_FULL[i]
                                    : true;
        L->is_kv_shared   = (int) i >= kv_share_threshold;
        L->head_dim       = L->is_full ? 512 : 256;
        L->q_out          = st->n_q_heads * L->head_dim;
        L->kv_out         = st->n_kv_heads * L->head_dim;
        L->intermediate   = L->is_kv_shared ? 12288 : 6144;
        L->sliding_window = L->is_full ? 0 : 512;
        L->rope_theta     = L->is_full ? 1000000.0f : 10000.0f;
        L->n_rotated_dims = L->is_full ? 128 : (int) L->head_dim;
    }
}

/* ---- Llama (P1.5.d) --------------------------------------------------- */

static void populate_llama(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Strip every Gemma-family default first; the loader uses these
     * flags to skip tensors that aren't present in Llama GGUFs. */
    st->config.has_ple              = false;
    st->config.logit_softcap        = 0.0f;
    st->config.has_gemma_attn_norms = false;
    st->config.kv_sliding_src       = -1;
    st->config.kv_full_src          = -1;
    st->config.ple_input_scale      = 0.0f;
    st->config.ple_model_proj_scale = 0.0f;
    st->config.ple_table_scale      = 0.0f;
    st->config.has_sub_ln           = false;
    st->config.ffn_activation       = GEIST_FFN_SWIGLU;
    st->config.rope_interleaved     = true; /* LLAMA_ARCH = NORM RoPE; weights are pre-permuted */

    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "llama.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "llama.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "llama.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "llama.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "llama.vocab_size", &u))
        st->vocab_size = u;
    if (gguf_get_meta_f32(gguf, "llama.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    /* Llama doesn't have PLE — zero out the PLE-derived dims so any
     * accidental access traps on size==0. */
    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static void populate_layers_llama(struct transformer_arch_state *st) {
    /* Llama is uniform: all layers full-attention, no KV sharing,
     * head_dim derived from d_model / n_q_heads. RoPE rotates the
     * full head_dim (no partial rotation like Gemma's 25% on full
     * layers). FFN intermediate is uniform via feed_forward_length.
     *
     * The actual feed_forward_length must come from meta — we cache
     * it from llama.feed_forward_length once. Pull it lazily here. */
    const size_t head_dim     = (st->n_q_heads > 0) ? (st->d_model / st->n_q_heads) : 0;
    uint32_t     intermediate = 0;
    /* Re-read from the GGUF cached on st. */
    if (st->gguf != nullptr) {
        gguf_get_meta_u32((struct gguf_ctx *) st->gguf, "llama.feed_forward_length", &intermediate);
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        L->is_full                          = true;
        L->is_kv_shared                     = false;
        L->head_dim                         = head_dim;
        L->q_out                            = st->n_q_heads * head_dim;
        L->kv_out                           = st->n_kv_heads * head_dim;
        L->intermediate                     = intermediate;
        L->sliding_window                   = 0;
        L->rope_theta                       = 100000.0f; /* default; populator should override */
        L->n_rotated_dims                   = (int) head_dim;
    }
    /* RoPE theta override from llama.rope.freq_base. */
    if (st->gguf != nullptr) {
        float freq_base;
        if (gguf_get_meta_f32((struct gguf_ctx *) st->gguf, "llama.rope.freq_base", &freq_base)) {
            for (size_t i = 0; i < st->n_layers; i++) {
                st->layers[i].rope_theta = freq_base;
            }
        }
    }
}

/* ---- BitNet b1.58 (P1.3) --------------------------------------------- */
/*
 * BitNet b1.58 is Llama-style transformer with two architectural
 * additions over the Llama family populator:
 *
 *   1. SubLN: an extra RMSNorm before each BitLinear (between
 *      attn-output and o_proj, between FFN activation and down_proj).
 *      Forward path is in P1.4; we set has_sub_ln so the loader knows
 *      to pull the extra norm weights.
 *
 *   2. FFN activation: BitNet b1.58 2B-4T (Microsoft flagship) uses *gated*
 *      squared-ReLU — relu(gate)^2 * up, with gate/up/down all present (HF
 *      hidden_act="relu2"). Community 1bitLLM/bitnet_b1_58-* use SwiGLU. The
 *      activation is not in the GGUF, so ffn_activation_from_meta picks the
 *      default by general.architecture ("bitnet-b1.58" -> gated squared-ReLU,
 *      "bitnet" -> SwiGLU) and a *.feed_forward_activation key overrides it.
 *
 * Tensor weights are TQ2_0 (see P1.2). Tokenizer is Llama3-style BPE
 * for 2B-4T, Llama2-style SentencePiece for 3B — both routed through
 * the existing GGUF-embedded tokenizer path.
 */
static enum geist_ffn_activation_kind ffn_activation_from_meta(struct gguf_ctx *gguf) {
    /* Default depends on the architecture, because the activation is NOT in the
     * GGUF metadata and the two BitNet families differ:
     *   - "bitnet-b1.58": Microsoft's official 2B-4T uses gated squared-ReLU
     *     (HF config hidden_act="relu2"): relu(gate)^2 * up. Its GGUF carries no
     *     activation key, so this MUST be the default — verified: MMLU on the
     *     i2_s 2B-4T jumps 25.5% (SwiGLU, chance) -> 50% (relu2, ~the published
     *     ~53%); SwiGLU also breaks greedy coherence.
     *   - "bitnet" (community 1bitLLM reproductions): SwiGLU (silu(gate) * up) —
     *     relu2 garbles those, so they keep the SwiGLU default.
     * An explicit *.feed_forward_activation key (below) overrides either. */
    size_t                         al   = 0;
    const char                    *arch = gguf_get_meta_string(gguf, "general.architecture", &al);
    enum geist_ffn_activation_kind out  = (arch != nullptr && al == sizeof("bitnet-b1.58") - 1 &&
                                           memcmp(arch, "bitnet-b1.58", al) == 0)
                                                  ? GEIST_FFN_GATED_SQUARED_RELU
                                                  : GEIST_FFN_SWIGLU;
    size_t                         len  = 0;
    const char *s = gguf_get_meta_string(gguf, "bitnet-b1.58.feed_forward_activation", &len);
    if (s == nullptr) {
        s = gguf_get_meta_string(gguf, "bitnet.feed_forward_activation", &len);
    }
    if (s == nullptr) {
        s = gguf_get_meta_string(gguf, "general.feed_forward_activation", &len);
    }
    if (s != nullptr) {
        if (len == sizeof("swiglu") - 1 && memcmp(s, "swiglu", len) == 0) {
            out = GEIST_FFN_SWIGLU;
        } else if (len == sizeof("squared_relu") - 1 && memcmp(s, "squared_relu", len) == 0) {
            out = GEIST_FFN_SQUARED_RELU;
        } else if (len == sizeof("geglu") - 1 && memcmp(s, "geglu", len) == 0) {
            out = GEIST_FFN_GEGLU;
        } else if ((len == sizeof("relu2") - 1 && memcmp(s, "relu2", len) == 0) ||
                   (len == sizeof("gated_squared_relu") - 1 &&
                    memcmp(s, "gated_squared_relu", len) == 0)) {
            /* Microsoft BitNet 2B-4T uses `hidden_act = "relu2"` which
             * in their config means gated squared-ReLU: relu(g)^2 * up.
             * gguf-py / mainline llama.cpp doesn't emit this key today;
             * a local converter patch (see docs/bitnet_conversion.md)
             * adds it. */
            out = GEIST_FFN_GATED_SQUARED_RELU;
        }
    }
    return out;
}

static void populate_bitnet_b158(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Strip Gemma-family defaults — BitNet is Llama-style underneath. */
    st->config.has_ple              = false;
    st->config.logit_softcap        = 0.0f;
    st->config.has_gemma_attn_norms = false;
    st->config.kv_sliding_src       = -1;
    st->config.kv_full_src          = -1;
    st->config.ple_input_scale      = 0.0f;
    st->config.ple_model_proj_scale = 0.0f;
    st->config.ple_table_scale      = 0.0f;

    /* BitNet additions. */
    st->config.has_sub_ln     = true;
    st->config.ffn_activation = ffn_activation_from_meta(gguf);

    /* Microsoft's bitnet.cpp converter writes metadata keys with the
     * full architecture string as the prefix ("bitnet-b1.58.*"), unlike
     * the llama family which uses just "llama.*". Try both — the
     * official GGUF uses "bitnet-b1.58.*", community converters may
     * use "bitnet.*". */
    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.block_count", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.embedding_length", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.attention.head_count", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.attention.head_count_kv", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.vocab_size", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.vocab_size", &u)) {
        st->vocab_size = u;
    } else {
        /* Community BitNet GGUFs (e.g. gianni-cor's TQ2_0 / Q4_0 builds)
         * omit the vocab_size metadata key. Fall back to the
         * tokenizer.ggml.tokens array length, which is always present
         * because the tokenizer rides inside the GGUF. */
        uint32_t       elem_vt = 0;
        uint64_t       count   = 0;
        const uint8_t *payload = nullptr;
        if (gguf_get_meta_array_info(gguf, "tokenizer.ggml.tokens", &elem_vt, &count, &payload)) {
            st->vocab_size = (size_t) count;
        }
    }
    if (gguf_get_meta_f32(gguf, "bitnet-b1.58.attention.layer_norm_rms_epsilon", &f) ||
        gguf_get_meta_f32(gguf, "bitnet.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    /* No PLE — keep the PLE-derived dims at zero so any accidental
     * access traps on size==0. */
    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static void populate_layers_bitnet_b158(struct transformer_arch_state *st) {
    /* BitNet is Llama-uniform: all layers full attention, no KV sharing,
     * head_dim = d_model / n_q_heads, single FFN intermediate from
     * bitnet.feed_forward_length. RoPE rotates the full head_dim. */
    const size_t head_dim     = (st->n_q_heads > 0) ? (st->d_model / st->n_q_heads) : 0;
    uint32_t     intermediate = 0;
    if (st->gguf != nullptr) {
        if (!gguf_get_meta_u32((struct gguf_ctx *) st->gguf,
                               "bitnet-b1.58.feed_forward_length",
                               &intermediate)) {
            gguf_get_meta_u32(
                    (struct gguf_ctx *) st->gguf, "bitnet.feed_forward_length", &intermediate);
        }
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        L->is_full                          = true;
        L->is_kv_shared                     = false;
        L->head_dim                         = head_dim;
        L->q_out                            = st->n_q_heads * head_dim;
        L->kv_out                           = st->n_kv_heads * head_dim;
        L->intermediate                     = intermediate;
        L->sliding_window                   = 0;
        /* Default: Llama2 convention (10000). BitNet b1.58 3B uses the
         * Llama2 rope_theta; the 2B-4T flagship is Llama3-derived and
         * sets the metadata key, which the loop below picks up. */
        L->rope_theta     = 10000.0f;
        L->n_rotated_dims = (int) head_dim;
    }
    if (st->gguf != nullptr) {
        float freq_base;
        if (gguf_get_meta_f32(
                    (struct gguf_ctx *) st->gguf, "bitnet-b1.58.rope.freq_base", &freq_base) ||
            gguf_get_meta_f32((struct gguf_ctx *) st->gguf, "bitnet.rope.freq_base", &freq_base)) {
            for (size_t i = 0; i < st->n_layers; i++) {
                st->layers[i].rope_theta = freq_base;
            }
        }
    }
}

/* ---- Registry --------------------------------------------------------- */

static const struct transformer_family FAMILY_GEMMA4 = {
        .name            = "gemma4",
        .populate        = populate_gemma4,
        .populate_layers = populate_layers_gemma4,
};

static const struct transformer_family FAMILY_LLAMA = {
        .name            = "llama",
        .populate        = populate_llama,
        .populate_layers = populate_layers_llama,
};

static const struct transformer_family FAMILY_BITNET_B158 = {
        .name            = "bitnet-b1.58",
        .populate        = populate_bitnet_b158,
        .populate_layers = populate_layers_bitnet_b158,
};

/* Community converters (e.g. gianni-cor/bitnet_b1_58-3B-TQ2_0) sometimes
 * emit "general.architecture" = "bitnet" without the b1.58 suffix. Same
 * populator works for both — the metadata-key fallback chain in
 * populate_bitnet_b158 already tries "bitnet.*" after "bitnet-b1.58.*". */
static const struct transformer_family FAMILY_BITNET = {
        .name            = "bitnet",
        .populate        = populate_bitnet_b158,
        .populate_layers = populate_layers_bitnet_b158,
};

/* Future: register sibling families here, e.g.
 *
 *   static void populate_llama(struct gguf_ctx *g, struct transformer_arch_state *st) {
 *       st->config.has_ple       = false;
 *       st->config.logit_softcap = 0.0f;
 *       st->config.kv_sliding_src = -1;
 *       st->config.kv_full_src    = -1;
 *       if (gguf_get_meta_u32(g, "llama.block_count", &u)) st->n_layers = u;
 *       ...
 *   }
 *   static const struct transformer_family FAMILY_LLAMA = { "llama", populate_llama };
 *
 * Once Llama lands, also turn on forward.c family-conditional paths
 * (PLE skip when !has_ple, softcap skip when logit_softcap==0; the
 * latter is in place as of P1.5).
 */

static const struct transformer_family *const REGISTRY[] = {
        &FAMILY_GEMMA4,
        &FAMILY_LLAMA,
        &FAMILY_BITNET_B158,
        &FAMILY_BITNET,
};
static const size_t REGISTRY_N = sizeof REGISTRY / sizeof REGISTRY[0];

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
    /* Fall back to the legacy default. Honest failure later if the
     * GGUF isn't actually a Gemma-4 (tensor-shape mismatch at load). */
    return &FAMILY_GEMMA4;
}
