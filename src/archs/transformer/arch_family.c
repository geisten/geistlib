/*
 * src/archs/transformer/arch_family.c — family-populator implementations
 * and the `general.architecture` → family registry.
 *
 * Layer: ARCHITECTURE.
 *
 * Registry: gemma4, llama, bitnet-b1.58, bitnet — a static array iterated
 * linearly; no hash table needed at this size.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch_family.h"
#include "arch_state.h"

#include "gguf_reader.h"

#include <stddef.h>
#include <string.h>

/* ---- Gemma 4 (current default) ---------------------------------------- */

/* Fallback attention pattern for GGUFs without the
 * gemma4.attention.sliding_window_pattern key: the E2B layout — 4
 * sliding-window layers + 1 full-attention layer, repeated 7 times =
 * 35 layers. E4B (5+1 × 7 = 42 layers) and anything else MUST carry
 * the metadata key (#258). */
static const bool GEMMA4_LAYER_IS_FULL[] = {
        false, false, false, false, true,  false, false, false, false, true,  false, false,
        false, false, true,  false, false, false, false, true,  false, false, false, false,
        true,  false, false, false, false, true,  false, false, false, false, true,
};

static void populate_gemma4(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Gemma-4 owns its defaults — state_create installs a NEUTRAL config
     * and each family declares what it HAS. Structural values are the
     * E2B geometry, kept as fallbacks for pre-key GGUFs; current unsloth
     * GGUFs override every one of them from metadata below. PLE scales
     * are architectural constants (not in GGUF metadata). */
    st->n_layers         = 35;
    st->d_model          = 1536;
    st->vocab_size       = 262144; /* no gemma4.vocab_size key exists */
    st->n_q_heads        = 8;
    st->n_kv_heads       = 1;
    st->hidden_per_layer = 256;

    st->config.logit_softcap        = 30.0f;
    st->config.has_ple              = true;
    st->config.ple_input_scale      = 0.7071067811865476f;
    st->config.ple_model_proj_scale = 0.02551551815399144f;
    st->config.ple_table_scale      = 16.0f;
    /* E2B KV-share sources; populate_layers_gemma4 re-derives them
     * from the attention pattern (#258). */
    st->config.kv_sliding_src       = 13;
    st->config.kv_full_src          = 14;
    st->config.has_gemma_attn_norms = true;
    st->config.has_qk_norms         = true;
    st->config.ffn_activation       = GEIST_FFN_GEGLU;

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
}

static bool populate_layers_gemma4(struct transformer_arch_state *st) {
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    /* Attention pattern: bool array, true = sliding-window layer. E2B
     * is 4+1 × 7 (35 layers), E4B is 5+1 × 7 (42 layers) — variant
     * geometry, so it must come from metadata. GGUFs without the key
     * fall back to the static E2B table; any other layer count without
     * the key is not derivable → fail (caller names the geometry). */
    uint32_t       pat_vt = 0;
    uint64_t       pat_n  = 0;
    const uint8_t *pat    = nullptr;
    if (g == nullptr ||
        !gguf_get_meta_array_info(
                g, "gemma4.attention.sliding_window_pattern", &pat_vt, &pat_n, &pat) ||
        pat_vt != GGUF_META_VT_BOOL || pat_n != st->n_layers) {
        pat = nullptr;
        if (st->n_layers != sizeof GEMMA4_LAYER_IS_FULL / sizeof GEMMA4_LAYER_IS_FULL[0]) {
            return false;
        }
    }

    /* KV sharing: the LAST shared_kv_layers layers alias earlier
     * caches. E2B: 35-20 → owners 0..14 (matches the old hardcoded
     * threshold 15). E4B: 42-18 → owners 0..23. */
    uint32_t shared_kv = 20;
    if (g != nullptr)
        gguf_get_meta_u32(g, "gemma4.attention.shared_kv_layers", &shared_kv);
    if (shared_kv > st->n_layers) {
        return false;
    }
    const size_t kv_share_threshold = st->n_layers - shared_kv;

    /* Per-type head dims + sliding window + RoPE bases, E2B defaults. */
    uint32_t full_hd = 512, swa_hd = 256, sliding_window = 512;
    float    theta_full = 1000000.0f, theta_swa = 10000.0f;
    /* FFN inner dim: scalar key (uniform, E4B) or per-layer u32/i32
     * array (E2B: 6144 owners / 12288 shared). */
    uint32_t       ffn_uniform = 0;
    const uint8_t *ffn_arr     = nullptr;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "gemma4.attention.key_length", &full_hd);
        gguf_get_meta_u32(g, "gemma4.attention.key_length_swa", &swa_hd);
        gguf_get_meta_u32(g, "gemma4.attention.sliding_window", &sliding_window);
        gguf_get_meta_f32(g, "gemma4.rope.freq_base", &theta_full);
        gguf_get_meta_f32(g, "gemma4.rope.freq_base_swa", &theta_swa);
        uint32_t ffn_vt = 0;
        uint64_t ffn_n  = 0;
        if (!gguf_get_meta_u32(g, "gemma4.feed_forward_length", &ffn_uniform) &&
            gguf_get_meta_array_info(g, "gemma4.feed_forward_length", &ffn_vt, &ffn_n, &ffn_arr)) {
            if ((ffn_vt != GGUF_META_VT_U32 && ffn_vt != GGUF_META_VT_I32) ||
                ffn_n != st->n_layers) {
                ffn_arr = nullptr;
            }
        }
    }

    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        L->is_full      = pat != nullptr ? pat[i] == 0 : GEMMA4_LAYER_IS_FULL[i];
        L->is_kv_shared = i >= kv_share_threshold;
        L->head_dim     = L->is_full ? full_hd : swa_hd;
        L->q_out        = st->n_q_heads * L->head_dim;
        L->kv_out       = st->n_kv_heads * L->head_dim;
        if (ffn_arr != nullptr) {
            uint32_t v;
            memcpy(&v, ffn_arr + i * sizeof v, sizeof v);
            L->intermediate = v;
        } else if (ffn_uniform > 0) {
            L->intermediate = ffn_uniform;
        } else {
            L->intermediate = L->is_kv_shared ? 12288 : 6144; /* pre-key E2B GGUFs */
        }
        L->sliding_window = L->is_full ? 0 : sliding_window;
        L->rope_theta     = L->is_full ? theta_full : theta_swa;
        /* Full-attn layers rotate 25% of head_dim (partial_rotary_factor
         * 0.25 — family constant; the GGUF's rope.dimension_count key
         * carries head_dim, not the rotated width, so it can't be used).
         * Sliding layers rotate the full head_dim. */
        L->n_rotated_dims = L->is_full ? (int) (full_hd / 4) : (int) L->head_dim;
    }

    /* KV-share sources: each shared layer reads the cache of the LAST
     * owner layer of its own attention type. E2B: full←14, sliding←13
     * (the previously hardcoded values). E4B: full←23, sliding←22. */
    int last_full = -1, last_sliding = -1;
    for (size_t i = 0; i < kv_share_threshold; i++) {
        if (st->layers[i].is_full)
            last_full = (int) i;
        else
            last_sliding = (int) i;
    }
    if (shared_kv > 0 && (last_full < 0 || last_sliding < 0)) {
        return false; /* shared layers with no owner of one type */
    }
    st->config.kv_full_src    = last_full;
    st->config.kv_sliding_src = last_sliding;
    return true;
}

/* ---- Llama (P1.5.d) --------------------------------------------------- */

static void populate_llama(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Neutral config from state_create covers Llama; the two family
     * specifics: SwiGLU (already the neutral default, stated for
     * clarity) and interleaved RoPE (weights are pre-permuted). */
    st->config.ffn_activation   = GEIST_FFN_SWIGLU;
    st->config.rope_interleaved = true; /* LLAMA_ARCH = NORM RoPE */

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

static bool populate_layers_llama(struct transformer_arch_state *st) {
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
    return true;
}

/* ---- Qwen3 (#275) ------------------------------------------------------ */
/*
 * Llama-style uniform stack (GQA, SwiGLU, full attention, no KV sharing,
 * no PLE, no softcap) with two deviations:
 *
 *   1. Per-head QK-norm: RMSNorm over [head_dim] on Q and K before RoPE
 *      (has_qk_norms — the Gemma V-norm / scale semantics do NOT apply,
 *      Qwen3 keeps the standard 1/sqrt(head_dim) scale).
 *   2. head_dim is decoupled from d_model: qwen3.attention.key_length
 *      (128 on 0.6B where d_model/n_heads would give 64), so
 *      q_out = n_heads * head_dim can EXCEED d_model. o_proj is
 *      [d_model, q_out] and scratch sizes from q_out — already the
 *      Gemma-4 full-attn shape, no special casing needed.
 *
 * RoPE is NEOX-style half-split like Gemma (no weight pre-permute), so
 * rope_interleaved stays false — unlike llama.
 */
static void populate_qwen3(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Neutral config + the two qwen3 specifics: per-head QK-norm and
     * SwiGLU. NEOX RoPE (no pre-permute) is the neutral default. */
    st->config.has_qk_norms   = true;
    st->config.ffn_activation = GEIST_FFN_SWIGLU;

    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "qwen3.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "qwen3.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "qwen3.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen3.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen3.vocab_size", &u)) {
        st->vocab_size = u;
    } else {
        /* Qwen's GGUFs omit the vocab_size key — fall back to the
         * tokenizer.ggml.tokens array length (same as BitNet). */
        uint32_t       elem_vt = 0;
        uint64_t       count   = 0;
        const uint8_t *payload = nullptr;
        if (gguf_get_meta_array_info(gguf, "tokenizer.ggml.tokens", &elem_vt, &count, &payload)) {
            st->vocab_size = (size_t) count;
        }
    }
    if (gguf_get_meta_f32(gguf, "qwen3.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static bool populate_layers_qwen3(struct transformer_arch_state *st) {
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    /* head_dim from metadata — NOT d_model / n_heads (0.6B: key_length
     * 128 vs quotient 64). Missing key on a geometry where the quotient
     * would be wrong is exactly the #258 failure mode, so fall back to
     * the quotient only as a last resort and fail on zero. */
    uint32_t head_dim = 0, intermediate = 0;
    float    freq_base = 1000000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "qwen3.attention.key_length", &head_dim);
        gguf_get_meta_u32(g, "qwen3.feed_forward_length", &intermediate);
        gguf_get_meta_f32(g, "qwen3.rope.freq_base", &freq_base);
    }
    if (head_dim == 0 && st->n_q_heads > 0) {
        head_dim = (uint32_t) (st->d_model / st->n_q_heads);
    }
    if (head_dim == 0 || intermediate == 0) {
        return false;
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
        L->rope_theta                       = freq_base;
        L->n_rotated_dims                   = (int) head_dim;
    }
    return true;
}

/* ---- qwen35 (Qwen3.5/3.6/3.8 hybrid, #281) ----------------------------- */
/*
 * Hybrid token-mixer stack: with full_attention_interval N (default 4),
 * layer i is softmax attention iff (i+1) % N == 0, else gated DeltaNet
 * (recurrent conv + delta-rule state — see forward/layer_deltanet.c).
 * The attention layers are standard GQA with QK-norm, partial NEOX RoPE
 * (rope.dimension_count of head_dim, theta 1e7) and a sigmoid output
 * gate fed by a joint q+gate projection (has_attn_output_gate).
 *
 * block_count INCLUDES nextn_predict_layers trailing MTP block(s); they
 * are not part of the autoregressive stack — n_layers excludes them and
 * their blk.N tensors are simply never referenced.
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
    if (nextn < st->n_layers)
        st->n_layers -= nextn; /* trailing MTP blocks: skip */
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
    /* DeltaNet geometry. GGUF key mapping (see #281 spec): state_size =
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
    return true;
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
    /* Read general.architecture + an optional *.feed_forward_activation override
     * and let geist_ffn_activation_select (pure, unit-tested) decide. The default
     * is arch-keyed because the activation is NOT in the GGUF and the BitNet
     * families differ: "bitnet-b1.58" (Microsoft 2B-4T) needs gated squared-ReLU
     * — verified MMLU 25.5% (SwiGLU, chance) -> 50% (relu2, ~published ~53%) —
     * while community "bitnet" uses SwiGLU. The official 2B-4T converter / mainline
     * llama.cpp don't emit the activation key (see docs/bitnet_conversion.md). */
    size_t      al = 0, len = 0;
    const char *arch = gguf_get_meta_string(gguf, "general.architecture", &al);
    const char *s    = gguf_get_meta_string(gguf, "bitnet-b1.58.feed_forward_activation", &len);
    if (s == nullptr) {
        s = gguf_get_meta_string(gguf, "bitnet.feed_forward_activation", &len);
    }
    if (s == nullptr) {
        s = gguf_get_meta_string(gguf, "general.feed_forward_activation", &len);
    }
    return geist_ffn_activation_select(arch, al, s, s != nullptr ? len : 0);
}

static void populate_bitnet_b158(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Neutral config + the BitNet additions: SubLN and the arch-keyed
     * FFN activation. Llama-style underneath, but NEOX RoPE (the
     * bitnet.cpp converter does NOT pre-permute Q/K) — the neutral
     * rope_interleaved=false is correct, unlike the llama family. */
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

static bool populate_layers_bitnet_b158(struct transformer_arch_state *st) {
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
    return true;
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

static const struct transformer_family FAMILY_QWEN3 = {
        .name            = "qwen3",
        .populate        = populate_qwen3,
        .populate_layers = populate_layers_qwen3,
};

static const struct transformer_family FAMILY_QWEN35 = {
        .name            = "qwen35",
        .populate        = populate_qwen35,
        .populate_layers = populate_layers_qwen35,
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

static const struct transformer_family *const REGISTRY[] = {
        &FAMILY_GEMMA4,
        &FAMILY_LLAMA,
        &FAMILY_QWEN3,
        &FAMILY_QWEN35,
        &FAMILY_BITNET_B158,
        &FAMILY_BITNET,
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
