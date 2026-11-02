/*
 * src/archs/transformer/arch_config.h — per-model architecture config.
 *
 * Layer: ARCHITECTURE (internal).
 *
 * Filled at load time by the family populator (arch_family.c) from GGUF
 * metadata and family defaults. The model state holds one instance; the
 * forward pass reads family-specific knobs and feature flags through
 * st->config.X. Structural dimensions live on transformer_arch_state.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_CONFIG_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_CONFIG_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/arch_config.h is internal to the architecture layer."
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* FFN activation kind, consumed by the FFN forward branch
 * (forward/blocks.c).
 *   GEGLU              — gelu_tanh(gate) * up; Gemma 3/4.
 *   SWIGLU             — silu(gate) * up; Llama 2/3, Qwen, BitNet 3B
 *                        (community). The neutral default.
 *   SQUARED_RELU       — relu(up)^2; gateless. No validated model uses it.
 *   GATED_SQUARED_RELU — relu(gate)^2 * up; Microsoft BitNet b1.58 2B-4T
 *                        (HF config: hidden_act = "relu2").
 */
enum geist_ffn_activation_kind {
    GEIST_FFN_GEGLU = 0,
    GEIST_FFN_SWIGLU,
    GEIST_FFN_SQUARED_RELU,
    GEIST_FFN_GATED_SQUARED_RELU,
};

/* How a model turns a sequence of hidden states into one vector.
 *
 *   NONE       — generative model: no pooling, the forward pass ends in the
 *                LM head and the session emits tokens.
 *   LAST_TOKEN — the last non-padding position's hidden state, L2-normalised.
 *                What the BitNet embedding models use.
 *   MEAN       — the mean over all prefilled positions, L2-normalised.
 */
enum geist_pooling_kind {
    GEIST_POOLING_NONE = 0,
    GEIST_POOLING_LAST_TOKEN,
    GEIST_POOLING_MEAN,
};

/* Map a GGUF `*.pooling` string to a kind. `s`/`len` is the raw metadata
 * value; pass s=nullptr when the key is absent, which yields NONE (a
 * generative model).
 *
 * Returns false for a value this build does not understand, so the loader
 * refuses the model instead of guessing a pooling that would produce
 * plausible-looking but wrong vectors. Pure, so it is unit-testable. */
static inline bool geist_pooling_select(size_t len, const char *s, enum geist_pooling_kind *out) {
    if (s == nullptr || len == 0) {
        *out = GEIST_POOLING_NONE;
        return true;
    }
    if (len == sizeof("last_token") - 1 && memcmp(s, "last_token", len) == 0) {
        *out = GEIST_POOLING_LAST_TOKEN;
        return true;
    }
    if (len == sizeof("mean") - 1 && memcmp(s, "mean", len) == 0) {
        *out = GEIST_POOLING_MEAN;
        return true;
    }
    return false;
}

/* Map gguf-py's numeric PoolingType to a kind. That is the `{arch}.pooling_type`
 * key upstream's own published GGUFs carry, where ours writes the
 * `bitnet.embedding.pooling` string — a file converted by anyone but
 * tools/convert_bitnet_embedding.py has only this one.
 *
 * Values are gguf-py/gguf/constants.py: NONE=0, MEAN=1, CLS=2, LAST=3, RANK=4.
 * CLS and RANK are not implemented, so they are refused here exactly as an
 * unrecognised string is — a pooling this build cannot perform must fail the
 * load, not be approximated. Pure, so it is unit-testable. */
static inline bool geist_pooling_from_gguf_type(uint32_t t, enum geist_pooling_kind *out) {
    switch (t) {
    case 0u:
        *out = GEIST_POOLING_NONE;
        return true;
    case 1u:
        *out = GEIST_POOLING_MEAN;
        return true;
    case 3u:
        *out = GEIST_POOLING_LAST_TOKEN;
        return true;
    default:
        *out = GEIST_POOLING_NONE;
        return false;
    }
}

/* True iff this pooling kind makes the model an embedding model: no LM
 * head, no tokens, a vector out. */
static inline bool geist_pooling_is_embedding(enum geist_pooling_kind k) {
    return k != GEIST_POOLING_NONE;
}

/* Pick the FFN activation. An explicit *.feed_forward_activation value (`act`,
 * `act_len`; pass act=nullptr when the GGUF has no such key) wins; otherwise,
 * and for a value it does not know, the family's default `dflt` (BitNet:
 * gated squared-ReLU for "bitnet-b1.58", SwiGLU for community "bitnet").
 * Pure (no GGUF) so it is unit-testable. Keep test_bitnet_arch_unit in sync. */
static inline enum geist_ffn_activation_kind
geist_ffn_activation_select(size_t act_len, enum geist_ffn_activation_kind dflt, const char *act) {
    enum geist_ffn_activation_kind out = dflt;
    if (act == nullptr) {
        return out;
    }
    if (act_len == sizeof("swiglu") - 1 && memcmp(act, "swiglu", act_len) == 0) {
        out = GEIST_FFN_SWIGLU;
    } else if (act_len == sizeof("squared_relu") - 1 && memcmp(act, "squared_relu", act_len) == 0) {
        out = GEIST_FFN_SQUARED_RELU;
    } else if (act_len == sizeof("geglu") - 1 && memcmp(act, "geglu", act_len) == 0) {
        out = GEIST_FFN_GEGLU;
    } else if ((act_len == sizeof("relu2") - 1 && memcmp(act, "relu2", act_len) == 0) ||
               (act_len == sizeof("gated_squared_relu") - 1 &&
                memcmp(act, "gated_squared_relu", act_len) == 0)) {
        out = GEIST_FFN_GATED_SQUARED_RELU;
    }
    return out;
}

/* True iff this activation needs the relu_squared primitive. It has no
 * bound alternative — it cannot be composed from the other prims (there is
 * no relu, and no max) — so a backend that lacks it cannot run such a
 * model, and transformer_exec_plan_build refuses at load rather than
 * calling a null pointer per layer (#352). Pure, so the list of kinds that
 * need it is testable without a backend. */
[[nodiscard]] static inline bool geist_ffn_needs_relu_squared(enum geist_ffn_activation_kind act) {
    return act == GEIST_FFN_SQUARED_RELU || act == GEIST_FFN_GATED_SQUARED_RELU;
}

struct geist_arch_config {
    /* ---- Numerics. */
    float rms_eps;       /* RMSNorm epsilon. Gemma 4: 1e-6f. */
    float logit_softcap; /* tanh(p/softcap)*softcap; 0 = disabled. */

    /* ---- Embedding scale: multiply looked-up token embeddings by
     * sqrt(d_model). The Gemma families do this; Llama/Qwen/BitNet do not.
     * Independent of has_ple: Gemma 3 has the scale and no PLE. */
    bool has_embed_scale;

    /* ---- PLE (Per-Layer Embedding, Gemma 4 family only). When
     * has_ple == false, the precompute path is skipped entirely. */
    bool  has_ple;
    float ple_input_scale;      /* multiplied onto (model_proj + lookup) */
    float ple_model_proj_scale; /* multiplied onto model_proj(h) */
    float ple_table_scale;      /* multiplied onto dequant'd PLE row */

    /* ---- KV-shared layer mapping (Gemma 4 only). When a layer has
     * `is_kv_shared == true`, its K/V cache aliases the source layer's.
     * The pattern is full-attn vs sliding-attn, with two distinct
     * source layers. -1 = sharing not used by this family. */
    int kv_sliding_src;
    int kv_full_src;

    /* ---- Gemma-family extra per-layer norms. Gemma 3/4
     * adds q_norm and k_norm after the Q/K projections, plus
     * post_attention_norm and post_ffw_norm in the residual pipeline.
     * Other families don't have these; the loader skips the tensor
     * lookup and the forward pass skips the rmsnorm calls. */
    bool has_gemma_attn_norms;

    /* ---- Gemma 3 attention (#674): the Gemma norms above, but with the
     * queries scaled by query_pre_attn_scalar^-1/2 and no V norm — Gemma 4
     * folds the scale into its Q/K norms and normalizes V, Gemma 3 (as
     * Hugging Face and llama.cpp run it) does neither. 0 = not Gemma 3. */
    float gemma3_q_scale;

    /* ---- Per-head Q/K RMSNorm alone (qwen3). Gates ONLY the
     * attn_q_norm / attn_k_norm load + the pre-RoPE rmsnorm on Q and K.
     * Distinct from has_gemma_attn_norms, which additionally V-norms,
     * drops the 1/sqrt(head_dim) Q scale and adds the post_attention /
     * post_ffw residual norms. Gemma sets both flags; qwen3 only this
     * one (standard scale, plain residual adds). */
    bool has_qk_norms;
    /* The QK norms run over a token's whole q/k row, before the split into
     * heads, instead of per head (OLMoE). */
    bool qk_norm_whole_row;

    /* ---- BitNet b1.58 family knobs.
     *
     * has_sub_ln: BitNet inserts an extra RMSNorm before each
     *   BitLinear (between attn-output and o_proj, between FFN
     *   activation and down_proj). The forward path skips these
     *   norms when the flag is false. The norm weight tensors are
     *   loaded as L->attn_sub_norm / L->ffn_sub_norm.
     *
     * ffn_activation: which FFN structure the layer runs. Gemma 3/4
     *   use GEGLU. Llama family is SWIGLU. BitNet b1.58
     *   2B-4T is GATED_SQUARED_RELU (relu(gate)^2 * up; gate/up/down all
     *   present). The gateless SQUARED_RELU variant skips the gate
     *   projection but isn't what the official 2B-4T uses. */
    bool                           has_sub_ln;
    enum geist_ffn_activation_kind ffn_activation;

    /* ---- BitNet embedding models: an RMSNorm on the INPUT of
     * every BitLinear projection, seven per layer — the pattern upstream
     * calls "per-projection norms" and that no standard architecture has
     * (docs/BITNET_EMBEDDINGS_PLAN.md).
     *
     * Two of the seven already have a home: the norm before o_proj is
     * shaped [q_out] and the one before down_proj is [intermediate],
     * which is exactly where has_sub_ln's attn_sub_norm / ffn_sub_norm
     * sit. Those two load into the existing fields and the existing
     * forward code applies them, so this flag adds only the five that
     * precede q, k, v, gate and up — all shaped [d_model].
     *
     * The five are the reason the flag also has a cost: q/k/v otherwise
     * share one normalised input (and so do gate/up), which is what lets the
     * fused triple-QKV and gate_up kernels run. Per-projection norms give
     * each its own input, so those fusions are disabled while this is
     * set — see exec_plan.c. */
    bool has_projection_input_norms;

    /* ---- Mixture of experts (#48). moe_n_experts == 0 is a dense FFN.
     * Otherwise the layer has no ffn_gate/up/down: a router (ffn_gate_inp)
     * scores moe_n_experts expert FFNs per token, softmax over all of them,
     * keeps the moe_n_used best and, when moe_norm_topk, renormalises their
     * weights to sum 1 (blk_moe_ffn). Each expert is a SwiGLU FFN of width
     * transformer_layer_weights.intermediate. */
    size_t moe_n_experts;
    size_t moe_n_used;
    bool   moe_norm_topk;
    /* 0 or 1: a shared expert every token runs, weighted by
     * sigmoid(x . moe_shexp_gate) (qwen35moe, #169). It is routed as one
     * more expert, slot moe_n_experts, so every placement of the routed
     * experts (CPU, hot on a GPU) takes it as it is. */
    size_t moe_n_shared;
    /* A family refuses more experts than this at load (blk_moe_ffn lists
     * the active experts of a layer on the stack). */
#define GEIST_MOE_MAX_EXPERTS 512
    /* How the router's logits become the selection and its weights
     * (llama.cpp's expert_gating_func, moe_route):
     *   SOFTMAX         p = softmax(logits) over all experts; the k largest
     *                   p, weights p (renormalised when moe_norm_topk).
     *   SOFTMAX_WEIGHT  the k largest logits; weights the softmax over those
     *                   k logits alone (GPT-OSS, #243).
     * The router's bias (ffn_gate_inp.bias), where a layer has one, is added
     * to the logits first. Values are the GGUF key's. */
    enum geist_moe_gating {
        GEIST_MOE_GATING_SOFTMAX        = 1,
        GEIST_MOE_GATING_SIGMOID        = 2,
        GEIST_MOE_GATING_SOFTMAX_WEIGHT = 3,
    } moe_gating;
    /* DeepSeek-V3's router (#242), as llama.cpp's build_moe_ffn runs it:
     * moe_w_scale multiplies the final weights (0: 1, i.e. none); with
     * moe_n_group > 1 the experts form that many equal groups, a group
     * scores the sum of its two best selection scores and only the
     * moe_topk_group best groups stay eligible. A per-layer selection bias
     * (exp_probs_b, transformer_layer_weights.moe_sel_b) shifts the scores
     * for the selection alone; the weights are the unbiased scores. */
    float  moe_w_scale;
    size_t moe_n_group, moe_topk_group;
    /* The expert FFN's activation (moe_expert_rows):
     *   SWIGLU      silu(gate) * up.
     *   SWIGLU_OAI  g = min(gate, limit), u = clamp(up, -limit, limit),
     *               g * sigmoid(alpha * g) * (u + 1), alpha 1.702, limit 7
     *               (GPT-OSS, llama.cpp's ggml_swiglu_oai, #243). */
    enum geist_moe_act {
        GEIST_MOE_ACT_SWIGLU = 0,
        GEIST_MOE_ACT_SWIGLU_OAI,
    } moe_act;

    /* ---- Pooling. A model with a pooling kind other than NONE is an
     * EMBEDDING model: it has no LM head, emits no tokens, and its forward
     * pass ends at output_norm + pooling + L2 normalisation instead of a
     * vocab projection.
     *
     * Read from the GGUF rather than hardcoded so a model with different
     * pooling does not need a second API — an unimplemented kind is
     * refused loudly at load, never silently treated as last-token. */
    enum geist_pooling_kind pooling;
    /* The GGUF named a pooling this build does not implement. Set instead
     * of guessing; the layer populator turns it into a load failure. */
    bool pooling_unsupported;

    /* RoPE pair convention.
     *   false: NEOX-style split pairs (i, i + head_dim/2). Gemma 3/4,
     *          BitNet 2B-4T, every arch where llama.cpp's
     *          `llama_rope_type()` returns LLAMA_ROPE_TYPE_NEOX. This
     *          matches HF transformers' standard `rotate_half`.
     *   true:  NORM-style interleaved pairs (2i, 2i+1). Arches where
     *          llama.cpp returns LLAMA_ROPE_TYPE_NORM (notably
     *          LLM_ARCH_LLAMA itself). llama.cpp's convert script
     *          permutes HF Q/K weights at conversion time
     *          (LlamaModel.permute) so that interleaved RoPE on the
     *          permuted weights yields the same result as HF's split
     *          RoPE on original weights. arch="llama" GGUFs ship
     *          pre-permuted — we have to apply interleaved RoPE to
     *          match. */
    bool rope_interleaved;
    /* Rows the loader reorders for interleaved RoPE (load_layer_matrix_rope_il,
     * #242): 0 = the whole group of rows (a llama head); else only the last
     * rope_il_n rows of each group rotate (DeepSeek-V2: a q head is
     * [nope | pe], kv_a_mqa [latent | k_pe]) and they move to the front,
     * deinterleaved, so the forward pass rotates the first rope_il_n dims
     * with half-split pairs. */
    size_t rope_il_n;

    /* How a PARTIAL rotation (n_rotated_dims < head_dim) is laid out. The
     * two shipped families disagree, and each matches its own reference:
     *
     *   false (Gemma 4): the cos/sin table is head_dim wide, frequencies
     *     beyond the rotated quarter are zero, and the rotation pairs
     *     channel i with i + head_dim/2. Validated against the reference
     *     logit dumps in test_full_logits_gguf_int.
     *
     *   true (qwen35): ggml's layout — the table is n_rotated_dims wide,
     *     pairs are (i, i + n_rotated_dims/2) inside that block, the
     *     frequency exponent divides by n_rotated_dims, and channels at
     *     or above it are left alone. The Gemma layout on qwen35 costs
     *     in-context recall (#432).
     *
     * With n_rotated_dims == head_dim both collapse to the same thing. */
    bool rope_partial_block;

    /* YaRN context extension (rope.scaling.type = yarn; GPT-OSS, #243), as
     * ggml's rope_yarn computes it: rope_yarn_factor > 1 turns it on.
     * Frequency i (pair index) blends the interpolated angle (divided by the
     * factor) with the original one by a ramp over the correction dims of
     * rope_yarn_orig_ctx / beta_fast / beta_slow, and cos/sin are scaled by
     * rope_yarn_mscale (llama.cpp's attn_factor times 1 + 0.1 ln factor). */
    float  rope_yarn_factor;
    size_t rope_yarn_orig_ctx;
    float  rope_yarn_beta_fast;
    float  rope_yarn_beta_slow;
    float  rope_yarn_mscale;

    /* ---- Multi-head latent attention (DeepSeek-V2/V3, #242). mla_kv_rank
     * == 0: none. Otherwise K and V come from a compressed latent: per
     * token kv_a = [k_pe (mla_rope) | latent (mla_kv_rank)], the latent
     * RMS-normalised and expanded by kv_b to every head's [k_nope | v];
     * q (from q_a / q_a_norm / q_b when mla_q_rank > 0) is per head
     * [q_pe | q_nope] after the load-time row reorder (rope_il_n). The
     * cache holds full heads of head_dim = mla_rope + nope; V heads of
     * mla_v_dim < head_dim are zero-padded to it, and the first mla_v_dim
     * of each output head go to o_proj. attn_q_scale replaces
     * 1/sqrt(head_dim) (YaRN's mscale squared over it). */
    size_t mla_kv_rank;
    size_t mla_q_rank;
    size_t mla_rope;
    size_t mla_v_dim;
    float  attn_q_scale;

    /* ---- qwen35 hybrid family ------------------------------------------ *
     *
     * has_attn_output_gate: the attention q_proj jointly produces
     *   query + a per-head gate (2x rows, per-head layout
     *   [query(hd) | gate(hd)]); after attention the concatenated head
     *   outputs are multiplied by sigmoid(gate) before o_proj.
     *
     * dn_*: gated-DeltaNet geometry, uniform across the family's
     *   linear-attention layers (mixer == GEIST_MIXER_DELTANET).
     *   Derived: key_dim = dn_n_k_heads*dn_head_k, value_dim =
     *   dn_n_v_heads*dn_head_v, conv_dim = 2*key_dim + value_dim. */
    bool   has_attn_output_gate;
    size_t dn_n_k_heads;   /* linear_num_key_heads   (0.8B: 16) */
    size_t dn_n_v_heads;   /* linear_num_value_heads (0.8B: 16) */
    size_t dn_head_k;      /* linear_key_head_dim    (0.8B: 128) */
    size_t dn_head_v;      /* linear_value_head_dim  (0.8B: 128) */
    size_t dn_conv_kernel; /* causal depthwise conv width (4) */
};

/* The MoE's expert slots (routed, then shared) and selections per token. */
static inline size_t moe_slots(const struct geist_arch_config *c) {
    return c->moe_n_experts + c->moe_n_shared;
}
static inline size_t moe_k(const struct geist_arch_config *c) {
    return c->moe_n_used + c->moe_n_shared;
}

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_CONFIG_H */
