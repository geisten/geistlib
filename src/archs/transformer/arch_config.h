/*
 * src/archs/transformer/arch_config.h — per-model architecture config.
 *
 * Layer: ARCHITECTURE (internal).
 *
 * Populated at load time from a combination of GGUF tensor shapes
 * (where derivable) and arch-family defaults. The model state holds
 * one instance; every per-call access to a "Gemma-specific numeric
 * knob" goes through st->config.X rather than a global macro, so
 * adding a sibling arch family (Llama / Mistral / Phi) becomes a
 * matter of swapping the populator, not the consumer.
 *
 * Scope at P1.4:
 *   - Gemma-specific numeric knobs (RMS eps, logit softcap, PLE
 *     scales, KV-shared layer mapping) are migrated to this struct.
 *   - Structural dimensions (HIDDEN, NUM_LAYERS, VOCAB, head counts)
 *     remain as compile-time macros in arch_state.h because they
 *     size fixed-length arrays in struct transformer_arch_state.
 *     Migration of those follows in P1.4.b (the struct must lose its
 *     compile-time-sized member arrays first).
 *
 * Future: extend the GGUF reader to expose metadata KV pairs and
 * derive the metadata-only fields (RoPE theta, RMS eps,
 * sliding-window length, logit softcap) from GGUF directly.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_CONFIG_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_CONFIG_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/arch_config.h is internal to the architecture layer."
#endif

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* FFN activation kind. Constexpr-able (single enum byte), consumed by
 * the FFN forward branch in transformer/forward.c.
 *   GEGLU              — gelu_tanh(gate) * up; Gemma 3/4. Default.
 *   SWIGLU             — silu(gate) * up; Llama 2/3, BitNet 3B (community).
 *   SQUARED_RELU       — relu(up)^2; gateless. (Not currently emitted by
 *                        any model we've validated — reserved.)
 *   GATED_SQUARED_RELU — relu(gate)^2 * up; Microsoft BitNet b1.58 2B-4T
 *                        (HF config: hidden_act = "relu2").
 */
enum geist_ffn_activation_kind {
    GEIST_FFN_GEGLU = 0,
    GEIST_FFN_SWIGLU,
    GEIST_FFN_SQUARED_RELU,
    GEIST_FFN_GATED_SQUARED_RELU,
};

/* Pick the FFN activation. An explicit *.feed_forward_activation value (`act`,
 * `act_len`; pass act=nullptr when the GGUF has no such key) wins. Otherwise the
 * default is keyed on `general.architecture`: only "bitnet-b1.58" (Microsoft's
 * official 2B-4T — gated squared-ReLU, and its GGUF carries NO activation key)
 * defaults to GATED_SQUARED_RELU; everything else (community "bitnet", llama,
 * gemma) defaults to SwiGLU. Pure (no GGUF) so it is unit-testable — this is the
 * exact decision that, when it wrongly defaulted 2B-4T to SwiGLU, dropped MMLU
 * to chance. Keep test_bitnet_arch_unit in sync. */
static inline enum geist_ffn_activation_kind
geist_ffn_activation_select(size_t arch_len, size_t act_len, const char *arch, const char *act) {
    enum geist_ffn_activation_kind out =
            (arch != nullptr && arch_len == sizeof("bitnet-b1.58") - 1 &&
             memcmp(arch, "bitnet-b1.58", arch_len) == 0)
                    ? GEIST_FFN_GATED_SQUARED_RELU
                    : GEIST_FFN_SWIGLU;
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
/* Mirrors gguf-py's PoolingType so `{arch}.pooling_type` maps across
 * unchanged. NONE/CLS/RANK are listed for that correspondence; only MEAN and
 * LAST are implemented — the others return GEIST_E_UNSUPPORTED from embed. */
enum geist_pooling {
    GEIST_POOL_NONE = 0,
    GEIST_POOL_MEAN = 1,
    GEIST_POOL_CLS  = 2,
    GEIST_POOL_LAST = 3,
    GEIST_POOL_RANK = 4,
};

[[nodiscard]] static inline bool geist_ffn_needs_relu_squared(enum geist_ffn_activation_kind act) {
    return act == GEIST_FFN_SQUARED_RELU || act == GEIST_FFN_GATED_SQUARED_RELU;
}

struct geist_arch_config {
    /* ---- Family identity. Future use by sub-vtable dispatch
     * (PLE precompute / logit softcap routing). */
    const char *family; /* "gemma4" today; "llama", "mistral", … later */

    /* ---- Numerics. */
    float rms_eps;       /* RMSNorm epsilon. Gemma 4: 1e-6f. */
    float logit_softcap; /* tanh(p/softcap)*softcap; 0 = disabled. */

    /* ---- PLE (Per-Layer Embedding, Gemma 3/4 family only). When
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

    /* ---- Gemma-family extra per-layer norms (P1.5.d). Gemma 3/4
     * adds q_norm and k_norm after the Q/K projections, plus
     * post_attention_norm and post_ffw_norm in the residual pipeline.
     * Llama / Mistral don't have these; the loader skips the tensor
     * lookup and the forward pass skips the rmsnorm calls. */
    bool has_gemma_attn_norms;

    /* ---- Per-head Q/K RMSNorm alone (qwen3, #275). Gates ONLY the
     * attn_q_norm / attn_k_norm load + the pre-RoPE rmsnorm on Q and K.
     * Distinct from has_gemma_attn_norms, which additionally V-norms,
     * drops the 1/sqrt(head_dim) Q scale and adds the post_attention /
     * post_ffw residual norms. Gemma sets both flags; qwen3 only this
     * one (standard scale, plain residual adds). */
    bool has_qk_norms;

    /* ---- BitNet b1.58 family knobs.
     *
     * has_sub_ln: BitNet inserts an extra RMSNorm before each
     *   BitLinear (between attn-output and o_proj, between FFN
     *   activation and down_proj). The forward path skips these
     *   norms when the flag is false. The norm weight tensors are
     *   loaded as L->attn_sub_norm / L->ffn_sub_norm (P1.4 weight loader).
     *
     * ffn_activation: which FFN structure the layer runs. Gemma 3/4
     *   keeps GEGLU (default 0). Llama family is SWIGLU. BitNet b1.58
     *   2B-4T is GATED_SQUARED_RELU (relu(gate)^2 * up; gate/up/down all
     *   present). The gateless SQUARED_RELU variant skips the gate
     *   projection but isn't what the official 2B-4T uses. */
    bool                           has_sub_ln;
    enum geist_ffn_activation_kind ffn_activation;

    /* ---- BitNet-embedding SubLN (microsoft/bitnet-embedding-{270m,0.6b}).
     * Those models put an RMSNorm plus the A8 activation quantization
     * inside EVERY BitLinear, not just the two positions has_sub_ln
     * covers:
     *
     *   x -> RMSNorm(x, norm.weight) -> activation_quant(8 bit) -> matmul
     *
     * for q, k, v, o, gate, up and down (microsoft/BitNet
     * docs/bitnet-embeddings-i2s-guide.md §3.1). Two of the seven sit
     * exactly where has_sub_ln already puts them — o_proj's input and
     * down_proj's input — so those reuse L->attn_sub_norm / L->ffn_sub_norm
     * and only the tensor NAMES differ (*_norm_in vs *_sub_norm). The other
     * five are L->{q,k,v,gate,up}_norm_in.
     *
     * Additive and orthogonal to has_sub_ln: BitNet b1.58 2B-4T keeps its
     * two-norm layout untouched. A family sets one or the other, never
     * both — the two disagree about where the activation quant belongs. */
    bool has_bitlinear_subln;

    /* ---- How geist_session_embed pools the per-position hidden states.
     * Read from the GGUF's `{arch}.pooling_type` key (arch_state.c), whose
     * values are gguf-py's PoolingType — hence the numbering, which must not
     * drift from it. Absent key = MEAN.
     *
     * Worth knowing before trusting either: microsoft/bitnet-embedding-270m
     * and -0.6b both write 1 (MEAN) while their model cards say last-token.
     * The card's own reproducible llama-embedding invocation passes no
     * --pooling override, so what the vendor published as reference output
     * is what the key says. The file wins; the prose is not reproducible. */
    enum geist_pooling pooling;

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

    /* ---- qwen35 hybrid family (#281) ---------------------------------- *
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

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_CONFIG_H */
