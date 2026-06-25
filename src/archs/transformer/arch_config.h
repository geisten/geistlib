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
geist_ffn_activation_select(const char *arch, size_t arch_len, const char *act, size_t act_len) {
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
};

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_CONFIG_H */
