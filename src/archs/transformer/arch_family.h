/*
 * src/archs/transformer/arch_family.h — per-family populator dispatch.
 *
 * Layer: ARCHITECTURE (internal).
 *
 * A `struct transformer_family` is the small sub-vtable that turns a
 * GGUF metadata blob into a populated `struct transformer_arch_state`
 * + matching `struct geist_arch_config`. One entry per architecture
 * family the engine supports.
 *
 * Selection happens at state_create from the `general.architecture`
 * metadata string. Unknown or missing arch fails closed — no family,
 * no load. The engine gate (arch_registry lookup in model.c) rejects
 * such GGUFs before state_create; the nullptr return here is the
 * belt-and-suspenders for callers that bypass the engine.
 *
 * Families differ in:
 *   - which meta keys carry the geometry (`gemma4.*` vs `llama.*` ...)
 *   - the feature flags in geist_arch_config (PLE, logit softcap, KV
 *     sharing, SubLN, DeltaNet, ...); the forward pass reads those flags
 *     to skip family-specific stages.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/arch_family.h is internal to the architecture layer."
#endif

#include "arch_state.h"
#include "prefill_tuning.h"
#include "gguf_reader.h"

struct transformer_layer_forward_ctx;

struct transformer_family {
    /* Lowercase identifier matching the `general.architecture` GGUF
     * value (e.g. "gemma4", "llama"). */
    const char *name;

    /* Populator: read the family-specific metadata keys and write
     * the runtime fields on `st` (n_layers, d_model, n_q_heads, ...)
     * + the family-specific bits of `st->config` (rms_eps,
     * logit_softcap, has_ple, kv_*_src, ...). Called on a NEUTRAL config
     * (every feature off), so the populator declares only what the family
     * has. */
    void (*populate)(struct gguf_ctx *gguf, struct transformer_arch_state *st);

    /* Per-layer geometry filler. Called AFTER state_create
     * has heap-allocated st->layers to the right count. Fills the
     * geometry fields of every layer slot — `is_full`,
     * `is_kv_shared`, `head_dim`, `q_out`, `kv_out`, `intermediate`,
     * `sliding_window`, `rope_theta`, `n_rotated_dims`, `layer_idx`.
     *
     * load_one_layer reads these pre-filled fields, so the layer pattern
     * (Gemma's sliding/full mix + KV sharing vs Llama's uniform full-attn)
     * is a family concern, not the loader's.
     *
     * Returns false when the geometry is NOT derivable (metadata keys
     * missing for a non-default layer count, inconsistent pattern
     * length, ...) — state_create then fails with a message naming
     * the geometry instead of a downstream weight-wiring error (#258). */
    bool (*populate_layers)(struct transformer_arch_state *st);

    /* ---- Family unit hooks (#37).
     *
     * wire_layer (optional): load the family's own per-layer tensors after
     *   the generic set (load_one_layer).
     * attention_block, ffn_block (required): the layer, composed of the
     *   shared blocks in forward/blocks.h. ffn_block owns the layer tail
     *   (PLE, layer scale) where the family has one. They read no family
     *   flag; whatever the family cannot run is refused at load. DeltaNet
     *   layers are dispatched to the DeltaNet block instead of
     *   attention_block. state_create refuses a family without them. */
    /* ---- Family capabilities shared code asks about (#37). Zero = no.
     *
     * rotation: prism.hadamard rotated checkpoints are wired for this
     *   family (rotation.c refuses them otherwise).
     * prefill_delta: a larger prefill chunk from a model size on
     *   (prefill_tuning.h).
     * towers: the bundled audio/vision towers' soft tokens fit this family's
     *   residual stream (geist_arch_ops_decoder.supports_towers). */
    bool                             rotation;
    struct transformer_prefill_delta prefill_delta;
    bool                             towers;

    enum geist_status (*wire_layer)(struct transformer_arch_state    *st,
                                    struct gguf_ctx                  *gguf,
                                    struct transformer_layer_weights *L);
    enum geist_status (*attention_block)(struct transformer_layer_forward_ctx *ctx);
    enum geist_status (*ffn_block)(struct transformer_layer_forward_ctx *ctx);
};

/* Families that live in their own unit (families/<name>.c). */
extern const struct transformer_family transformer_family_bitnet_b158;
extern const struct transformer_family transformer_family_bitnet;
extern const struct transformer_family transformer_family_llama;
extern const struct transformer_family transformer_family_qwen3;
extern const struct transformer_family transformer_family_gemma3;
extern const struct transformer_family transformer_family_gemma4;
extern const struct transformer_family transformer_family_qwen35;
extern const struct transformer_family transformer_family_qwen3moe;

/* arch_family.c — the BitNet embedding variant's metadata (per-projection
 * input norms, pooling), read by the qwen3 and gemma3 populators. */
void populate_bitnet_embedding_meta(struct gguf_ctx               *gguf,
                                    const char                    *arch,
                                    struct transformer_arch_state *st);

/* Select a family by `general.architecture` string. Returns the
 * matching family entry, or nullptr when the key is absent or no
 * entry matches (fail closed — caller aborts the load). */
const struct transformer_family *transformer_family_select(struct gguf_ctx *gguf);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_H */
