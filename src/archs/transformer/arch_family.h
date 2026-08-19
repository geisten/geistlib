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
 *   - which meta keys carry the geometry (`gemma4.*` vs `llama.*` vs
 *     `mistral.*`)
 *   - the Gemma-specific bits (PLE precompute, logit softcap, KV-
 *     shared layer pattern) — these flip via the config flags
 *     `has_ple`, `logit_softcap > 0`, and (forthcoming) the per-
 *     layer mask. Forward.c reads those flags to skip family-
 *     specific stages.
 *
 * P1.5: registry + dispatch + softcap guard land. Llama / Mistral
 * populators are TODO — see ARCHITECTURE.md P1.5.b/.c.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/arch_family.h is internal to the architecture layer."
#endif

#include "arch_state.h"
#include "gguf_reader.h"

struct transformer_family {
    /* Lowercase identifier matching the `general.architecture` GGUF
     * value (e.g. "gemma4", "llama", "mistral"). */
    const char *name;

    /* Populator: read the family-specific metadata keys and write
     * the runtime fields on `st` (n_layers, d_model, n_q_heads, ...)
     * + the family-specific bits of `st->config` (rms_eps,
     * logit_softcap, has_ple, kv_*_src, ...). Called AFTER state has
     * Gemma-4 defaults already installed, so the populator only needs
     * to override what differs. */
    void (*populate)(struct gguf_ctx *gguf, struct transformer_arch_state *st);

    /* Per-layer geometry filler (P1.5.c). Called AFTER state_create
     * has heap-allocated st->layers to the right count. Fills the
     * geometry fields of every layer slot — `is_full`,
     * `is_kv_shared`, `head_dim`, `q_out`, `kv_out`, `intermediate`,
     * `sliding_window`, `rope_theta`, `n_rotated_dims`, `layer_idx`.
     *
     * weight_load.c::load_one_layer reads these pre-filled fields
     * instead of deriving them; that way the layer pattern (Gemma's
     * sliding/full mix + KV sharing vs Llama's uniform full-attn)
     * becomes a family concern, not the loader's.
     *
     * Returns false when the geometry is NOT derivable (metadata keys
     * missing for a non-default layer count, inconsistent pattern
     * length, ...) — state_create then fails with a message naming
     * the geometry instead of a downstream weight-wiring error (#258). */
    bool (*populate_layers)(struct transformer_arch_state *st);
};

/* Select a family by `general.architecture` string. Returns the
 * matching family entry, or nullptr when the key is absent or no
 * entry matches (fail closed — caller aborts the load). */
const struct transformer_family *transformer_family_select(struct gguf_ctx *gguf);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_H */
