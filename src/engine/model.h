/*
 * src/engine/model.h — internal definition of struct geist_model + arch
 * descriptor table.
 *
 * Layer: ENGINE. Cannot include Architecture or Backend internals.
 *
 * Defined in:
 *   src/engine/model.c          — load, lifecycle, GGUF arch detection
 *   src/engine/arch_registry.c  — compiled-in architecture list (#if-gated)
 */
#ifndef GEIST_INTERNAL_MODEL_H
#define GEIST_INTERNAL_MODEL_H

#ifndef GEIST_INTERNAL_ENGINE_LAYER
#error "model.h is internal to the engine layer. Use <geist.h> from outside."
#endif

#include <geist.h>
#include <geist_types.h>

/* Forward decls for arch op-tables — Architecture layer fills these in. */
struct geist_arch_ops_decoder;
struct geist_arch_ops_encoder;
struct geist_arch_ops_vision;

/* Composite Model structure: text decoder + optional encoders + projectors. */
struct geist_model {
    /* Required: the text decoder. */
    struct {
        const struct geist_arch_ops_decoder *arch_ops;
        void                                *arch_meta;
    } text_decoder;

    /* Optional: audio encoder (Conformer for Gemma 4). nullptr if absent. */
    struct {
        const struct geist_arch_ops_encoder *arch_ops;
        void                                *arch_meta;
    } audio_encoder;

    /* Optional: vision encoder (SigLIP-derived ViT for Gemma 4 vision).
     * nullptr if vision_tower.safetensors not found. */
    struct {
        const struct geist_arch_ops_vision *arch_ops;
        void                               *arch_meta;
    } vision_encoder;

    /* Engine-side state (struct model_engine_state in model.c). */
    void *weights;

    /* The loaded tokenizer: sp_bpe_tokenizer or gguf_tokenizer. */
    void *tokenizer;

    /* Owning backend (weights live in its buffers). */
    struct geist_backend *backend;
};

/* Architecture registry — array of descriptors, gated by #if GEIST_ARCH_*.
 * Defined in src/engine/arch_registry.c. NULL-terminated. */
struct geist_arch_descriptor;
extern const struct geist_arch_descriptor *const geist_arch_registry[];

/* External tokenizer.bin tokenizer. nullptr if not loaded. */
struct sp_bpe_tokenizer;
struct sp_bpe_tokenizer *geist_model_internal_tokenizer(struct geist_model *m);

/* GGUF-embedded tokenizer. Tried first; tokenizer.bin is the fallback. */
struct gguf_tokenizer;
struct gguf_tokenizer *geist_model_internal_gguf_tokenizer(struct geist_model *m);

/* Internal-test accessor: returns the text decoder's arch_meta pointer
 * (e.g. transformer_arch_state*). Tests in tests/ use this to reach
 * internal primitives like transformer_verify_forward that aren't
 * exposed in the public session API. Production code must not depend on
 * this — go through arch_ops vtable instead. */
void *geist_model_internal_arch_meta(struct geist_model *m);

/* Engine-internal: the arch session handle this geist_session dispatches
 * with (its own arch session, or the arch_state for single-session
 * archs). For tests that cross from the engine into the arch layer. */
void *geist_session_internal_arch_session(struct geist_session *s);

#endif /* GEIST_INTERNAL_MODEL_H */
