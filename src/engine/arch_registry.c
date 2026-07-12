/*
 * src/engine/arch_registry.c — compiled-in architecture list.
 *
 * Layer: ENGINE. Mirror of backend_registry.c — each architecture is
 * gated by GEIST_ARCH_<NAME> at compile time; the registry is NULL-
 * terminated and ordered by GGUF-name match preference.
 *
 * Adding a new architecture:
 *   1. Implement src/archs/<name>/arch.c exporting
 *      'extern const struct geist_arch_ops_decoder geist_arch_<name>'.
 *   2. Add mk/arch-<name>.mk setting SRCS_ARCH += src/archs/<name>/...c
 *   3. Add #if GEIST_ARCH_<NAME> block below.
 *   4. Build with `make ARCHS="transformer <name>"`.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "arch_registry.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../archs/audio_conformer/arch.h"
#include "../archs/transformer/arch.h"
#include "../archs/vision_siglip/arch.h"
#undef GEIST_INTERNAL_ARCH_LAYER

/* Compiled-in list. GEIST_ARCH_TRANSFORMER is defined by default since
 * the only model the engine currently supports is Gemma 4. */
#define GEIST_ARCH_TRANSFORMER 1

#if GEIST_ARCH_TRANSFORMER
static const struct geist_arch_descriptor desc_transformer = {
        .name               = "transformer",
        .decoder_ops        = &geist_arch_transformer,
        .audio_encoder_ops  = &geist_arch_audio_conformer,
        .vision_encoder_ops = &geist_arch_vision_siglip,
};
#endif

const struct geist_arch_descriptor *const geist_arch_registry[] = {
#if GEIST_ARCH_TRANSFORMER
        &desc_transformer,
#endif
        nullptr,
};

const struct geist_arch_descriptor *geist_arch_registry_lookup(const char *gguf_arch) {
    /* ponytail: single-arch build — every GGUF maps to the one registered
     * descriptor (transformer covers gemma/llama/mistral). Restore
     * gguf_arch name-matching over the registry when a second arch lands. */
    (void) gguf_arch;
    return geist_arch_registry[0];
}
