/*
 * src/engine/arch_registry.c — compiled-in architecture list.
 *
 * Layer: ENGINE. Mirror of backend_registry.c — the registry is NULL-
 * terminated and ordered by GGUF-name match preference. Lookup matches
 * a GGUF's `general.architecture` against each descriptor's gguf_names
 * list and fails closed on no match — no fallback descriptor.
 *
 * Adding a new architecture:
 *   1. Implement src/archs/<name>/arch.c exporting
 *      'extern const struct geist_arch_ops_decoder geist_arch_<name>'
 *      plus a NULL-terminated 'geist_arch_<name>_gguf_names' list of
 *      the general.architecture values it accepts.
 *   2. Add its sources to mk/common.mk.
 *   3. Add a descriptor + registry entry below.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "arch_registry.h"

#include <string.h>

#define GEIST_INTERNAL_ARCH_LAYER
#include "../archs/audio_conformer/arch.h"
#include "../archs/transformer/arch.h"
#include "../archs/vision_siglip/arch.h"
#undef GEIST_INTERNAL_ARCH_LAYER

static const struct geist_arch_descriptor desc_transformer = {
        .name               = "transformer",
        .gguf_names         = geist_arch_transformer_gguf_names,
        .decoder_ops        = &geist_arch_transformer,
        .audio_encoder_ops  = &geist_arch_audio_conformer,
        .vision_encoder_ops = &geist_arch_vision_siglip,
};

const struct geist_arch_descriptor *const geist_arch_registry[] = {
        &desc_transformer,
        nullptr,
};

const struct geist_arch_descriptor *geist_arch_registry_lookup(const char *gguf_arch) {
    if (gguf_arch == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; geist_arch_registry[i] != nullptr; i++) {
        const struct geist_arch_descriptor *d = geist_arch_registry[i];
        for (const char *const *n = d->gguf_names; n != nullptr && *n != nullptr; n++) {
            if (strcmp(*n, gguf_arch) == 0) {
                return d;
            }
        }
    }
    return nullptr;
}
