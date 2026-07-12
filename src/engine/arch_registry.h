/*
 * src/engine/arch_registry.h — full struct geist_arch_descriptor + lookup.
 *
 * Layer: ENGINE.
 *
 * Defined here so model.c and arch_registry.c share the type. Forward-
 * declared in model.h.
 */
#ifndef GEIST_INTERNAL_ARCH_REGISTRY_H
#define GEIST_INTERNAL_ARCH_REGISTRY_H

#ifndef GEIST_INTERNAL_ENGINE_LAYER
#error "arch_registry.h is internal to the engine layer."
#endif

#include <geist.h>
#include <geist_arch.h> /* geist_arch_ops_{decoder,encoder,vision} vtables */

struct geist_arch_descriptor {
    const char                          *name;
    const struct geist_arch_ops_decoder *decoder_ops;
    const struct geist_arch_ops_encoder *audio_encoder_ops;  /* Phase B-5 */
    const struct geist_arch_ops_vision  *vision_encoder_ops; /* Phase P1 */
};

/* NULL-terminated. Defined in arch_registry.c. */
extern const struct geist_arch_descriptor *const geist_arch_registry[];

/* Return the architecture descriptor for a GGUF general.architecture
 * value. Single-arch build: always the first-registered descriptor
 * (transformer). Returns nullptr only if the registry is empty. */
const struct geist_arch_descriptor *geist_arch_registry_lookup(const char *gguf_arch);

#endif /* GEIST_INTERNAL_ARCH_REGISTRY_H */
