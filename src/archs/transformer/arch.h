/*
 * src/archs/transformer/arch.h — transformer decoder architecture.
 *
 * Layer: ARCHITECTURE. Implements the geist_arch_ops_decoder vtable for the
 * transformer-style decoders in arch_family.c (Gemma, Llama, Qwen, BitNet).
 *
 * The vtable shape itself lives in <geist_arch.h> (engine-owned interface);
 * this header only exports the concrete descriptor.
 *
 * Defined in (src/archs/transformer/):
 *   arch.c        — descriptor; thunks into arch_ops.c and forward/
 *   arch_ops.c    — op entry points (prefill, decode, pin_prefix, reset, ...)
 *   arch_state.c  — model state from a GGUF, session lifecycle
 *   arch_family.c — `general.architecture` → family registry, hparams
 *   weight_load/  — GGUF tensors → backend buffers and weight views
 *   exec_plan.c, scratch_plan.c, rotation.c — per-layer plan, scratch
 *                   sizing, prism.hadamard rotation
 *   forward/      — the per-token forward pass (step.c first)
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/arch.h is internal to the architecture layer."
#endif

#include <geist.h>
#include <geist_arch.h>

/* Concrete descriptor for the transformer decoder. */
extern const struct geist_arch_ops_decoder geist_arch_transformer;

/* NULL-terminated list of the exact `general.architecture` values the
 * transformer arch accepts ("gemma4", "llama", ...). Single source of
 * truth in arch_family.c, next to the family registry it mirrors; the
 * engine's arch_registry gate matches against it fail-closed. */
extern const char *const geist_arch_transformer_gguf_names[];

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_H */
