/*
 * src/archs/transformer/weight_load.h — internal weight-load surface.
 *
 * Layer: ARCHITECTURE (internal). Implemented in weight_load/; only
 * arch_state.c (state_create) calls into these.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_WEIGHT_LOAD_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_WEIGHT_LOAD_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/weight_load.h is internal to the architecture layer."
#endif

#include "arch_state.h"
#include "gguf_reader.h"

/* Size of the first weight-arena chunk. Without caps.weights_device_copy
 * every tensor lives in the arena, so this is the sum of the file's tensor
 * payloads (64-byte aligned each) plus the F32 widening of small
 * half-precision matrices and headroom: one chunk holds the model. With it,
 * only the tensors the loader marks WEIGHT_BIND do, a decision made per
 * call site that a scan of the file cannot repeat; the arena then starts
 * at one chunk and grows (arena_alloc). */
[[nodiscard]] enum geist_status compute_weight_arena_capacity(const struct geist_backend *be,
                                                              struct gguf_ctx            *gguf,
                                                              size_t *out_bytes);

/* Open the β-mode weight arena with a first chunk of `bytes`. */
[[nodiscard]] enum geist_status weight_arena_open(struct transformer_arch_state *st, size_t bytes);

/* Load one transformer block (layer L) from the GGUF: attention or
 * DeltaNet, FFN, and per-layer norm + scalar tensors. L's geometry is
 * pre-filled by the family's populate_layers. Allocates layer buffers via
 * the backend (arena slice in arena mode, aliased in mmap-alias mode). */
[[nodiscard]] enum geist_status load_one_layer(struct transformer_arch_state    *st,
                                               struct gguf_ctx                  *gguf,
                                               struct transformer_layer_weights *L);

/* Load one trailing Qwen3.5 MTP block. The regular attention/FFN block and
 * its nextn-specific fusion/norm tensors share one owning-buffer list. */
[[nodiscard]] enum geist_status load_mtp_layer(struct transformer_arch_state        *st,
                                               struct gguf_ctx                      *gguf,
                                               struct transformer_mtp_layer_weights *M);

/* Load all non-per-layer tensors: embed_table, output_table, ple_table,
 * model_proj, model_proj_norm, output_norm. Resolves geist_weight wrappers
 * for each. */
[[nodiscard]] enum geist_status
load_globals(struct geist_backend *be, struct gguf_ctx *gguf, struct transformer_arch_state *st);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_WEIGHT_LOAD_H */
