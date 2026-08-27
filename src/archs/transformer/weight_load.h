/*
 * src/archs/transformer/weight_load.h — internal weight-load surface.
 *
 * Layer: ARCHITECTURE (internal). Owned by weight_load.c; only
 * arch_state.c (state_create) calls into these.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_WEIGHT_LOAD_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_WEIGHT_LOAD_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/weight_load.h is internal to the architecture layer."
#endif

#include "arch_state.h"
#include "gguf_reader.h"

/* Pre-scan the GGUF to compute the total byte budget for the backend
 * weight arena. Sum of every weight tensor's payload + per-tensor
 * 64-byte alignment slack. */
[[nodiscard]] enum geist_status compute_weight_arena_capacity(struct gguf_ctx *gguf,
                                                              size_t          *out_bytes);

/* Load one transformer block (layer L) from the GGUF. Reads attention,
 * FFN, and per-layer norm + scalar tensors; populates L->is_full,
 * L->head_dim, L->q_out, L->kv_out, L->intermediate. Allocates layer
 * buffers via the backend (arena slice when arena mode, raw upload
 * when mmap-alias mode). */
[[nodiscard]] enum geist_status load_one_layer(struct transformer_arch_state    *st,
                                               struct gguf_ctx                  *gguf,
                                               struct transformer_layer_weights *L);

/* Load one trailing Qwen3.5 MTP block. The regular attention/FFN block and
 * its nextn-specific fusion/norm tensors share one owning-buffer list. */
[[nodiscard]] enum geist_status load_mtp_layer(struct transformer_arch_state        *st,
                                               struct gguf_ctx                      *gguf,
                                               struct transformer_mtp_layer_weights *M);

/* Load all non-per-layer tensors: embed_table, ple_table, model_proj,
 * model_proj_norm, output_norm, and the per-layer-input embed for the
 * audio path. Resolves geist_weight wrappers for each. */
[[nodiscard]] enum geist_status
load_globals(struct geist_backend *be, struct gguf_ctx *gguf, struct transformer_arch_state *st);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_WEIGHT_LOAD_H */
