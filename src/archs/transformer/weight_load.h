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
 * at one chunk and grows (arena_alloc). Only the tensors `st` holds
 * count: the layers in [layer_lo, layer_hi) and, but on a split stage, the
 * globals (#49). */
[[nodiscard]] enum geist_status
compute_weight_arena_capacity(const struct geist_backend          *be,
                              struct gguf_ctx                     *gguf,
                              const struct transformer_arch_state *st,
                              size_t                              *out_bytes);

/* Open the β-mode weight arena with a first chunk of `bytes`. */
[[nodiscard]] enum geist_status weight_arena_open(struct transformer_arch_state *st, size_t bytes);

/* Load one transformer block (layer L) from the GGUF: the family's own
 * tensors (its wire_layer hook), then the ones every family shares —
 * attn_norm, the attention projections and Q/K norms of a softmax-attention
 * layer, ffn_norm and the dense FFN — skipping a slot the family filled.
 * L's geometry is pre-filled by the family's populate_layers. Allocates
 * layer buffers via the backend (arena slice in arena mode, aliased in
 * mmap-alias mode). */
[[nodiscard]] enum geist_status load_one_layer(struct transformer_arch_state    *st,
                                               struct gguf_ctx                  *gguf,
                                               struct transformer_layer_weights *L);

/* Load a family's own per-layer projection `name` ([n_out, n_in], any
 * supported dtype) and resolve its kernel, like load_one_layer's own. With
 * `exact` the backend must keep the weight's precision (GEIST_W_EXACT). */
[[nodiscard]] enum geist_status load_layer_matrix(struct transformer_arch_state    *st,
                                                  struct gguf_ctx                  *gguf,
                                                  struct transformer_layer_weights *L,
                                                  const char                       *name,
                                                  size_t                            n_out,
                                                  size_t                            n_in,
                                                  bool                              exact,
                                                  struct geist_tensor              *out_view,
                                                  struct geist_weight              *out_weight);

/* Load the 3-D expert tensor `name`, [n_experts][n_out][n_in] in row-major
 * order, as n_experts projections: one buffer (tracked by L), a heap array
 * of n_experts views into it and one of resolved weights, both owned by L
 * (freed with the layer). Fails when a backend has no kernel for the
 * experts' dtype, naming the tensor. */
[[nodiscard]] enum geist_status load_layer_experts(struct transformer_arch_state    *st,
                                                   struct gguf_ctx                  *gguf,
                                                   struct transformer_layer_weights *L,
                                                   const char                       *name,
                                                   size_t                            n_experts,
                                                   size_t                            n_out,
                                                   size_t                            n_in,
                                                   struct geist_tensor             **out_views,
                                                   struct geist_weight             **out_weights);

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
