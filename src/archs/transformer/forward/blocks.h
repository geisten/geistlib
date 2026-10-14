/*
 * src/archs/transformer/forward/blocks.h - shared layer building blocks.
 *
 * Layer: ARCHITECTURE (transformer forward).
 *
 * The pieces a family unit (families/<name>.c) composes its layer from
 * (#37). A block knows HOW an op runs on the session's buffers: tensor
 * views over ctx->seq rows, backend dispatch, host fallbacks. It never
 * knows WHICH family calls it: no block reads a family flag. The order of
 * the blocks is the family's business.
 *
 * Every block works on rows [ctx->seq, n] of session or layer buffers and
 * returns the backend status; outputs may alias inputs where noted.
 */
#pragma once

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "forward/blocks.h is a private architecture-layer header"
#endif

#include "internal.h"

/* y = rmsnorm(x) * w over rows of n; y may be x. */
[[nodiscard]] enum geist_status blk_rmsnorm(struct transformer_layer_forward_ctx *ctx,
                                            size_t                                n,
                                            struct geist_buffer                  *x,
                                            const struct geist_tensor            *w,
                                            struct geist_buffer                  *y);

/* y = a + b over rows of n; y may be a or b. */
[[nodiscard]] enum geist_status blk_add(struct transformer_layer_forward_ctx *ctx,
                                        size_t                                n,
                                        struct geist_buffer                  *a,
                                        struct geist_buffer                  *b,
                                        struct geist_buffer                  *y);

/* y = x * W^T for a projection loaded as [n_out, n_in]; dims come from w. */
[[nodiscard]] enum geist_status blk_linear(struct transformer_layer_forward_ctx *ctx,
                                           struct geist_buffer                  *x,
                                           const struct geist_tensor            *w,
                                           const struct geist_weight            *wq,
                                           struct geist_buffer                  *y);

/* x[t, j] *= scale[j] over rows of n (AWQ inverse scales); no-op for nullptr. */
void blk_scale_cols(struct transformer_layer_forward_ctx *ctx,
                    size_t                                n,
                    const float                          *scale,
                    struct geist_buffer                  *x);

/* scratch_q / scratch_k / scratch_v = x projected by the layer's q/k/v. */
[[nodiscard]] enum geist_status blk_qkv(struct transformer_layer_forward_ctx *ctx,
                                        struct geist_buffer                  *x);

/* RoPE on q, q *= q_scale, RoPE on k, in that order (scratch_q, scratch_k). */
[[nodiscard]] enum geist_status blk_rope_qk(struct transformer_layer_forward_ctx *ctx,
                                            float                                 q_scale);

/* Append k/v to the session KV cache (any KV mode) and attend:
 * scratch_attn = attention(scratch_q, cache). */
[[nodiscard]] enum geist_status blk_attend(struct transformer_layer_forward_ctx *ctx);

/* scratch_gate / scratch_up = x projected by the layer's gate/up. */
[[nodiscard]] enum geist_status blk_gate_up(struct transformer_layer_forward_ctx *ctx,
                                            struct geist_buffer                  *x);

/* x = relu(x)^2 over rows of n, in place. */
[[nodiscard]] enum geist_status
blk_relu_sq(struct transformer_layer_forward_ctx *ctx, size_t n, struct geist_buffer *x);

/* gate = gate * up over rows of n. */
[[nodiscard]] enum geist_status blk_mul(struct transformer_layer_forward_ctx *ctx,
                                        size_t                                n,
                                        struct geist_buffer                  *gate,
                                        struct geist_buffer                  *up);

/* gate = silu(gate) * up over rows of n; the fused op when the plan bound it. */
[[nodiscard]] enum geist_status blk_silu_mul(struct transformer_layer_forward_ctx *ctx,
                                             size_t                                n,
                                             struct geist_buffer                  *gate,
                                             struct geist_buffer                  *up);
