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
 *
 * Blocks may read MODEL FEATURES that any family can carry — AWQ scales,
 * prism rotation, per-projection input norms (BitNet embedding models), a
 * SubLN slot, the KV mode, plan-bound fusions — but never which family is
 * running.
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

/* Per-head RMSNorm over head_dim on scratch_q and scratch_k (QK-norm). */
[[nodiscard]] enum geist_status blk_qk_norm(struct transformer_layer_forward_ctx *ctx,
                                            const struct geist_tensor            *q_w,
                                            const struct geist_tensor            *k_w);

/* q: optional per-head RMSNorm (q_norm, nullptr for none), RoPE, then
 * q *= q_scale unless q_scale is 1 (scratch_q). */
[[nodiscard]] enum geist_status blk_q_prep(struct transformer_layer_forward_ctx *ctx,
                                           const struct geist_tensor            *q_norm,
                                           float                                 q_scale);

/* k: optional per-head RMSNorm (k_norm), optional scale-free RMSNorm of v
 * (v_norm), then RoPE on k (scratch_k, scratch_v). */
[[nodiscard]] enum geist_status blk_k_prep(struct transformer_layer_forward_ctx *ctx,
                                           const struct geist_tensor            *k_norm,
                                           bool                                  v_norm);

/* RoPE on q, q *= q_scale, RoPE on k, in that order (scratch_q, scratch_k). */
[[nodiscard]] enum geist_status blk_rope_qk(struct transformer_layer_forward_ctx *ctx,
                                            float                                 q_scale);

/* The plan-bound fused q/k/v prep: per-head norms, V norm, RoPE and the KV
 * append in one backend op; q only on a KV-shared layer. */
[[nodiscard]] enum geist_status blk_qkv_prep_fused(struct transformer_layer_forward_ctx *ctx);

/* Llama / Qwen3: q/k norms (where q_norm / k_norm are non-null), RoPE, the
 * q scale and the KV append in one fused->attn_rope_append call (#17);
 * blk_attend(ctx, false) follows. */
[[nodiscard]] enum geist_status blk_rope_append_fused(struct transformer_layer_forward_ctx *ctx,
                                                      const struct geist_tensor            *q_norm,
                                                      const struct geist_tensor            *k_norm,
                                                      float q_scale);

/* Append k/v to the session KV cache (any KV mode) when `append`, then
 * attend: scratch_attn = attention(scratch_q, cache). A KV-shared layer or
 * a fused prep that already appended passes false. */
[[nodiscard]] enum geist_status blk_attend(struct transformer_layer_forward_ctx *ctx, bool append);

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

/* ---- Composite blocks ---------------------------------------------------
 * A standard sequence several families share verbatim. Still family-free:
 * what varies between families is a different block, not a flag here. */

/* scratch_o = o_proj(attn) with, when the model has them, the AWQ inverse
 * scale, the SubLN / per-projection norm on o_proj's input, and rotation. */
[[nodiscard]] enum geist_status blk_o_proj(struct transformer_layer_forward_ctx *ctx);

/* scratch_ffn_out = down(mid), with the FFN sub-norm when the model has one,
 * the sparsity probe, rotation and down_scale (AWQ; nullptr if already
 * applied) on down's input. */
[[nodiscard]] enum geist_status blk_down(struct transformer_layer_forward_ctx *ctx,
                                         struct geist_buffer                  *mid,
                                         const float                          *down_scale);

/* out = res + rmsnorm(x) * w over d_model rows; the plan-bound fused op when
 * `fused`, else rmsnorm into tmp and add. */
[[nodiscard]] enum geist_status blk_norm_add(struct transformer_layer_forward_ctx *ctx,
                                             bool                                  fused,
                                             const struct geist_tensor            *w,
                                             struct geist_buffer                  *x,
                                             struct geist_buffer                  *res,
                                             struct geist_buffer                  *tmp,
                                             struct geist_buffer                  *out);

/* attn_norm(h_in) into x (scratch_normed), unless the previous layer's
 * tail already wrote it there (ctx->x_prenormed, #17). */
[[nodiscard]] enum geist_status blk_attn_norm(struct transformer_layer_forward_ctx *ctx,
                                              struct geist_buffer                  *x);

/* The post-attention residual step into scratch_h_post_attn:
 * h_in + rmsnorm(scratch_o) * post_w (post_w nullptr: h_in + scratch_o).
 * Where the plan binds fuse_attn_add_norm, the same op also writes the
 * FFN's input ffn_norm(h_post_attn) into scratch_pre_ff and sets
 * ctx->ffn_x_prenormed (#17). */
[[nodiscard]] enum geist_status blk_attn_residual(struct transformer_layer_forward_ctx *ctx,
                                                  const struct geist_tensor            *post_w);

/* The layer's last residual step into ctx->h_out_buf: res + rmsnorm(x) * w
 * (blk_norm_add, `fused` as there; w nullptr: res + x). Where the plan
 * binds fuse_tail_norm, one op also scales the row by `scale` (setting
 * ctx->out_scaled) and writes the next layer's attn_norm of it into
 * scratch_normed, and the session notes that for the next layer call
 * (#17); otherwise `scale` is the caller's to apply. */
[[nodiscard]] enum geist_status blk_residual_out(struct transformer_layer_forward_ctx *ctx,
                                                 float                                 scale,
                                                 bool                                  fused,
                                                 const struct geist_tensor            *w,
                                                 struct geist_buffer                  *x,
                                                 struct geist_buffer                  *res,
                                                 struct geist_buffer                  *tmp);

/* The standard attention block into scratch_h_post_attn (Llama, Qwen3,
 * Gemma 3, Qwen3-MoE, Qwen3.5):
 *   h_post_attn = h_in + post_w(o_proj(attend(rope(qk_norm(qkv(rot(attn_norm(h_in))))))))
 * q_norm / k_norm nullptr: no QK-norm; post_w nullptr: plain residual. With
 * allow_fused_rope, the plan-bound fused prep (blk_rope_append_fused) takes
 * the QK-norm, RoPE, q scale and KV append on the plain (f32/f16) cache. */
[[nodiscard]] enum geist_status blk_attention(struct transformer_layer_forward_ctx *ctx,
                                              const struct geist_tensor            *q_norm,
                                              const struct geist_tensor            *k_norm,
                                              float                                 q_scale,
                                              const struct geist_tensor            *post_w,
                                              bool allow_fused_rope);

/* Pre-norm SwiGLU FFN with residual (Llama, Qwen3):
 *   h_out = h_post_attn + down(silu(gate(x)) * up(x)),  x = rmsnorm(h_post_attn) */
[[nodiscard]] enum geist_status blk_ffn_swiglu(struct transformer_layer_forward_ctx *ctx);

/* Sandwich-norm GEGLU FFN with residual (Gemma):
 *   out = h_post_attn + post_norm(down(gelu(gate(x)) * up(x))),  x = rmsnorm(h_post_attn)
 * with the plan-bound fusions (fused front, gelu*mul). */
[[nodiscard]] enum geist_status blk_ffn_geglu(struct transformer_layer_forward_ctx *ctx,
                                              struct geist_buffer                  *out);

/* ---- Mixture of experts (forward/moe.c, #48) ---------------------------- */

/* Router top-k for one token: softmax over the n logits, the k largest
 * probabilities ranked (probability descending, expert id ascending), into
 * ids[0..k) and w[0..k) (plain pointers: k may be 0, which is refused);
 * renormalised to sum 1 when `norm`. Returns false,
 * leaving ids / w unspecified, when k is 0 or exceeds n or a logit is not
 * finite. */
[[nodiscard]] bool moe_route_topk(
        size_t n, size_t k, bool norm, const float logits[static n], uint32_t *ids, float *w);

/* Pre-norm mixture-of-experts SwiGLU FFN with residual:
 *   h_out = h_post_attn + sum_{e in topk} w_e * down_e(silu(gate_e(x)) * up_e(x)),
 *   x = rmsnorm(h_post_attn)
 * on the layer's moe_router / moe_{gate,up,down}; routing on the host. */
[[nodiscard]] enum geist_status blk_moe_ffn(struct transformer_layer_forward_ctx *ctx);

/* The session state blk_moe_ffn needs (router logits, routing, gathered
 * rows, the GPU's hot-expert buffers); a MoE family's session hooks. */
[[nodiscard]] enum geist_status moe_session_alloc(struct transformer_arch_session *sess);
void                            moe_session_free(struct transformer_arch_session *sess);
