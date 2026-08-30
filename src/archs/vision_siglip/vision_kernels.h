/*
 * vision_kernels — arch-local kernels for the Gemma 4 vision tower.
 *
 * Per the Q5/Q8 design decisions, heavy ops (linear, RMSNorm, softmax,
 * GELU) lower to the existing backend catalog. This module owns only
 * vision-specific ops that don't fit there:
 *
 *   - patch_embed_reshape: (h_p, w_p, patch_px) layout glue feeding into
 *                          a backend SGEMM (no compute itself)
 *   - avgpool2d_k3: kernel-3 stride-3 average pool over a 2D patch grid
 *   - rope_2d: 2D RoPE applied to Q,K (theta=100, head_dim=64)
 *
 * Phase status:
 *   P1 ⇐ THIS — header + stubs
 *   P3-P4     — fleshed out, NEON specialization for Pi 5
 */
#ifndef VISION_KERNELS_H
#define VISION_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* Kernel-3 stride-3 2D average pool over a patch-grid feature tensor.
 *   in:  (grid_h, grid_w, hidden) fp32
 *   out: (pool_h, pool_w, hidden) fp32
 *     pool_h = grid_h / 3, pool_w = grid_w / 3
 *     grid_h / grid_w are guaranteed multiples of 3 (HF planner rounds to
 *     side_mult = pool_kernel * patch_size = 48 px = 3 patches). */
void avgpool2d_k3_fp32(const float *in, float *out, size_t grid_h, size_t grid_w, size_t hidden);

/* 2D split-axis rotary position embedding, applied in-place.
 *
 * Matches Gemma 4 vision's apply_multidimensional_rope with ndim=2:
 *   - Split each head_dim vector in half: [x-axis half | y-axis half].
 *     For head_dim=64, halves are 32 dims each.
 *   - Each half is treated as a standard RoPE rotation over its own
 *     position (x for first half, y for second half) with shared
 *     inv_freq computed at the half-dim granularity.
 *   - Within a half of size H/2 = 32, RoPE pairs dim k with dim k + H/4
 *     (concat-style RoPE, NOT interleaved): result[k] = x[k]*cos -
 *     x[k+H/4]*sin, result[k+H/4] = x[k+H/4]*cos + x[k]*sin.
 *
 *   x         (n_tokens, n_heads, head_dim) fp32 — modified in place
 *   positions (n_tokens, 2) int32 — (x_pos, y_pos) per token
 *   theta     rope base (100 for Gemma 4 vision)
 *
 * head_dim must be a multiple of 4 (each half is a multiple of 2). */
void rope_2d_split_fp32(size_t         n_tokens,
                        size_t         n_heads,
                        size_t         head_dim,
                        float          theta,
                        float         *x,
                        const int32_t *positions);

/* Bidirectional (non-causal) multi-head attention.
 *
 *   q, k, v   (n_tokens, n_heads, head_dim) fp32 row-major
 *   out       (n_tokens, n_heads, head_dim) fp32 row-major
 *
 * scale = 1 / sqrt(head_dim) (no separate softcap).
 * No mask: every query attends to every key (vision tower has only
 * padding masking, and the HF planner sizes the patch grid exactly so
 * there is no padding for the active region in single-image batches).
 *
 * Implementation runs per-head GEMMs via cblas to amortize the n²
 * score-matrix workload. */
void vision_attention_bidir_fp32(size_t       n_tokens,
                                 size_t       n_heads,
                                 size_t       head_dim,
                                 const float *q,
                                 const float *k,
                                 const float *v,
                                 float       *out);

#endif
