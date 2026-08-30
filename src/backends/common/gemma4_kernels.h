/*
 * gemma4_kernels — math kernels for Gemma 4 forward pass (FP32).
 *
 * All kernels operate on FP32 buffers in standard row-major layout.
 * Weights stored as BF16 in safetensors are converted to FP32 at load
 * time (see helpers below) since we are computing in FP32 throughout.
 */
#ifndef GEMMA4_KERNELS_H
#define GEMMA4_KERNELS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* BF16 (uint16 storage) -> FP32. */
static inline float bf16_to_fp32(uint16_t bf) {
    uint32_t bits = (uint32_t) bf << 16;
    float    f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

/* Convert n BF16 values into a FP32 buffer. dst MUST be at least n*4 bytes.
 * src is void* on purpose: safetensors packs tensor data at arbitrary byte
 * offsets in the mapping, so it is read unaligned (see the .c) — which is
 * also why it gets no [static] contract: a void* has no element count.
 * dst does, and states it. */
void bf16_array_to_fp32(size_t n, const void *src, float dst[static n]);

/* Allocate (calloc-style) FP32 array and convert n BF16 values into it.
 * Returns nullptr on alloc failure. */
float *bf16_alloc_fp32(size_t n, const void *src);

/* RMSNorm — Gemma 4 style:
 *   for each row of x [n_rows][hidden]:
 *     mean_sq = sum(x[i]²) / hidden
 *     rsqrt   = 1 / sqrt(mean_sq + eps)
 *     y[i]    = x[i] * rsqrt * weight[i]
 *
 * weight may be nullptr (skip the per-element scale, e.g. for "with_scale=False").
 * x and y may alias (in-place is supported).
 */
void rmsnorm_fp32(size_t       n_rows,
                  size_t       hidden,
                  const float  x[static n_rows * hidden],
                  const float *weight, /* nullable: skips the per-element scale */
                  float        eps,
                  float        y[static n_rows * hidden]);

/* Compute RoPE cos/sin tables for positions 0..seq_len-1 over `head_dim`
 * dimensions (must be even) using base `theta`. Output buffers are
 * [seq_len][head_dim] row-major; emb is the standard [freqs, freqs]
 * duplicated form, with cos = cos(emb), sin = sin(emb).
 *
 * `n_rotated_dims` controls partial rotary: only the first n_rotated_dims/2
 * frequencies are populated; the rest get inv_freq=0 (cos=1, sin=0, i.e.
 * identity rotation). For full RoPE, pass n_rotated_dims == head_dim.
 * For Gemma 4 full-attention with partial_rotary_factor=0.25 and
 * head_dim=512, pass n_rotated_dims = 128.
 */
void rope_compute(size_t seq_len,
                  size_t head_dim,
                  size_t n_rotated_dims,
                  float  theta,
                  float *cos_out,
                  float *sin_out);

/* Apply RoPE in-place. x has shape [seq_len, n_heads, head_dim].
 * cos/sin are [seq_len, head_dim]. */
void rope_apply(size_t      seq_len,
                size_t      n_heads,
                size_t      head_dim,
                float       x[static seq_len * n_heads * head_dim],
                const float cos[static seq_len * head_dim],
                const float sin[static seq_len * head_dim]);

/* Scaled dot-product attention with MQA broadcast and causal mask.
 *   q   shape [seq_len, n_q_heads,  head_dim]
 *   k   shape [seq_len, n_kv_heads, head_dim]
 *   v   shape [seq_len, n_kv_heads, head_dim]
 *   out shape [seq_len, n_q_heads,  head_dim]
 * sliding_window: 0 = unbounded causal; >0 = q at position t only sees
 *   k positions in (t - sliding_window, t]. */
void attention_mqa_causal(size_t      seq_len,
                          size_t      n_q_heads,
                          size_t      n_kv_heads,
                          size_t      head_dim,
                          size_t      sliding_window,
                          const float q[static seq_len * n_q_heads * head_dim],
                          const float k[static seq_len * n_kv_heads * head_dim],
                          const float v[static seq_len * n_kv_heads * head_dim],
                          float       out[static seq_len * n_q_heads * head_dim]);

/* Decoupled-length variant for KV-cached inference.
 *   q       shape [n_q,  n_q_heads,  head_dim]
 *   k       shape [n_kv, n_kv_heads, head_dim]
 *   v       shape [n_kv, n_kv_heads, head_dim]
 *   out     shape [n_q,  n_q_heads,  head_dim]
 * Position of q[t] in the absolute sequence is (q_offset + t). Causal mask
 * permits q[t] to attend to k[s] iff s <= q_offset + t. With
 * sliding_window > 0, additionally s > q_offset + t - sliding_window.
 *
 * For prefill: n_q = n_kv, q_offset = 0  (equivalent to attention_mqa_causal).
 * For decode:  n_q = 1, n_kv = cache_len_after_append, q_offset = cache_len_before. */
void attention_mqa_causal_kv(size_t      n_q,
                             size_t      n_kv,
                             size_t      q_offset,
                             size_t      n_q_heads,
                             size_t      n_kv_heads,
                             size_t      head_dim,
                             size_t      sliding_window,
                             const float q[static n_q * n_q_heads * head_dim],
                             const float k[static n_kv * n_kv_heads * head_dim],
                             const float v[static n_kv * n_kv_heads * head_dim],
                             float       out[static n_q * n_q_heads * head_dim]);

/* Rotary position embedding is defined on PAIRS of channels: element i is
 * rotated against element i + head_dim/2. An odd head_dim leaves the last
 * channel with no partner, and every function in this family then quietly
 * skips it — rope_compute_at fills only 2*(head_dim/2) table entries and
 * leaves the last one as the allocator left it, and the interleaved-layout
 * permutation in the arch layer copies that same uninitialized tail into
 * the activation. head_dim is model metadata (d_model / n_q_heads for the
 * Llama and BitNet families), so a malformed file can pick it.
 *
 * Rather than invent a meaning for a half-pair, the contract is: rotary
 * requires an even, non-zero head_dim, checked once when the RoPE plan is
 * built. This predicate is that check, in one place, so the load-time
 * rejection and the hot-path guard cannot drift apart. */
[[nodiscard]] static inline bool rope_head_dim_supported(const size_t head_dim) {
    return head_dim != 0u && (head_dim % 2u) == 0u;
}

/* Compute RoPE cos/sin tables starting from a position offset.
 * For decode: pos_offset = cache_length (so the new token gets the right pos).
 * Requires rope_head_dim_supported(head_dim); the caller checks. Writes
 * exactly n_positions * head_dim entries to each of cos_out and sin_out. */
void rope_compute_at(size_t pos_offset,
                     size_t n_positions,
                     size_t head_dim,
                     size_t n_rotated_dims,
                     float  theta,
                     float *cos_out,
                     float *sin_out);

/* GELU-tanh activation in-place (or out-of-place if y != x):
 *   y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/π) * (x[i] + 0.044715 * x[i]³)))
 * Matches PyTorch's "gelu_pytorch_tanh" / config "gelu_pytorch_tanh". */
void gelu_tanh_fp32(size_t n, const float x[static n], float y[static n]);

/* Fused GEGLU inner activation: y[i] = gelu_tanh(x[i]) * z[i].
 * y may alias x or z. */
void gelu_tanh_mul_fp32(size_t      n,
                        const float x[static n],
                        const float z[static n],
                        float       y[static n]);

/* Squared-ReLU activation: y[i] = max(x[i], 0) * max(x[i], 0). BitNet
 * b1.58 2B-4T FFN activation. y may alias x. */
void relu_squared_fp32(size_t n, const float x[static n], float y[static n]);

/* SiLU activation: y[i] = x[i] / (1 + exp(-x[i])). Llama / BitNet 3B
 * SwiGLU activation function. y may alias x.
 * Named *_ooo (out-of-place) to avoid collision with audio_conformer's
 * in-place silu_fp32(x, n). */
void silu_fp32_ooo(size_t n, const float x[static n], float y[static n]);

/* Element-wise addition: y[i] = a[i] + b[i].  y may alias a or b. */
void add_fp32(size_t n, const float a[static n], const float b[static n], float y[static n]);

/* Element-wise multiplication: y[i] = a[i] * b[i].  y may alias a or b. */
void mul_fp32(size_t n, const float a[static n], const float b[static n], float y[static n]);

/* Linear: y = x @ weight^T + (bias if non-null)
 *
 * x      shape [m, n_in]   row-major
 * weight shape [n_out, n_in] row-major (PyTorch convention)
 * bias   shape [n_out]      row-major, may be nullptr
 * y      shape [m, n_out]   row-major
 *
 * Uses Apple Accelerate sgemm on darwin; falls back to a naive triple
 * loop otherwise. For seq_len = 1 this still goes through sgemm — the
 * library handles the gemv case efficiently.
 */
void linear_fp32(size_t       m,
                 size_t       n_in,
                 size_t       n_out,
                 const float  x[static m * n_in],
                 const float  weight[static n_out * n_in],
                 const float *bias, /* nullable */
                 float        y[static m * n_out]);

#endif
