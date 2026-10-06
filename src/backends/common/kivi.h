/*
 * kivi.h — 2-bit asymmetric KV-cache quantization (KIVI).
 *
 * Reference: Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization
 * for KV Cache", ICML 2024 (arXiv:2402.02750).
 *
 * Two regimes:
 *
 *   V-cache: per-token, per-head. Each row [head_dim floats] gets its own
 *     min and range; V values rarely cluster in specific channels.
 *
 *   K-cache: per-channel, per-head, groupwise along sequence. A "group" is
 *     KIVI_K_GROUP_SIZE consecutive tokens (128); within one group each
 *     channel gets its own min/range computed across the group. Captures
 *     K's channel-wise outlier pattern (which per-token quant smears).
 *
 * 2-bit encoding: 4 values per byte, little-endian within byte:
 *     byte = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0
 * Reconstruction: x ≈ q * scale + zero with q ∈ {0,1,2,3}.
 *
 * All inputs must be aligned to factors of 4 (size_t n, ...). Outputs
 * fully overwrite the destination — caller owns all buffers.
 */
#ifndef GEIST_KIVI_H
#define GEIST_KIVI_H

#include <stddef.h>
#include <stdint.h>

constexpr size_t KIVI_K_GROUP_SIZE = 128;

/* Quantize one V row (e.g. one head's worth of values for one token).
 * Writes n/4 bytes of packed 2-bit quants and one (scale, zero) pair.
 * n must be a multiple of 4. */
void kivi_pack_v_row(size_t      n,
                     const float in[static n],
                     uint8_t     out_q[static n / 4],
                     float      *out_scale,
                     float      *out_zero);

/* Inverse of kivi_pack_v_row. */
void kivi_unpack_v_row(
        size_t n, const uint8_t in_q[static n / 4], float scale, float zero, float out[static n]);

/* Quantize one K group: g_tokens rows of n_channels floats each
 * (row-major [g_tokens, n_channels]). Per-channel min/range across the
 * group → n_channels scales + n_channels zeros. Packs (g_tokens *
 * n_channels) values into (g_tokens * n_channels)/4 bytes. Both
 * g_tokens and n_channels must be multiples of 4 for clean packing
 * (Gemma 4 head_dim is 256 or 512, group size is 128 — both fit). */
void kivi_pack_k_group(size_t      g_tokens,
                       size_t      n_channels,
                       const float in[static g_tokens * n_channels],
                       uint8_t     out_q[static(g_tokens * n_channels) / 4],
                       float       out_scales[static n_channels],
                       float       out_zeros[static n_channels]);

/* Inverse of kivi_pack_k_group. */
void kivi_unpack_k_group(size_t        g_tokens,
                         size_t        n_channels,
                         const uint8_t in_q[static(g_tokens * n_channels) / 4],
                         const float   scales[static n_channels],
                         const float   zeros[static n_channels],
                         float         out[static g_tokens * n_channels]);

#endif /* GEIST_KIVI_H */
