/*
 * quant — quantization subsystem contract shared by formats/ and backends/.
 *
 * Defines the block layouts (element/byte sizes) for every supported quant
 * dtype, the fp16->fp32 helper, the dequant-to-fp32 row codecs, and the
 * quantized linear-kernel API. Deliberately format-agnostic: it operates on
 * raw block memory (const void *) and scalars, with no GGUF (or any
 * file-format) types. formats/gguf and the CPU backends provide the
 * implementations; archs and backends consume this contract. The GGUF
 * tensor-aware dispatch (gguf_dequant_to_fp32) lives in formats/gguf/gguf_dequant.h.
 *
 * Block dequant kernels mirror the reference algorithms in
 * llama.cpp/ggml-quants.c — see comments per function for exact line refs.
 */
#ifndef QUANT_H
#define QUANT_H

#include <stddef.h>
#include <stdint.h>

/* FP16 -> FP32. On ARM64 (and Apple Silicon) where __fp16 is hardware-
 * native, this is a single `fcvt` instruction and inlines to zero call
 * overhead — important for the per-super-block scale conversion in the
 * Q4_K / Q6_K / Q3_K decode kernels (profile showed ~5% of decode time
 * was function-call overhead on this path).
 *
 * Falls back to the bit-exact IEEE-754 decode in src/formats/gguf/common.c
 * when no hardware fp16 is available. */
#if defined(__ARM_FP) && (__ARM_FP & 2)
#include <string.h>
static inline float fp16_to_fp32(uint16_t h) {
    __fp16 f;
    memcpy(&f, &h, sizeof(f));
    return (float) f;
}
#else
float fp16_to_fp32(uint16_t h);
#endif

/* Q8_0 block: 32 elements, 1 fp16 scale + 32 int8 quants = 34 bytes. */
constexpr size_t Q8_0_BLOCK_ELEMS = 32;
constexpr size_t Q8_0_BLOCK_BYTES = 34;
void             dequant_q8_0_row(const void *blocks, float *out, size_t n_elems);

/* Q4_0 block: 32 elements, 1 fp16 scale + 16 int4 quant bytes = 18 bytes.
 * Each byte packs two nibbles: low nibble is element [2*i], high is
 * [2*i+1]. Values are unsigned [0..15] biased to signed [-8..+7]. */
constexpr size_t Q4_0_BLOCK_ELEMS = 32;
constexpr size_t Q4_0_BLOCK_BYTES = 18;
void             dequant_q4_0_row(const void *blocks, float *out, size_t n_elems);

/* Q3_K super-block: 256 elements, 110 bytes:
 *   32 hmask (high bit), 64 qs (low 2 bits), 12 scales (16 × 6-bit signed), 1 fp16 d */
constexpr size_t Q3_K_BLOCK_ELEMS = 256;
constexpr size_t Q3_K_BLOCK_BYTES = 110;
void             dequant_q3_K_row(const void *blocks, float *out, size_t n_elems);

/* Q4_K (k-quants) super-block: 256 elements, 144 bytes:
 *   fp16 d, fp16 dmin, 12 bytes scales/mins (8 × 6-bit pairs), 128 bytes 4-bit qs */
constexpr size_t Q4_K_BLOCK_ELEMS = 256;
constexpr size_t Q4_K_BLOCK_BYTES = 144;
void             dequant_q4_K_row(const void *blocks, float *out, size_t n_elems);

/* Q5_K super-block: 256 elements, 176 bytes:
 *   fp16 d, fp16 dmin, 12 bytes scales/mins (Q4_K-style), 32 qh (high bit), 128 qs (low 4) */
constexpr size_t Q5_K_BLOCK_ELEMS = 256;
constexpr size_t Q5_K_BLOCK_BYTES = 176;
void             dequant_q5_K_row(const void *blocks, float *out, size_t n_elems);

/* Q6_K super-block: 256 elements, 210 bytes:
 *   128 lower-4-bit qs, 64 upper-2-bit qh, 16 scales (int8), 1 fp16 d */
constexpr size_t Q6_K_BLOCK_ELEMS = 256;
constexpr size_t Q6_K_BLOCK_BYTES = 210;
void             dequant_q6_K_row(const void *blocks, float *out, size_t n_elems);

/* IQ2_S super-block: 256 elements, 82 bytes, ~2.56 bits/value.
 * Codebook-based: ggml's 1024-entry 8-element grid + per-sub-block scale.
 * Used by IQ2_M quantization variants (most weight tensors). */
constexpr size_t IQ2_S_BLOCK_ELEMS = 256;
constexpr size_t IQ2_S_BLOCK_BYTES = 82;
void             dequant_iq2_s_row(const void *blocks, float *out, size_t n_elems);

/* IQ3_S super-block: 256 elements, 110 bytes, ~3.44 bits/value.
 * Codebook-based: ggml's 512-entry 8-element grid + per-sub-block scale.
 * Used by IQ2_M for higher-precision weight tensors (attn projections etc.). */
constexpr size_t IQ3_S_BLOCK_ELEMS = 256;
constexpr size_t IQ3_S_BLOCK_BYTES = 110;
void             dequant_iq3_s_row(const void *blocks, float *out, size_t n_elems);

/* TQ2_0 super-block: 256 elements, 66 bytes, 2.0625 bpw.
 *   qs[64]: 256 trits packed 4-per-byte (encoding: 0/1/2 mapped to -1/0/+1)
 *   d: fp16 scale.
 * Used by BitNet b1.58 (W1.58A8) weight tensors. */
constexpr size_t TQ2_0_BLOCK_ELEMS = 256;
constexpr size_t TQ2_0_BLOCK_BYTES = 66;
void             dequant_tq2_0_row(const void *blocks, float *out, size_t n_elems);

/* Hard cap on the M dimension of native prefill kernels. The engine's
 * default m_max remains lower, but sessions can opt into larger prefill
 * chunks up to this cap when memory allows. Lets kernels stack-allocate
 * per-row accumulators without heap in the inner loop. */
constexpr size_t GEIST_QUANT_M_CAP = 128;

/* W2A8 fast path for IQ2_S. Reconstructs 32 int8 weights per sub-block
 * from the 1024-entry codebook + sign byte, dots against pre-quantized
 * x_q8 via vdotq_s32. Per-sub-half int scale (2*s+1) folds via
 * vmlaq_n_s32; the (d/8 * scale_x) float multiply happens once per
 * super-block (256 elems). Replaces the FP32 dequant + sgemm slow path
 * (~1.7 GB/s) with int8 NEON dots (~5-7 GB/s target on Pi 5). */
void linear_iq2s_decode_w2a8(
        const float *x, const void *w_iq2s, size_t n_in, size_t n_out, float *y);
void linear_iq2s_decode_w2a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_iq2s, size_t n_in, size_t n_out, float *y);

/* W3A8 fast path for IQ3_S. Mirrors IQ2_S design: 32 int8 weights per
 * sub-block built from the 512-entry codebook + sign byte. Per-sub-block
 * int scale (2*s+1) folds via vmlaq_n_s32; d * scale_x float multiply
 * once per super-block. */
void linear_iq3s_decode_w3a8(
        const float *x, const void *w_iq3s, size_t n_in, size_t n_out, float *y);
void linear_iq3s_decode_w3a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_iq3s, size_t n_in, size_t n_out, float *y);

/* W2A8 prefill (m>1) variant for IQ2_S. Mirrors W4A8 / W6A8 prefill
 * design: read each super-block once per output row, reconstruct the
 * 32 int8 weights per sub-block ONCE via the codebook+sign helper,
 * dot against m activation rows via vdotq_s32, fold sub-block int
 * scale (2*s+1) via vmlaq_n_s32 into per-row int32 accumulators.
 * One float multiply (d/8 * scale_x[i]) per super-block per row.
 *
 * Replaces the FP32 dequant + sgemm fallback for M>1 IQ2_S — Pi 5
 * primary win (no AMX). x is row-major (m, n_in); y row-major (m, n_out). */
void linear_iq2s_w2a8_prefill(
        const float *x, const void *w_iq2s, size_t m, size_t n_in, size_t n_out, float *y);
void linear_iq2s_w2a8_prefill_pre(const int8_t *x_q8,
                                  const float  *scale_x,
                                  size_t        m,
                                  const void   *w_iq2s,
                                  size_t        n_in,
                                  size_t        n_out,
                                  float        *y);

/* W3A8 prefill (m>1) variant for IQ3_S. Same shape as IQ2_S prefill
 * but uses iq3s codebook (512-entry, 4 elements per grid entry) and
 * the IQ3_S sub-block organisation (4 outer × 2 halves). */
void linear_iq3s_w3a8_prefill(
        const float *x, const void *w_iq3s, size_t m, size_t n_in, size_t n_out, float *y);
void linear_iq3s_w3a8_prefill_pre(const int8_t *x_q8,
                                  const float  *scale_x,
                                  size_t        m,
                                  const void   *w_iq3s,
                                  size_t        n_in,
                                  size_t        n_out,
                                  float        *y);

/* Pre-decode one row of IQ2_S / IQ3_S weights into 1 byte per element
 * (signed int8 magnitude; sub-block scale stays in the original blocks).
 * `n_in` must be a multiple of the IQ block elem-count. `flat` must hold
 * `n_in` bytes. Output layout matches the dot order used by the matching
 * flat-decode kernel below. */
void iq2s_decode_to_int8_row(const void *w_iq2s_row, int8_t *flat, size_t n_in);
void iq3s_decode_to_int8_row(const void *w_iq3s_row, int8_t *flat, size_t n_in);

/* Selective-flat-decode M=1 W2A8 / W3A8 kernels.
 *
 * Identical numerics to linear_iq{2s,3s}_decode_w{2,3}a8 but skips the
 * codebook lookup + sign-apply in the per-token hot path by reading
 * pre-decoded int8 weights from `w_flat`. FP16 d and sub-block scales
 * still come from the original IQ blocks. Caller allocates and populates
 * `w_flat` (1 byte per element of the original weight tensor) via the
 * iq{2s,3s}_decode_to_int8_row helpers — typically lazy-cached for
 * ffn_down / ffn_up where the per-token decode cost dominates. */
void linear_iq2s_flat_w2a8(const float  *x,
                           const void   *w_iq2s_blocks,
                           const int8_t *w_flat,
                           size_t        n_in,
                           size_t        n_out,
                           float        *y);
void linear_iq2s_flat_w2a8_pre(const int8_t *x_q8,
                               float         scale_x,
                               const void   *w_iq2s_blocks,
                               const int8_t *w_flat,
                               size_t        n_in,
                               size_t        n_out,
                               float        *y);
void linear_iq3s_flat_w3a8(const float  *x,
                           const void   *w_iq3s_blocks,
                           const int8_t *w_flat,
                           size_t        n_in,
                           size_t        n_out,
                           float        *y);
void linear_iq3s_flat_w3a8_pre(const int8_t *x_q8,
                               float         scale_x,
                               const void   *w_iq3s_blocks,
                               const int8_t *w_flat,
                               size_t        n_in,
                               size_t        n_out,
                               float        *y);

/* Selective-flat-decode M>1 W2A8 / W3A8 prefill kernels. Same numerics
 * as linear_iq{2s,3s}_w{2,3}a8_prefill_pre but consumes the pre-decoded
 * int8 buffer instead of recomputing weights per sub-block. Hot for the
 * speculative-decode verify pass — cache-hits eliminate the dominant
 * per-call weight-reconstruction cost. */
void linear_iq2s_flat_w2a8_prefill(const float  *x,
                                   const void   *w_iq2s_blocks,
                                   const int8_t *w_flat,
                                   size_t        m,
                                   size_t        n_in,
                                   size_t        n_out,
                                   float        *y);
void linear_iq2s_flat_w2a8_prefill_pre(const int8_t *x_q8,
                                       const float  *scale_x,
                                       size_t        m,
                                       const void   *w_iq2s_blocks,
                                       const int8_t *w_flat,
                                       size_t        n_in,
                                       size_t        n_out,
                                       float        *y);
void linear_iq3s_flat_w3a8_prefill(const float  *x,
                                   const void   *w_iq3s_blocks,
                                   const int8_t *w_flat,
                                   size_t        m,
                                   size_t        n_in,
                                   size_t        n_out,
                                   float        *y);
void linear_iq3s_flat_w3a8_prefill_pre(const int8_t *x_q8,
                                       const float  *scale_x,
                                       size_t        m,
                                       const void   *w_iq3s_blocks,
                                       const int8_t *w_flat,
                                       size_t        n_in,
                                       size_t        n_out,
                                       float        *y);

/* GGUF tensor-aware dequant dispatch (gguf_dequant_to_fp32 /
 * gguf_dequant_row_to_fp32) lives in formats/gguf/gguf_dequant.h — it
 * depends on struct gguf_tensor_t and so is kept out of this neutral header. */

/* Fused Q4_K vec-matmul (decode case, M=1) — FP32 reference path.
 *
 * Test/debug reference (kept for kernel A/B comparisons in test_q4k_kernel.c).
 * Production decode dispatch (linear_w) routes Q4_K through linear_q4k_decode_w4a8.
 *
 * W is in GGUF Q4_K layout: n_out rows of (n_in/256) super-blocks each.
 * n_in must be a multiple of Q4_K_BLOCK_ELEMS (256).
 */
void linear_q4k_decode_fp32(const float *x, const void *w_q4k, size_t n_in, size_t n_out, float *y);

/* Same for Q6_K weights. */
void linear_q6k_decode_fp32(const float *x, const void *w_q6k, size_t n_in, size_t n_out, float *y);

/* W4A8 path: quantize input vector to INT8 (symmetric, per-row scale)
 * once, then dot with Q4_K weights using NEON vdotq_s32 for ~4× FMA
 * throughput vs FP32 path. Used in linear_w when m=1 and weight is Q4_K.
 *
 * Workspace requirements (caller-allocated):
 *   x_q8: int8_t[n_in]   — quantized input
 * Returns: scale_x = max|x[i]| / 127 (one float).
 */
float quantize_x_int8_sym(const float *x, size_t n, int8_t *x_q8);

void linear_q4k_decode_w4a8(const float *x, const void *w_q4k, size_t n_in, size_t n_out, float *y);
void linear_q4k_decode_w4a8_pair(const float *x,
                                 const void  *w0_q4k,
                                 const void  *w1_q4k,
                                 size_t       n_in,
                                 size_t       n_out0,
                                 size_t       n_out1,
                                 float       *y0,
                                 float       *y1);

/* W4A8 prefill (m>1) variant. Mirror of W3A8 prefill for Q4_K weights —
 * eliminates the slow gguf_dequant_to_fp32 + cblas_sgemm path for prefill
 * on Pi 5 / non-AMX targets. x is row-major (m, n_in); y row-major (m, n_out). */
void linear_q4k_w4a8_prefill(
        const float *x, const void *w_q4k, size_t m, size_t n_in, size_t n_out, float *y);
void   linear_q4k_w4a8_prefill_pre(const int8_t  *x_q8,
                                   const float   *scale_x,
                                   const int32_t *sum32,
                                   size_t         m,
                                   const void    *w_q4k,
                                   size_t         n_in,
                                   size_t         n_out,
                                   float         *y);
size_t q4k_predecode_size_bytes(size_t n_in, size_t n_out);
int    q4k_predecode_pack(const void *w_q4k, size_t n_in, size_t n_out, void *dst);
size_t q4k_predecode_ntile4_size_bytes(size_t n_in, size_t n_out);
int    q4k_predecode_ntile4_pack(const void *w_q4k, size_t n_in, size_t n_out, void *dst);
void   linear_q4k_decode_w4a8_predecoded(
        const float *x, const void *packed, size_t n_in, size_t n_out, float *y);
void linear_q4k_w4a8_prefill_predecoded(const int8_t  *x_q8,
                                        const float   *scale_x,
                                        const int32_t *sum32,
                                        size_t         m,
                                        const void    *packed,
                                        size_t         n_in,
                                        size_t         n_out,
                                        float         *y);
void linear_q4k_w4a8_prefill_predecoded_mtile4(const int8_t  *x_q8,
                                               const float   *scale_x,
                                               const int32_t *sum32,
                                               size_t         m,
                                               const void    *packed,
                                               size_t         n_in,
                                               size_t         n_out,
                                               float         *y);
void linear_q4k_w4a8_prefill_predecoded_mtile8(const int8_t  *x_q8,
                                               const float   *scale_x,
                                               const int32_t *sum32,
                                               size_t         m,
                                               const void    *packed,
                                               size_t         n_in,
                                               size_t         n_out,
                                               float         *y);
void linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4(const int8_t  *x_q8,
                                                      const float   *scale_x,
                                                      const int32_t *sum32,
                                                      size_t         m,
                                                      const void    *packed,
                                                      size_t         n_in,
                                                      size_t         n_out,
                                                      float         *y);
void linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(const int8_t  *x_q8,
                                                             const float   *scale_x,
                                                             const int32_t *sum32,
                                                             size_t         m,
                                                             const void    *packed,
                                                             size_t         n_in,
                                                             size_t         n_out,
                                                             float         *y);
void linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed(const int8_t  *x_q8,
                                                             const float   *scale_x,
                                                             const int32_t *sum32,
                                                             size_t         m,
                                                             const void    *packed,
                                                             size_t         n_in,
                                                             size_t         n_out,
                                                             float         *y);
void linear_q4k_w4a8_prefill_pair_predecoded_mtile4_ntile4_packed(const int8_t  *x_q8,
                                                                  const float   *scale_x,
                                                                  const int32_t *sum32,
                                                                  size_t         m,
                                                                  const void    *packed0,
                                                                  const void    *packed1,
                                                                  size_t         n_in,
                                                                  size_t         n_out,
                                                                  float         *y0,
                                                                  float         *y1);
void linear_q4k_w4a8_prefill_predecoded_mtile4_bscale(const int8_t  *x_q8,
                                                      const float   *scale_blocks,
                                                      const int32_t *sum32,
                                                      size_t         m,
                                                      const void    *packed,
                                                      size_t         n_in,
                                                      size_t         n_out,
                                                      float         *y);

/* P2.b: native W5A8 kernels for Q5_K. Same activation-quant shape as
 * Q4_K (reuse quantize_x_for_q4k). The decode (M=1) and prefill (M>1)
 * kernels both use NEON vdotq_s32 on per-row reconstructed 5-bit
 * values; replaces the dequant-and-cblas trampoline path. */
void linear_q5k_decode_w5a8(const float *x, const void *w_q5k, size_t n_in, size_t n_out, float *y);
void linear_q5k_decode_w5a8_pre(const int8_t  *x_q8,
                                float          scale_x,
                                const int32_t *sum32,
                                const void    *w_q5k,
                                size_t         n_in,
                                size_t         n_out,
                                float         *y);
void linear_q5k_w5a8_prefill(
        const float *x, const void *w_q5k, size_t m, size_t n_in, size_t n_out, float *y);
void linear_q5k_w5a8_prefill_pre(const int8_t  *x_q8,
                                 const float   *scale_x,
                                 const int32_t *sum32,
                                 size_t         m,
                                 const void    *w_q5k,
                                 size_t         n_in,
                                 size_t         n_out,
                                 float         *y);

/* Pre-quantized W4A8 path: caller owns x_q8[n_in] and sum32[n_in/32], having
 * filled them via quantize_x_for_q4k. Lets callers share one quantization of
 * x across multiple matmul targets (q/k/v from attn_norm out, gate/up from
 * pre_ffn_norm out). n_in must be a multiple of 32.
 */
float quantize_x_for_q4k(const float *x, size_t n, int8_t *x_q8, int32_t *sum32);
void  quantize_x_for_q4k_blocks(
        const float *x, size_t n, int8_t *x_q8, int32_t *sum32, float *scale_blocks);
void linear_q4k_decode_w4a8_pre(const int8_t  *x_q8,
                                float          scale_x,
                                const int32_t *sum32,
                                const void    *w_q4k,
                                size_t         n_in,
                                size_t         n_out,
                                float         *y);

/* W6A8 fast path for Q6_K. Reconstructs 6-bit quants (4-bit ql + 2-bit qh
 * → unsigned 6-bit, then minus 32 → int8 in [-32, 31]) one 16-element
 * sub-block at a time and dots against pre-quantized x_q8 via vdotq_s32.
 * Per-sub-block scale s_j (int8) and per-super-block d (fp16) are folded
 * in scalar after the dot.
 *
 * On Pi 5: replaces the FP32-input reference (1.7 GB/s) with int8 NEON
 * dots (~10 GB/s target). x_q8 must be pre-quantized via quantize_x_int8_sym;
 * no per-block sum32 needed (Q6_K has no min-offset like Q4_K). */
void linear_q6k_decode_w6a8(const float *x, const void *w_q6k, size_t n_in, size_t n_out, float *y);
void linear_q6k_decode_w6a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_q6k, size_t n_in, size_t n_out, float *y);
size_t q6k_x8_gemv_size_bytes(size_t n_in, size_t n_out);
int    q6k_x8_gemv_pack(const void *w_q6k, size_t n_in, size_t n_out, void *dst);
void   linear_q6k_decode_w6a8_x8(
        const float *x, const void *packed, size_t n_in, size_t n_out, float *y);
void linear_q6k_decode_w6a8_x8_pre(
        const int8_t *x_q8, float scale_x, const void *packed, size_t n_in, size_t n_out, float *y);

/* W6A8 prefill (m>1) variant for Q6_K. Mirrors the W3A8 / W4A8 prefill
 * design: read each super-block once per output row, extract 4 reconstructed
 * int8 streams, dot against m activation rows via vdotq_s32. Per-row float
 * accumulator folds in d * scale_x[i] * sub-block-scale at the end of each
 * 16-element chunk. Used by speculative-decode verify (M=K) and any other
 * M>1 path that targets a Q6_K weight (e.g. lm_head with K-wide verify).
 * x is row-major (m, n_in); y row-major (m, n_out). */
void linear_q6k_w6a8_prefill(
        const float *x, const void *w_q6k, size_t m, size_t n_in, size_t n_out, float *y);
void   linear_q6k_w6a8_prefill_pre(const int8_t *x_q8,
                                   const float  *scale_x,
                                   size_t        m,
                                   const void   *w_q6k,
                                   size_t        n_in,
                                   size_t        n_out,
                                   float        *y);
void   linear_q6k_w6a8_prefill_pre_accum_blocks(const int8_t *x_q8,
                                                const float  *scale_x,
                                                size_t        m,
                                                const void   *w_q6k,
                                                size_t        n_in_total,
                                                size_t        n_out,
                                                size_t        block_start,
                                                size_t        n_blocks,
                                                float        *y);
size_t q6k_predecode_ntile4_size_bytes(size_t n_in, size_t n_out);
int    q6k_predecode_ntile4_pack(const void *w_q6k, size_t n_in, size_t n_out, void *dst);
size_t q6k_predecode_ntile4_stream_size_bytes(size_t n_in, size_t n_out);
int    q6k_predecode_ntile4_stream_pack(const void *w_q6k, size_t n_in, size_t n_out, void *dst);
void   linear_q6k_w6a8_prefill_predecoded_ntile4(const int8_t *x_q8,
                                                 const float  *scale_x,
                                                 size_t        m,
                                                 const void   *packed,
                                                 size_t        n_in,
                                                 size_t        n_out,
                                                 float        *y);
void   linear_q6k_w6a8_prefill_predecoded_ntile4_stream(const int8_t *x_q8,
                                                        const float  *scale_x,
                                                        size_t        m,
                                                        const void   *packed,
                                                        size_t        n_in,
                                                        size_t        n_out,
                                                        float        *y);

/* W3A8 fast path for Q3_K. Reconstructs 3-bit signed quants (low 2 bits from
 * qs + high bit from hmask → q ∈ [-4, 3]) and dots against pre-quantized
 * x_q8 via vdotq_s32. Per-sub-group int8 scale and per-super-block fp16 d
 * folded scalarly after each dot. No min-offset (unlike Q4_K/Q5_K). */
void linear_q3k_decode_w3a8(const float *x, const void *w_q3k, size_t n_in, size_t n_out, float *y);
void linear_q3k_decode_w3a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_q3k, size_t n_in, size_t n_out, float *y);

/* W3A8 prefill (m>1) variant. Reads weights once per output row and dots
 * against M activation rows — bandwidth amortization for prefill, replacing
 * the slow gguf_dequant_to_fp32 + cblas_sgemm fallback. x is row-major
 * (m, n_in); y is row-major (m, n_out). */
void linear_q3k_w3a8_prefill(
        const float *x, const void *w_q3k, size_t m, size_t n_in, size_t n_out, float *y);
void linear_q3k_w3a8_prefill_pre(const int8_t *x_q8,
                                 const float  *scale_x,
                                 size_t        m,
                                 const void   *w_q3k,
                                 size_t        n_in,
                                 size_t        n_out,
                                 float        *y);

/* PTQTP kernels live in their own module — see ptqtp_kernel.h.
 * Public entry points: ptqtp_gemv_2plane_fp16alpha, ptqtp_gemv_3plane_fp32alpha,
 * ptqtp_gemm_2plane_fp32alpha. */

/* W8A8 fast path for Q8_0. Mirrors the W4A8 design but for the simpler
 * Q8_0 block layout (32 elements, 1 fp16 scale + 32 int8 quants, no offset).
 *
 * Math: dequant(q[i]) = d · q[i], so
 *   sum_i x[i] · dequant(q[i]) = d · scale_x · sum_i (x_q8[i] · q[i])
 *
 * Caller-friendly variants:
 *   linear_q8_0_decode_w8a8        — single shot: quantize x → matmul → free.
 *   linear_q8_0_decode_w8a8_pre    — caller owns x_q8 + scale_x (lets multiple
 *                                    matmuls share one quantization of x).
 *
 * NEON: uses vdotq_s32 (int8x16 · int8x16 → int32x4). */
void linear_q8_0_decode_w8a8(const float *x, const void *w_q8, size_t n_in, size_t n_out, float *y);

/* P2.d: W8A8 prefill (m>1) for Q8_0. Replaces the dequant-and-cblas
 * trampoline for SmolLM2 / Llama-family Q8_0 prefill on platforms
 * where the native NEON kernel beats the SGEMM (Pi 5). On Mac the
 * resolver may still prefer the trampoline (Apple AMX). */
void linear_q8_0_w8a8_prefill(
        const float *x, const void *w_q8, size_t m, size_t n_in, size_t n_out, float *y);
void linear_q8_0_w8a8_prefill_pre(const int8_t *x_q8,
                                  const float  *scale_x,
                                  size_t        m,
                                  const void   *w_q8,
                                  size_t        n_in,
                                  size_t        n_out,
                                  float        *y);

void linear_q8_0_decode_w8a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_q8, size_t n_in, size_t n_out, float *y);

#endif
