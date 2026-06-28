/*
 * src/backends/cpu_x86/kernel_w8a8.h — W8A8 dot/GEMV inner kernel.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * --- Contract ----------------------------------------------------------------
 *
 * Inner kernel: W8A8 row-dot, designed to map cleanly onto VPDPBUSD
 * (u8 × s8 → s32) and onto the GGUF Q6_K weight layout. Same algebraic
 * shape as kernel_w4a8.h but with 16-element blocks and 1-byte-per-weight
 * storage (Q6_K's sub-block granularity is 16, so the per-block scale and
 * offset map directly).
 *
 *   for each block b in [0, n_blocks):
 *       d_b = sum over i in [0, 16) of u_w[b, i] * a[b, i]
 *   y = scale_x * sum_b ( w_scales[b] * d_b - w_offsets[b] * sum_a[b] )
 *
 * where:
 *   u_w[b, i] ∈ [0, 255] unsigned 8-bit (Q6_K predecoder writes [0, 63]).
 *   a[b, i]   ∈ [-127, 127] signed int8, pre-quantized per row.
 *   w_scales[b], w_offsets[b] fp32 per-block.
 *   sum_a[b]  int32 per-block sum of a[b, .].
 *   scale_x   per-row fp32 activation scale.
 *
 * The hot path is allocation-free. The caller owns every buffer.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W8A8_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W8A8_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_w8a8.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

constexpr size_t W8A8_BLOCK_ELEMS = 16;

[[nodiscard]] float w8a8_dot_scalar(size_t        n_blocks,
                                    const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
                                    const float   w_scales[static n_blocks],
                                    const float   w_offsets[static n_blocks],
                                    const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
                                    const int32_t sum_a_per_block[static n_blocks],
                                    float         scale_x);

[[nodiscard]] float w8a8_dot(size_t        n_blocks,
                             const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
                             const float   w_scales[static n_blocks],
                             const float   w_offsets[static n_blocks],
                             const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
                             const int32_t sum_a_per_block[static n_blocks],
                             float         scale_x);

/* Multi-row GEMV: n_rows independent dots, OMP-parallel internally. */
void w8a8_gemv(size_t        n_rows,
               size_t        n_blocks_per_row,
               const uint8_t weights[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const float   w_scales[static n_rows * n_blocks_per_row],
               const float   w_offsets[static n_rows * n_blocks_per_row],
               const int8_t  acts[static n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const int32_t sum_a_per_block[static n_blocks_per_row],
               float         scale_x,
               float         out[static n_rows]);

/* Prefill GEMM: M tokens × n_rows output rows. Y is token-major row-major
 * (Y[j * n_rows + r]). Weights are read once per output row and reused
 * across all M tokens (the tiled-GEMM amortization); JT independent fp32
 * accumulators per inner step keep the back-end fed. OMP-parallel over
 * output rows internally. Allocation-free; caller owns every buffer. */
void w8a8_gemm(size_t        n_tokens,
               size_t        n_rows,
               size_t        n_blocks_per_row,
               const uint8_t weights[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const float   w_scales[static n_rows * n_blocks_per_row],
               const float   w_offsets[static n_rows * n_blocks_per_row],
               const int8_t  acts[static n_tokens * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const int32_t sum_a_per_block[static n_tokens * n_blocks_per_row],
               const float   scale_x[static n_tokens],
               float         out[static n_tokens * n_rows]);

/* --- Lane-parallel W8A8 (W8x8) ---------------------------------------------
 *
 * The GEMM above reduces each 16-element block to a scalar per output row
 * (2 hadd per 2-block) and applies the per-block fp32 scale once per row —
 * the two costs that cap its density. W8x8 removes both: it interleaves 8
 * output rows so one VPDPBUSD lands 8 rows in 8 int32 lanes (no hadd), and
 * the per-block scale/offset become one 8-wide fp32 FMA for all 8 rows.
 * This mirrors the block_q4_Kx8 lane-parallel layout used for Q4_K.
 *
 * Interleaved layout for one matrix (n_out rows, n_out % 8 == 0; n_in
 * elements; NB = n_in / 16 blocks; NG = n_out / 8 row-groups):
 *   qs:      NG groups × (n_in * 8) bytes. Per group, 4-element stripes:
 *            qs[grp*32 + r*4 + e] = row r's weight at element grp*4 + e.
 *   scales:  NG × NB × 8 fp32, block-major (scales[b*8 + r]).
 *   offsets: NG × NB × 8 fp32, same shape.
 * Total bytes == the row-major W8A8 blob (pure permutation). */
constexpr size_t W8X8_NROWS = 8;

/* W8x16: same idea, 16 rows per group → one 512-bit VPDPBUSD lands 16
 * output rows in the 16 int32 lanes. On Zen 5's full-width AVX-512 datapath
 * this is ~1.5× the 256-bit W8x8 (measured 2393→3720 GFLOP/s). Layout is
 * identical to W8x8 with NROWS=16 (qs[grp*64 + r*4 + e], scales[b*16 + r]).
 * Requires n_out % 16 == 0. */
constexpr size_t W8X16_NROWS = 16;

/* True iff the W8A8 dispatcher resolved to the AVX-512+VNNI tier, i.e.
 * w8x8_gemm is usable on this host. Host-constant after first call. */
[[nodiscard]] int w8a8_isa_is_vnni(void);

/* Repack a row-major W8A8 matrix (the q6k_to_w8a8 output) into the W8x8
 * interleaved layout. n_out must be a multiple of W8X8_NROWS. Layout-only
 * (no ISA); runs once at model load. Caller owns the outputs. */
void w8x8_repack(size_t        n_out,
                 size_t        n_in,
                 const uint8_t weights[static n_out * n_in],
                 const float   w_scales[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
                 const float   w_offsets[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
                 uint8_t       qs_out[static n_out * n_in],
                 float         scales_out[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
                 float         offsets_out[static n_out * (n_in / W8A8_BLOCK_ELEMS)]);

/* W8x16 repack (16-row interleave) + 512-bit GEMM. n_out/n_rows % 16 == 0. */
void w8x16_repack(size_t        n_out,
                  size_t        n_in,
                  const uint8_t weights[static n_out * n_in],
                  const float   w_scales[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
                  const float   w_offsets[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
                  uint8_t       qs_out[static n_out * n_in],
                  float         scales_out[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
                  float         offsets_out[static n_out * (n_in / W8A8_BLOCK_ELEMS)]);

void w8x16_gemm(size_t        n_tokens,
                size_t        n_rows,
                size_t        n_blocks_per_row,
                const uint8_t qs[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
                const float   scales[static n_rows * n_blocks_per_row],
                const float   offsets[static n_rows * n_blocks_per_row],
                const int8_t  acts[static n_tokens * n_blocks_per_row * W8A8_BLOCK_ELEMS],
                const int32_t sum_a_per_block[static n_tokens * n_blocks_per_row],
                const float   scale_x[static n_tokens],
                float         out[static n_tokens * n_rows]);

/* Lane-parallel prefill GEMM. n_rows % W8X8_NROWS == 0. Y is token-major
 * row-major (Y[j * n_rows + r]). AVX-512+VNNI only — call w8a8_isa_is_vnni()
 * first. Allocation-free; caller owns every buffer. */
void w8x8_gemm(size_t        n_tokens,
               size_t        n_rows,
               size_t        n_blocks_per_row,
               const uint8_t qs[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const float   scales[static n_rows * n_blocks_per_row],
               const float   offsets[static n_rows * n_blocks_per_row],
               const int8_t  acts[static n_tokens * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const int32_t sum_a_per_block[static n_tokens * n_blocks_per_row],
               const float   scale_x[static n_tokens],
               float         out[static n_tokens * n_rows]);

/* Reuses the per-row int8 quantization + sum_a logic from kernel_w4a8.h
 * (W8A8 and W4A8 share the activation pipeline). The caller is expected
 * to use that path; nothing new is needed here. */

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W8A8_H */
