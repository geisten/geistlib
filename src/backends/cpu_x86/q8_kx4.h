/*
 * src/backends/cpu_x86/q8_kx4.h — Q8_Kx4 4-row interleaved activation
 * quantizer with precomputed bsums.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Mirrors llama.cpp's block_q8_Kx4 (ggml/src/ggml-cpu/repack.h:96): 4
 * Q8_K activation rows packed into one block. Per super-block of 256
 * elements per row:
 *   - d[4]:    fp32 super-block scale per row
 *   - qs[1024]: 4 rows × 256 int8 quants, interleaved in 8-byte stripes
 *               within each 64-element sub-block (4 sub-blocks per
 *               super-block × 4 rows × 8 bytes = 256 bytes per sb chunk)
 *   - bsums[64]: 4 rows × 16 sub-block sums (int16) — sum of 16 int8s
 *               per group, precomputed so the GEMM kernel can compute
 *               the Q4_K min-term once per super-block instead of
 *               per-inner-iteration.
 *
 * Quantization is symmetric int8 per row: scale_x = max|x| / 127.
 *
 * The interleaved qs layout matches what the lane-parallel GEMM kernel
 * loads via 32-byte vector reads: bytes 0..7 = row 0, bytes 8..15 = row 1,
 * etc., for each 32-byte stripe in the sub-block.
 *
 * --- Contract --------------------------------------------------------------
 *
 * Caller owns input buffers. Output is one block_q8_Kx4 per super-block.
 * Runs once per linear_mN call, not inside the per-cell hot loop.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_Q8_KX4_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_Q8_KX4_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/q8_kx4.h is internal to the backend layer."
#endif

#include "quant.h"

#include <stddef.h>
#include <stdint.h>

/* QK_K = 256 — Q8_K super-block size matches Q4_K. */
constexpr size_t Q8_KX4_BLOCK_BYTES = 4 * sizeof(float) + 4 * 256 + 4 * 16 * sizeof(int16_t);

struct block_q8_Kx4 {
    float   d[4];          /* per-row super-block scale (fp32) */
    int8_t  qs[4 * 256];   /* 4 rows × 256 quants, 8-byte-row-interleaved */
    int16_t bsums[4 * 16]; /* 4 rows × 16 sub-block sums */
} __attribute__((packed));

_Static_assert(sizeof(struct block_q8_Kx4) == Q8_KX4_BLOCK_BYTES, "block_q8_Kx4 size mismatch");

/* Quantize 4 contiguous fp32 activation rows into one block_q8_Kx4 per
 * super-block. Each row has n_in fp32 elements; n_in must be a multiple
 * of 256. Output buffer must hold (n_in / 256) blocks.
 *
 * After this call, out[s].qs[256*s + 8*k*4 + r*8 + b] holds row r's
 * element at K-position (s*256 + k*64 + ...) in the interleaved layout,
 * and out[s].bsums[r*16 + g] holds row r's sum of int8s for sub-block g.
 */
void quantize_q8_Kx4(size_t n_in, const float x_rows[static 4 * n_in], struct block_q8_Kx4 *out);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_Q8_KX4_H */
