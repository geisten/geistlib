/*
 * src/backends/cpu_x86/q4k_to_q4kx8.h — Q4_K → Q4_Kx8 interleaved repack.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Repacks 8 contiguous Q4_K super-blocks (= 8 output rows × 256 K-elements)
 * into one block_q4_Kx8 whose qs[] field interleaves the 4-bit nibbles
 * across the 8 rows in 8-byte stripes. This is the same layout llama.cpp
 * uses in its AVX-512 q4_K GEMM kernel (ggml/src/ggml-cpu/repack.h and
 * arch/x86/repack.cpp), which lets one 256-bit weight load feed VPMADDUBSW
 * across 8 output cells simultaneously — the lane-parallel structure that
 * makes their prefill ~18× faster than our independent-row W4A8 SoA.
 *
 * --- Block layout ----------------------------------------------------------
 *
 *   block_q4_Kx8 {
 *     uint16_t d[8];       // 16 B — super-block scale per source row (fp16)
 *     uint16_t dmin[8];    // 16 B — super-block min   per source row (fp16)
 *     uint8_t  scales[96]; // 96 B — repacked 6-bit sub-scales + sub-mins
 *     uint8_t  qs[1024];   // 1024 B — 8-byte-interleaved 4-bit quants
 *   }                      // total 1152 B for 8 rows × 256 elements
 *
 * Memory footprint per matrix: same as 8 separate Q4_K rows (8 × 144 = 1152).
 * The repack is purely a permutation of bytes plus an unpack/repack of the
 * 12-byte scales descriptor.
 *
 * --- Contract --------------------------------------------------------------
 *
 * Allocation-free; caller owns the output buffer. Runs once per weight at
 * model load.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_Q4K_TO_Q4KX8_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_Q4K_TO_Q4KX8_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/q4k_to_q4kx8.h is internal to the backend layer."
#endif

#include "quant.h"
#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>

/* Q4_Kx8 block size: 8 × sizeof(block_q4_K_t) = 8 × 144 = 1152 bytes. */
constexpr size_t Q4_KX8_BLOCK_BYTES = 8 * 144;

/* Interleaved layout matching llama.cpp's block_q4_Kx8 (repack.h:43). */
struct block_q4_Kx8 {
    uint16_t d[8];       /* fp16 super-block scales for 8 source rows */
    uint16_t dmin[8];    /* fp16 super-block mins  for 8 source rows */
    uint8_t  scales[96]; /* repacked 6-bit sub-scales + sub-mins */
    uint8_t  qs[1024];   /* 8-byte-interleaved 4-bit quants */
} __attribute__((packed));

_Static_assert(sizeof(struct block_q4_Kx8) == Q4_KX8_BLOCK_BYTES, "block_q4_Kx8 size mismatch");

/* Repack one row-octet (8 contiguous Q4_K rows × n_super super-blocks per
 * row) into a Q4_Kx8 array of length n_super.
 *
 * Inputs:
 *   n_super: number of Q4_K super-blocks per source row (n_in / 256).
 *   q4k_rows: pointer to 8 contiguous Q4_K rows in source order. Source
 *     stride per row is n_super × Q4_K_BLOCK_BYTES.
 *
 * Output:
 *   q4kx8_out: n_super interleaved Q4_Kx8 blocks (n_super × Q4_KX8_BLOCK_BYTES).
 */
void q4k_to_q4kx8_octet(size_t               n_super,
                        const uint8_t        q4k_rows[static 8 * n_super * Q4_K_BLOCK_BYTES],
                        struct block_q4_Kx8 *q4kx8_out);

/* Bulk repack: n_out / 8 octets covered in one call. n_out must be a
 * multiple of 8. */
void q4k_to_q4kx8_matrix(size_t               n_in,
                         size_t               n_out,
                         const uint8_t       *q4k_data,
                         struct block_q4_Kx8 *q4kx8_out);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_Q4K_TO_Q4KX8_H */
