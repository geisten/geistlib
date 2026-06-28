/*
 * src/backends/cpu_x86/q4k_to_w4a8.h — Q4_K → W4A8 row predecoder.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * One-time conversion at model-load that takes a GGUF Q4_K row and
 * produces the W4A8 layout the inner-dot kernel consumes (see
 * kernel_w4a8.h).
 *
 * Q4_K formula per element i (see src/formats/gguf/q4_K.c):
 *   y[i] = d * sub_scale * u_w[i] - dmin * sub_min
 * Where:
 *   d, dmin: fp16, one per Q4_K super-block (256 elements).
 *   sub_scale, sub_min: 6-bit unsigned, one pair per 32-element sub-block.
 *   u_w[i]: unsigned 4-bit weight quantization in [0, 15].
 *
 * W4A8 formula per block b (32 elements):
 *   block_term = w_scale * sum_i u_w[i]*a[i] - w_offset * sum_i a[i]
 *
 * Mapping is direct: w_scale[b] = d * sub_scale[b], w_offset[b] =
 * dmin * sub_min[b]. The unsigned 4-bit nibbles are repacked from
 * Q4_K's sub-block-strided form (low nibble of byte k holds element k of
 * one sub-block, high nibble holds element k of the next) to W4A8's
 * within-block consecutive form (byte k holds elements 2k and 2k+1 of
 * the same block).
 *
 * --- Contract ---------------------------------------------------------------
 *
 * Allocation-free. Caller owns every output buffer. Inputs/outputs have no
 * alignment requirement — byte-loop, no SIMD inside the predecoder (it runs
 * once per weight at model-load, not in the inference hot path).
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_Q4K_TO_W4A8_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_Q4K_TO_W4A8_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/q4k_to_w4a8.h is internal to the backend layer."
#endif

#include "kernel_w4a8.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>

/* Predecode one row of Q4_K weights to W4A8.
 *
 * Requires n_in to be a positive multiple of Q4_K_BLOCK_ELEMS (256) AND
 * therefore also a multiple of W4A8_BLOCK_ELEMS (32). The caller is
 * responsible for that — the kernel does not validate.
 *
 * Sizes (computed from n_in):
 *   n_super = n_in / Q4_K_BLOCK_ELEMS               (Q4_K super-blocks)
 *   n_block = n_in / W4A8_BLOCK_ELEMS                (W4A8 blocks)
 *
 *   q4k_row:   n_super * Q4_K_BLOCK_BYTES bytes  (= n_in * 9/16)
 *   weights:   n_block * W4A8_BLOCK_BYTES_WEIGHTS  (= n_in / 2)
 *   w_scales:  n_block
 *   w_offsets: n_block
 */
void q4k_to_w4a8_row(size_t        n_in,
                     const uint8_t q4k_row[static(n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES],
                     uint8_t weights[static(n_in / W4A8_BLOCK_ELEMS) * W4A8_BLOCK_BYTES_WEIGHTS],
                     float   w_scales[static n_in / W4A8_BLOCK_ELEMS],
                     float   w_offsets[static n_in / W4A8_BLOCK_ELEMS]);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_Q4K_TO_W4A8_H */
