/*
 * src/backends/cpu_x86/q6k_to_w8a8.h — Q6_K → W8A8 row predecoder.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Q6_K dequant per element:  y[i] = d * sub_scale[k(i)] * (q[i] - 32)
 * with q ∈ [0, 63], k(i) the 16-elem sub-block index, sub_scale signed
 * int8, d fp16 super-scale.
 *
 * W8A8 inner kernel computes:
 *   block_term = w_scale[b] * sum_i u_w[b,i]*a[b,i] - w_offset[b] * sum_a[b]
 *
 * Mapping:
 *   u_w[b, i]   = q[b*16 + i]                   (unsigned 6-bit in u8)
 *   w_scale[b]  = d * sub_scale[b]              (fp32)
 *   w_offset[b] = d * sub_scale[b] * 32         (fp32; absorbs -32 in y)
 *
 * Run once at model load. Allocation-free; caller owns outputs.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_Q6K_TO_W8A8_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_Q6K_TO_W8A8_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/q6k_to_w8a8.h is internal to the backend layer."
#endif

#include "kernel_w8a8.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>

void q6k_to_w8a8_row(size_t        n_in,
                     const uint8_t q6k_row[static(n_in / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES],
                     uint8_t       weights[static(n_in / W8A8_BLOCK_ELEMS) * W8A8_BLOCK_ELEMS],
                     float         w_scales[static n_in / W8A8_BLOCK_ELEMS],
                     float         w_offsets[static n_in / W8A8_BLOCK_ELEMS]);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_Q6K_TO_W8A8_H */
