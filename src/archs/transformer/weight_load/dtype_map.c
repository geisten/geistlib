/*
 * src/archs/transformer/weight_load/dtype_map.c — GGUF dtype → geist
 * dtype/layout dispatcher.
 *
 * Layer: ARCHITECTURE.
 *
 * Extracted from weight_load.c during R5 of the C23/AGENT.md cleanup.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include <geist_types.h>

struct dtype_map_entry map_gguf_dtype(gguf_dtype_t gd) {
    switch (gd) {
    case GGUF_TYPE_F32:
        return (struct dtype_map_entry) {GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, true};
    case GGUF_TYPE_F16:
        return (struct dtype_map_entry) {GEIST_DTYPE_F16, GEIST_LAYOUT_DENSE, true};
    case GGUF_TYPE_BF16:
        return (struct dtype_map_entry) {GEIST_DTYPE_BF16, GEIST_LAYOUT_DENSE, true};
    case GGUF_TYPE_Q4_0:
        return (struct dtype_map_entry) {GEIST_DTYPE_Q4_0, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_Q4_1:
        return (struct dtype_map_entry) {GEIST_DTYPE_Q4_1, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_Q3_K:
        return (struct dtype_map_entry) {GEIST_DTYPE_Q3_K, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_Q4_K:
        return (struct dtype_map_entry) {GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_Q5_K:
        return (struct dtype_map_entry) {GEIST_DTYPE_Q5_K, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_Q6_K:
        return (struct dtype_map_entry) {GEIST_DTYPE_Q6_K, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_Q8_0:
        return (struct dtype_map_entry) {GEIST_DTYPE_Q8_0, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_IQ2_S:
        return (struct dtype_map_entry) {GEIST_DTYPE_IQ2_S, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_IQ3_S:
        return (struct dtype_map_entry) {GEIST_DTYPE_IQ3_S, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_IQ4_NL:
        return (struct dtype_map_entry) {GEIST_DTYPE_IQ4_NL, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_IQ4_XS:
        return (struct dtype_map_entry) {GEIST_DTYPE_IQ4_XS, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_TQ1_0:
        return (struct dtype_map_entry) {GEIST_DTYPE_TQ1_0, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_TQ2_0:
        return (struct dtype_map_entry) {GEIST_DTYPE_TQ2_0, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    case GGUF_TYPE_I2_S:
        return (struct dtype_map_entry) {GEIST_DTYPE_I2_S, GEIST_LAYOUT_BLOCK_QUANTIZED, true};
    default:
        return (struct dtype_map_entry) {0, 0, false};
    }
}

/* ---- Buffer/tensor loading primitives ---------------------------------- */

/* Build a 2D tensor view [shape0, shape1] (row-major, contiguous) for a
 * tensor whose bytes have just been uploaded into `buf`. For BLOCK_QUANTIZED
 * layouts stride is unused; for DENSE layouts we set it. */
