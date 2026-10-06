/*
 * gguf_dequant — GGUF tensor-aware dequant dispatch.
 *
 * Bridges a parsed GGUF tensor (struct gguf_tensor_t) to the format-agnostic
 * row codecs in quant.h: it reads the tensor's dtype and routes to the right
 * dequant_*_row kernel. Kept separate from quant.h so the neutral quant
 * contract carries no GGUF (file-format) types.
 */
#ifndef GGUF_DEQUANT_H
#define GGUF_DEQUANT_H

#include <stdbool.h>
#include <stddef.h>

#include "gguf_reader.h"

/* Generic dispatch: dequantize a GGUF tensor of any supported dtype to a
 * fresh heap.h FP32 array; caller frees with safe_free. Returns nullptr on error
 * (unsupported dtype or alloc failure). */
float *gguf_dequant_to_fp32(const struct gguf_tensor_t *t);

/* Dequant a single row of a 2D tensor into the caller-provided buffer.
 * row_elems must equal the row length (= columns); must be a multiple of
 * the block size for Q-formats. Used by callers that can't afford to
 * dequant the whole tensor (e.g. PLE table on memory-constrained Pi 5). */
bool gguf_dequant_row_to_fp32(const struct gguf_tensor_t *t,
                              size_t                      row_idx,
                              size_t                      row_elems,
                              float                      *out);

#endif /* GGUF_DEQUANT_H */
