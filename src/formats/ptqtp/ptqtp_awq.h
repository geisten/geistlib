/*
 * ptqtp_awq.h — AWQ (Activation-aware Weight Quantization) scale loader.
 *
 * Reads an awq_scales.bin file produced by tools/awq_compute_scales.py.
 * Provides name-keyed lookup of per-input-channel scale vectors. The
 * runtime uses these to:
 *   1. Pre-divide RMSNorm gamma by s for foldable norms (zero runtime cost).
 *   2. Apply 1/s as extra runtime scaling before non-foldable linears (o, down).
 *   3. Quantize PTQTP weights as W ⊙ s (done in the Python quantizer, not here).
 *
 * File format ("AWQS"):
 *   magic[4] = "AWQS"
 *   version: u32 = 1
 *   n_norms: u32
 *   per norm:
 *     name_len: u32
 *     name:     char[name_len]   e.g. "blk.0.attn_norm.out"
 *     n_in:     u32
 *     scales:   float[n_in]
 *
 * Memory model: file is read once at open and parsed into an in-memory
 * arena. No mmap (size is small — ~200 KB total). Scale pointers are
 * valid for the lifetime of ctx. ptqtp_awq_close frees everything.
 */
#ifndef GEIST_PTQTP_AWQ_H
#define GEIST_PTQTP_AWQ_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ptqtp_awq_ctx;

[[nodiscard]] struct ptqtp_awq_ctx *ptqtp_awq_open(const char *path, const char **err);

void ptqtp_awq_close(struct ptqtp_awq_ctx *ctx);

/* Lookup. Returns nullptr if not found. *n_out receives the channel count
 * on success. */
[[nodiscard]] const float *
ptqtp_awq_get(const struct ptqtp_awq_ctx *ctx, const char *name, size_t *n_out);

#ifdef __cplusplus
}
#endif

#endif /* GEIST_PTQTP_AWQ_H */
