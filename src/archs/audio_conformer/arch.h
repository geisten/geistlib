/*
 * src/archs/audio_conformer/arch.h — audio Conformer encoder (Gemma 4).
 *
 * Layer: ARCHITECTURE. Implements the geist_arch_ops_encoder vtable for
 * the Gemma 4 audio tower (Conformer-based, stateless forward).
 *
 * The vtable shape itself lives in <geist_arch.h> (engine-owned interface);
 * this header only exports the concrete descriptor.
 *
 * Defined in (src/archs/audio_conformer/):
 *   arch.c            — descriptor, encode entry
 *   audio_encoder.c   — encoder orchestration (start reading here)
 *   encoder_forward.c — the Conformer forward stages
 *   encoder_stream.c  — the streaming API and its worker thread
 *   encoder_weights.c — weight loading, per-class precision, teardown
 *   audio_linear.c    — quantized matmuls
 *   audio_kernels.c   — conv2d, LayerNorm, ReLU (FP32)
 *   mel_pipeline.c    — log-mel spectrogram (vDSP on Apple, vendored FFT
 *                       elsewhere)
 */
#ifndef GEIST_INTERNAL_ARCH_AUDIO_CONFORMER_H
#define GEIST_INTERNAL_ARCH_AUDIO_CONFORMER_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "audio_conformer/arch.h is internal to the architecture layer."
#endif

#include <geist.h>
#include <geist_arch.h>

/* Concrete descriptor for the Gemma 4 audio Conformer. */
extern const struct geist_arch_ops_encoder geist_arch_audio_conformer;

#endif /* GEIST_INTERNAL_ARCH_AUDIO_CONFORMER_H */
