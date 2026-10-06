/*
 * mel_pipeline — log-mel spectrogram for Gemma 4 audio encoder.
 *
 * Input  : 320 fp32 PCM samples (one analysis frame, 20ms @ 16kHz)
 * Output : 128 fp32 log-mel coefficients
 *
 * Matches transformers.Gemma4AudioFeatureExtractor exactly:
 *   periodic Hann (length 320), zero-pad to 512, real FFT, magnitude,
 *   HTK mel filterbank (257 → 128, no norm, [0..8000] Hz), log(x + 1e-3).
 *
 * The Hann window and mel filterbank are precomputed in Python
 * (tools/gen_mel_constants.py) and loaded as a single binary blob.
 */
#ifndef MEL_PIPELINE_H
#define MEL_PIPELINE_H

#include <stddef.h>

#define MEL_FRAME_LENGTH 320
#define MEL_FFT_LENGTH 512
#define MEL_N_FFT_BINS 257 /* MEL_FFT_LENGTH / 2 + 1 */
#define MEL_N_MEL 128

struct MelState;

struct MelState *mel_create(const char *constants_bin_path);
void             mel_destroy(struct MelState *);

/* Compute one log-mel frame from MEL_FRAME_LENGTH (=320) fp32 PCM samples. */
void mel_frame_compute(struct MelState *, const float *pcm_320, float *out_mel_128);

#endif
