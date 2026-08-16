#!/usr/bin/env python3
"""Precompute Hann window + Mel filterbank for the Gemma 4 audio encoder.

Generates a binary blob that mel_pipeline.c mmaps at startup. Matches
Gemma4AudioFeatureExtractor exactly:
  - frame_length = 320 (20ms @ 16kHz)
  - fft_length   = 512 (next power of 2 above 320)
  - n_mel        = 128, htk scale, no norm, [0..8000] Hz
  - periodic Hann (np.hanning(N+1)[:-1])

Layout (little-endian fp32):
  [0 .. 320)            : Hann window (320 floats)
  [320 .. 320 + 257*128): Mel filterbank (257 rows × 128 cols, row-major)

Run once: python gen_mel_constants.py > audio_test_data/mel_constants.bin
"""
import sys
import warnings
import numpy as np
from transformers.audio_utils import window_function, mel_filter_bank

FRAME_LENGTH = 320
FFT_LENGTH = 512
N_FFT_BINS = FFT_LENGTH // 2 + 1   # 257
N_MEL = 128
SR = 16000

window = window_function(FRAME_LENGTH).astype(np.float32)
assert window.shape == (FRAME_LENGTH,)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mel = mel_filter_bank(
        num_frequency_bins=N_FFT_BINS,
        num_mel_filters=N_MEL,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=SR,
        norm=None,
        mel_scale="htk",
    ).astype(np.float32)
assert mel.shape == (N_FFT_BINS, N_MEL)

blob = window.tobytes() + mel.tobytes()
expected_bytes = (FRAME_LENGTH + N_FFT_BINS * N_MEL) * 4
assert len(blob) == expected_bytes, f"got {len(blob)}, expect {expected_bytes}"

sys.stdout.buffer.write(blob)
print(
    f"[gen_mel_constants] wrote {len(blob)} bytes "
    f"(window {FRAME_LENGTH}f + mel {N_FFT_BINS}x{N_MEL}f)",
    file=sys.stderr,
)
