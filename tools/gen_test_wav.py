#!/usr/bin/env python3
"""gen_test_wav.py — deterministic synthetic WAV for the audio CI smoke.

Emits 16-bit mono PCM at 16 kHz: a 440->880 Hz sweep with an amplitude
envelope, so the mel pipeline sees real spectral structure (a constant tone
lights up one filter bank bin; silence lights up none). No voice recording —
nothing here is speech, it only has to drive the encoder end to end.

Usage: python3 tools/gen_test_wav.py [out.wav] [seconds]
"""
import math
import struct
import sys
import wave

out = sys.argv[1] if len(sys.argv) > 1 else "audio_test_data/smoke.wav"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
sr = 16000
n = int(sr * seconds)

frames = bytearray()
for i in range(n):
    t = i / sr
    freq = 440.0 * (2.0 ** (t / seconds))          # one-octave sweep
    env = math.sin(math.pi * t / seconds)          # fade in/out, no clicks
    s = 0.5 * env * math.sin(2.0 * math.pi * freq * t)
    frames += struct.pack("<h", int(s * 32767))

with wave.open(out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(bytes(frames))
print(f"wrote {out}: {seconds}s, {n} samples @ {sr} Hz")
