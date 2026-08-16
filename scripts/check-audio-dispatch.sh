#!/bin/sh
# check-audio-dispatch.sh — the audio tower selects its quantized matmul
# kernels at LOAD TIME from the hardware probe (#236), never at compile time.
#
# Enforced mechanically: the forward pass must contain no ISA conditionals.
# Intrinsics belong in audio_linear.c, behind the runtime binding — a new
# `#if __ARM_NEON` / `#ifdef __AVX...` creeping back into encoder_forward.c
# reintroduces the platform fork this check exists to prevent.
set -eu

file="src/archs/audio_conformer/encoder_forward.c"
pattern='__ARM_NEON|__ARM_FEATURE|__AVX|__SSE|arm_neon\.h|immintrin\.h'

if hits=$(grep -nE "$pattern" "$file"); then
    echo "check-audio-dispatch: compile-time ISA selection is back in $file:" >&2
    echo "$hits" >&2
    echo "Route kernels through audio_linear.c's runtime binding instead." >&2
    exit 1
fi
echo "audio dispatch OK: no compile-time ISA branch in $file"
