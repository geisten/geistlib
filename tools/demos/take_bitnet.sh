#!/usr/bin/env bash
# take_bitnet.sh — record the bitnet.cpp half of assets/versus-bitnetcpp.gif.
# Reconstruction of the /tmp take script from PR #214 (issue #218).
# Baseline is microsoft/BitNet @ 404980e built with clang (see README.md:
# current main mis-decodes canonical i2_s on ARM, gcc-12 ICEs their build).
set -euo pipefail
cd "$(dirname "$0")"

BITNET_CLI=${BITNET_CLI:-$HOME/BitNet/build/bin/llama-cli}
GGUF=${GGUF:-$HOME/BitNet/models/ggml-model-i2_s.gguf} # canonical Microsoft file
PROMPT=${PROMPT:-"The three largest moons of Jupiter are"}
MAX_NEW=${MAX_NEW:-110}
CAST=${CAST:-take_bitnet.cast}
GATE_C=${GATE_C:-57.0}

temp() { vcgencmd measure_temp | grep -Eo '[0-9]+\.[0-9]+'; }

echo "thermal gate: waiting for <= ${GATE_C} C ..."
while (($(echo "$(temp) > $GATE_C" | bc -l))); do
    sleep 10
done

# Pre-warm bitnet.cpp's pages (geist's mmap evicted them — see take_geist.sh).
"$BITNET_CLI" -m "$GGUF" -t 4 --temp 0 -n 8 -p "warm-up" >/dev/null 2>&1

asciinema rec --overwrite -c "
  echo \"microsoft/BitNet 404980e  |  \$(vcgencmd measure_temp)\";
  $BITNET_CLI -m $GGUF -t 4 --temp 0 -n $MAX_NEW -p \"$PROMPT\";
  vcgencmd measure_temp
" "$CAST"

echo "wrote $CAST — see README.md for the render + hstack steps"
