#!/usr/bin/env bash
# take_geist.sh — record the geist half of assets/versus-bitnetcpp.gif.
# Reconstruction of the /tmp take script from PR #214 (issue #218); the
# method is documented in tools/demos/README.md and docs/DEMOS.md.
set -euo pipefail
cd "$(dirname "$0")"

BIN=${BIN:-./geist-bitnet}                       # release asset (arm64)
VERSION=${VERSION:-$(basename "$BIN")}           # shown in the banner; the CLI
                                                 # has no --version flag (any
                                                 # argument becomes a prompt)
PROMPT=${PROMPT:-"The three largest moons of Jupiter are"}
MAX_NEW=${MAX_NEW:-110}
CAST=${CAST:-take_geist.cast}
GATE_C=${GATE_C:-57.0}                           # thermal gate, °C

temp() { vcgencmd measure_temp | grep -Eo '[0-9]+\.[0-9]+'; }

echo "thermal gate: waiting for <= ${GATE_C} C ..."
while (($(echo "$(temp) > $GATE_C" | bc -l))); do
    sleep 10
done

# Pre-warm THIS engine's pages: on a 4 GB board the other engine's 1.2 GB
# mmap has evicted them, and a cold first run halves tok/s.
OMP_WAIT_POLICY=active "$BIN" "warm-up" 8 >/dev/null 2>&1

asciinema rec --overwrite -c "
  echo \"geistlib $VERSION  |  \$(vcgencmd measure_temp)\";
  time OMP_WAIT_POLICY=active $BIN \"$PROMPT\" $MAX_NEW;
  vcgencmd measure_temp
" "$CAST"

echo "wrote $CAST — see README.md for the render + hstack steps"
