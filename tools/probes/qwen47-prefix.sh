#!/bin/sh
# Replay the prompt and inspect the pinned engine without changing the campaign.
set -eu
diag=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo=$(git rev-parse --show-toplevel)
out="$1"
experiment=${2:-simd}
case "$experiment" in
  simd | causal) ;;
  *) exit 2 ;;
esac
host=$(/usr/sbin/sysctl -n hw.model)
if [ "$host" != "Mac18,5" ]; then
  printf '%s\n' "This probe requires the Mac mini M6." >&2
  exit 1
fi
assets="$HOME/geist-lab-qwen32k/incoming/assets"
reference="$HOME/geist-lab-qwen32k/incoming/snapshot-00000083"
omp="$HOME/geist-lab-qwen32k/state/runtime/libomp/23.1.5"
engine=$(mktemp -d "$RUNNER_TEMP/qwen47-engine.XXXXXX")
trap 'rm -rf "$engine"' EXIT
mkdir -p "$out"
model="$assets/models/base-f32.gguf"
printf '%s  %s\n' \
  689a06b8bbd262db30fbe001287edcc88f4124f8074c72d3fa419300942db1fe "$model" \
  7285a6bbafef71b0275912ad005f04179e971abe79a1093aa9a8cf674e21d3cd "$omp/lib/libomp.dylib" \
  | /usr/bin/shasum -a 256 -c
git -C "$repo" archive --output="$engine/source.tar" 916f0e2bbc76fe19af5278ee0922e02ba000e065
tar -xf "$engine/source.tar" -C "$engine"
/usr/bin/python3 "$diag/qwen47-instrument.py" "$engine" "$diag"
driver="$engine/qwen47-prefix"
build() {
  make -C "$engine" -s -j 4 TARGET=mac-omp LIBOMP_PREFIX="$omp" lib
  xcrun clang -std=c23 -O2 -Wall -Wextra -Werror -I"$engine/include" "$diag/qwen47-prefix.c" \
    "$engine/lib/mac-omp/release/libgeist.a" -framework Accelerate -L"$omp/lib" -lomp -lm \
    -o "$driver"
}
build
/usr/bin/python3 - "$reference" "$engine" <<'PY'
from pathlib import Path
import hashlib
import json
import re
import struct
import sys

reference, engine = map(Path, sys.argv[1:])
record = json.loads((reference / "cases/00000046.json").read_bytes())
relative = record["artifacts"]
if record["index"] != 46 or not re.fullmatch(r"artifacts/case-00000046-attempt-\d+", relative):
    raise ValueError("Unexpected case 47 artifact path")
raw = (reference / relative / "generation.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != record["generation_sha256"]:
    raise ValueError("Case 47 reference checksum differs")
trace = json.loads(raw)
if trace["n_prompt"] != 91 or not trace["sampled"] or trace["sampled"][0]["id"] != trace["ids"][91]:
    raise ValueError("Unexpected case 47 token fixture")
(engine / "prompt.bin").write_bytes(struct.pack("<92i", *trace["ids"][:92]))
PY
input="$engine/prompt.bin"
{
  sw_vers
  printf '%s\n' "$host"
  xcrun clang --version
  otool -L "$driver"
} > "$out/platform.txt"
replay() {
  "$HOME/bin/with-macmini-lock" /usr/bin/env DYLD_LIBRARY_PATH="$omp/lib" \
    "$@" "$driver" "$model" "$input"
}
replay > "$out/default.txt"
replay GEIST_METAL_TENSOR=0 > "$out/tensor-off.txt"
mkdir -p "$out/default-vectors"
replay QWEN47_TRACE=1 QWEN47_RAW_DIR="$out/default-vectors" > "$out/trace.txt"
for file in default tensor-off trace; do
  printf '%s ' "$file"
  tail -1 "$out/$file.txt"
done
awk '$1 == "logits"' "$out/trace.txt" > "$out/trace-logits.txt"
diff -u "$out/default.txt" "$out/trace-logits.txt"
if [ "$experiment" = causal ]; then
  mkdir -p "$out/patched-vectors"
  replay QWEN47_TRACE=1 QWEN47_PATCH_QKV=1 QWEN47_RAW_DIR="$out/patched-vectors" > "$out/patched.txt"
  shader="$engine/src/backends/metal/metal_shaders.h"
  cp "$shader" "$engine/metal_shaders.template"
  for variant in precise-rsqrt precise-exp-log; do
    cp "$engine/metal_shaders.template" "$shader"
    /usr/bin/python3 "$diag/qwen47-deltanet-variant.py" "$engine" "$variant"
    build
    mkdir -p "$out/$variant-vectors"
    replay QWEN47_TRACE=1 QWEN47_PATCH_QKV=1 QWEN47_RAW_DIR="$out/$variant-vectors" > "$out/$variant.txt"
  done
  for file in default patched precise-rsqrt precise-exp-log; do
    awk '$1 == "logits" && $2 == 0 {print FILENAME, $0}' "$out/$file.txt"
  done
  exit 0
fi
shader="$engine/src/backends/metal/metal_shaders.h"
cp "$shader" "$engine/metal_shaders.template"
for order in descending ascending; do
  cp "$engine/metal_shaders.template" "$shader"
  /usr/bin/python3 "$diag/qwen47-matvec-reduction.py" "$engine" "$order"
  build
  mkdir -p "$out/$order-vectors"
  replay QWEN47_TRACE=1 QWEN47_RAW_DIR="$out/$order-vectors" > "$out/$order.txt"
  printf '%s ' "$order"
  tail -1 "$out/$order.txt"
done
