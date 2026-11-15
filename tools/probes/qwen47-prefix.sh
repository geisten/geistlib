#!/bin/sh
# Replay the prompt and inspect the pinned engine without changing the campaign.
set -eu
diag=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo=$(git rev-parse --show-toplevel)
out="$1"
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
make -C "$engine" -s -j 4 TARGET=mac-omp LIBOMP_PREFIX="$omp" lib
xcrun clang -std=c23 -O2 -Wall -Wextra -Werror -I"$engine/include" "$diag/qwen47-prefix.c" \
  "$engine/lib/mac-omp/release/libgeist.a" -framework Accelerate -L"$omp/lib" -lomp -lm \
  -o "$engine/qwen47-prefix"
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
driver="$engine/qwen47-prefix"
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
replay QWEN47_TRACE=1 > "$out/trace.txt"
for file in default tensor-off trace; do
  printf '%s ' "$file"
  tail -1 "$out/$file.txt"
done
awk '$1 == "logits"' "$out/trace.txt" > "$out/trace-logits.txt"
diff -u "$out/default.txt" "$out/trace-logits.txt"
