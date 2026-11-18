#!/bin/sh
# Relink the frozen C benchmark frontend with the reviewed Metal arithmetic.
set -eu
repo=$(git rev-parse --show-toplevel)
calibration="$1"
out="$2"
bundle="$HOME/workspace/geist-lab/runs/macmini-handoff-2026-11-10"
work=$(mktemp -d "${RUNNER_TEMP:-/private/tmp}/qwen47-build.XXXXXX")
trap 'rm -rf "$work"' EXIT
[ "$(/usr/sbin/sysctl -n hw.model)" = MacBookPro18,4 ] || exit 2
mkdir -p "$work/engine" "$out"
awk '$2 ~ /^runner-source\// {print}' "$bundle/SHA256SUMS" > "$work/runner-source.sha256"
printf '%s  %s\n' 1e4971cb7fb4833e756736ead26d629b3217afe8c3659d139ccf86c0affc63bb \
  "$work/runner-source.sha256" | /usr/bin/shasum -a 256 -c
/usr/bin/python3 - "$bundle" "$work" <<'PY'
import hashlib
from pathlib import Path
import sys

source, scratch = map(Path, sys.argv[1:])
for line in (scratch / "runner-source.sha256").read_text().splitlines():
    digest, relative = line.split(maxsplit=1)
    path = source / relative
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Archived benchmark source changed: {relative}")
    dst = scratch / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
PY
revision=$(cat "$work/runner-source/geistlib-revision")
[ "$revision" = 916f0e2bbc76fe19af5278ee0922e02ba000e065 ] || exit 2
git archive --output="$work/engine.tar" "$revision"
/usr/bin/tar -xf "$work/engine.tar" -C "$work/engine"
/usr/bin/python3 "$repo/tools/probes/qwen47-m1-rounding-patch.py" "$work/engine" m1-ffn
# The checked git archive above takes the place of the normal working-tree
# check. The scoped Metal patch is intentionally applied to that archive.
make -C "$work/runner-source" -f Makefile -o check-engine -s -j 4 all \
  TARGET=mac-omp MODE=release GEIST_DIR="$work/engine" WITH_MATH_VERIFY=1 \
  PYTHON_CONFIG=/opt/homebrew/opt/python@3.12/bin/python3.12-config
cp "$work/runner-source/build/geist-lab" "$out/runner"
for table in exp rcp beta-neg rsqrt softplus-pos softplus-neg; do
  cp "$calibration/$table.qm1" "$out/$table.qm1"
done
for manifest in checksums beta-k-checksums softplus-checksums source-platform; do
  cp "$calibration/$manifest.txt" "$out/$manifest.txt"
done
/usr/bin/python3 - "$out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
names = ("runner", "exp.qm1", "rcp.qm1", "beta-neg.qm1", "rsqrt.qm1", "softplus-pos.qm1", "softplus-neg.qm1")
hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
(root / "checksums.json").write_text(json.dumps(hashes, sort_keys=True) + "\n")
PY
