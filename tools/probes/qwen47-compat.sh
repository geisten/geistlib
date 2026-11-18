#!/bin/sh
# Compare the pinned M1 engine and the M6 compatibility build at full vocabulary.
set -eu
repo=$(git rev-parse --show-toplevel)
diag="$repo/tools/probes"
role="$1"
out="$2"
calibration=${3:-}
revision=916f0e2bbc76fe19af5278ee0922e02ba000e065
model_id=$(/usr/sbin/sysctl -n hw.model)
case "$role:$model_id" in
  reference:MacBookPro18,4)
    bundle="$HOME/workspace/geist-lab/runs/macmini-handoff-2026-11-10"
    omp=/opt/homebrew/opt/libomp
    lock="$HOME/bin/with-mac-lock"
    ;;
  candidate:Mac18,5)
    bundle="$HOME/geist-lab-qwen32k/incoming"
    omp="$HOME/geist-lab-qwen32k/state/runtime/libomp/23.1.5"
    lock="$HOME/bin/with-macmini-lock"
    ;;
  *) printf '%s\n' "Invalid replay role or host: $role / $model_id" >&2; exit 2 ;;
esac
models="$bundle/assets/models"
snapshot="$bundle/snapshot-00000083"
mkdir -p "$out"
engine=$(mktemp -d "${RUNNER_TEMP:-/private/tmp}/qwen47-compat.XXXXXX")
trap 'rm -rf "$engine"' EXIT

/usr/bin/shasum -a 256 -c <<HASHES
689a06b8bbd262db30fbe001287edcc88f4124f8074c72d3fa419300942db1fe  $models/base-f32.gguf
ebdc74d2bc7c24246ae0beb8d7ff54f5125633ee7075aa5b23658c2bfa9826b4  $models/p50c-f32.gguf
7285a6bbafef71b0275912ad005f04179e971abe79a1093aa9a8cf674e21d3cd  $omp/lib/libomp.dylib
HASHES
git archive --output="$engine/source.tar" "$revision"
/usr/bin/tar -xf "$engine/source.tar" -C "$engine"
/usr/bin/python3 "$diag/qwen47-instrument.py" "$engine" "$diag"
/usr/bin/python3 "$diag/qwen47-ffn-trace.py" "$engine"
for id in 47 48; do
  /usr/bin/python3 "$diag/qwen47-compare-compat.py" fixture "$snapshot" "$id" "$engine/case-$id.bin"
done
build() {
  make -C "$engine" -s -j 4 TARGET=mac-omp LIBOMP_PREFIX="$omp" lib
  xcrun clang -std=c23 -O2 -Wall -Wextra -Werror -I"$engine/include" "$diag/qwen47-prefix.c" \
    "$engine/lib/mac-omp/release/libgeist.a" -framework Accelerate -L"$omp/lib" -lomp -lm \
    -o "$engine/replay"
}
replay() {
  target="$1"
  shift
  mkdir -p "$target/case-47" "$target/case-48"
  # shellcheck disable=SC2016  # The locked child shell expands its own arguments.
  LOCK_MAX=5 LOCK_NOTE="Qwen $role two-model compatibility replay" \
    "$lock" /bin/sh -eu -c '
      driver=$1; omp=$2; models=$3; dir=$4; fixtures=$5
      shift 5
      unset QWEN47_PATCH_QKV
      export DYLD_LIBRARY_PATH="$omp/lib" QWEN47_TRACE=1
      env "$@" QWEN47_RAW_DIR="$dir/case-47" "$driver" "$models/base-f32.gguf" "$fixtures/case-47.bin" >"$dir/case-47.txt"
      env "$@" QWEN47_RAW_DIR="$dir/case-48" "$driver" "$models/p50c-f32.gguf" "$fixtures/case-48.bin" >"$dir/case-48.txt"
    ' replay "$engine/replay" "$omp" "$models" "$target" "$engine" "$@"
}
build
{
  sw_vers
  printf '%s\n' "$model_id"
  xcrun clang --version
  otool -L "$engine/replay"
} > "$out/platform.txt"
if [ "$role" = reference ]; then
  replay "$out"
  awk '$1 == "logits" && $2 < 91 {print $2, $3}' "$out/case-47.txt" > "$engine/case-47-prefix.txt"
  diff -u "$diag/qwen47-m1-logits.txt" "$engine/case-47-prefix.txt"
  exit 0
fi
replay "$out/native"
(
  cd "$calibration"
  for name in checksums beta-k-checksums softplus-checksums; do
    /usr/bin/shasum -a 256 -c "$name.txt"
  done
)
/usr/bin/python3 "$diag/qwen47-m1-rounding-patch.py" "$engine" m1-ffn
for unit in lifecycle ops pipelines; do
  rm -f "$engine/build/mac-omp/release/src/backends/metal/$unit.o"
done
build
replay "$out/compatible" QWEN47_M1_EXP="$calibration/exp.qm1" QWEN47_M1_RCP="$calibration/rcp.qm1" \
  QWEN47_M1_BETA_NEG="$calibration/beta-neg.qm1" QWEN47_M1_RSQRT="$calibration/rsqrt.qm1" \
  QWEN47_M1_SOFTPLUS_POS="$calibration/softplus-pos.qm1" QWEN47_M1_SOFTPLUS_NEG="$calibration/softplus-neg.qm1"
