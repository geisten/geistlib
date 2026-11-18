#!/bin/sh
# Sample the native M1 FP32 operations without loading a model.
set -eu
diag=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
if [ "$1" = --sample ]; then
    sample=$2
    work=$3
    "$sample" check-exp 00000000 32ffffff 3f800000
    "$sample" check-exp 43000001 7f800000 00000000
    "$sample" exp 33000000 43000000 "$work/exp.raw"
    "$sample" rcp 3f800000 40000000 "$work/rcp.raw"
    "$sample" check-beta-neg 00000000 32ffffff 3f000000
    "$sample" check-beta-neg 43000001 7f800000 00000000
    "$sample" beta-neg 33000000 43000000 "$work/beta-neg.raw"
    "$sample" check-rsqrt-scale 00800000 7f7fffff 0
    "$sample" rsqrt 3f800000 407fffff "$work/rsqrt.raw"
    if [ "${4:-}" = compatibility ]; then
        "$sample" check-rcp-scale 3f800000 47000000 0
        "$sample" check-softplus-pos 00000000 32ffffff 3f317218
        "$sample" check-softplus-neg 00000000 32ffffff 3f317218
        "$sample" check-softplus-pos 43000001 7f800000 7f800000
        "$sample" check-softplus-neg 43000001 7f800000 0
        "$sample" softplus-pos 33000000 43000000 "$work/softplus-pos.raw"
        "$sample" softplus-neg 33000000 43000000 "$work/softplus-neg.raw"
    fi
    exit 0
fi
out="$1"
mode=${2:-}
[ "$(/usr/sbin/sysctl -n hw.model)" = MacBookPro18,4 ] || exit 1
mkdir -p "$out"
work=$(mktemp -d "${RUNNER_TEMP:-/private/tmp}/qwen47-calibration.XXXXXX")
trap 'rm -rf "$work"' EXIT
sw_vers > "$out/source-platform.txt"
printf '%s\n' MacBookPro18,4 >> "$out/source-platform.txt"
xcrun clang --version >> "$out/source-platform.txt"
xcrun clang -std=c23 -O2 -Wall -Wextra -Werror "$diag/qwen-m1-metal-sample.c" \
  -framework Metal -framework Foundation -o "$work/sample"
xcrun clang -std=c23 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror \
  "$diag/qwen-m1-rounding.c" -o "$work/pack"
LOCK_MAX=3 LOCK_NOTE='M1 DeltaNet FP32 calibration without a model' \
  "$HOME/bin/with-mac-lock" /bin/sh "$diag/qwen47-m1-calibrate.sh" --sample "$work/sample" "$work" "$mode"
"$work/pack" encode 33000000 43000000 "$work/exp.raw" "$out/exp.qm1"
"$work/pack" encode 3f800000 40000000 "$work/rcp.raw" "$out/rcp.qm1"
"$work/pack" verify "$work/exp.raw" "$out/exp.qm1"
"$work/pack" verify "$work/rcp.raw" "$out/rcp.qm1"
"$work/pack" encode 33000000 43000000 "$work/beta-neg.raw" "$out/beta-neg.qm1"
"$work/pack" encode 3f800000 407fffff "$work/rsqrt.raw" "$out/rsqrt.qm1"
"$work/pack" verify "$work/beta-neg.raw" "$out/beta-neg.qm1"
"$work/pack" verify "$work/rsqrt.raw" "$out/rsqrt.qm1"
"$work/pack" check-deltanet "$out/exp.qm1" "$out/rcp.qm1" "$out/beta-neg.qm1" "$out/rsqrt.qm1" \
  "$diag/qwen47-m1-deltanet-reference.txt"
(
  cd "$out"
  /usr/bin/shasum -a 256 exp.qm1 rcp.qm1 source-platform.txt > checksums.txt
  /usr/bin/shasum -a 256 beta-neg.qm1 rsqrt.qm1 > beta-k-checksums.txt
)
if [ "$mode" = compatibility ]; then
    for name in softplus-pos softplus-neg; do
        "$work/pack" encode 33000000 43000000 "$work/$name.raw" "$out/$name.qm1"
        "$work/pack" verify "$work/$name.raw" "$out/$name.qm1"
    done
    (cd "$out" && /usr/bin/shasum -a 256 softplus-pos.qm1 softplus-neg.qm1 > softplus-checksums.txt)
fi
