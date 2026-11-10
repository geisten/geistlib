#!/bin/sh
# ab374b.sh <out dir> — #374 bench scratch (not for main): one
# bench_revision_ab per model in $MODELS, each after quiet_wait.sh. The
# caller holds with-desktop-lock once around this whole script.
# Env: MODELS (gptoss;deepseek;qwen3moe), BASE, NEW, SEQ, DN, CYC, XENV,
# GPTOSS / DEEPSEEK / QWEN3MOE (model paths).
set -u
O=$1
mkdir -p "$O"
envs="--env GEIST_BENCH_BACKEND=cpu_x86"
for e in $(echo "$XENV" | tr ';' ' '); do envs="$envs --env $e"; done
for m in $(echo "$MODELS" | tr ';' ' '); do
    case $m in gptoss) F=$GPTOSS ;; deepseek) F=$DEEPSEEK ;; qwen3moe) F=$QWEN3MOE ;; *) continue ;; esac
    echo "== $m  base=$BASE ($(git rev-parse --short "$BASE")) new=$NEW ($(git rev-parse --short "$NEW"))" | tee -a "$O/report.txt"
    sh tools/probes/quiet_wait.sh 2>&1 | tee -a "$O/report.txt"
    # shellcheck disable=SC2086
    python3 tools/bench_revision_ab.py \
        --rev "base=$BASE" --rev "new=$NEW" --gguf "$F" --seq-lens "$SEQ" \
        --decode-n "$DN" --repeats 1 --warmup 1 --cycles "$CYC" $envs \
        --prompt-file docs/OFFLOAD.md \
        --make-arg TARGET=linux --make-arg "BACKENDS=cpu_x86 cpu_scalar" \
        --make-arg GEMM_PROVIDER=native --out-dir "$O/ab-$m" 2>&1 | tee -a "$O/report.txt"
done
