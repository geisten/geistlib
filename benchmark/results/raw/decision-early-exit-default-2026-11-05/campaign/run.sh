#!/bin/sh
# run.sh <name> <ids|-> <tau>: one chunk of the #286 campaign (docs/DECISION_CAMPAIGN_PLAN.json early_exit_default_286)
set -e
D=/Users/germar/workspace/geistlib-campaigns/286
W=$D/src
MODEL="/Users/germar/Library/Application Support/geisten/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
IDS=""
[ "$2" != "-" ] && IDS="--question-ids $2"
# latency phase: re-check the power gate under the lock (exit 75: campaign.sh retries)
case "$1" in latency*) true;; *) false;; esac && ! $D/power.sh ok && exit 75
# power state every 30 s while the chunk runs (sidecar, outside the evaluator's out-dir)
( while :; do $D/power.sh state >> $D/power-$1.log; sleep 30; done ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
cd $W && caffeinate -i env OMP_NUM_THREADS=6 OMP_WAIT_POLICY=passive /opt/homebrew/bin/python3.14 tools/eval_decision_batched_reasoning.py \
  --cases-dir $D/cases-test --split test --model "$MODEL" --binary $W/bin/mac-omp/release/tools/bench_reasoning_batch \
  --backend metal --threads 6 --align 320 --cap 8192 --batch 8 --probe-interval 128 --exit-tau $3 $IDS \
  --out-dir $D/$1 --chunk-questions 64 --deadline-minutes 28
