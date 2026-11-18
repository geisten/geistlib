#!/bin/sh
# run.sh <name> <ids|-> <tau>: one chunk of the #467 campaign (docs/DECISION_CAMPAIGN_PLAN.json early_exit_tau09_467)
set -e
D=/Users/germar/workspace/geistlib-campaigns/467
W=$D/src
MODEL="/Users/germar/Library/Application Support/geisten/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
IDS=""
[ "$2" != "-" ] && IDS="--question-ids $2"
# power state every 30 s while the chunk runs (accuracy does not depend on it)
( while :; do $D/power.sh state >> $D/power-$1.log; sleep 30; done ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
cd $W && caffeinate -i -s env OMP_NUM_THREADS=6 OMP_WAIT_POLICY=passive /opt/homebrew/bin/python3.14 tools/eval_decision_batched_reasoning.py \
  --cases-dir $D/cases-test --split test --model "$MODEL" --binary $W/bin/mac-omp/release/tools/bench_reasoning_batch \
  --backend metal --threads 6 --align 320 --cap 8192 --batch 8 --probe-interval 128 --exit-tau $3 $IDS \
  --out-dir $D/$1 --chunk-questions 64 --deadline-minutes 28
