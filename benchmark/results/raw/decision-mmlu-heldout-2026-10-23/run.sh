#!/bin/sh
# #171 one chunk: run.sh <name> <split> <purpose> <K> [extra runner args]
set -e
name=$1 split=$2 purpose=$3 k=$4; shift 4
W=${W:-/tmp/geistlib-171}
C=/Users/germar/workspace/geistlib-campaigns/171
MODEL="/Users/germar/Library/Application Support/geisten/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
[ -f "$C/K.$name" ] && k=$(cat "$C/K.$name")  # chunk size override, read per chunk
cases=${CASES:-$C/cases-$name}; [ -d "$cases" ] || cases=$C/cases-$split
cd $W && caffeinate -i env OMP_NUM_THREADS=6 OMP_WAIT_POLICY=passive /opt/homebrew/bin/python3.14 tools/eval_decision_campaign.py \
  --cases-dir "$cases" --split $split --purpose $purpose \
  --model "$MODEL" --binary $W/bin/mac-omp/release/tools/bench_decision \
  --backend metal --threads 6 --decode-cap 512 --margin-pp 2 --warmup 0 --repeats ${REPEATS:-1} \
  --out-dir "$C/run-$name" --chunk-questions $k --deadline-minutes ${DEADLINE:-25} "$@"
