#!/bin/sh
# #171 amendment 3: chat_reasoning at cap 2048 on the frozen 167-question list
# (3 capped dry-run, 3 finished determinism checks, 161 capped). Resumable.
# Usage: nohup ./campaign-cap2048.sh K >> campaign-cap2048.log 2>&1 &
set -u
C=/Users/germar/workspace/geistlib-campaigns/171
cd $C
export W=$C/src REPEATS=0 DECODE_CAP=2048 CASES=$C/cases-test
K=$1
pending() { /opt/homebrew/bin/python3.14 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["pending"])
except Exception: print(99999)' "$C/run-cap2048/status.json"; }
n=0
while [ "$(pending)" != 0 ]; do
  [ -e STOP ] && { echo "$(date) STOP file: halting"; exit 0; }
  n=$((n+1)); echo "$(date) cap2048 chunk $n (pending $(pending))"
  LOCK_PRIO=batch LOCK_MAX=30 LOCK_NOTE="#171 cap2048 chunk $n" ~/bin/with-mac-lock sh -c \
    "./run.sh cap2048 test quality $K --question-ids $C/cap2048_question_ids.json --reasoning-only" \
    || { echo "$(date) cap2048 chunk $n failed"; exit 1; }
  [ "${ONE:-0}" = 1 ] && exit 0
  sleep 30
done
echo "$(date) cap2048 complete"
