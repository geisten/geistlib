#!/bin/sh
# campaign.sh <name> <ids|-> <tau>: chunks until done or STOP (the night scheduler sets STOP at 07:00)
D=/Users/germar/workspace/geistlib-campaigns/286
pending() { /opt/homebrew/bin/python3.14 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["pending"])
except Exception: print(99999)' "$D/$1/status.json"; }
n=0
while [ "$(pending $1)" != 0 ]; do
  [ -e $D/STOP ] && { echo "$(date) STOP"; exit 2; }
  # latency phase: only on AC with Low Power Mode off (#286, PR #321); wait without holding the lock
  if case "$1" in latency*) true;; *) false;; esac && ! $D/power.sh ok; then
    echo "$(date) power gate: $($D/power.sh state), waiting"; sleep 60; continue
  fi
  n=$((n+1)); echo "$(date) $1 chunk $n (pending $(pending $1))"
  LOCK_PRIO=batch LOCK_MAX=30 LOCK_NOTE="#286 $1 chunk $n" ~/bin/with-mac-lock sh $D/run.sh "$@"; rc=$?
  [ $rc = 75 ] && { echo "$(date) power gate: chunk $n not started ($($D/power.sh state))"; sleep 60; continue; }
  [ $rc = 0 ] || { echo "$(date) $1 chunk $n failed"; exit 1; }
  sleep 30
done
echo "$(date) $1 complete"
