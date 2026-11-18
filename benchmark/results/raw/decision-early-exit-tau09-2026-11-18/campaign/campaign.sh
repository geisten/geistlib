#!/bin/sh
# campaign.sh <name> <ids|-> <tau>: chunks until done or STOP (the night scheduler sets STOP at 07:00)
D=/Users/germar/workspace/geistlib-campaigns/467
pending() { /opt/homebrew/bin/python3.14 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["pending"])
except Exception: print(99999)' "$D/$1/status.json"; }
n=0
while [ "$(pending $1)" != 0 ]; do
  [ -e $D/STOP ] && { echo "$(date) STOP"; exit 2; }
  n=$((n+1)); echo "$(date) $1 chunk $n (pending $(pending $1))"
  LOCK_PRIO=batch LOCK_MAX=30 LOCK_NOTE="#467 $1 chunk $n" ~/bin/with-mac-lock sh $D/run.sh "$@"; rc=$?
  [ $rc = 0 ] || { echo "$(date) $1 chunk $n failed"; exit 1; }
  sleep 30
done
echo "$(date) $1 complete"
