#!/bin/sh
# #467 night-only scheduler: chunks start only 21:00-07:00 Berlin; at 07:00 STOP (a running chunk finishes).
# Waits until the declaration (early_exit_tau09_467) is on origin/main: no inference before it is merged.
D=/Users/germar/workspace/geistlib-campaigns/467
pending() { /opt/homebrew/bin/python3.14 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["pending"])
except Exception: print(99999)' "$D/$1/status.json"; }
declared() { git -C $D/src fetch -q origin main 2>/dev/null; git -C $D/src show origin/main:docs/DECISION_CAMPAIGN_PLAN.json 2>/dev/null | grep -q '"early_exit_tau09_467"'; }
until declared; do echo "$(date) declaration not on main yet, waiting"; sleep 300; done
echo "$(date) declaration on main"
while [ "$(pending main)" != 0 ]; do
  h=$(date +%H%M)
  if [ "$h" -ge 2100 ] || [ "$h" -lt 0700 ]; then
    rm -f $D/STOP
    if ! pgrep -f "467/campaign.sh" >/dev/null; then
      grep -q "chunk.*failed" $D/night.log 2>/dev/null && { echo "$(date) a chunk failed: scheduler stops"; exit 1; }
      echo "$(date) night: run main (pending $(pending main))"
      nohup $D/campaign.sh main - 0 >> $D/night.log 2>&1 &
    fi
  else
    [ -e $D/STOP ] || { touch $D/STOP; echo "$(date) day: no new chunks"; }
  fi
  sleep 60
done
echo "$(date) main complete"
