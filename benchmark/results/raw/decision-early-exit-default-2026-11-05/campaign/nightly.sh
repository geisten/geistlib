#!/bin/sh
# #286 night-only scheduler: chunks start only 21:00-07:00 Berlin; at 07:00 STOP (a running chunk finishes).
# Main run, then the latency check: 300 live with tau 0.8, then the same 300 with full reasoning (amendment, PR #322).
D=/Users/germar/workspace/geistlib-campaigns/286
pending() { /opt/homebrew/bin/python3.14 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["pending"])
except Exception: print(99999)' "$D/$1/status.json"; }
for job in main latency latency-full; do
  case $job in main) args="main - 0";; latency) args="latency $D/latency_ids.json 0.8";;
    latency-full) args="latency-full $D/latency_ids.json 0";; esac
  while [ "$(pending $job)" != 0 ]; do
    h=$(date +%H%M)
    if [ "$h" -ge 2100 ] || [ "$h" -lt 0700 ]; then
      rm -f $D/STOP
      if ! pgrep -f "286/campaign.sh" >/dev/null; then
        grep -q "chunk.*failed" $D/night.log 2>/dev/null && { echo "$(date) a chunk failed: scheduler stops"; exit 1; }
        echo "$(date) night: run $job (pending $(pending $job))"
        nohup $D/campaign.sh $args >> $D/night.log 2>&1 &
      fi
    else
      [ -e $D/STOP ] || { touch $D/STOP; echo "$(date) day: no new chunks"; }
    fi
    sleep 60
  done
  echo "$(date) $job complete"
done
