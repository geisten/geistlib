#!/bin/sh
# #171 detached campaign: prepare, then calibration chunks, then 500-question
# test chunks. Each step takes its own batch lease (<= 30 min) and releases it,
# so Mac CI and other agents get turns. Resumable: rerun this script any time.
# Usage: nohup ./campaign.sh KCAL KTEST >> campaign.log 2>&1 &
set -u
C=/Users/germar/workspace/geistlib-campaigns/171
KCAL=$1 KTEST=$2
cd $C
export W=$C/src
lock() { LOCK_PRIO=batch LOCK_MAX=30 LOCK_NOTE="$1" ~/bin/with-mac-lock sh -c "$2"; }
pending() {  # pending questions per status.json; unknown = 99999
  /opt/homebrew/bin/python3.14 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["pending"])
except Exception: print(99999)' "$C/run-$1/status.json"
}
[ -e STOP ] && { echo "STOP file present"; exit 0; }
for b in calibration test; do
  [ -e cases-$b/metadata.json ] || lock "#171 prepare $b" "./prepare.sh $b" || { echo "prepare $b failed"; exit 1; }
done
phase() {  # name purpose K total-questions [extra]
  name=$1 split=$2 purpose=$3 k=$4 total=$5; shift 5
  n=0
  while [ "$(pending $name)" != 0 ]; do
    [ -e STOP ] && { echo "$(date) STOP file: halting"; exit 0; }
    n=$((n+1)); m=$(( ( $(pending $name) > total ? total : $(pending $name) ) ))
    left=$(( (m + k - 1) / k ))
    echo "$(date) $name chunk $n (pending $m, ~$left chunks left)"
    lock "#171 $name chunk $n/~$left left" "./run.sh $name $split $purpose $k $*" \
      || { echo "$(date) $name chunk $n failed"; exit 1; }
    sleep 30   # let the broker hand the lease to someone else
  done
}
# Amendment 2 (user, 2026-10-22 evening): one trial per arm, core arms, 20-question
# determinism spot-check; the binary changed, so calibration is re-run (250, dense, 1 trial).
export REPEATS=0
export CASES=$C/cases-calibration250; phase calibration250b calibration calibration $KCAL 250 --calibration-modes decision_dense
CAL=$C/run-calibration250b/calibration.json
[ -e $CAL ] || { echo "no calibration.json"; exit 1; }
export CASES=$C/cases-test; phase test2 test quality $KTEST 520 --calibration $CAL --core-arms --spotcheck 20
echo "$(date) campaign complete"
