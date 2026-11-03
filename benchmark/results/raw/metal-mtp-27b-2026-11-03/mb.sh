#!/bin/bash
# mb.sh: GPU time per linear call (sequence of 30, best of 5), rows 1..8, 27B shapes
C=${C:-.}
for f in "q40 5120 17408 18 32" "q40 17408 5120 18 32" "q41 17408 5120 20 32" "q5k 6144 5120 176 256" "q80 10240 5120 34 32"; do
 for arm in base new n4row; do
  case $arm in base) b=$C/mb_base; e="";; new) b=$C/mb_new; e="";; n4row) b=$C/mb_new; e="GEIST_METAL_MV_M8=0";; esac
  echo "$f $arm: $(env $e $b $f 30 | awk '{printf "%s ",$6}')"
 done
done
