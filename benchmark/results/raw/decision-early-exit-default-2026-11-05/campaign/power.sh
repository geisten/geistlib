#!/bin/sh
# power.sh ok: exit 0 iff on AC and Low Power Mode off (#286, PR #321's rule)
# power.sh state: one line "unix AC|Batt lpm=0|1"
src=$(pmset -g batt | grep -q "'AC Power'" && echo AC || echo Batt)
lpm=$(pmset -g | grep -Eq '^ *(lowpowermode|powermode) +1$' && echo 1 || echo 0)
case $1 in
  ok) [ $src = AC ] && [ $lpm = 0 ] ;;
  state) echo "$(date +%s) $src lpm=$lpm" ;;
esac
