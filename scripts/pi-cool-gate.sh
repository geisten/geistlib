#!/bin/sh
# pi-cool-gate.sh [--report] — the Raspberry Pi 5 thermal gate of pi5.yml and
# pi5-ab.yml. Without an argument: wait up to 5 min for the board to be cool
# (< 60 °C) and idle (1-min load < 1) before a measurement, exit 1 if it
# never is. --report: print the thermal state after a run.
zone=/sys/class/thermal/thermal_zone0/temp
if [ "${1:-}" = --report ]; then
  echo "temp=$(cat $zone) throttled=$(vcgencmd get_throttled 2>/dev/null || echo n/a)"
  exit 0
fi
for _ in $(seq 1 30); do
  t=$(cat $zone); l=$(cut -d' ' -f1 /proc/loadavg)
  echo "temp=$t load=$l"
  [ "$t" -lt 60000 ] && [ "${l%.*}" -lt 1 ] && exit 0
  sleep 10
done
echo "board not cool/idle after 5 min; skipping the measurement" && exit 1
