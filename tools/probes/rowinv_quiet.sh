#!/bin/sh
# rowinv_quiet.sh — before a measurement on the desktop: wait until no
# GITEA-ACTIONS-TASK container runs and the CPU was >= 97 % idle over 10 s,
# then log the top of ps. Exit 1 after QUIET_TIMEOUT seconds (default 1800).
set -u
LOG=${QUIET_LOG:-/dev/stderr}
deadline=$(( $(date +%s) + ${QUIET_TIMEOUT:-1800} ))
idle_pct() {
    set -- $(head -1 /proc/stat); shift
    t1=0; for v in "$@"; do t1=$((t1 + v)); done; i1=$(($4 + $5))
    sleep 10
    set -- $(head -1 /proc/stat); shift
    t2=0; for v in "$@"; do t2=$((t2 + v)); done; i2=$(($4 + $5))
    echo $(( 1000 * (i2 - i1) / (t2 - t1) ))
}
while :; do
    c=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c GITEA-ACTIONS-TASK)
    idle=$(idle_pct)
    if [ "$c" -eq 0 ] && [ "$idle" -ge 970 ]; then
        { echo "quiet $(date -u +%FT%TZ): idle=$((idle / 10)).$((idle % 10)) % containers=0"
          ps -eo pid,pcpu,etime,comm --sort=-pcpu | head -8; } >> "$LOG"
        exit 0
    fi
    echo "waiting $(date -u +%FT%TZ): idle=$((idle / 10)).$((idle % 10)) % task containers=$c" >> "$LOG"
    [ "$(date +%s)" -lt "$deadline" ] || { echo "host did not become quiet" >> "$LOG"; exit 1; }
    sleep 20
done
