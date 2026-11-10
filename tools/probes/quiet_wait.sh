#!/bin/sh
# quiet_wait.sh — #374 bench scratch (not for main): block until no Gitea
# job container runs on this host and the CPU is >= 97 % idle over 10 s,
# logging every probe and, when quiet, the top of ps. Gives up after ~40 min
# and says so (the measurement then carries the note).
i=0
while :; do
    n=$(docker ps --filter name=GITEA-ACTIONS-TASK -q 2>/dev/null | wc -l)
    idle=$(python3 -c '
import time
def r():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return v[3] + v[4], sum(v)
a = r(); time.sleep(10); b = r()
print(int(1000 * (b[0] - a[0]) / max(1, b[1] - a[1])))')
    echo "quiet_wait $(date +%T): task containers $n, idle ${idle} permille, load $(cut -d' ' -f1-3 /proc/loadavg)"
    if [ "$n" -eq 0 ] && [ "$idle" -ge 970 ]; then
        break
    fi
    docker ps --filter name=GITEA-ACTIONS-TASK --format '  {{.Names}} {{.RunningFor}}' 2>/dev/null
    ps -eo pid,pcpu,etime,comm --sort=-pcpu | head -6 | sed 's/^/  /'
    i=$((i + 1))
    if [ $i -ge 200 ]; then
        echo "quiet_wait: HOST NEVER GOT QUIET, measuring anyway"
        break
    fi
    sleep 2
done
echo "quiet_wait: ps at start of measurement"
ps -eo pid,user,pcpu,pmem,etime,args --sort=-pcpu | head -12 | cut -c1-160
