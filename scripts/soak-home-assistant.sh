#!/bin/sh
# Long-running resident-daemon reliability check with a JSON result artifact.
set -eu

usage() {
    cat <<'EOF'
usage: scripts/soak-home-assistant.sh [options]

Options:
  --socket PATH       daemon Unix socket (default: /config/geist.sock)
  --service NAME      systemd unit (default: geist-home)
  --duration SEC      total duration (default: 86400)
  --interval SEC      sample interval (default: 300)
  --output PATH       JSON result (default: ./geist-home-soak.json)
  --request TEXT      read-only status request
  --help
EOF
}

socket_path=/config/geist.sock
service=geist-home
duration=86400
interval=300
output=./geist-home-soak.json
request='Ist das Licht im Flur an?'
while [ "$#" -gt 0 ]; do
    case "$1" in
        --socket) socket_path=${2-}; shift 2 ;;
        --service) service=${2-}; shift 2 ;;
        --duration) duration=${2-}; shift 2 ;;
        --interval) interval=${2-}; shift 2 ;;
        --output) output=${2-}; shift 2 ;;
        --request) request=${2-}; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "soak-home-assistant: unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$duration:$interval" in
    *[!0-9:]*|0:*|*:0) echo "soak-home-assistant: duration/interval must be positive integers" >&2; exit 2 ;;
esac
command -v systemctl >/dev/null 2>&1 || { echo "soak-home-assistant: systemd required" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "soak-home-assistant: python3 required" >&2; exit 2; }
[ -S "$socket_path" ] || { echo "soak-home-assistant: socket missing: $socket_path" >&2; exit 1; }

start=$(date +%s)
deadline=$((start + duration))
start_pid=$(systemctl show "$service" -p MainPID --value)
start_restarts=$(systemctl show "$service" -p NRestarts --value)
samples=0
failures=0
min_rss=0
max_rss=0
latency_file=${output}.latency
: >"$latency_file"

while [ "$(date +%s)" -lt "$deadline" ]; do
    pid=$(systemctl show "$service" -p MainPID --value 2>/dev/null || printf 0)
    rss=$(systemctl show "$service" -p MemoryCurrent --value 2>/dev/null || printf 0)
    case "$rss" in ''|*[!0-9]*) rss=0 ;; esac
    if [ "$rss" -eq 0 ] && [ -r "/proc/$pid/status" ]; then
        rss_kb=$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$pid/status")
        case "$rss_kb" in ''|*[!0-9]*) rss_kb=0 ;; esac
        rss=$((rss_kb * 1024))
    fi
    if [ "$pid" != "$start_pid" ] || [ "$pid" = 0 ]; then failures=$((failures + 1)); fi
    if [ "$rss" -gt 0 ]; then
        if [ "$min_rss" -eq 0 ] || [ "$rss" -lt "$min_rss" ]; then min_rss=$rss; fi
        if [ "$rss" -gt "$max_rss" ]; then max_rss=$rss; fi
    fi
    if ! python3 -c 'import socket,sys,time
s=socket.socket(socket.AF_UNIX); s.settimeout(60); t=time.monotonic(); s.connect(sys.argv[1]); s.sendall(sys.argv[2].encode()+b"\n"); s.shutdown(socket.SHUT_WR); data=b""
while True:
 c=s.recv(4096)
 if not c: break
 data+=c
assert data.strip()
print(round((time.monotonic()-t)*1000))' "$socket_path" "$request" >>"$latency_file"; then
        failures=$((failures + 1))
    fi
    samples=$((samples + 1))
    now=$(date +%s)
    remaining=$((deadline - now))
    [ "$remaining" -le 0 ] && break
    sleep_for=$interval
    [ "$remaining" -lt "$sleep_for" ] && sleep_for=$remaining
    sleep "$sleep_for"
done

end=$(date +%s)
end_pid=$(systemctl show "$service" -p MainPID --value 2>/dev/null || printf 0)
end_restarts=$(systemctl show "$service" -p NRestarts --value 2>/dev/null || printf 999999)
if [ "$end_pid" != "$start_pid" ] || [ "$end_restarts" != "$start_restarts" ]; then failures=$((failures + 1)); fi
python3 -c 'import json,math,statistics,sys
v=sorted(map(int,open(sys.argv[1]).read().split()))
result={"status":"PASS" if int(sys.argv[2])==0 else "FAIL","duration_s":int(sys.argv[3]),"samples":int(sys.argv[4]),"failures":int(sys.argv[2]),"start_pid":int(sys.argv[5]),"end_pid":int(sys.argv[6]),"start_restarts":int(sys.argv[7]),"end_restarts":int(sys.argv[8]),"min_rss_bytes":int(sys.argv[9]),"max_rss_bytes":int(sys.argv[10]),"rss_growth_bytes":int(sys.argv[10])-int(sys.argv[9]),"latency_p50_ms":round(statistics.median(v)) if v else None,"latency_p95_ms":v[math.ceil(.95*len(v))-1] if v else None}
open(sys.argv[11],"w").write(json.dumps(result,indent=2)+"\n")' \
    "$latency_file" "$failures" "$((end-start))" "$samples" "$start_pid" "$end_pid" \
    "$start_restarts" "$end_restarts" "$min_rss" "$max_rss" "$output"
rm -f "$latency_file"
cat "$output"
[ "$failures" -eq 0 ]
