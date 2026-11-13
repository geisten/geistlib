#!/bin/bash
# #444: the four nightly M1 Max cells at 6642260d, unchanged protocol/gates.
# Outside the lock: wait (bounded) until Spotlight/media daemons are idle and
# the CPU is >= 85 % idle; then one lease per cell. A failed cell (quiet
# timeout or spread gate) is retried later in the window.
S=/Users/germar/workspace/geistlib-campaigns/444/run
B=/Users/germar/workspace/geistlib-campaigns/444/tools
LOG=$S/run444.log
DEADLINE_HOUR=${DEADLINE_HOUR:-7}   # do not start a cell at/after 06:xx local
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

busy_procs() {
  ps -Ao pcpu=,comm= | LC_ALL=C awk '{gsub(",",".",$1); n=$2; sub(".*/","",n);
    if ($1+0 >= 10 && n ~ /^(mds|mds_stores|mdworker|mdworker_shared|mediaanalysisd|corespotlightd|spotlightknowledged|hybridsearchd|mobileassetd|photoanalysisd|bench_perf_sweep|llama-bench|clang|ld|make|eval_geist)$/) printf "%s(%s) ", n, $1}'
}
idle_pct() { top -l 2 -n 0 -s 5 | grep 'CPU usage' | tail -1 | tr , . | LC_ALL=C awk '{for(i=1;i<=NF;i++) if($i=="idle") print $(i-1)+0}' | tr -d %; }

wait_quiet() {   # up to $1 minutes; returns 0 when quiet for 2 consecutive checks
  local end=$(( $(date +%s) + $1*60 )) ok=0
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(date +%H)" -ge "$DEADLINE_HOUR" ] && [ "$(date +%H)" -lt 12 ] && { log "past window"; return 2; }
    local bp idle; bp=$(busy_procs); idle=$(idle_pct)
    if [ -z "$bp" ] && [ "${idle%.*}" -ge 85 ]; then ok=$((ok+1)); [ $ok -ge 2 ] && { log "quiet: idle=$idle"; return 0; }
    else ok=0; log "not quiet: idle=$idle busy=[$bp] load=$(LC_ALL=C sysctl -n vm.loadavg)"; fi
    sleep 60
  done
  return 1
}

run_cell() {  # column model protocol variant lockmax
  local col=$1 model=$2 proto=$3 var=$4 lmax=$5 repo=$6 pin=$7
  local llama gguf out=$S/out/$col/$model
  mkdir -p "$out"
  llama=$("$B"/scripts/ensure-llama-bench.sh "$var" ${repo:+"$repo"} ${pin:+"$pin"} 2>>"$LOG")
  gguf=$(cd "$B" && python3 tools/fetch_bench_model.py --model "$model" --protocol benchmark/cross_engine_${proto}_protocol.json)
  log "start $col/$model llama=$llama gguf=$gguf"
  ( cd "$B" && LOCK_PRIO=batch LOCK_MAX=$lmax LOCK_NOTE="#444 M1 Max baseline cells: $col/$model" \
      ~/bin/with-mac-lock python3 tools/bench_cross_engine.py \
        --geist "/Users/germar/workspace/geistlib-campaigns/444/metal/bin/mac-omp/release/tests/bench_perf_sweep" --llama "$llama" --gguf "$gguf" \
        --protocol benchmark/cross_engine_${proto}_protocol.json \
        --host-profile "$col" --model "$model" --out-dir "$out" ) >> "$out/stdout.log" 2>> "$out/stderr.log"
  local rc=$?
  log "end $col/$model rc=$rc; $(grep -E 'spread|quiet' "$out/stderr.log" | tail -3 | tr '\n' '|')"
  return $rc
}

CELLS=(
  "apple_m1max_metal gemma4-e2b-q4km gpu metal 120"
  "apple_m1max_metal bonsai2-27b-pq2 gpu metal 240 PrismML-Eng/llama.cpp adfffbe41"
)
mkdir -p $S/state
log "=== start"
for round in 1 2; do
  for c in "${CELLS[@]}"; do
    set -- $c; key="$1/$2"; sk="$S/state/$1_$2"
    [ -e "$sk.done" ] && continue
    wait_quiet 180; w=$?
    [ $w = 2 ] && { log "window over"; break 2; }
    [ $w = 1 ] && { log "host never quiet in 3 h before $key"; break 2; }
    t=$(( $(cat "$sk.tries" 2>/dev/null || echo 0) + 1 )); echo $t > "$sk.tries"
    if run_cell "$@"; then touch "$sk.done"; log "OK $key (try $t)"; else log "FAIL $key (try $t)"; fi
  done
done
for c in "${CELLS[@]}"; do set -- $c; log "summary $1/$2 done=$([ -e $S/state/$1_$2.done ] && echo 1 || echo 0) tries=$(cat $S/state/$1_$2.tries 2>/dev/null)"; done
log "=== end"
