#!/usr/bin/env bash
# nightly-home-gate.sh — run the home-appliance eval gate on a schedule.
#
# The nightly answers one question: does the DEPLOYED tree on this box still
# pass the fixed home gate (AGENT_EVAL_HOME_MIN, forced mode)? It runs whatever
# is built here — deployment (rsync + make bin) is a deliberate, manual step,
# so a red nightly always means "this deployed state regressed", never "someone
# pushed something".
#
# Install (on the Pi):
#   crontab -e   ->   30 3 * * *  ~/geist-public/scripts/nightly-home-gate.sh
#
# Results: one line per night appended to ~/geist-nightly/home-gate.log,
# the full harness output in ~/geist-nightly/home-gate-<date>.log, and a
# ~/geist-nightly/FAILED marker while the latest run is red (removed on green —
# check it in the morning; there is deliberately no notification infra here).
#
# ponytail: cron + flock + log files, no systemd units, no mailer. The gate
# result also lands in benchmark/AGENT_EVAL.md by hand when it changes.
set -u

REPO="${GEIST_REPO:-$HOME/geist-public}"
OUT="${GEIST_NIGHTLY_DIR:-$HOME/geist-nightly}"
GGUF="${GEIST_GGUF_PATH:-$REPO/gguf_artifacts/bitnet-2b4t-i2_s.gguf}"
LOCK="$OUT/.lock"

mkdir -p "$OUT"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date '+%F %T') SKIP overlapping run" >> "$OUT/home-gate.log"
    exit 0
fi

STAMP=$(date '+%F')
LOG="$OUT/home-gate-$STAMP.log"
BIN="$REPO/bin/pi5/release/tests/bench_agent_eval"
[ -x "$BIN" ] || BIN="$(ls "$REPO"/bin/*/release/tests/bench_agent_eval 2>/dev/null | head -1)"

# Subshell, not a brace group: the inner exit must end the MEASURED run only,
# not the whole script — the summary line and FAILED marker below still run.
(
    date '+%F %T'
    # Quiesce check, not a hard stop: the HA container idles alongside — note
    # the load so a slow-looking run can be read with its context.
    uptime
    if [ ! -x "$BIN" ] || [ ! -f "$GGUF" ]; then
        echo "SETUP_MISSING bin=$BIN gguf=$GGUF"
        exit 99
    fi
    cd "$REPO" || exit 99
    T0=$(date +%s)
    GEIST_GGUF_PATH="$GGUF" OMP_NUM_THREADS=4 \
        "$BIN" --tools home --mode forced --min-pass "${AGENT_EVAL_HOME_MIN:-56}"
    RC=$?
    echo "EVAL_EXIT=$RC WALL_SECONDS=$(( $(date +%s) - T0 ))"
    exit $RC
) > "$LOG" 2>&1
RC=$?

SUMMARY=$(grep -m1 '^SUMMARY mode=forced' "$LOG" || echo "SUMMARY missing")
WALL=$(grep -o 'WALL_SECONDS=[0-9]*' "$LOG" || true)
if [ "$RC" -eq 0 ]; then
    rm -f "$OUT/FAILED"
    echo "$(date '+%F %T') PASS $SUMMARY $WALL" >> "$OUT/home-gate.log"
else
    echo "$LOG" > "$OUT/FAILED"
    echo "$(date '+%F %T') FAIL rc=$RC $SUMMARY $WALL" >> "$OUT/home-gate.log"
fi
# keep two weeks of full logs
find "$OUT" -name 'home-gate-*.log' -mtime +14 -delete 2>/dev/null
exit "$RC"
