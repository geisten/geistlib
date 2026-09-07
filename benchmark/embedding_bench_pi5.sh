#!/usr/bin/env bash
# benchmark/embedding_bench_pi5.sh — run benchmark/embedding_protocol.json.
#
# The protocol has been pinned since 0.11.0 and nothing executed it, so
# `status` stayed "UNMEASURED". This is the missing half: it reads the protocol
# as the source of truth, measures prefill throughput for the BitNet embedding
# models, and writes the numbers back — but only from a run it can vouch for.
#
# Run ON the Pi 5, not cross-invoked, like compare_ternary_pi5.sh.
#
# Usage:
#   MODEL=~/models/bitnet-embeddings-0.6b-bf16-i2_s.gguf \
#   ./benchmark/embedding_bench_pi5.sh            # measure, print, save raw
#   ./benchmark/embedding_bench_pi5.sh --write    # ... and update the protocol
#
# Env knobs (all optional except MODEL):
#   MODEL         path to the embedding GGUF (required)
#   MODEL_KEY     which protocol `models` entry this is
#                 (default: guessed from the filename)
#   PROFILE       host profile in the protocol (default: pi5)
#   GEIST_BIN     bench binary (default: bin/pi5/release/tests/bench_perf_sweep)
#   THREADS       override the profile's prefill_threads
#   SEQ_LENS      override the protocol's seq_lens (comma list)
#   REPEATS       override the protocol's repeats
#   MAX_TEMP_C    start/continue only below this (default 56, per METHODOLOGY.md)
#   MAX_LOAD      refuse above this 1-minute load average (default 0.5)
#   COOL_TIMEOUT  seconds to wait for the board to cool between points (default 600)
#   SKIP_COOL     =1 bypasses the thermal gate — TAINTS the run
#   FORCE_WRITE   =1 writes the protocol even from a tainted run (don't)
#
# WHY THE GATE IS PER SEQUENCE LENGTH, not just at the start:
# METHODOLOGY.md records that a large prefill drives this passively-cooled
# board to ~78 °C and trips the soft temp limit in under a minute — that is how
# an earlier revision understated llama's pp128 to 22 t/s when the true figure
# was ~37. This sweep runs to pp4096. Gating only at t=0 would measure the
# cooling system from pp1024 onward, and the longest sequences — the ones that
# matter most for an embedding model — would be the most wrong.
#
# WHY A TAINTED RUN IS NOT WRITTEN BACK:
# `vcgencmd get_throttled` reports throttling that HAPPENED, not just what is
# happening now, so a run can look fine at the end and still have been
# throttled in the middle. A pinned protocol whose numbers came from a
# throttled board is worse than one that still says UNMEASURED: the first
# invites comparison, the second says plainly that there is nothing to compare.
set -u

MODEL="${MODEL:-}"
PROTOCOL="benchmark/embedding_protocol.json"
PROFILE="${PROFILE:-pi5}"
GEIST_BIN="${GEIST_BIN:-bin/pi5/release/tests/bench_perf_sweep}"
MAX_TEMP_C="${MAX_TEMP_C:-56}"
MAX_LOAD="${MAX_LOAD:-0.5}"
COOL_TIMEOUT="${COOL_TIMEOUT:-600}"
SKIP_COOL="${SKIP_COOL:-0}"
FORCE_WRITE="${FORCE_WRITE:-0}"
WRITE=0
[ "${1:-}" = "--write" ] && WRITE=1

die()  { echo "error: $*" >&2; exit 1; }
note() { echo ">> $*"; }

TAINT=""
taint() { TAINT="${TAINT:+$TAINT; }$*"; note "TAINTED: $*"; }

[ -n "$MODEL" ] || die "set MODEL=path/to/embedding.gguf"
[ -f "$MODEL" ] || die "MODEL not found: $MODEL"
[ -f "$PROTOCOL" ] || die "$PROTOCOL not found — run from the repository root"
command -v python3 >/dev/null 2>&1 || die "python3 is required (statistics + JSON)"

MODEL_KEY="${MODEL_KEY:-}"
if [ -z "$MODEL_KEY" ]; then
  case "$(basename "$MODEL")" in
    *270m*|*270M*) MODEL_KEY="bitnet-embedding-270m" ;;
    *f16*|*F16*)   MODEL_KEY="bitnet-embedding-0.6b-f16" ;;
    *)             MODEL_KEY="bitnet-embedding-0.6b-i2_s" ;;
  esac
  note "MODEL_KEY guessed as $MODEL_KEY (override with MODEL_KEY=...)"
fi

# ---- the protocol is the source of truth, not this script's defaults -------
read -r P_SEQ P_REPEATS P_DECODE P_THREADS P_BACKEND <<EOF
$(python3 - "$PROTOCOL" "$PROFILE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
w = p["workload"]
prof = p["host_profiles"].get(sys.argv[2])
if prof is None:
    sys.exit(f"no host profile {sys.argv[2]!r} in the protocol")
print(",".join(str(s) for s in w["seq_lens"]), w["repeats"], w["decode_n"],
      prof["prefill_threads"], prof["expect_backend"])
PY
)
EOF
[ -n "${P_SEQ:-}" ] || die "could not read $PROTOCOL"

SEQ_LENS="${SEQ_LENS:-$P_SEQ}"
REPEATS="${REPEATS:-$P_REPEATS}"
THREADS="${THREADS:-$P_THREADS}"

# These models emit no tokens: geist_session_decode_step returns
# GEIST_E_UNSUPPORTED on them, so decode throughput is not a defined quantity.
# The protocol says decode_n: 0 for that reason; a protocol that said otherwise
# would be measuring a refusal.
[ "$P_DECODE" = "0" ] || die "protocol decode_n is $P_DECODE, expected 0 — embedding models emit no tokens"

note "protocol: seq_lens=$SEQ_LENS repeats=$REPEATS decode_n=0"
note "profile $PROFILE: ${THREADS} threads, expects $P_BACKEND"

OUT="benchmark/results/embedding-$(hostname)-${MODEL_KEY}-t${THREADS}"
mkdir -p "$OUT"
note "results dir: $OUT"

# ---- box hygiene ----------------------------------------------------------
LOAD1=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
note "uptime / load: $(uptime)"
if ! awk -v l="$LOAD1" -v m="$MAX_LOAD" 'BEGIN{exit !(l<m)}'; then
  taint "1-minute load $LOAD1 >= $MAX_LOAD (a single stray process inverts 4-thread numbers)"
fi

have_vcgencmd=0
command -v vcgencmd >/dev/null 2>&1 && have_vcgencmd=1
[ "$have_vcgencmd" = "1" ] || taint "vcgencmd not found — thermals and throttling unverifiable"

board_temp() {  # whole degrees, or empty
  [ "$have_vcgencmd" = "1" ] || return 0
  vcgencmd measure_temp 2>/dev/null | sed -E 's/[^0-9.]//g' | cut -d. -f1
}
throttled_word() {
  [ "$have_vcgencmd" = "1" ] || return 0
  vcgencmd get_throttled 2>/dev/null | sed 's/.*=//'
}

# Block until the board is below MAX_TEMP_C, or give up after COOL_TIMEOUT.
cool_gate() {  # $1 = what we are about to run, for the message
  [ "$SKIP_COOL" = "1" ] && return 0
  [ "$have_vcgencmd" = "1" ] || return 0
  local waited=0 t
  while :; do
    t=$(board_temp)
    [ -n "$t" ] || return 0
    [ "$t" -lt "$MAX_TEMP_C" ] && { note "$1: board at ${t}C — go"; return 0; }
    if [ "$waited" -ge "$COOL_TIMEOUT" ]; then
      taint "$1 started at ${t}C >= ${MAX_TEMP_C}C after ${waited}s of cooling"
      return 0
    fi
    note "$1: board at ${t}C, waiting for < ${MAX_TEMP_C}C (${waited}s/${COOL_TIMEOUT}s)"
    sleep 20; waited=$((waited + 20))
  done
}
[ "$SKIP_COOL" = "1" ] && taint "thermal gate bypassed (SKIP_COOL=1)"

if [ ! -x "$GEIST_BIN" ]; then
  note "building $GEIST_BIN ..."
  make TARGET=pi5 "$GEIST_BIN" >/dev/null 2>&1 \
    || make TARGET=pi5 tests >/dev/null 2>&1 \
    || die "could not build $GEIST_BIN — build geist first (make TARGET=pi5)"
fi

THROT_BEFORE=$(throttled_word)
note "throttled flags before: ${THROT_BEFORE:-n/a}"

# ---- measure, one sequence length at a time so the gate can run between ----
RAW="$OUT/raw.jsonl"
: > "$RAW"
echo
for n in $(echo "$SEQ_LENS" | tr ',' ' '); do
  cool_gate "pp$n"
  note "=== pp$n — $REPEATS repeat(s), warmup discarded ==="
  OMP_WAIT_POLICY=active OMP_NUM_THREADS="$THREADS" \
    "$GEIST_BIN" --gguf "$MODEL" --seq-lens "$n" \
      --decode-n 0 --warmup "$n" --repeats "$REPEATS" \
    | tee -a "$RAW" \
    || taint "bench failed at pp$n"
  t=$(board_temp); [ -n "${t:-}" ] && note "pp$n finished at ${t}C"
done

THROT_AFTER=$(throttled_word)
note "throttled flags after: ${THROT_AFTER:-n/a}"
if [ -n "${THROT_AFTER:-}" ] && [ "$THROT_AFTER" != "0x0" ]; then
  taint "vcgencmd reports throttling ($THROT_AFTER) — the board was limited during the run"
fi

# ---- median + MAD, exactly as the protocol pins them ----------------------
# bench_perf_sweep reports the MEAN; the protocol pins median and median
# absolute deviation, so the statistics come from the raw per-repeat samples,
# not from the tool's summary. METHODOLOGY.md: "Aggregation is never selected
# after looking at the result" — that is why this is not a choice made here.
echo
python3 - "$RAW" "$OUT/summary.json" "$MODEL_KEY" "$PROFILE" "$THREADS" "${TAINT:-}" <<'PY'
import json, statistics, sys

raw, out, key, profile, threads, taint = sys.argv[1:7]
rows = []
for line in open(raw):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass

points = []
for r in rows:
    s = r.get("samples", {}).get("prefill_ms") or []
    if not s:
        continue
    med = statistics.median(s)
    mad = statistics.median([abs(x - med) for x in s])
    n = r["seq_len"]
    points.append({
        "seq_len": n,
        "repeats": len(s),
        "prefill_ms_median": round(med, 3),
        "prefill_ms_mad": round(mad, 3),
        # tok/s from the median, not the median of the tok/s: the protocol
        # aggregates the measured quantity (time), and throughput is derived.
        "prefill_tps": round(n / (med / 1000.0), 2) if med > 0 else None,
    })
points.sort(key=lambda p: p["seq_len"])

summary = {"model": key, "profile": profile, "threads": int(threads),
           "decode_n": 0, "aggregation": "median", "dispersion": "median_absolute_deviation",
           "tainted": bool(taint), "taint_reason": taint or None, "points": points}
json.dump(summary, open(out, "w"), indent=2)

if not points:
    print("no usable samples — nothing measured")
    sys.exit(1)
w = max(len(str(p["seq_len"])) for p in points)
print(f"  {'pp':>{w+2}}  {'tok/s':>9}  {'median ms':>10}  {'MAD ms':>8}")
for p in points:
    print(f"  pp{p['seq_len']:<{w}}  {p['prefill_tps']:>9.2f}  "
          f"{p['prefill_ms_median']:>10.1f}  {p['prefill_ms_mad']:>8.1f}")
PY
PYRC=$?
echo
note "raw: $RAW"
note "summary: $OUT/summary.json"

# ---- write back, only from a run we can vouch for -------------------------
if [ "$WRITE" != "1" ]; then
  echo
  note "protocol NOT updated (pass --write to record these numbers)"
  exit $PYRC
fi
if [ $PYRC -ne 0 ]; then
  die "nothing measured — refusing to write $PROTOCOL"
fi
if [ -n "$TAINT" ] && [ "$FORCE_WRITE" != "1" ]; then
  echo
  note "protocol NOT updated — this run is tainted: $TAINT"
  note "a pinned protocol carrying throttled numbers is worse than one that"
  note "still says UNMEASURED. Fix the cause and re-run, or FORCE_WRITE=1."
  exit 1
fi

python3 - "$PROTOCOL" "$OUT/summary.json" <<'PY'
import json, subprocess, sys, datetime

proto_path, summary_path = sys.argv[1], sys.argv[2]
proto = json.load(open(proto_path))
s = json.load(open(summary_path))

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or "unknown"
runs = proto.setdefault("runs", [])
runs = [r for r in runs
        if not (r.get("model") == s["model"] and r.get("profile") == s["profile"])]
runs.append({
    "model": s["model"], "profile": s["profile"], "threads": s["threads"],
    "measured_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "geist_commit": commit,
    "aggregation": s["aggregation"], "dispersion": s["dispersion"],
    "points": s["points"],
})
proto["runs"] = runs
proto["status"] = (
    "MEASURED — see runs[]. Prefill only; these models emit no tokens, so there "
    "is no decode figure and none is implied."
)
json.dump(proto, open(proto_path, "w"), indent=2)
open(proto_path, "a").write("\n")
print(f"updated {proto_path}: {s['model']} on {s['profile']} @ {commit}")
PY
