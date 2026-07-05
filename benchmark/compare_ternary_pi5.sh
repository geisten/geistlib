#!/usr/bin/env bash
# benchmark/compare_ternary_pi5.sh — ternary (BitNet b1.58 / TQ2_0) head-to-head
# on a Raspberry Pi 5: geist vs llama.cpp vs bitnet.cpp, one identical protocol.
#
# Goal this serves: establish whether geist's ternary decode/prefill is >=
# MAX(bitnet.cpp, llama.cpp) on the A76. It measures all three engines on the
# SAME GGUF, SAME thread count, from a COOL baseline (the Pi 5 is passively
# cooled and throttles — see benchmark/BENCHMARK_PI5.md), reporting the MEAN of
# N repeats after a discarded warm-up. Raw outputs are saved so nothing is lost.
#
# Run ON the Pi 5 (not cross-invoked). Reference binaries are optional — the
# script measures whatever it can find and skips the rest with a clear note.
#
# Usage:
#   MODEL=~/models/bitnet-2b4t-TQ2_0-v2.gguf \
#   LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench \
#   BITNET_BENCH=~/BitNet/build/bin/llama-bench \
#   ./benchmark/compare_ternary_pi5.sh
#
# Env knobs (all optional except MODEL):
#   MODEL         path to the TQ2_0 GGUF (required)
#   GEIST_BIN     geist bench (default: bin/pi5/release/tests/bench_perf_sweep)
#   LLAMA_BENCH   path to llama.cpp llama-bench (skipped if unset/not found)
#   BITNET_BENCH  path to bitnet.cpp llama-bench (skipped if unset/not found)
#   THREADS       thread count (default 4; decode is often best at 3 — see doc)
#   SEQ_LENS      prefill lengths (default 128,256,512)
#   DECODE_N      decode steps measured (default 64)
#   REPEATS       measured repeats, mean reported (default 10)
#   MAX_TEMP_C    refuse to start above this (default 56)
#   SKIP_COOL     set =1 to bypass the thermal gate (not recommended)
#   SETUP_REFS    set =1 to clone+build llama.cpp and bitnet.cpp under ./benchmark/ref/
#                 if their binaries aren't already provided (one-time, ~10 min)
#   LLAMA_REF     llama.cpp commit/tag to pin when SETUP_REFS builds it
#                 (default b9010 = d05fe1d, the documented Pi baseline)
#   BITNET_REF    microsoft/BitNet commit to pin when SETUP_REFS builds it
#                 (default 01eb415… — pins FUTURE runs; the numbers already in
#                 benchmark/headline_results.json predate this pin, see their
#                 baseline_version "master")
set -u

MODEL="${MODEL:-}"
GEIST_BIN="${GEIST_BIN:-bin/pi5/release/tests/bench_perf_sweep}"
LLAMA_BENCH="${LLAMA_BENCH:-}"
BITNET_BENCH="${BITNET_BENCH:-}"
THREADS="${THREADS:-4}"
SEQ_LENS="${SEQ_LENS:-128,256,512}"
DECODE_N="${DECODE_N:-64}"
REPEATS="${REPEATS:-10}"
MAX_TEMP_C="${MAX_TEMP_C:-56}"
SKIP_COOL="${SKIP_COOL:-0}"
LLAMA_REF="${LLAMA_REF:-b9010}"                                    # = d05fe1d, documented Pi baseline
BITNET_REF="${BITNET_REF:-01eb415772c342d9f20dc42772f1583ae1e5b102}"

die() { echo "error: $*" >&2; exit 1; }
note() { echo ">> $*"; }

# Pin a freshly-cloned reference repo to a fixed commit/tag (reproducibility).
# No-op with a warning if ref is empty. subs=1 re-syncs submodules after checkout.
pin_ref() {  # $1=dir  $2=ref  $3=1 to update submodules
  local dir="$1" ref="$2" subs="${3:-0}"
  [ -n "$ref" ] || { note "$dir: UNPINNED (default branch — run not reproducible)"; return 0; }
  if git -C "$dir" fetch --depth 1 origin "$ref" >/dev/null 2>&1; then
    git -C "$dir" checkout -q FETCH_HEAD
  elif ! git -C "$dir" checkout -q "$ref" 2>/dev/null; then
    note "$dir: could not checkout $ref — using default branch"; return 0
  fi
  [ "$subs" = "1" ] && git -C "$dir" submodule update --init --recursive >/dev/null 2>&1
  note "$dir pinned @ $ref ($(git -C "$dir" rev-parse --short HEAD 2>/dev/null))"
}

[ -n "$MODEL" ] || die "set MODEL=path/to/ternary.gguf (a TQ2_0 BitNet GGUF)"
[ -f "$MODEL" ] || die "MODEL not found: $MODEL"

OUT="benchmark/results/ternary-$(hostname)-t${THREADS}"
mkdir -p "$OUT"
note "results dir: $OUT"

# --- box hygiene: a stray process halves 4-thread numbers; heat throttles ---
note "uptime / load: $(uptime)"
if command -v vcgencmd >/dev/null 2>&1; then
  TEMP=$(vcgencmd measure_temp 2>/dev/null | sed -E 's/[^0-9.]//g')
  THROT=$(vcgencmd get_throttled 2>/dev/null)
  note "temp=${TEMP}C  ${THROT}"
  if [ "$SKIP_COOL" != "1" ] && [ -n "${TEMP:-}" ]; then
    # integer compare on the whole-degree part
    if [ "${TEMP%.*}" -ge "$MAX_TEMP_C" ]; then
      die "board at ${TEMP}C >= ${MAX_TEMP_C}C — let it cool (passively-cooled Pi throttles; the engine measured second loses unfairly). Set SKIP_COOL=1 to override."
    fi
  fi
else
  note "vcgencmd not found — cannot verify thermals; ensure the board is cool."
fi

# Build geist's bench on demand.
if [ ! -x "$GEIST_BIN" ]; then
  note "building geist bench ($GEIST_BIN) ..."
  make TARGET=pi5 "bin/pi5/release/tests/bench_perf_sweep" >/dev/null 2>&1 \
    || make TARGET=pi5 tests >/dev/null 2>&1 \
    || die "could not build $GEIST_BIN — build geist first (make TARGET=pi5)"
fi

# Optional one-time bootstrap of the reference engines on the Pi itself.
if [ "${SETUP_REFS:-0}" = "1" ]; then
  REFDIR="benchmark/ref"; mkdir -p "$REFDIR"
  if [ -z "$LLAMA_BENCH" ] || [ ! -x "$LLAMA_BENCH" ]; then
    note "building llama.cpp under $REFDIR/llama.cpp (OpenBLAS) ..."
    [ -d "$REFDIR/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$REFDIR/llama.cpp"
    pin_ref "$REFDIR/llama.cpp" "$LLAMA_REF"
    cmake -S "$REFDIR/llama.cpp" -B "$REFDIR/llama.cpp/build" \
      -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON -DLLAMA_CURL=OFF >/dev/null \
      && cmake --build "$REFDIR/llama.cpp/build" --target llama-bench -j4 >/dev/null \
      && LLAMA_BENCH="$REFDIR/llama.cpp/build/bin/llama-bench" \
      && note "llama-bench: $LLAMA_BENCH" || note "llama.cpp build FAILED — set LLAMA_BENCH manually"
  fi
  if [ -z "$BITNET_BENCH" ] || [ ! -x "$BITNET_BENCH" ]; then
    note "building bitnet.cpp under $REFDIR/BitNet ..."
    [ -d "$REFDIR/BitNet" ] || git clone --depth 1 --recursive https://github.com/microsoft/BitNet "$REFDIR/BitNet"
    pin_ref "$REFDIR/BitNet" "$BITNET_REF" 1
    # bitnet.cpp ships a llama.cpp fork in 3rdparty; build its llama-bench.
    cmake -S "$REFDIR/BitNet/3rdparty/llama.cpp" -B "$REFDIR/BitNet/build" -DGGML_NATIVE=ON >/dev/null 2>&1 \
      && cmake --build "$REFDIR/BitNet/build" --target llama-bench -j4 >/dev/null 2>&1 \
      && BITNET_BENCH="$REFDIR/BitNet/build/bin/llama-bench" \
      && note "bitnet-bench: $BITNET_BENCH" \
      || note "bitnet.cpp build needs its own setup (python setup_env.py) — see its README; set BITNET_BENCH manually"
  fi
fi

# llama-bench wants -p <prefill lens> -n <decode> -t <threads>; -r <repeats>.
LLAMA_PN=( -p "$SEQ_LENS" -n "$DECODE_N" -t "$THREADS" -r "$REPEATS" )

echo
note "=== geist (TQ2_0, SDOT default) ==="
OMP_WAIT_POLICY=active OMP_NUM_THREADS="$THREADS" \
  "$GEIST_BIN" --gguf "$MODEL" --seq-lens "$SEQ_LENS" \
  --decode-n "$DECODE_N" --warmup "$DECODE_N" --repeats "$REPEATS" \
  | tee "$OUT/geist.jsonl"

# Optional: geist with TL1 LUT decode (Apple-tuned path; measure on A76 too).
note "=== geist (TQ2_0, GEIST_TL1=1 decode) ==="
GEIST_TL1=1 OMP_WAIT_POLICY=active OMP_NUM_THREADS="$THREADS" \
  "$GEIST_BIN" --gguf "$MODEL" --seq-lens "$SEQ_LENS" \
  --decode-n "$DECODE_N" --warmup "$DECODE_N" --repeats "$REPEATS" \
  | tee "$OUT/geist-tl1.jsonl"

if [ -n "$LLAMA_BENCH" ] && [ -x "$LLAMA_BENCH" ]; then
  note "=== llama.cpp ($LLAMA_BENCH) ==="
  "$LLAMA_BENCH" -m "$MODEL" "${LLAMA_PN[@]}" | tee "$OUT/llama.txt"
else
  note "llama.cpp: SKIPPED (set LLAMA_BENCH to an executable llama-bench)"
fi

if [ -n "$BITNET_BENCH" ] && [ -x "$BITNET_BENCH" ]; then
  note "=== bitnet.cpp ($BITNET_BENCH) ==="
  "$BITNET_BENCH" -m "$MODEL" "${LLAMA_PN[@]}" | tee "$OUT/bitnet.txt"
else
  note "bitnet.cpp: SKIPPED (set BITNET_BENCH to an executable llama-bench)"
fi

echo
note "=== geist summary (mean tps) ==="
# Pull prefill_tps / decode_tps per seq_len from geist's JSONL (no jq needed).
extract_geist() {
  sed -nE 's/.*"seq_len":([0-9]+).*"prefill_tps":([0-9.]+).*"decode_tps":([0-9.]+).*/seq=\1  prefill=\2  decode=\3/p' "$1"
}
echo "[SDOT]"; extract_geist "$OUT/geist.jsonl"
echo "[TL1] "; extract_geist "$OUT/geist-tl1.jsonl"

echo
note "Reference tables are raw in $OUT/{llama,bitnet}.txt (llama-bench format:"
note "t/s columns 'pp<N>' = prefill, 'tg<N>' = decode). Compare against geist above."
note "Verdict rule for the goal: geist prefill@each seq AND decode must be >="
note "MAX(llama, bitnet). Record the winning config in BENCHMARK_PI5.md."
