#!/usr/bin/env bash
# compare_metal.sh — cool-state Metal A/B: geist vs llama.cpp, with the
# pre-merge wip engine and llama as fixed reference targets.
#
# Reuses tests/bench_session_throughput (geist) + llama-bench. No framework.
#
#   ./benchmark/compare_metal.sh [model.gguf] [cooldown_s]
#
# Env:
#   GEIST_M_MAX   prefill chunk cap (default 128 — the metal sweet spot)
#   SKIP_LLAMA=1  geist-only run (skip the llama-bench half)
#   PP, TG        workload sizes (default 512 / 64, matches llama-bench -p/-n)
#
# ponytail: shell + grep, not a harness. The reference column is hard-coded
# from docs/proposals/metal-beat-llamacpp-plan.md — update it there, not here.
set -euo pipefail

MODEL="${1:-models/gemma-4-E2B-it-Q4_K_M.gguf}"
COOLDOWN="${2:-240}"
M_MAX="${GEIST_M_MAX:-128}"
PP="${PP:-512}"
TG="${TG:-64}"
BIN="bin/mac-omp/release/tests/bench_session_throughput"

# Fixed reference targets (cool, M1 Max, gemma-4-E2B Q4_K_M). Two bars:
#   wip   = the GPU-first engine before the main-contract re-port
#   llama = llama.cpp Metal (BLAS,MTL), its stable cool best
REF_WIP_PP=1072; REF_WIP_TG=61
REF_LLAMA_PP=1548; REF_LLAMA_TG=93

[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
[ -x "$BIN" ] || { echo "build first: make BACKENDS=metal (missing $BIN)" >&2; exit 1; }

num() { grep -oE '[0-9]+\.[0-9]+|[0-9]+' | head -1; }

echo "cooldown ${COOLDOWN}s (let the GPU settle to cool clocks)…" >&2
sleep "$COOLDOWN"

echo "geist metal pp${PP}/tg${TG} (GEIST_M_MAX=${M_MAX})…" >&2
OUT=$(GEIST_GGUF_PATH="$MODEL" GEIST_BENCH_BACKEND=metal GEIST_M_MAX="$M_MAX" \
      GEIST_BENCH_PP="$PP" GEIST_BENCH_TG="$TG" "$BIN" 2>&1)
G_PP=$(echo "$OUT" | grep "prefill (" | grep -oE '[0-9]+\.[0-9] tok' | num)
G_TG=$(echo "$OUT" | grep "decode  (" | grep -oE '[0-9]+\.[0-9] tok' | num)
G_TOT=$(echo "$OUT" | grep "total   (" | grep -oE '[0-9]+\.[0-9] tok' | num)

L_PP="$REF_LLAMA_PP"; L_TG="$REF_LLAMA_TG"
if [ "${SKIP_LLAMA:-0}" != "1" ] && command -v llama-bench >/dev/null; then
  echo "llama-bench back-to-back…" >&2
  LOUT=$(llama-bench -m "$MODEL" -p "$PP" -n "$TG" 2>/dev/null || true)
  LP=$(echo "$LOUT" | grep "pp${PP}" | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  LT=$(echo "$LOUT" | grep "tg${TG}" | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  [ -n "$LP" ] && L_PP="$LP"; [ -n "$LT" ] && L_TG="$LT"
fi

ratio() { awk -v a="$1" -v b="$2" 'BEGIN{ if(b>0) printf "%.2f", a/b; else printf "-" }'; }

printf '\n'
printf '  metric        geist    wip(ref)   llama(ref)   gap→wip  gap→llama\n'
printf '  ------------  -------  ---------  -----------  -------  ---------\n'
printf '  prefill pp%-3s %7s  %9s  %11s  %6sx  %7sx\n' \
  "$PP" "$G_PP" "$REF_WIP_PP" "$L_PP" "$(ratio "$REF_WIP_PP" "$G_PP")" "$(ratio "$L_PP" "$G_PP")"
printf '  decode  tg%-3s %7s  %9s  %11s  %6sx  %7sx\n' \
  "$TG" "$G_TG" "$REF_WIP_TG" "$L_TG" "$(ratio "$REF_WIP_TG" "$G_TG")" "$(ratio "$L_TG" "$G_TG")"
printf '  total(pp+tg)  %7s  %9s  %11s\n' "$G_TOT" "-" "-"
printf '\n  (gap = target/geist; 1.00 = parity. wip = pre-re-port GPU-first engine.)\n\n'
