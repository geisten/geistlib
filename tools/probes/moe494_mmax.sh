#!/bin/sh
# moe494_mmax.sh MODEL NEW [ROUNDS [MMAX [MODELS [SEQS]]]] — #494 on the M6:
# MoE prefill of NEW (Metal, bench_perf_sweep) per prefill chunk size
# GEIST_M_MAX (comma list MMAX, default 256,512) against llama-bench
# (pinned build, -ngl 99, default ubatch 512), ROUNDS (default 3)
# alternating rounds, the order rotating. MODELS: comma list of ol, ds, gc,
# go (OLMoE-1B-7B, DeepSeek-V2-Lite, GigaChat3-10B Q4_K_M, gpt-oss-20b
# MXFP4; default all); SEQS default 128,512,2048. MODEL (mac-probe's) is
# unused. Run by .gitea/workflows/mac-probe.yml under the host lock; writes
# to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision}
ROUNDS=${2:-3}
MMAX=${3:-256,512}
MODELS=${4:-ol,ds,gc,go}
SEQS=${5:-128,512,2048}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
case $NEW in origin/*) git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW" ;; esac
WT="$(mktemp -d)/new"
git worktree add --detach "$WT" "$NEW" >"$OUT/worktree.log" 2>&1
(cd "$WT" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build.log" 2>&1 || {
    echo "build of $NEW failed" >"$OUT/summary.md"
    exit 1
}
G="$WT/bin/mac-omp/release/tests/bench_perf_sweep"
LB=$(scripts/ensure-llama-bench.sh metal 2>"$OUT/llama-build.log") || LB=""
fetch() { python3 tools/fetch_bench_model.py --url "$1" --sha256 "$2"; }
path() {
    case "$1" in
    ol) fetch https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/resolve/2ac1d2731792/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
        4ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f ;;
    ds) fetch https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
        70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a ;;
    gc) fetch https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
        88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1 ;;
    go) fetch https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
        27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901 ;;
    esac
}
: >"$OUT/raw.txt"
for key in $(echo "$MODELS" | tr ',' ' '); do
    m=$(path "$key")
    set -- $(echo "$MMAX" | tr ',' ' ') llama
    i=0
    while [ $i -lt "$ROUNDS" ]; do
        for v in "$@"; do
            sleep 10
            if [ "$v" = llama ]; then
                [ -n "$LB" ] || continue
                "$LB" -m "$m" -p "$SEQS" -n 0 -r 3 -ngl 99 -o csv 2>/dev/null |
                    sed "s/^/$key llama r$i /" >>"$OUT/raw.txt"
            else
                GEIST_M_MAX=$v GEIST_BENCH_BACKEND=metal "$G" --gguf "$m" --seq-lens "$SEQS" \
                    --decode-n 1 --warmup 2 --repeats 3 --emit-jsonl 2>/dev/null | grep '^{' |
                    sed "s/^/$key mmax$v r$i /" >>"$OUT/raw.txt"
            fi
        done
        first=$1
        shift
        set -- "$@" "$first"
        i=$((i + 1))
    done
done
LC_ALL=C sysctl -n vm.loadavg >>"$OUT/env.txt"
{
    echo "## #494 MoE prefill per chunk size on the M6: $NEW ($(git -C "$WT" rev-parse HEAD))"
    echo
    echo '```'
    cat "$OUT/env.txt"
    echo '```'
    echo
    python3 - "$OUT/raw.txt" <<'EOF'
import csv, io, json, sys, statistics as st
from collections import defaultdict
d = defaultdict(list)
hdr = None
for line in open(sys.argv[1]):
    key, var, rnd, rest = line.split(" ", 3)
    if var == "llama":
        r = next(csv.reader(io.StringIO(rest)))
        if "n_prompt" in r:
            hdr = r
        elif hdr is not None:
            rec = dict(zip(hdr, r))
            if int(rec.get("n_gen", 0)) == 0 and int(rec.get("n_prompt", 0)) > 0:
                d[(key, "llama", int(rec["n_prompt"]))].append(float(rec["avg_ts"]))
        continue
    try:
        j = json.loads(rest)
    except ValueError:
        continue
    if j.get("prefill_ms") and "seq_len" in j:
        d[(key, var, int(j["seq_len"]))].append(j["seq_len"] / j["prefill_ms"] * 1e3)
vars_ = sorted({k[1] for k in d if len(k) == 3}, key=lambda v: (v == "llama", v))
seqs = sorted({k[2] for k in d if len(k) == 3})
keys = sorted({k[0] for k in d if len(k) == 3})
print("prefill tok/s, median of rounds (ratio to llama.cpp)")
print()
print("| model | tokens | " + " | ".join(vars_) + " |")
print("|---|--:|" + "--:|" * len(vars_))
for k in keys:
    for s in seqs:
        ll = d.get((k, "llama", s))
        lm = st.median(ll) if ll else None
        cells = []
        for v in vars_:
            xs = d.get((k, v, s))
            if not xs:
                cells.append("-")
                continue
            md = st.median(xs)
            c = f"{md:.0f}"
            if lm and v != "llama":
                c += f" ({md / lm:.2f}x)"
            cells.append(c)
        print(f"| {k} | {s} | " + " | ".join(cells) + " |")
EOF
} >"$OUT/summary.md"
git worktree remove --force "$WT" >/dev/null 2>&1 || true
