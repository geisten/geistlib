#!/bin/sh
# parity374b.sh <base bin dir> <new bin dir> <out dir> <model.gguf>... —
# #374 bench scratch (not for main): logits of a kernel change on real
# models. cpu_x86 of the base and the new revision, and cpu_scalar (the
# exact-dequant reference), on the same tokens: final logits after a short
# prompt (first $BYTES bytes of docs/OFFLOAD.md; cpu_scalar on x86 gcc turns
# gpt-oss's layer-19 router logits non-finite on a ~1100-token prefill) and
# $TF teacher-forced positions after a $PRE-token prefill. Also new cpu_x86
# at 64-row chunks against 128 (row-count independence).
set -u
BB=$1 NB=$2 O=$3; shift 3
BYTES=${BYTES:-2000} PRE=${PRE:-384} TF=${TF:-128}
mkdir -p "$O"
head -c "$BYTES" docs/OFFLOAD.md > "$O/prompt.txt"
for m in "$@"; do
    n=$(basename "$m" .gguf)
    for v in ref:$NB:cpu_scalar:0 base:$BB:cpu_x86:0 new:$NB:cpu_x86:0 new64:$NB:cpu_x86:64; do
        lab=${v%%:*}; rest=${v#*:}; bin=${rest%%:*}; rest=${rest#*:}; be=${rest%%:*}; mm=${rest#*:}
        export GEIST_DUMP_BACKEND=$be GEIST_BACKEND=$be
        if [ "$mm" = 0 ]; then unset GEIST_M_MAX; else export GEIST_M_MAX=$mm; fi
        echo "-- $n $lab ($be, $bin)" >> "$O/log_$n.txt"
        "$bin/tools/dump_geist_logits" "$m" "$O/prompt.txt" "$O/ids_$n.bin" "$O/final_${n}_$lab.bin" 2>>"$O/log_$n.txt"
        python3 -c "import sys; d=open(sys.argv[1],'rb').read(); open(sys.argv[2],'wb').write(d[:($PRE+$TF)*4])" "$O/ids_$n.bin" "$O/idstf_$n.bin"
        "$bin/tools/dump_geist_tf_logits" "$m" "$O/idstf_$n.bin" "$PRE" "$O/tf_${n}_$lab.bin" 2>>"$O/log_$n.txt"
    done
    unset GEIST_M_MAX
    python3 - "$O" "$n" <<'PY' | tee -a "$O/result.md"
import sys, struct, math, os
o, n = sys.argv[1:3]
def rows(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0: return None
    d = open(p, 'rb').read(); return struct.unpack(f'{len(d)//4}f', d)
am = lambda a: max(range(len(a)), key=a.__getitem__)
def rel(a, b): return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)) / sum(q * q for q in b))
def mx(a, b): return max(abs(p - q) for p, q in zip(a, b))
def js(a, b):
    ma, mb = max(a), max(b); ea = [math.exp(x - ma) for x in a]; eb = [math.exp(x - mb) for x in b]
    sa, sb = sum(ea), sum(eb); s = 0.0
    for x, y in zip(ea, eb):
        p, q = x / sa, y / sb; mm = (p + q) / 2
        if p > 0: s += 0.5 * p * math.log(p / mm)
        if q > 0: s += 0.5 * q * math.log(q / mm)
    return s
F = {l: rows(f'{o}/final_{n}_{l}.bin') for l in ('ref', 'base', 'new', 'new64')}
T = {l: rows(f'{o}/tf_{n}_{l}.bin') for l in ('ref', 'base', 'new', 'new64')}
print(f"### {n}\n\n| pair | final max|d| / rel L2 | final top-1 same | tf rel L2 mean / worst | tf mean JS | tf top-1 agree | bit-identical |\n|:--|:--|:--|:--|:--|:--|:--|")
for a, b in (('base', 'ref'), ('new', 'ref'), ('new', 'base'), ('new64', 'new')):
    if F[a] is None or F[b] is None or T[a] is None or T[b] is None:
        print(f"| {a} vs {b} | missing output | | | | | |"); continue
    nv = len(F[b]); k = len(T[b]) // nv
    rels, jss, top = [], [], 0
    for i in range(k):
        x, y = T[a][i*nv:(i+1)*nv], T[b][i*nv:(i+1)*nv]
        rels.append(rel(x, y)); jss.append(js(x, y)); top += am(x) == am(y)
    same = F[a] == F[b] and T[a] == T[b]
    print(f"| {a} vs {b} | {mx(F[a], F[b]):.4g} / {100*rel(F[a], F[b]):.2f} % | {am(F[a]) == am(F[b])} | {100*sum(rels)/k:.2f} % / {100*max(rels):.2f} % | {sum(jss)/k:.5f} | {top}/{k} | {same} |")
PY
    rm -f "$O"/*_"$n"_*.bin
done
