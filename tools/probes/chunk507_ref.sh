#!/bin/sh
# chunk507_ref.sh <bin dir> <out dir> <model.gguf>... — #507: cpu_x86 at 32-,
# 64- and 128-row prefill chunks, each against cpu_scalar (the exact-dequant
# reference) on the same tokens: final logits after a ~900-token prompt and
# 128 teacher-forced positions after a 640-token prefill. The question: is
# the 128-row chunk further from the reference than the 64-row one?
set -u
B=$1 O=$2; shift 2
mkdir -p "$O"
head -c 4000 docs/OFFLOAD.md > "$O/prompt.txt"
for m in "$@"; do
    n=$(basename "$m" .gguf)
    for v in ref:cpu_scalar:0 c32:cpu_x86:32 c64:cpu_x86:64 c128:cpu_x86:128; do
        lab=${v%%:*}; rest=${v#*:}; be=${rest%%:*}; mm=${rest#*:}
        export GEIST_DUMP_BACKEND=$be GEIST_BACKEND=$be
        if [ "$mm" = 0 ]; then unset GEIST_M_MAX; else export GEIST_M_MAX=$mm; fi
        "$B/tools/dump_geist_logits" "$m" "$O/prompt.txt" "$O/ids_$n.bin" "$O/final_${n}_$lab.bin" 2>>"$O/log_$n.txt"
        python3 -c "import sys; d=open(sys.argv[1],'rb').read(); open(sys.argv[2],'wb').write(d[:768*4])" "$O/ids_$n.bin" "$O/ids768_$n.bin"
        "$B/tools/dump_geist_tf_logits" "$m" "$O/ids768_$n.bin" 640 "$O/tf_${n}_$lab.bin" 2>>"$O/log_$n.txt"
    done
    unset GEIST_M_MAX
    python3 - "$O" "$n" <<'PY' | tee -a "$O/result.md"
import sys, struct, math
o, n = sys.argv[1:3]
def rows(p):
    d = open(p, 'rb').read(); return struct.unpack(f'{len(d)//4}f', d)
ref_f = rows(f'{o}/final_{n}_ref.bin'); nv = len(ref_f); ref_t = rows(f'{o}/tf_{n}_ref.bin'); k = len(ref_t) // nv
am = lambda a: max(range(len(a)), key=a.__getitem__)
def rel(a, b): return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)) / sum(q * q for q in b))
def js(a, b):
    ma, mb = max(a), max(b); ea = [math.exp(x - ma) for x in a]; eb = [math.exp(x - mb) for x in b]
    sa, sb = sum(ea), sum(eb); s = 0.0
    for x, y in zip(ea, eb):
        p, q = x / sa, y / sb; mm = (p + q) / 2
        if p > 0: s += 0.5 * p * math.log(p / mm)
        if q > 0: s += 0.5 * q * math.log(q / mm)
    return s
for lab in ('c32', 'c64', 'c128'):
    f = rows(f'{o}/final_{n}_{lab}.bin'); t = rows(f'{o}/tf_{n}_{lab}.bin')
    rels, jss, top = [], [], 0
    for i in range(k):
        a, b = t[i*nv:(i+1)*nv], ref_t[i*nv:(i+1)*nv]
        rels.append(rel(a, b)); jss.append(js(a, b)); top += am(a) == am(b)
    rels.sort()
    print(f"| {n} | {lab} | {rel(f, ref_f):.4f} | {sum(rels)/k:.4f} / {rels[k//2]:.4f} / {rels[-1]:.4f} | {sum(jss)/k:.5f} | {top}/{k} |")
PY
    rm -f "$O"/*_"$n"_*.bin
done
