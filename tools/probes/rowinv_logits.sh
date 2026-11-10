#!/bin/sh
# rowinv_logits.sh <bin dir> <backend> <out dir> <model.gguf>... — the same
# binary prefilling in 64-row and in 128-row chunks (GEIST_M_MAX):
#   final: logits after the first 4000 bytes of docs/OFFLOAD.md
#   tf:    128 teacher-forced positions after a 640-token prefill
# Prints whether each pair is bit-identical, else max |d| and rel L2.
set -u
B=$1 BE=$2 O=$3; shift 3
mkdir -p "$O"
head -c 4000 docs/OFFLOAD.md > "$O/prompt.txt"
export GEIST_DUMP_BACKEND=$BE
for m in "$@"; do
    n=$(basename "$m" .gguf)
    for c in 64 128; do
        GEIST_M_MAX=$c "$B/tools/dump_geist_logits" "$m" "$O/prompt.txt" "$O/ids_$n.bin" "$O/final_${n}_$c.bin" 2>"$O/log_${n}_$c.txt"
        python3 -c "import sys; d=open(sys.argv[1],'rb').read(); open(sys.argv[2],'wb').write(d[:768*4])" "$O/ids_$n.bin" "$O/ids768_$n.bin"
        GEIST_M_MAX=$c "$B/tools/dump_geist_tf_logits" "$m" "$O/ids768_$n.bin" 640 "$O/tf_${n}_$c.bin" 2>>"$O/log_${n}_$c.txt"
    done
    python3 - "$O" "$n" "$BE" <<'PY' | tee -a "$O/result.md"
import sys, struct, math
o, n, be = sys.argv[1:4]
def load(p):
    d = open(p, 'rb').read(); return d, struct.unpack(f'{len(d)//4}f', d)
def cmp(a, b):
    num = sum((p - q) ** 2 for p, q in zip(a, b)); den = sum(q * q for q in b) or 1.0
    return max(abs(p - q) for p, q in zip(a, b)), math.sqrt(num / den)
out = [n, be]
for k in ("final", "tf"):
    (d64, a), (d128, b) = load(f'{o}/{k}_{n}_64.bin'), load(f'{o}/{k}_{n}_128.bin')
    if len(d64) == 0 or len(d64) != len(d128):
        out.append(f"{k}: missing ({len(d64)} / {len(d128)} bytes)")
    elif d64 == d128:
        out.append(f"{k}: bit-identical ({len(a)} floats)")
    else:
        mx, rl = cmp(a, b); out.append(f"{k}: max|d| {mx:.3g}, rel L2 {rl:.3g}")
print("| " + " | ".join(out) + " |")
PY
    rm -f "$O"/final_*.bin "$O"/tf_*.bin
done
