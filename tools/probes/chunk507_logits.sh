#!/bin/sh
# chunk507_logits.sh <bin dir> <backend> <out dir> <model.gguf>... — #507's
# check: the same binary prefilling in 64-row chunks (GEIST_M_MAX=64) and in
# its default (128 on a CPU backend for an MoE model).
#   final: logits after a ~900-token prompt (first 4000 bytes of docs/OFFLOAD.md)
#   tf:    128 teacher-forced positions after a 640-token prefill (they read
#          the KV cache the chunked prefill wrote): max |d|, rel L2, top-1
#   greedy: 32 greedy tokens after the same 640-token prefill
set -u
B=$1 BE=$2 O=$3; shift 3
mkdir -p "$O"
head -c 4000 docs/OFFLOAD.md > "$O/prompt.txt"
export GEIST_DUMP_BACKEND=$BE
for m in "$@"; do
    n=$(basename "$m" .gguf)
    for c in 64 128; do
        if [ $c = 64 ]; then export GEIST_M_MAX=64; else unset GEIST_M_MAX; fi
        "$B/tools/dump_geist_logits" "$m" "$O/prompt.txt" "$O/ids_$n.bin" "$O/final_${n}_$c.bin" 2>"$O/log_${n}_$c.txt"
        python3 - "$O/ids_$n.bin" "$O/ids768_$n.bin" "$O/ids640_$n.bin" <<'PY'
import sys
d = open(sys.argv[1], 'rb').read()
assert len(d) >= 768 * 4, f"prompt has only {len(d)//4} ids"
open(sys.argv[2], 'wb').write(d[:768 * 4]); open(sys.argv[3], 'wb').write(d[:640 * 4])
PY
        "$B/tools/dump_geist_tf_logits" "$m" "$O/ids768_$n.bin" 640 "$O/tf_${n}_$c.bin" 2>>"$O/log_${n}_$c.txt"
        "$B/tools/dump_geist_greedy" "$m" "$O/ids640_$n.bin" 32 > "$O/greedy_${n}_$c.txt" 2>>"$O/log_${n}_$c.txt"
    done
    unset GEIST_M_MAX
    python3 - "$O" "$n" "$BE" <<'PY' | tee -a "$O/result.md"
import sys, struct, math
o, n, be = sys.argv[1:4]
def rows(p, nv=None):
    d = open(p, 'rb').read(); x = struct.unpack(f'{len(d)//4}f', d)
    return x
f64, f128 = rows(f'{o}/final_{n}_64.bin'), rows(f'{o}/final_{n}_128.bin')
nv = len(f64)
def cmp(a, b):
    num = sum((p - q) ** 2 for p, q in zip(a, b)); den = sum(q * q for q in b)
    return max(abs(p - q) for p, q in zip(a, b)), math.sqrt(num / den), max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
fm, fr, ft = cmp(f128, f64)
t64, t128 = rows(f'{o}/tf_{n}_64.bin'), rows(f'{o}/tf_{n}_128.bin')
k = len(t64) // nv
mx, rl, agree = 0.0, 0.0, 0
for i in range(k):
    m_, r_, a_ = cmp(t128[i*nv:(i+1)*nv], t64[i*nv:(i+1)*nv]); mx = max(mx, m_); rl = max(rl, r_); agree += a_
g64 = [l.split()[1] for l in open(f'{o}/greedy_{n}_64.txt') if l.strip()[:1].isdigit()]
g128 = [l.split()[1] for l in open(f'{o}/greedy_{n}_128.txt') if l.strip()[:1].isdigit()]
same = sum(a == b for a, b in zip(g64, g128))
first = next((i for i, (a, b) in enumerate(zip(g64, g128)) if a != b), None)
bits = open(f'{o}/final_{n}_64.bin','rb').read() == open(f'{o}/final_{n}_128.bin','rb').read()
print(f"| {n} | {be} | {'identical' if bits else f'{fm:.3g} / {fr:.3g}'} | {'yes' if ft else 'NO'} | {mx:.3g} / {rl:.3g} | {agree}/{k} | {same}/{len(g64)}" + ("" if first is None else f" (first differs at step {first})") + " |")
PY
done
rm -f "$O"/*.bin # large (vocab x 128 rows); the table above is the result
