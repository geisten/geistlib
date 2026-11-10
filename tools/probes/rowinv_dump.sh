#!/bin/sh
# rowinv_dump.sh <bin dir> <backend> <out dir> <d> <inter> <model.gguf> — layer 0's
# expert rows (x, act(g), y per token and expert) at 64- and 128-row chunks,
# compared: which stage first differs for an identical input row.
set -u
B=$1 BE=$2 O=$3 D=$4 I=$5 M=$6
mkdir -p "$O"
head -c 4000 docs/OFFLOAD.md > "$O/prompt.txt"
for c in 64 128; do
    rm -f "$O/dump_$c.bin"
    GEIST_DUMP_BACKEND=$BE GEIST_M_MAX=$c GEIST_ROWINV_DUMP="$O/dump_$c.bin" \
        "$B/tools/dump_geist_logits" "$M" "$O/prompt.txt" "$O/ids.bin" "$O/final_$c.bin" 2>&1 | tail -1
done
python3 - "$O" "$D" "$I" <<'PY'
import struct, sys, collections
o, d, I = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rec = 12 + 4 * (d + I + d)
def load(p):
    b = open(p, 'rb').read(); r = {}
    for off in range(0, len(b), rec):
        pos, e, m = struct.unpack_from('<III', b, off)
        r[(pos, e)] = (m, b[off+12:off+12+4*d], b[off+12+4*d:off+12+4*(d+I)], b[off+12+4*(d+I):off+rec])
    return r
a, c = load(f'{o}/dump_64.bin'), load(f'{o}/dump_128.bin')
nx = ng = ny = 0; ex = []
cnt = collections.Counter(); tot = collections.Counter()
for k in a:
    if k not in c: continue
    (ma, xa, ga, ya), (mc, xc, gc, yc) = a[k], c[k]
    if xa != xc: nx += 1; continue
    key = (ma % 4 == 0, mc % 4 == 0, ma >= 16, mc >= 16)
    tot[key] += 1
    if ga != gc:
        ng += 1; cnt[key] += 1
        if len(ex) < 10: ex.append(('g', k, ma, mc))
    elif ya != yc:
        ny += 1; cnt[key] += 1
        if len(ex) < 10: ex.append(('y', k, ma, mc))
print(f"records {len(a)} / {len(c)}; x differ {nx}; act(gate,up) differ {ng}; down differ {ny}")
for e in ex: print(" ", e)
for k in sorted(tot): print(" (m64%4==0, m128%4==0, m64>=16, m128>=16)", k, cnt[k], "/", tot[k])
PY
rm -f "$O"/dump_*.bin
