"""GEIST_MOE_TRACE records -> USAGE lines (the format of moe_usage.cpp, which
analyze.py and GEIST_MOE_PROFILE read).

Usage: trace_usage.py <model> <set> <n_experts> <k> <trace>...
A trace record (forward/moe.c): int32 layer, int32 position, k uint32 expert
ids, float32 margin. One file per corpus file; its name stands for the file."""
import os, struct, sys
from collections import defaultdict

model, sset, n_exp, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rec = struct.Struct(f"<ii{k}If")
for path in sys.argv[5:]:
    data = open(path, "rb").read()
    assert len(data) % rec.size == 0, f"{path}: not a whole number of {rec.size}-byte records"
    counts = defaultdict(lambda: [0] * n_exp)
    tokens = defaultdict(int)
    for layer, _pos, *rest in rec.iter_unpack(data):
        for e in rest[:k]:
            counts[layer][e] += 1
        tokens[layer] += 1
    name = os.path.basename(path).removesuffix(".trace")
    for layer in sorted(counts):
        print(f"USAGE {model} {sset} {name} {tokens[0]} {layer} {','.join(map(str, counts[layer]))}")
