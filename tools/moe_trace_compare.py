#!/usr/bin/env python3
"""moe_trace_compare.py A.trace B.trace k [--js per_step.txt --n-prompt N]

Two routing traces (GEIST_MOE_TRACE / benchmark/dump_llamacpp_tf.cpp: int32
layer, int32 position, k uint32 expert ids ascending, float margin) compared
per (layer, position): how often the chosen expert set differs, by layer, and
how close A's router was to its k-th choice where it differs. With --js, the
per-step JS of the matching logit rows (tools/js_streams per_step output; row
i predicts the token after position N - 1 + i) is split by whether that
position routed differently anywhere in the stack."""
import argparse
import math
import struct
from collections import defaultdict


def load(path, k):
    rec = struct.Struct(f"<ii{k}If")
    out = {}
    data = open(path, "rb").read()
    for off in range(0, len(data) - rec.size + 1, rec.size):
        v = rec.unpack_from(data, off)
        out[(v[0], v[1])] = (frozenset(v[2:2 + k]), v[2 + k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("k", type=int)
    ap.add_argument("--js")
    ap.add_argument("--n-prompt", type=int, default=0)
    args = ap.parse_args()
    A, B = load(args.a, args.k), load(args.b, args.k)
    keys = sorted(set(A) & set(B))
    by_layer = defaultdict(lambda: [0, 0])
    flip_margins, flip_pos, swaps = [], defaultdict(int), []
    for key in keys:
        sa, ma = A[key]
        sb, _ = B[key]
        by_layer[key[0]][1] += 1
        if sa != sb:
            by_layer[key[0]][0] += 1
            flip_margins.append(ma)
            flip_pos[key[1]] += 1
            swaps.append(len(sa - sb))
    total = sum(n for _, n in by_layer.values())
    flips = sum(f for f, _ in by_layer.values())
    print(f"records compared: {total}, routed differently: {flips} ({100.0 * flips / max(total, 1):.2f} %)")
    if swaps:
        print(f"experts swapped per differing record: mean {sum(swaps) / len(swaps):.2f}, max {max(swaps)}")
    if flip_margins:
        fm = sorted(m for m in flip_margins if not math.isnan(m))
        if fm:
            q = lambda p: fm[min(len(fm) - 1, int(p * len(fm)))]
            print(f"A's margin where they differ: median {q(0.5):.4f}, p90 {q(0.9):.4f}, max {fm[-1]:.4f}")
    allm = sorted(A[k][1] for k in keys if not math.isnan(A[k][1]))
    if allm:
        q = lambda p: allm[min(len(allm) - 1, int(p * len(allm)))]
        print(f"A's margin over all records: p1 {q(0.01):.4f}, p10 {q(0.1):.4f}, median {q(0.5):.4f}")
    print("layer: differing / records")
    print(" ".join(f"{l}:{f}/{n}" for l, (f, n) in sorted(by_layer.items())))
    if args.js:
        rows = [l.split() for l in open(args.js)]
        # How many layers routed differently at the position that produced
        # each logit row, against that row's JS: correlation and bins.
        pairs = [(flip_pos.get(args.n_prompt - 1 + int(st), 0), float(js)) for st, js, _ in rows]
        if len(pairs) > 2:
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cov = sum((x - mx) * (y - my) for x, y in pairs)
            vx = sum((x - mx) ** 2 for x in xs) ** 0.5
            vy = sum((y - my) ** 2 for y in ys) ** 0.5
            rank = lambda v: {i: r for r, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
            rx, ry = rank(xs), rank(ys)
            n = len(pairs)
            rho = 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
            print(f"layers routed differently per position: mean {mx:.1f}; Pearson r with JS {cov / (vx * vy) if vx and vy else 0:.3f}, Spearman rho {rho:.3f}")
            q = sorted(xs)
            edges = [q[len(q) // 4], q[len(q) // 2], q[(3 * len(q)) // 4]]
            for lo, hi, name in ((-1, edges[0], f"<= {edges[0]}"), (edges[0], edges[1], f"{edges[0] + 1}-{edges[1]}"),
                                 (edges[1], edges[2], f"{edges[1] + 1}-{edges[2]}"), (edges[2], 10**9, f"> {edges[2]}")):
                v = sorted(y for x, y in pairs if lo < x <= hi)
                if v:
                    print(f"  {name:>8} differing layers: n={len(v)} JS mean {sum(v) / len(v):.5f} median {v[len(v) // 2]:.5f} p90 {v[(9 * len(v)) // 10]:.5f}")
        with_flip, without = [], []
        for step, js, _ in rows:
            pos = args.n_prompt - 1 + int(step)
            (with_flip if flip_pos.get(pos, 0) else without).append(float(js))
        for name, v in (("positions routed differently", with_flip), ("positions routed alike", without)):
            if v:
                v.sort()
                print(f"JS at {name}: n={len(v)} mean {sum(v) / len(v):.5f} median {v[len(v) // 2]:.5f} max {v[-1]:.5f}")


if __name__ == "__main__":
    main()
