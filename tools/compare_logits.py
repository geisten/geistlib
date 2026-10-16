#!/usr/bin/env python3
"""compare_logits.py A.bin B.bin [--vocab N] — two next-token logit vectors
(float32, as tools/dump_geist_logits and benchmark/dump_llamacpp_logits.c
write them) compared the way benchmark/results/QWEN35_LOGIT_PARITY.md does:
top-1, top-5/top-10 overlap, centered cosine and RMSE (a constant offset
does not change the probabilities), and the Jensen-Shannon divergence of the
full-vocabulary softmax in nats. Prints one JSON line."""
import argparse
import json
import math
import struct


def load(path):
    data = open(path, "rb").read()
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    if len(a) != len(b) or not a:
        raise SystemExit(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    rank = lambda x: sorted(range(n), key=lambda i: -x[i])
    ra, rb = rank(a), rank(b)
    ma, mb = sum(a) / n, sum(b) / n
    ca, cb = [v - ma for v in a], [v - mb for v in b]
    dot = sum(x * y for x, y in zip(ca, cb))
    cos = dot / math.sqrt(sum(x * x for x in ca) * sum(y * y for y in cb))
    rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(ca, cb)) / n)
    pa, pb = softmax(a), softmax(b)
    mid = [(x + y) / 2 for x, y in zip(pa, pb)]
    kl = lambda p: sum(x * math.log(x / m) for x, m in zip(p, mid) if x > 0)
    print(json.dumps({
        "top1_a": ra[0], "top1_b": rb[0], "top1_same": ra[0] == rb[0],
        "top5_overlap": len(set(ra[:5]) & set(rb[:5])),
        "top10_overlap": len(set(ra[:10]) & set(rb[:10])),
        "centered_cosine": round(cos, 6), "centered_rmse": round(rmse, 6),
        "js_nats": round((kl(pa) + kl(pb)) / 2, 6),
        "p_top1_a": round(pa[ra[0]], 4), "p_top1_b": round(pb[rb[0]], 4),
    }))


if __name__ == "__main__":
    main()
