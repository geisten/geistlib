#!/usr/bin/env python3
"""logits_diff.py A B VOCAB — how far two GEIST_NLL_DUMP files are apart.

Both hold the scored logits vectors of tests/bench_nll as float32, VOCAB
values each. Prints one JSON line: the vectors, the largest absolute
difference, the largest magnitude of a logit, their ratio, the vectors
whose argmax differs, and the largest difference among each vector's top
logit.
"""
import array
import json
import sys


def main():
    a_path, b_path, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
    a, b = array.array("f"), array.array("f")
    with open(a_path, "rb") as f:
        a.frombytes(f.read())
    with open(b_path, "rb") as f:
        b.frombytes(f.read())
    if len(a) != len(b) or len(a) % vocab != 0:
        print(json.dumps({"error": f"sizes {len(a)} and {len(b)}, vocab {vocab}"}))
        return 1
    worst = top = largest = 0.0
    flips = 0
    for row in range(len(a) // vocab):
        va, vb = a[row * vocab:(row + 1) * vocab], b[row * vocab:(row + 1) * vocab]
        worst = max(worst, max(abs(x - y) for x, y in zip(va, vb)))
        largest = max(largest, max(abs(x) for x in va))
        ia = max(range(vocab), key=va.__getitem__)
        ib = max(range(vocab), key=vb.__getitem__)
        flips += ia != ib
        top = max(top, abs(va[ia] - vb[ia]))
    print(json.dumps({"vectors": len(a) // vocab, "max_abs_diff": worst, "max_abs_logit": largest,
                      "relative": worst / largest if largest else 0.0, "argmax_differs": flips,
                      "max_diff_at_top": top}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
