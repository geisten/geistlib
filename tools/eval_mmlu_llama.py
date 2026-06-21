#!/usr/bin/env python3
"""eval_mmlu_llama.py — same MMLU cloze as eval_mmlu.py, scored by llama.cpp.

Reuses eval_mmlu's dataset loader + prompt builder so geist and llama.cpp see
the *identical* questions (same --shuffle seed), 5-shot exemplars and prompt
text. Scores via a running `llama-server`: ask for one token with n_probs and
read the next-token logprob of " A"/" B"/" C"/" D" from completion_probabilities.

  llama-server -m model.gguf -c 4096 --port 8080   # start first, same GGUF
  python3 tools/eval_mmlu_llama.py --hf --shuffle --limit 500

This isolates the runtime: same weights, same method, only the kernels differ.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict

import eval_mmlu as E  # build_prompt, load_dataset, pick_shots, LETTERS


def score(url: str, prompt: str) -> list[float]:
    body = json.dumps({"prompt": prompt, "n_predict": 1, "n_probs": 60,
                       "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    # completion_probabilities[0].probs: [{tok_str/token, prob}, ...] top n_probs
    probs = r["completion_probabilities"][0]
    items = probs.get("probs") or probs.get("top_logprobs") or []
    table = {}
    for it in items:
        tok = it.get("tok_str", it.get("token", "")).strip()
        lp = it.get("logprob", it.get("prob"))
        # " C" and "C" both strip to "C"; keep the higher (the spaced variant
        # is the real post-"Answer:" continuation, the bare one is far lower).
        if tok not in table or lp > table[tok]:
            table[tok] = lp
    # ponytail: missing letter -> -inf (logprobs are negative; 0.0 would win).
    # With the "Answer:" prompt all 4 letters sit in the top-n_probs, so this
    # only bites pathological cases. Raise n_probs if a real letter is missed.
    return [table.get(L, -1e9) for L in E.LETTERS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--jsonl")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows, shots_by_subject = E.load_dataset(args)
    if args.shuffle:
        import random
        random.Random(1234).shuffle(rows)  # SAME seed as eval_mmlu.py
    if args.limit > 0:
        rows = rows[:args.limit]

    n_correct = 0
    per_subj = defaultdict(lambda: [0, 0])
    for i, (subject, question, choices, gold) in enumerate(rows):
        shots = E.pick_shots(shots_by_subject, subject, args.shots)
        prompt = E.build_prompt(subject, question, choices, shots)
        ps = score(args.url, prompt)
        pred = max(range(4), key=lambda k: ps[k])
        ok = (pred == gold)
        n_correct += ok
        per_subj[subject][0] += ok
        per_subj[subject][1] += 1
        if args.verbose:
            print(f"[{i+1}/{len(rows)}] {subject}: pred={E.LETTERS[pred]} "
                  f"gold={E.LETTERS[gold]} {'✓' if ok else '✗'}")
        elif (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(rows)}  running acc={n_correct/(i+1):.3f}",
                  file=sys.stderr)

    acc = n_correct / len(rows)
    print(f"\nllama.cpp MMLU accuracy: {acc:.4f}  ({n_correct}/{len(rows)})")
    if len(per_subj) > 1:
        print("per-subject:")
        for subj in sorted(per_subj):
            c, t = per_subj[subj]
            print(f"  {subj:40s} {c/t:.3f}  ({c}/{t})")
    print("(random-chance baseline = 0.25)")


if __name__ == "__main__":
    main()
