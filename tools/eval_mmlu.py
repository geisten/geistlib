#!/usr/bin/env python3
"""eval_mmlu.py — MMLU (and MMLU-style) accuracy for a geist GGUF.

Self-contained: drives the `eval_geist` REPL and tokenizes with the model's
OWN GGUF tokenizer (the `TOK` command), so there is no external HF-tokenizer
dependency and no risk of a tokenizer mismatch between scoring and the model.

Method (the standard log-likelihood cloze used by lm-eval-harness): build the
prompt

    The following are multiple choice questions (with answers) about {subject}.

    {question}
    A. {a}
    B. {b}
    C. {c}
    D. {d}
    Answer:

then score the next-token log-probability of " A" / " B" / " C" / " D" (each a
single token for the Gemma vocab) in ONE prefill via SCOREALT, and pick the
argmax. Accuracy = fraction matching the gold letter. This is a base-completion
eval (no chat template), which is what MMLU is conventionally run as, so it
sidesteps the chat-template parity question entirely.

Data:
  --hf            load `cais/mmlu` (config `all`, split `test`) via the
                  `datasets` library (pip install datasets).
  --jsonl FILE    load a local JSONL with keys: question, choices[4], answer
                  (0-3 or A-D), subject (optional).
  (default)       a tiny embedded sample, enough to smoke-test the harness.

Examples:
  python3 tools/eval_mmlu.py --bin bin/mac-omp/release/tools/eval_geist \
      --gguf gguf_artifacts/gemma4-e2b-Q4_K_M.gguf --hf --limit 200
  python3 tools/eval_mmlu.py --bin .../tools/eval_geist --gguf model.gguf --selftest
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_repl import EvalRepl  # noqa: E402

LETTERS = ["A", "B", "C", "D"]

# Minimal embedded sample (general knowledge) for a no-dataset smoke test.
EMBEDDED = [
    ("geography", "What is the capital of France?",
     ["London", "Berlin", "Paris", "Madrid"], 2),
    ("math", "What is 7 multiplied by 8?", ["54", "56", "48", "64"], 1),
    ("science", "What gas do plants primarily absorb for photosynthesis?",
     ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], 2),
    ("history", "In which year did World War II end?",
     ["1918", "1939", "1945", "1963"], 2),
    ("biology", "Which organ pumps blood through the human body?",
     ["Liver", "Heart", "Lung", "Kidney"], 1),
]

# Few-shot exemplars (disjoint from EMBEDDED) so the model sees the
# "Answer: <letter>" format. MMLU is conventionally a 5-shot eval; without
# shots small models collapse to a position bias (always "A").
EMBEDDED_SHOTS = [
    ("geography", "What is the largest ocean on Earth?",
     ["Atlantic", "Indian", "Arctic", "Pacific"], 3),
    ("math", "What is 12 divided by 4?", ["2", "3", "4", "6"], 1),
    ("science", "What is the chemical symbol for water?",
     ["O2", "H2O", "CO2", "NaCl"], 1),
    ("history", "Who was the first President of the United States?",
     ["Abraham Lincoln", "Thomas Jefferson", "George Washington", "John Adams"], 2),
    ("biology", "How many legs does a spider have?", ["6", "8", "10", "4"], 1),
]


def _format_q(question: str, choices: list[str]) -> str:
    body = question.strip() + "\n"
    for letter, choice in zip(LETTERS, choices):
        body += f"{letter}. {choice}\n"
    return body + "Answer:"


def build_prompt(subject: str, question: str, choices: list[str],
                 shots: list[tuple]) -> str:
    head = (f"The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n")
    pre = ""
    for (_subj, sq, sc, sgold) in shots:
        pre += _format_q(sq, sc) + f" {LETTERS[sgold]}\n\n"
    return head + pre + _format_q(question, choices)


def load_dataset(args) -> tuple[list[tuple], dict]:
    """Return (test_rows, shots_by_subject). shots_by_subject maps a subject to
    its few-shot exemplars; the special key '*' is the fallback pool."""
    if args.selftest or (not args.hf and not args.jsonl):
        return list(EMBEDDED), {"*": list(EMBEDDED_SHOTS)}
    if args.jsonl:
        import json
        rows = []
        for line in Path(args.jsonl).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ans = r["answer"]
            gold = ans if isinstance(ans, int) else LETTERS.index(str(ans).strip().upper())
            rows.append((r.get("subject", "misc"), r["question"], r["choices"], gold))
        return rows, {"*": list(EMBEDDED_SHOTS)}
    # --hf: cais/mmlu provides a 'dev' split with 5 exemplars per subject.
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        sys.exit("--hf needs `pip install datasets`. Use --jsonl or --selftest instead.")
    test = [(r["subject"], r["question"], r["choices"], int(r["answer"]))
            for r in hf_load("cais/mmlu", "all", split="test")]
    shots: dict = defaultdict(list)
    for r in hf_load("cais/mmlu", "all", split="dev"):
        shots[r["subject"]].append(
            (r["subject"], r["question"], r["choices"], int(r["answer"])))
    return test, shots


def pick_shots(shots_by_subject: dict, subject: str, n: int) -> list[tuple]:
    pool = shots_by_subject.get(subject) or shots_by_subject.get("*") or []
    return pool[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="path to eval_geist binary")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf", action="store_true", help="load cais/mmlu via datasets")
    ap.add_argument("--jsonl", help="local JSONL dataset")
    ap.add_argument("--selftest", action="store_true", help="run the embedded sample only")
    ap.add_argument("--limit", type=int, default=0, help="cap number of questions (0 = all)")
    ap.add_argument("--shots", type=int, default=5, help="few-shot exemplars (MMLU default 5)")
    ap.add_argument("--bos", type=int, default=-2,
                    help="BOS token id to prepend. Default -2 asks the model "
                         "(GGUF add_bos_token / bos_token_id via the REPL's BOS "
                         "command): Gemma says yes and gets its <bos>, a model "
                         "whose metadata says no gets none. -1 forces none, any "
                         "other value forces that id. Hardcoding one family's id "
                         "here fed Ternary-Bonsai an unrelated token 2 and cost it "
                         "MMLU points.")
    ap.add_argument("--shuffle", action="store_true",
                    help="deterministically shuffle before --limit (the MMLU test "
                         "split is subject-ordered, so an unshuffled --limit hits one subject)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not Path(args.bin).is_file():
        sys.exit(f"eval_mmlu: binary not found: {args.bin} (run `make` first)")

    rows, shots_by_subject = load_dataset(args)
    if args.shuffle:
        import random
        random.Random(1234).shuffle(rows)  # fixed seed → reproducible sample
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("eval_mmlu: no questions loaded")

    repl = EvalRepl(args.bin, args.gguf)
    try:
        # The four answer-letter tokens (one token each for the Gemma vocab).
        letter_ids = []
        for letter in LETTERS:
            ids = repl.encode(" " + letter)
            if len(ids) != 1:
                sys.exit(f"eval_mmlu: ' {letter}' is not a single token ({ids}); "
                         "this scorer assumes single-token answer letters.")
            letter_ids.append(ids[0])

        bos_id = (-1 if repl.bos is None else repl.bos) if args.bos == -2 else args.bos
        if args.verbose:
            print(f"BOS: {'none' if bos_id < 0 else bos_id}"
                  f"{' (from the model)' if args.bos == -2 else ' (forced)'}")

        n_correct = 0
        per_subj = defaultdict(lambda: [0, 0])  # subject -> [correct, total]
        for i, (subject, question, choices, gold) in enumerate(rows):
            shots = pick_shots(shots_by_subject, subject, args.shots)
            prompt_ids = repl.encode(build_prompt(subject, question, choices, shots))
            if bos_id >= 0:
                prompt_ids = [bos_id] + prompt_ids
            lps = [float(x) for x in repl.command(
                f"SCOREALT {len(prompt_ids)} {' '.join(map(str, prompt_ids))} "
                f"{len(letter_ids)} {' '.join(map(str, letter_ids))}")]
            pred = max(range(4), key=lambda k: lps[k])
            ok = (pred == gold)
            n_correct += ok
            per_subj[subject][0] += ok
            per_subj[subject][1] += 1
            if args.verbose:
                print(f"[{i+1}/{len(rows)}] {subject}: pred={LETTERS[pred]} "
                      f"gold={LETTERS[gold]} {'✓' if ok else '✗'}  lps={[round(x,2) for x in lps]}")
            elif (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(rows)}  running acc={n_correct/(i+1):.3f}",
                      file=sys.stderr)
    finally:
        repl.close()

    acc = n_correct / len(rows)
    print(f"\nMMLU accuracy: {acc:.4f}  ({n_correct}/{len(rows)})")
    if len(per_subj) > 1:
        print("per-subject:")
        for subj in sorted(per_subj):
            c, t = per_subj[subj]
            print(f"  {subj:40s} {c/t:.3f}  ({c}/{t})")
    # Random-chance baseline is 0.25; a working model should clear it comfortably.
    print(f"(random-chance baseline = 0.25)")


if __name__ == "__main__":
    main()
