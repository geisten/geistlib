#!/usr/bin/env python3
"""eval_runner.py — drive the `eval_geist` REPL for evaluation/scoring.

`eval_geist` loads a GGUF once and answers line-oriented commands on stdin
(DECODE / SCORE / SCOREALT / RESET / QUIT — see the header of eval_geist.c).
This wrapper speaks that protocol so you can:

  * generate  — greedy-decode a continuation from a text prompt;
  * mc        — multiple-choice: score each option's continuation logprob and
                pick the argmax (the MMLU/HellaSwag-style harness).

Tokenization uses a Hugging Face tokenizer for parity with reference
implementations (`pip install transformers`). For protocol smoke-testing
without a tokenizer, pass raw integer token IDs with --raw-ids.

Examples:
  # Greedy generate 32 tokens (HF tokenizer for the model's vocab):
  python3 tools/eval_runner.py --bin bin/mac-omp/release/tools/eval_geist \\
      --gguf model.gguf --tokenizer google/gemma-4-E2B-it \\
      generate --prompt "The capital of France is" --n 32

  # Protocol smoke test, no tokenizer:
  python3 tools/eval_runner.py --bin bin/.../tools/eval_geist --gguf model.gguf \\
      generate --raw-ids "2 651 1234" --n 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_repl import EvalRepl  # noqa: E402


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("eval_runner: --tokenizer requires `pip install transformers`. "
                 "Use --raw-ids to smoke-test the protocol without it.")
    return AutoTokenizer.from_pretrained(name)


def parse_ids(args, tok, text: str) -> list[int]:
    if args.raw_ids is not None:
        return [int(x) for x in args.raw_ids.split()]
    if tok is None:
        sys.exit("provide --tokenizer or --raw-ids")
    return tok.encode(text, add_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", required=True, help="path to eval_geist binary")
    p.add_argument("--gguf", required=True)
    p.add_argument("--awq", default=None)
    p.add_argument("--tokenizer", default=None, help="HF tokenizer name/path")
    sub = p.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("generate", help="greedy-decode a continuation")
    g.add_argument("--prompt", default="")
    g.add_argument("--raw-ids", default=None, help="space-separated token IDs")
    g.add_argument("--n", type=int, default=32)

    m = sub.add_parser("mc", help="multiple-choice: argmax option by logprob")
    m.add_argument("--prompt", required=True)
    m.add_argument("--options", required=True, nargs="+",
                   help="candidate continuations (text)")

    args = p.parse_args()
    if not Path(args.bin).is_file():
        sys.exit(f"eval_runner: binary not found: {args.bin} (run `make` first)")

    tok = load_tokenizer(args.tokenizer) if args.tokenizer else None
    repl = EvalRepl(args.bin, args.gguf, args=["--awq", args.awq] if args.awq else [])
    try:
        if args.mode == "generate":
            ids = parse_ids(args, tok, args.prompt)
            # DECODE -> OK <n> <id...>
            out = [int(x) for x in repl.command(
                f"DECODE {len(ids)} {' '.join(map(str, ids))} {args.n}")[1:]]
            text = tok.decode(out) if tok else None
            print(f"prompt_ids={ids}")
            print(f"output_ids={out}")
            if text is not None:
                print(f"output_text={text!r}")
        else:  # mc
            prompt_ids = tok.encode(args.prompt, add_special_tokens=True)
            best_i, best_lp = -1, float("-inf")
            for i, opt in enumerate(args.options):
                cont = tok.encode(opt, add_special_tokens=False)
                repl.command("RESET")
                lp = float(repl.command(  # SCORE -> OK <total continuation logprob> ...
                    f"SCORE {len(prompt_ids)} {' '.join(map(str, prompt_ids))} "
                    f"{len(cont)} {' '.join(map(str, cont))}")[0])
                print(f"  [{i}] logprob={lp:.3f}  {opt!r}")
                if lp > best_lp:
                    best_i, best_lp = i, lp
            print(f"answer={best_i}  {args.options[best_i]!r}")
    finally:
        repl.close()


if __name__ == "__main__":
    main()
