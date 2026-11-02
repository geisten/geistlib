#!/usr/bin/env python3
"""Perplexity of one GGUF in geist and in llama.cpp on the same tokens.

    tools/ppl_vs_llama.py MODEL.gguf TEXT --llama-bin DIR [--ctx 512]
                          [--chunks 8] [--eval-geist PATH] [--ngl 0]

llama-perplexity -c CTX scores, per chunk of CTX tokens of the whole text
(its first token replaced by BOS when the vocabulary adds one), the
predictions of positions CTX/2 .. CTX-2. This script runs llama-perplexity
for its number and repeats exactly that on geist: the token ids come from
llama-tokenize (so a tokenizer difference cannot hide in the comparison;
geist's own TOK ids are compared with them and the first difference
reported), each chunk is a fresh session (eval_geist SCORE: the first
CTX/2 + 1 tokens prefilled, the rest fed one at a time).

Prints one JSON line: both perplexities, the relative difference, the token
count and whether geist's tokenizer agrees. GEIST_BACKEND picks geist's
backend; GEIST_KV_* its KV cache (llama.cpp's default is F16, as most MoE
families' here).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_repl import EvalRepl  # noqa: E402


def llama_tokens(bin_dir: Path, model: Path, text: Path) -> list[int]:
    out = subprocess.run([str(bin_dir / "llama-tokenize"), "-m", str(model), "-f", str(text),
                          "--ids", "--log-disable"], check=True, capture_output=True, text=True)
    m = re.search(r"\[([0-9,\s]*)\]", out.stdout)
    if m is None:
        sys.exit("llama-tokenize printed no id list")
    return [int(x) for x in m.group(1).replace(",", " ").split()]


def llama_ppl(bin_dir: Path, model: Path, text: Path, ctx: int, chunks: int, ngl: int) -> float:
    out = subprocess.run([str(bin_dir / "llama-perplexity"), "-m", str(model), "-f", str(text),
                          "-c", str(ctx), "--chunks", str(chunks), "-ngl", str(ngl)],
                         capture_output=True, text=True)
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out.stdout + out.stderr)
    if m is None:
        sys.exit("llama-perplexity gave no estimate:\n" + (out.stdout + out.stderr)[-2000:])
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("text", type=Path)
    ap.add_argument("--llama-bin", type=Path, required=True)
    ap.add_argument("--eval-geist", type=Path,
                    default=Path(__file__).resolve().parent.parent / "bin" / "mac-omp" / "release" / "tools" / "eval_geist")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--ngl", type=int, default=0, help="llama.cpp GPU layers")
    ap.add_argument("--skip-llama", type=float, default=None,
                    help="use this llama.cpp perplexity instead of running it")
    a = ap.parse_args()

    ids = llama_tokens(a.llama_bin, a.model, a.text)
    ref = a.skip_llama if a.skip_llama is not None else llama_ppl(
        a.llama_bin, a.model, a.text, a.ctx, a.chunks, a.ngl)

    ev = EvalRepl(a.eval_geist, a.model)
    try:
        # llama-tokenize added BOS when the vocabulary asks for one; geist's
        # TOK never does.
        body = ids[1:] if ev.bos is not None and ids and ids[0] == ev.bos else ids
        own = [int(x) for x in ev.command("TOK " + a.text.read_text().replace("\\", "\\\\")
                                          .replace("\n", "\\n").replace("\t", "\\t"))[1:]]
        tok_ok = own == body
        first_diff = next((i for i, (x, y) in enumerate(zip(own, body)) if x != y),
                          None if len(own) == len(body) else min(len(own), len(body)))
        n_chunk = min(a.chunks, len(ids) // a.ctx)
        first = a.ctx // 2
        nll, count = 0.0, 0
        for c in range(n_chunk):
            chunk = ids[c * a.ctx:(c + 1) * a.ctx]
            if ev.bos is not None:
                chunk[0] = ev.bos
            prompt, cont = chunk[:first + 1], chunk[first + 1:]
            r = ev.command(f"SCORE {len(prompt)} " + " ".join(map(str, prompt)) +
                           f" {len(cont)} " + " ".join(map(str, cont)))
            lps = [float(x) for x in r[1:]]
            nll -= sum(lps)
            count += len(lps)
            print(f"chunk {c + 1}/{n_chunk}: running ppl {math.exp(nll / count):.4f}", file=sys.stderr)
    finally:
        ev.close()
    ppl = math.exp(nll / count)
    print(json.dumps({"model": a.model.name, "text": a.text.name, "ctx": a.ctx, "chunks": n_chunk,
                      "scored_tokens": count, "geist_ppl": round(ppl, 4), "llama_ppl": ref,
                      "rel_diff": round(ppl / ref - 1.0, 5), "backend": os.environ.get("GEIST_BACKEND", "auto"),
                      "tokenizer_agrees": tok_ok, "first_token_diff": first_diff}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
