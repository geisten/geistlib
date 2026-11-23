#!/usr/bin/env python3
"""metal_profile_mlx.py — mlx-lm under tools/probes/metal_trace.m (#590).

Runs in the MLX virtual environment of tools/probes/mlx_vs_geist.sh. Loads the
trace library before mlx, then does what mlx_vs_geist_mlx.py's bench does
(stream_generate on a token list, end-of-sequence ignored, a fresh cache per
run, one discarded warmup) and stamps every run's start, first token and last
token into the trace:

    metal_profile_mlx.py --trace-lib L --model DIR --prompt-file F --out T.jsonl
    metal_profile_mlx.py --no-hooks ...     the same runs without the library

Set MLX_MAX_OPS_PER_BUFFER=1 in the environment for one kernel per command
buffer (GPU time per kernel; the absolute times change, the shares are read).
Prints one JSON object with the per-run rates.
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trace-lib", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-n", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-hooks", action="store_true")
    args = parser.parse_args()

    if args.no_hooks:
        mark = dump = lambda _name: None
    else:
        trace = ctypes.CDLL(args.trace_lib)
        trace.metal_trace_mark.argtypes = [ctypes.c_char_p]
        trace.metal_trace_dump.argtypes = [ctypes.c_char_p]
        mark, dump = trace.metal_trace_mark, trace.metal_trace_dump
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    model, tokenizer = load(args.model)
    tokenizer._eos_token_ids = set()
    ids = list(tokenizer.encode(Path(args.prompt_file).read_text(encoding="utf-8")))
    ids = [ids[i % len(ids)] for i in range(args.prompt_tokens)]
    runs = []
    for rep in range(-1, args.repeats):  # -1: the discarded warmup
        tag = "warmup" if rep < 0 else f"r{rep}"
        stamps, tokens, last = [], [], None
        mark(f"{tag}.start".encode())
        tic = time.perf_counter()
        for last in stream_generate(model, tokenizer, ids, max_tokens=args.decode_n):
            if not stamps:
                mark(f"{tag}.first".encode())
            stamps.append(time.perf_counter())
            tokens.append(int(last.token))
        mark(f"{tag}.end".encode())
        runs.append({
            "run": tag,
            "prefill_ms": (stamps[0] - tic) * 1e3,
            "decode_ms": (stamps[-1] - stamps[0]) * 1e3,
            "prefill_tps": args.prompt_tokens / (stamps[0] - tic),
            "decode_tps": (len(stamps) - 1) / (stamps[-1] - stamps[0]),
            "lib_prompt_tps": last.prompt_tps,
            "lib_generation_tps": last.generation_tps,
            "tokens": tokens,
        })
    dump(args.out.encode())
    print("RESULT " + json.dumps({"mlx": mx.__version__, "runs": runs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
