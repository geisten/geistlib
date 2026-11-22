#!/usr/bin/env python3
"""mlx_vs_geist_mlx.py — the mlx-lm side of tools/probes/mlx_vs_geist.sh.

Runs in the probe's own virtual environment (mlx, mlx-lm pinned there), one
process per measurement, and prints one JSON object on stdout:

    download --repo R --revision SHA      snapshot into HF_HOME, sha256 of the weights
    load     --model DIR --prompt-file F  import, load, one token: time to ready
    bench    --model DIR --prompt-file F  warmup, then repeats of prefill + decode
    detok    --model DIR --ids 1,2,3      token ids as text (the GGUF's tokens)

bench follows mlx_lm.benchmark: stream_generate on a token list with the
end-of-sequence set emptied, one discarded run, no prompt cache between
runs. It reports the library's own prompt_tps / generation_tps and, beside
them, wall-clock times taken at each yielded token.
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

T0 = time.perf_counter()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_download(args):
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

    path = Path(snapshot_download(args.repo, revision=args.revision,
                                  allow_patterns=DEFAULT_ALLOW_PATTERNS))
    files = sorted(p for p in path.iterdir() if p.is_file())
    weights = {p.name: sha256_file(p) for p in files if p.suffix == ".safetensors"}
    return {
        "path": str(path),
        "weights": weights,
        "weight_bytes": sum(p.stat().st_size for p in files if p.suffix == ".safetensors"),
        "snapshot_bytes": sum(p.stat().st_size for p in files),
        "weight_files": [str(p.resolve()) for p in files if p.suffix == ".safetensors"],
    }


def prompt_ids(tokenizer, prompt_file, n):
    """The first n tokens of the file's text, the text repeated when it is
    shorter (bench_perf_sweep --prompt-file does the same on its side)."""
    ids = list(tokenizer.encode(Path(prompt_file).read_text(encoding="utf-8")))
    full = len(ids)
    out = [ids[i % full] for i in range(n)]
    return out, full


def model_stats(model, config):
    """Bytes the loaded parameters take and how many weights they stand for:
    a quantized layer packs its weights into uint32 words, so its count is
    words * 32 / bits; scales and biases count as bytes only."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    params = [v for _, v in tree_flatten(model.parameters()) if isinstance(v, mx.array)]
    nbytes = sum(v.nbytes for v in params)
    logical = sum(v.size for v in params)
    quantized = 0
    for _, module in model.named_modules():
        bits = getattr(module, "bits", None)
        if bits and "weight" in module and module["weight"].dtype == mx.uint32:
            packed = module["weight"].size
            extra = sum(module[k].size for k in ("scales", "biases") if k in module)
            logical += packed * 32 // bits - packed - extra
            quantized += 1
    return {
        "param_bytes": nbytes,
        "logical_weights": logical,
        "bits_per_weight": round(nbytes * 8 / logical, 3) if logical else None,
        "quantized_layers": quantized,
        "quantization": config.get("quantization"),
        "model_type": config.get("model_type"),
    }


def load_model(args):
    import mlx.core as mx
    t_import0 = time.perf_counter()
    from mlx_lm import load, stream_generate
    t_import = time.perf_counter()
    model, tokenizer, config = load(args.model, return_config=True)
    t_load = time.perf_counter()
    tokenizer._eos_token_ids = set()  # as mlx_lm.benchmark: never stop early
    times = {
        "python_start_to_main_s": round(t_import0 - T0, 4),
        "import_s": round(t_import - t_import0, 4),
        "load_s": round(t_load - t_import, 4),
    }
    return mx, stream_generate, model, tokenizer, config, times


def one_run(mx, stream_generate, model, tokenizer, ids, max_tokens):
    mx.reset_peak_memory()
    tokens, stamps = [], []
    tic = time.perf_counter()
    last = None
    for last in stream_generate(model, tokenizer, ids, max_tokens=max_tokens):
        tokens.append(int(last.token))
        stamps.append(time.perf_counter())
    n = len(tokens)
    return {
        "prompt_tokens": int(last.prompt_tokens),
        "generation_tokens": int(last.generation_tokens),
        "finish_reason": last.finish_reason,
        "lib_prompt_tps": last.prompt_tps,
        "lib_generation_tps": last.generation_tps,
        "lib_peak_memory_gb": last.peak_memory,
        "ttft_ms": (stamps[0] - tic) * 1000.0,
        "wall_decode_tps": (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else None,
        "wall_total_s": stamps[-1] - tic,
        "tokens": tokens,
    }


def cmd_load(args):
    mx, stream_generate, model, tokenizer, _config, times = load_model(args)
    ids, _full = prompt_ids(tokenizer, args.prompt_file, 1)
    tic = time.perf_counter()
    for _ in stream_generate(model, tokenizer, ids, max_tokens=1):
        pass
    times["first_token_s"] = round(time.perf_counter() - tic, 4)
    times["ready_s"] = round(time.perf_counter() - T0, 4)
    return times


def cmd_bench(args):
    mx, stream_generate, model, tokenizer, config, times = load_model(args)
    ids, full = prompt_ids(tokenizer, args.prompt_file, args.prompt_tokens)
    for _ in range(args.warmup):
        one_run(mx, stream_generate, model, tokenizer, ids, args.decode_n)
    runs = [one_run(mx, stream_generate, model, tokenizer, ids, args.decode_n)
            for _ in range(args.repeats)]
    first = runs[0]["tokens"]
    for run in runs[1:]:
        run["tokens_equal_first_repeat"] = run.pop("tokens") == first
    import mlx_lm
    try:
        stats = model_stats(model, config)
    except Exception as error:  # statistics only: keep the timings
        stats = {"error": repr(error)}
    return {
        "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
                     "python": sys.version.split()[0]},
        "times": times,
        "prompt_file_tokens": full,
        "prompt_tokens": len(ids),
        "prompt_head_ids": ids[:8],
        "decode_n": args.decode_n,
        "warmup": args.warmup,
        "runs": runs,
        "text_first_40": tokenizer.decode(first[:40]),
        "model": stats,
        "memory": {
            "mlx_peak_bytes_last_run": mx.get_peak_memory(),
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_cache_bytes": mx.get_cache_memory(),
            "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "device": mx.device_info(),
    }


def cmd_detok(args):
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(args.model))
    ids = [int(v) for v in args.ids.split(",") if v]
    prompt, full = prompt_ids(tokenizer, args.prompt_file, 8) if args.prompt_file else ([], 0)
    return {"text": tokenizer.decode(ids), "prompt_head_ids": prompt, "prompt_file_tokens": full}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["download", "load", "bench", "detok"])
    parser.add_argument("--repo")
    parser.add_argument("--revision")
    parser.add_argument("--model")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-n", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ids", default="")
    args = parser.parse_args()
    result = {"download": cmd_download, "load": cmd_load, "bench": cmd_bench,
              "detok": cmd_detok}[args.mode](args)
    print("RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
