# Benchmarking geist

How to produce trustworthy numbers. See [BENCHMARK.md](BENCHMARK.md) for the
Apple M1 Max comparison + auto-recorded results and
[BENCHMARK_PI5.md](BENCHMARK_PI5.md) for the Raspberry Pi 5 target.

## Perf (reproducible, in-tree)

```sh
make fetch-model                         # Gemma 4 E2B-it Q4_K_M, ~3.1 GB
OMP_WAIT_POLICY=active make bench-small   # best-of-2, records to benchmark/BENCHMARK.md
OMP_WAIT_POLICY=active make bench-detailed # best-of-5
BENCH_THREADS=6 OMP_WAIT_POLICY=active make bench-detailed  # pin thread count
```

These run `bench_session_throughput` (warm-up → prefill 200 → decode 50) via
`tools/bench_quality_perf.py`, which records a tagged row per
(model, host, os, target/mode, threads), keeping the best decode run.

Raw timing probes for individual subsystems:

```sh
make bench           # all bench_* binaries
make bench-vision    # vision encoder only
make bench-audio     # audio encoder only
```

## Comparison vs llama.cpp

**Speed** — pin the reference: build `llama.cpp` (`llama-bench`) from a known
commit against the **same** GGUF, then run `benchmark/total_tps.py` with
`LLAMA_CPU` / `LLAMA_BLAS` pointing at it (cross-engine total tok/s).

**Quality** — `make bench-quality-small` / `-detailed` run the MMLU cloze on
geist; `make bench-compare-ref` additionally scores a running `llama-server`
on the *same* GGUF and prints the gap:

```sh
pip install datasets
llama-server -m gguf_artifacts/gemma4-e2b-Q4_K_M.gguf -c 4096 &  # reference
make bench-compare-ref BENCH_REF_URL=http://127.0.0.1:8080       # default URL
```

Record, every time:

- the llama.cpp **commit hash** (`llama-bench --version` or `git rev-parse`);
- thread count and `OMP_WAIT_POLICY` / llama.cpp `-t`;
- host CPU + OS;
- that the GGUF is byte-identical for both engines.

Use bit-identical greedy output as the correctness gate before quoting any
speedup — a faster engine that produces different tokens is not iso-quality.

> `quality-*` / `compare-ref` need `pip install datasets` (and `compare-ref` a
> running `llama-server`); they're kept out of the hermetic build because of
> those external deps, but are otherwise fully wired (`tools/bench_quality_perf.py`).
> A cross-engine **PPL/KL** ranking is still unavailable — see below.

## Quality (perplexity / KL / MMLU)

Quality eval drives the `eval_geist` REPL through `tools/eval_runner.py`, which
tokenizes with a Hugging Face tokenizer for parity with reference
implementations:

```sh
pip install transformers
python3 tools/eval_runner.py --bin bin/<target>/release/tools/eval_geist \
    --gguf gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \
    --tokenizer google/gemma-4-E2B-it \
    generate --prompt "The capital of France is" --n 16
```

`eval_runner.py mc` scores multiple-choice options by continuation logprob
(MMLU/HellaSwag style). For PPL/KL against the reference model, capture
per-token logits with `eval_geist`'s `SCORE` command over a held-out corpus and
compare distributions; this harness is documented here rather than wired into
`make` because it depends on external datasets.

### MMLU accuracy (self-contained, `make bench-mmlu`)

`tools/eval_mmlu.py` measures MMLU accuracy through the `eval_geist` REPL,
tokenizing with the model's **own** GGUF tokenizer (the `TOK` command) — so
there is no external HF-tokenizer dependency and no risk of a tokenizer mismatch
between scoring and the model. It uses the standard log-likelihood **cloze**:
build the 5-shot prompt, then score the next-token log-prob of " A"/" B"/" C"/
" D" (each a single token in the Gemma vocab) in one `SCORE`-style prefill and
take the argmax. This is a **base-completion** eval (no chat template), which is
how MMLU is conventionally run, so it sidesteps the chat-template parity
question entirely.

```sh
pip install datasets
make bench-mmlu                       # 200 shuffled questions, 5-shot
make bench-mmlu MMLU_LIMIT=0          # full ~14k-question set
# or directly, incl. a no-dataset smoke test:
python3 tools/eval_mmlu.py --bin bin/<target>/release/tools/eval_geist \
    --gguf model.gguf --selftest --verbose     # embedded sample
python3 tools/eval_mmlu.py --bin ... --gguf model.gguf --hf --shuffle --limit 200
```

Few-shot matters: MMLU is conventionally 5-shot. Zero-shot, a small model
collapses to a position bias (always "A") — the harness reproduces this (0/5 on
the embedded sample at `--shots 0`, 5/5 at `--shots 5`), which is a property of
the model, not the scorer. This gives a real **absolute** MMLU number for geist;
a cross-*engine* MMLU/PPL ranking still needs matched conditions (below).

### Cross-engine MMLU ranking (resolved for the cloze path)

A cross-*engine* MMLU comparison needs logprob-identical conditions on both
sides. The MMLU **cloze** path now meets that bar:

```sh
llama-server -m gguf_artifacts/gemma4-e2b-Q4_K_M.gguf -c 4096   # reference
python3 tools/eval_mmlu_llama.py --hf --shuffle --limit 500     # llama.cpp
python3 tools/eval_mmlu.py --bin .../eval_geist --gguf model.gguf --hf --shuffle --limit 500
```

`eval_mmlu_llama.py` reuses `eval_mmlu.py`'s dataset loader + prompt builder, so
both engines see the *identical* questions (same `--shuffle` seed), 5-shot
exemplars and prompt text — only the kernels differ. Result on Gemma 4 E2B
Q4_K_M (500 q): **geist 52.8% vs llama.cpp 54.0%** — inside the ±4.4% binomial CI
at n=500, i.e. iso-quality (see [BENCHMARK_PI5.md](BENCHMARK_PI5.md#quality)).

Two gotchas that make or break the comparison:
- **BOS:** Gemma needs `<bos>` (id 2) prepended. `eval_mmlu.py --bos` defaults to
  it; llama-server's `/completion` adds it itself. Without it the model goes
  out-of-distribution and predicts a newline after `Answer:` (~37%).
- **Strip collisions:** ` C` and `C` both strip to `C` in the server's top-logprob
  table; keep the higher (the spaced variant is the real continuation).

Still **not** published as a leaderboard number: a cross-engine **PPL/KL** ranking.
llama.cpp's perplexity reports abnormally high absolute values on Gemma 4 E2B, so
it is not a reliable PPL baseline for this model yet, and geist's `bench_quality`
chat-template markers are unverified for logprob scoring (fine for greedy). The
cloze MMLU above sidesteps both (base completion, no chat template).

## Reporting (charts from data, not by hand)

The measuring scripts emit result JSON (stdlib-only — runs on a bare Pi);
`tools/bench_report.py` renders it to a grouped-bar chart with matplotlib (a
dev-box dep, kept off the measuring path). This keeps chart bars from drifting
away from the numbers:

```sh
JSON_OUT=benchmark/pi5_results.json python3 benchmark/total_tps.py   # measure (no deps)

# README chart — total tok/s, geist vs llama.cpp (no deps, pure stdlib):
python3 benchmark/chart_total_tps.py                                 # -> assets/pi5_total_tps.svg

# optional detailed prefill/decode/total breakdown (needs matplotlib):
pip install matplotlib
python3 tools/bench_report.py benchmark/pi5_results.json -o assets/pi5_pp_decode_total.svg
```

`bench_report.py` takes multiple JSON files (each `panels` entry → a panel) and
`-` for stdin, writes `.svg` or `.png`, and draws error bars when a metric value
is `{"value": x, "err": e}` (or `{"value": x, "lo": l, "hi": h}`) — e.g. a
best-of-N spread or the MMLU binomial CI.
