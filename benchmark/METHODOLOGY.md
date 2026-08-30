# Benchmarking geist

How to produce trustworthy numbers. See [results/APPLE.md](results/APPLE.md) for the
Apple M1 Max comparison + auto-recorded results and
[results/PI5.md](results/PI5.md) for the Raspberry Pi 5 target.

## Perf (reproducible, in-tree)

```sh
make fetch-model                         # Gemma 4 E2B-it Q4_K_M, ~3.1 GB
OMP_WAIT_POLICY=active make bench-small    # pp128/tg32, warmup 8, mean of 3
OMP_WAIT_POLICY=active make bench-detailed # pp512/tg64, warmup 64, mean of 10
BENCH_THREADS=6 OMP_WAIT_POLICY=active make bench-detailed  # pin thread count
```

These run `bench_perf_sweep` (warm-up, then N measured repeats) via
`tools/bench_quality_perf.py`, which records a tagged row per
(model, host, os, target/mode, threads), keeping the best decode run. The row
carries mean throughput, the run-to-run spread, and a derived TTFT. The exact
constants come from `benchmark/apple_cpu_protocol.json`; both this document and
the driver refer to that source. The timestamped raw JSONL retains every ordered
sample plus full commit/model/binary hashes, compiler, linked libraries,
environment and diagnostics.

For a performance-relevant revision, do not compare two back-to-back summary
rows. `tools/bench_mac_ab.py` rotates variant order across three cycles, waits
for the load/cooldown gate before every run, evaluates sequences
128/256/512/1024 and reports median plus median absolute deviation. See
[`results/APPLE.md`](results/APPLE.md#methodology) for the exact command and
guard band. It records the `pmset` thermal/power state before and after each
measurement, and marks interrupted artifacts as partial.
An interrupted campaign can be continued with `--resume` only when its model,
protocol, environment, baseline, binary hashes and rotated run prefix still
match exactly.

Raw timing probes for individual subsystems:

```sh
make bench           # all bench_* binaries
make bench-vision    # vision encoder only
make bench-audio     # audio encoder only
```

## Audio-tower baseline (Pi 5)

`benchmark/results/PI5-audio.md` records the per-stage audio baseline the
audio optimizations are judged against. Its protocol is self-contained and
fixture-reproducible: the tower comes from `make fetch-audio-tower`
(SHA-pinned Range extraction, see the Makefile), mel constants are checked
in, and the input clips are **synthesized** by `tools/gen_test_wav.py`
(deterministic — the doc records their SHA-256; no voice recordings enter
the repository). Protocol: quiesced board started < 55 °C, 1 warmup +
10 repeats per clip length (2 s / 10 s / 28 s), medians, spread and
temperatures recorded; stage shares from a `-DGEIST_AUDIO_PROFILE` build.
`reference_runs.json` stays owned by `make bench --record` and carries no
audio rows.

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
at n=500, i.e. iso-quality (see [results/PI5.md](results/PI5.md#quality)).

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

The measuring scripts emit result JSON (stdlib-only — runs on a bare Pi), and
the chart is rendered from that JSON. This keeps chart bars from drifting away
from the numbers:

```sh
# cross-engine measurement (no deps, thermal-quiesced on a Pi)
JSON_OUT=/tmp/total_tps.json python3 benchmark/total_tps.py

# the README scoreboard — pure stdlib, numbers straight from headline_results.json
python3 benchmark/chart_headline.py     # -> assets/headline_benchmarks.svg
```

`headline_results.json` is curated input, not raw output: each row names its
baseline engine and pinned version, and sub-parity rows are kept on purpose.

## Historical cross-engine aggregation

Both machines run the **same model and quantization**, both engines CPU-only, each
at its best thread count, after a **discarded warm-up run** (the runtime pages
weights resident and spins up the OpenMP pool, so timings reflect steady state, not
cold-start). llama.cpp build `d05fe1d`.

- **Raspberry Pi 5 — `mean of 10`, cool start.** A dedicated headless box,
  genuinely quiesced (load 0.0). The mean is meaningful: spread is <2 % run-to-run.
  Crucially, **both engines are started from a cool baseline (<56 °C)** — a 4.6 B
  prefill drives this passively-cooled board to ~78 °C and trips the soft temp
  limit in under a minute, so benchmarking one engine right after the other
  throttles the second (this is exactly what understated llama's pp128 to 22 t/s
  in an earlier revision — cool, it is ~37).
- **Apple M1 Max historical llama.cpp comparison — `best of 10`.** A developer workstation that *cannot* be
  quiesced while in use (WindowServer, browser, IDE all contend for the P-cores).
  On a contended box the **mean** is dominated by interference spikes (±20 %
  run-to-run), so we report the **best** of 10 repeats — the least-interrupted run,
  which approximates the uncontended ceiling and is stable across independent
  campaigns. Both engines use best-of here, so the comparison stays
  apples-to-apples.

That paragraph describes the frozen June cross-engine table only. Current
`small`/`detailed` rows retain and report the mean of their ordered repeats;
revision comparisons use three quiet-gated, order-rotated cycles and report
median plus median absolute deviation. Aggregation is never selected after
looking at the result.

**Always measure on a quiesced box.** On the 4-core Pi a single stray process
eating one core inverts the 4-thread numbers. Reproduce with `bench_perf_sweep`
(geist) and `llama-bench` (reference) — see the sections above for
the exact commands, the correctness gate (bit-identical greedy output on the same
weights), and the quality/PPL caveats.
