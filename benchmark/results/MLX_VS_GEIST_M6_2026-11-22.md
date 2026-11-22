# mlx-lm against geist on the Mac mini M6 (2026-11-22)

Six models, each as a GGUF for geist and llama.cpp and as an MLX conversion
for mlx-lm, on the Apple M6 Mac mini (Mac18,5, 12 cores, 32 GB, macOS 27.2;
runner `macmini-m6`).

**Result.** mlx-lm is faster than geist's Metal backend on every pair
measured, in prefill and in decode. Prefill at 512 prompt tokens: 1.18–1.39×
on five pairs and 2.88× on Gemma 4 E2B. Decode over 128 tokens: 1.08–1.30×.
geist is ahead in one thing measured here: the time from process start to the
first token, 0.05–0.09 s against 0.79–1.29 s with the weights in the page
cache (9–19×) and 0.18–0.93 s against 1.00–1.80 s with the weights evicted
(1.5–5.6×). Memory in use is about the same; the small "peak footprint" of
geist is an accounting difference, not a saving (see Memory).

The comparison is engine **and** format: the MLX files and the GGUF files are
different quantizations of the same model, with different bits per weight
(table under Models). Only the 8-bit pair of Qwen3.5 0.8B is close to like for
like: its two engines produced the same 128 greedy tokens.

## Throughput

Median (minimum–maximum) over all repeats of all cycles; n = 9 (3 cycles × 3
repeats) except the two geist rows marked n = 6. Rates are per the engine's
own tokens; the token counts are the same on both sides for these prompts
(see Prompt). TTFT is the time from the start of the 512-token prompt to the
first generated token.

| model | engine | prefill tok/s @ 512 | decode tok/s (128) | TTFT ms @ 512 | n |
| :-- | :-- | --: | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | geist Metal | 7104 (6958–7114) | 149.8 (149.6–149.9) | 72.1 (72.0–73.6) | 6 |
| | mlx-lm | 8518 (8484–8536) | 162.0 (161.6–162.1) | 60.3 (60.2–60.5) | 9 |
| | llama.cpp Metal | 6234 (6187–6283) | 144.6 (143.5–145.2) | — | 9 |
| | geist CPU (`cpu_neon`) | 979 (972–981) | 141.3 (140.7–141.5) | 522.8 (522.0–526.8) | 9 |
| Qwen3.5 0.8B, 4-bit | geist Metal | 7388 (6560–7412) | 205.0 (204.9–205.2) | 69.3 (69.1–78.0) | 9 |
| | mlx-lm | 9248 (9149–9277) | 266.6 (266.2–266.9) | 55.6 (55.4–56.1) | 9 |
| | llama.cpp Metal | 6345 (6281–6468) | 193.1 (190.5–198.5) | — | 9 |
| Llama 3.2 3B Instruct, 4-bit | geist Metal | 2041 (2035–2050) | 66.8 (66.7–67.2) | 250.8 (249.8–251.5) | 9 |
| | mlx-lm | 2634 (2630–2641) | 77.0 (76.1–77.2) | 194.5 (194.1–194.8) | 9 |
| | llama.cpp Metal | 2131 (2124–2135) | 67.3 (66.8–67.5) | — | 9 |
| Qwen3.5 4B, 4-bit | geist Metal | 1527 (1524–1530) | 48.2 (48.1–48.2) | 335.2 (334.6–335.9) | 9 |
| | mlx-lm | 1804 (1798–1809) | 57.1 (57.1–57.3) | 284.1 (283.3–284.9) | 9 |
| | llama.cpp Metal | 1420 (1412–1424) | 48.2 (48.2–48.3) | — | 9 |
| Gemma 4 E2B-it, 4-bit | geist Metal | 2676 (2542–2682) | 86.3 (86.2–86.3) | 191.4 (190.9–201.4) | 9 |
| | mlx-lm | 7693 (7677–7730) | 97.3 (95.8–97.4) | 66.8 (66.5–67.0) | 9 |
| | llama.cpp Metal | 2808 (2801–2820) | 83.6 (83.4–83.7) | — | 9 |
| Qwen3.5 9B, 4-bit | geist Metal | 748 (734–749) | 27.3 (27.0–27.3) | 684.9 (683.4–698.0) | 6 |
| | mlx-lm | 1042 (1038–1044) | 32.3 (32.2–32.4) | 491.7 (490.8–493.5) | 9 |
| | llama.cpp Metal | 860 (857–863) | 27.9 (27.8–27.9) | — | 9 |

Ratios of the medians:

| model | prefill mlx-lm / geist | decode mlx-lm / geist | prefill geist / llama.cpp | decode geist / llama.cpp |
| :-- | --: | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | 1.20 | 1.08 | 1.14 | 1.04 |
| Qwen3.5 0.8B, 4-bit | 1.25 | 1.30 | 1.16 | 1.06 |
| Llama 3.2 3B, 4-bit | 1.29 | 1.15 | 0.96 | 0.99 |
| Qwen3.5 4B, 4-bit | 1.18 | 1.19 | 1.08 | 1.00 |
| Gemma 4 E2B, 4-bit | 2.88 | 1.13 | 0.95 | 1.03 |
| Qwen3.5 9B, 4-bit | 1.39 | 1.18 | 0.87 | 0.98 |

geist Metal is behind llama.cpp Metal in prefill on Llama 3.2 3B (0.96),
Gemma 4 E2B (0.95) and Qwen3.5 9B (0.87), and level with it in decode
(0.98–1.06).

How the numbers are defined, since the engines do not time the same span:

- geist: `bench_perf_sweep` times `prefill_tokens` of 512 tokens, then 128
  `decode_step` calls. The first of those steps only picks from the logits
  the prefill left (it measures 0–2 µs), so the decode rate is 128 tokens
  over 127 forward passes.
- mlx-lm: `stream_generate`'s own `prompt_tps` (512 tokens over the time to
  the first generated token, so it includes the first sampling step) and
  `generation_tps` (128 tokens over the time after the first token, i.e.
  127 forward passes). The probe also stamped every yielded token: 127 tokens
  over those 127 intervals give 160.7, 264.5, 76.4, 56.7, 96.6 and 32.0 tok/s
  in table order, 0.8 % below the library's figure, as the definition
  predicts. With that stricter count the decode ratios are 1.07–1.29×.
- TTFT is the same span on both sides, and its ratio equals the prefill
  ratio on every pair.
- llama.cpp: `llama-bench` as the cross-engine protocol calls it (`-p 512
  -n 0`, then `-p 0 -n 128 -d 512`, `-ngl 99`), on its own random prompt
  tokens. It anchors the GGUF side; it reports no TTFT.

### Gemma 4 E2B: why 2.88×

The figure is real for a 512-token prompt, and it is not an engine speed
ratio. Evidence and reading:

- Wall clock agrees with the library: 66.8 ms to the first token and 1.383 s
  for the whole run (66.8 ms + 127 tokens at 96.6 tok/s) in every repeat;
  512 prompt tokens were passed and reported; the 128 generated tokens are
  the same in all nine repeats and continue the text sensibly.
- Gemma 4 E2B has 35 layers, of which the upper 20 own no key/value cache:
  they read the cache of an earlier layer (`num_kv_shared_layers: 20` in the
  conversion's `config.json`; `is_kv_shared` for layers 15–34 in geist). The
  upper layers of a prompt token are therefore needed only for the token
  whose logits are sampled. mlx-lm's prefill evaluates nothing but the cache
  state (`mx.eval([c.state for c in prompt_cache])` in `generate_step`,
  mlx-lm 0.32.2), and MLX is lazy, so for the first 511 prompt tokens the
  upper 20 layers, which also carry the double-width feed-forward blocks, are
  never computed. The conversion skips no weights for this: the per-layer
  embeddings are loaded and used.
- geist and llama.cpp run all 35 layers for every prompt token. A search of
  geist's transformer sources found no such shortcut, and geist's prefill
  rate equals llama.cpp's here (0.95). Decode, where every layer is needed,
  shows the ordinary gap (1.13×).

This reading comes from the sources and the arithmetic; no profile of either
engine was taken. If it holds, skipping the KV-shared layers for all but the
last prompt token is a prefill optimization open to geist on Gemma 4.

## Time to ready

Wall time of a fresh process from start to its first token on a one-token
prompt (`/usr/bin/time`), median (minimum–maximum) of one run per cycle.
"Cold": the weight file was evicted from the page cache right before
(`msync(MS_INVALIDATE)`; `mincore` showed 0 % of its pages resident afterwards
in every case). "Warm": the same command again.

| model | geist Metal cold s | mlx-lm cold s | geist Metal warm s | mlx-lm warm s |
| :-- | --: | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | 0.41 (0.28–0.53) | 1.07 (1.06–1.07) | 0.05 (0.05–0.05) | 0.97 (0.96–0.97) |
| Qwen3.5 0.8B, 4-bit | 0.18 (0.18–0.18) | 1.01 (1.00–1.07) | 0.05 (0.05–0.05) | 0.94 (0.94–0.94) |
| Llama 3.2 3B, 4-bit | 0.65 (0.65–0.65) | 1.00 (1.00–1.36) | 0.09 (0.09–0.10) | 0.79 (0.78–0.79) |
| Qwen3.5 4B, 4-bit | 0.65 (0.65–0.89) | 1.33 (1.33–1.43) | 0.08 (0.08–0.09) | 1.06 (1.05–1.06) |
| Gemma 4 E2B, 4-bit | 0.49 (0.48–0.49) | 1.62 (1.62–2.39) | 0.08 (0.08–0.08) | 1.29 (1.29–1.31) |
| Qwen3.5 9B, 4-bit | 0.93 (0.93–0.93) | 1.80 (1.79–1.81) | 0.09 (0.09–0.09) | 1.18 (1.18–1.19) |

The mlx-lm figure includes the Python interpreter and its imports (0.38–0.40 s) and `mlx_lm.load` (0.29–0.80 s warm, 0.50–1.20 s cold); a
long-running Python process pays the imports once. geist maps the GGUF and
wraps it for Metal without copying, so its warm figure hardly depends on the
model size. llama.cpp was not measured here: `llama-bench` does not expose it.

## Memory

No column measures the same thing for all three engines, and none shows a
saving for geist:

| model | engine | peak footprint GiB | peak RSS GiB | engine's own counter GiB |
| :-- | :-- | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | geist Metal | 0.18 | 0.16 | 0.88 |
| | mlx-lm | 1.87 | 1.32 | 1.33 |
| | llama.cpp Metal | 0.23 | 0.99 | — |
| Qwen3.5 0.8B, 4-bit | geist Metal | 0.18 | 0.16 | 0.60 |
| | mlx-lm | 1.52 | 0.97 | 0.99 |
| | llama.cpp Metal | 0.23 | 0.71 | — |
| Llama 3.2 3B, 4-bit | geist Metal | 0.43 | 0.61 | 2.05 |
| | mlx-lm | 2.51 | 2.07 | 2.18 |
| | llama.cpp Metal | 0.31 | 2.19 | — |
| Qwen3.5 4B, 4-bit | geist Metal | 0.28 | 0.25 | 2.66 |
| | mlx-lm | 3.56 | 2.78 | 2.99 |
| | llama.cpp Metal | 0.34 | 2.74 | — |
| Gemma 4 E2B, 4-bit | geist Metal | 0.32 | 0.35 | 3.02 |
| | mlx-lm | 3.52 | 3.54 | 2.94 |
| | llama.cpp Metal | 0.25 | 3.14 | — |
| Qwen3.5 9B, 4-bit | geist Metal | 0.34 | 0.31 | 5.32 |
| | mlx-lm | 5.95 | 5.27 | 5.39 |
| | llama.cpp Metal | 0.35 | 5.37 | — |

- **Peak footprint** and **peak RSS** are what `/usr/bin/time -l` reports
  for the throughput process (the larger of llama.cpp's two calls). The
  footprint counts the process's own dirty and GPU-allocated memory, not
  clean pages of a mapped file. geist and llama.cpp map the GGUF; geist hands
  those pages to Metal in place (`newBufferWithBytesNoCopy`,
  `docs/BACKENDS.md`), so its weights appear in neither figure, and
  llama.cpp's appear in RSS only. mlx-lm reads the safetensors into memory it
  allocates, so its weights appear in both.
- **Engine's own counter**: for geist `MTLDevice.currentAllocatedSize` after
  the run (`backend_allocated_mb`: the wrapped weights, the KV cache and
  scratch); for mlx-lm `mx.get_peak_memory()` of a measured run (weights,
  cache and transient arrays). These two are the closest pair, and they
  differ by −0.08 to +0.45 GiB (mlx-lm minus geist), with geist's host-side
  0.2–0.4 GiB not included in its counter.

What can be said: all three engines keep the weights in RAM once, and the
totals are within about half a GiB of each other. What differs is the kind of
memory: geist's and llama.cpp's weights are clean file pages, mlx-lm's are
anonymous memory. Whether that matters under memory pressure was not measured.
A system-wide measurement (wired and compressed memory before and during a
run) would be the like-for-like figure; this probe did not take one.

## Models

| model | GGUF (geist, llama.cpp) | bits/weight | MLX (mlx-lm) | bits/weight |
| :-- | :-- | --: | :-- | --: |
| Qwen3.5 0.8B, 8-bit | `unsloth/Qwen3.5-0.8B-GGUF` Q8_0, 0.81 GB | 8.63 | `mlx-community/Qwen3.5-0.8B-8bit` @ `87e768fb`, 1.00 GB file, 0.80 GB loaded | 8.51 |
| Qwen3.5 0.8B, 4-bit | `unsloth/Qwen3.5-0.8B-GGUF` Q4_0, 0.51 GB | 5.39 | `mlx-community/Qwen3.5-0.8B-4bit` @ `da28692b`, 0.63 GB file, 0.42 GB loaded | 4.51 |
| Llama 3.2 3B, 4-bit | `bartowski/Llama-3.2-3B-Instruct-GGUF` Q4_K_M, 2.02 GB | 5.03 | `mlx-community/Llama-3.2-3B-Instruct-4bit` @ `7f0dc925`, 1.81 GB | 4.50 |
| Qwen3.5 4B, 4-bit | `unsloth/Qwen3.5-4B-GGUF` Q4_0, 2.58 GB | 4.91 | `mlx-community/Qwen3.5-4B-4bit` @ `0e7ffd5c`, 3.03 GB file, 2.37 GB loaded | 4.50 |
| Gemma 4 E2B, 4-bit | `unsloth/gemma-4-E2B-it-GGUF` Q4_K_M, 3.11 GB | 5.35 | `mlx-community/gemma-4-e2b-it-4bit` @ `23876752`, 3.55 GB file, 2.60 GB loaded | 4.50 |
| Qwen3.5 9B, 4-bit | `unsloth/Qwen3.5-9B-GGUF` Q4_0, 5.38 GB | 4.81 | `mlx-community/Qwen3.5-9B-4bit` @ `8b2b98c0`, 5.95 GB file, 5.04 GB loaded | 4.50 |

Full URLs, revisions and SHA-256 of every file: `tools/probes/mlx_vs_geist_models.json`
(four GGUFs are the protocol's, `benchmark/model_sources.json`; the 0.8B Q4_0
and the 9B Q4_0 are pinned in the probe). The probe verified every hash on
the host. Bits per weight: GGUF file size over the element count of its
tensors; MLX loaded parameter bytes over the weights they stand for
(group-64 affine quantization: 4 or 8 bits plus a scale and a bias per 64
weights). "Loaded" is what mlx-lm keeps of the file: the MLX repositories
also carry vision or audio weights that the text model does not load. A
"Q4_0" file from this source is mixed (Q4_0 with Q4_1, Q5_K, Q6_K, Q8_0 and
F32 tensors), which is why it sits at 4.8–5.4 bits; the GGUF side has 0.3–0.9
bits per weight more than the MLX side on every 4-bit pair, and 0.13 more on
the 8-bit pair. More bits mean more memory traffic per token, which favours
the MLX files in decode by an amount this measurement cannot separate from
the engine.

## Method

- **Probe**: `tools/probes/mlx_vs_geist.sh` (driver `mlx_vs_geist.py`, MLX
  side `mlx_vs_geist_mlx.py`), run by `.gitea/workflows/mac-probe.yml` under
  the host lock. Runs 4039 (first three models) and 4041 (last three) at
  geistlib `3365493a` (main `d46135a2` plus the probe); run 4037 was a
  one-repeat check of the plumbing and is not used.
- **Schedule**: per model three cycles, the engines in rotating order
  (geist, mlx-lm, llama.cpp; then mlx-lm first; then llama.cpp first), the
  geist CPU cell appended for the 8-bit 0.8B pair. An engine's turn is: gate,
  cold time to ready, warm time to ready, throughput run.
- **Gate before every cell**: AC power with Low Power Mode off, CPU ≥ 80 %
  idle, chip ≤ 50 °C by `macmon` (45 °C CPU for the CPU cell). Recorded per
  cell: idle 96.6–98.5 %, GPU 32.3–44.4 °C, CPU ≤ 44.5 °C at the start, no
  cell waited for either gate. The 1-minute load average was 1.3–6.5; on
  macOS it does not track idleness (the Apple host profiles of
  `benchmark/cross_engine_gpu_protocol.json` gate on idle CPU for that
  reason), so the idle share is the gate.
- **Workload**: the first 512 tokens of `benchmark/prompts/moe-long.txt`
  (the opening of Moby-Dick, 862–889 tokens) in the engine's own tokenizer,
  then 128 greedy tokens with end-of-sequence ignored. One discarded warmup,
  3 measured repeats per process: geist's warmup is the protocol's (64-token
  prefill and 4 decode steps), mlx-lm's is one full run as in
  `mlx_lm.benchmark`. mlx-lm gets a fresh cache for every repeat (no prompt
  caching).
- **geist**: `bench_perf_sweep`, `TARGET=mac-omp BACKENDS="cpu_neon
  cpu_scalar metal"`, `GEIST_BENCH_BACKEND=metal`, 4 dispatch threads
  (profile `apple_m6_mini_metal`), default Metal settings. CPU cell:
  `cpu_neon`, 6 prefill and 5 decode threads (profile `apple_m6_mini`).
- **mlx-lm**: mlx 0.32.4, mlx-lm 0.32.2 (mlx-metal 0.32.4, transformers
  5.24.0, tokenizers 0.23.3, numpy 2.5.5; the full `pip freeze` is in the raw
  metadata), CPython 3.12.16 from uv 0.13.9, in a directory of its own.
  Downloads were anonymous.
- **llama.cpp**: `d8123504`, the repository's pin
  (`scripts/ensure-llama-bench.sh metal`), `llama-bench -ngl 99 -t 4`.
- **Aggregation**: median over all repeats of all cycles, with minimum and
  maximum. Two geist prefill rows contain one slow repeat each: the
  4-bit 0.8B (6560 against a median of 7388, a spread of 11.5 %) and Gemma 4
  E2B (2542 against 2676, 5.2 %). Every other row spreads by at most 4.1 %,
  most by under 1 %.

### Prompt and output check

The driver decoded the first 40 generated tokens of geist and of mlx-lm with
the MLX repository's tokenizer (raw data, `kind: sanity`); all twelve
continuations are English prose in the style of the prompt. Afterwards
geist's tokenizer was run over the prompt file for three of the GGUFs and
compared with the Hugging Face tokenizer of the MLX repository:

- Qwen3.5 (4B checked) and Gemma 4 E2B: identical ids, 889 and 862 tokens,
  no BOS on either side. On the 8-bit 0.8B pair the two engines then
  generated the same 128 tokens; on the 4-bit pairs they diverge after 0–2
  tokens, as different quantizations of a model do.
- Llama 3.2 3B: mlx-lm's tokenizer prepends BOS, `bench_perf_sweep
  --prompt-file` does not (`geist_session_tokenize` returns content tokens
  only, although the model's `add_bos` is set). So mlx-lm ran BOS and 511
  text tokens, geist 512 text tokens. The amount of work is the same; the
  texts generated are not comparable on this pair.
- Gemma 4 E2B: both engines got the same 512 ids, without the BOS the model
  expects. mlx-lm continues the passage; geist continues it for 25 tokens,
  emits `<turn|>` (its end-of-sequence, id 106) and, since the benchmark does
  not stop there, repeats it for the rest of the run. Whether geist's
  continuation is the right one for its Q4_K_M file was not checked against
  llama.cpp here; the two engines hold different weights and the prompt is
  outside what the model was trained on. geist's decode rate on this row is
  therefore measured mostly on a repeated token, which does not change the
  work per step of a dense model.

## Failed cells and a bench defect

Two geist Metal cells are missing (Qwen3.5 0.8B 8-bit, third cycle; Qwen3.5
9B, second cycle), which is why those rows have n = 6. Cause: for the time
to ready the driver ran `bench_perf_sweep --seq-lens 1 --decode-n 1` and
parsed its row, although it only needs the process's wall time. The single
decode step after a prefill does no forward pass, so its time is 0–2 µs, and
when the clock reads 0 `bench_perf_sweep` divides by it and prints
`"decode_tps":inf`, which is not JSON (`tests/bench_perf_sweep.c`, `dec_tps`;
reproduced on an M1 Max, where the same command prints 1000000.000 and
500000.000). The exception discarded the whole cell including its throughput
run. The driver no longer parses that row; `bench_perf_sweep` still prints
`inf` for a zero decode time and should print a number or null. The library
is not involved.

## Limits

- One machine, one day, one prompt, one prompt length. No sweep over prompt
  lengths; at other lengths the ratios may differ, on Gemma 4 E2B certainly.
- Engine and format are not separated (see Models). No pair runs the same
  weights; the 8-bit pair comes closest.
- Output quality was not compared. Different quantizations give different
  text; nothing here says which is better.
- Greedy decoding of raw text without a chat template; no sampling, no
  batching, no prompt caching, no speculative decoding, no MLX-specific
  conversions (DWQ, mixed-bit, MXFP4).
- Default settings of every engine; none was tuned for this machine.
- Memory: see Memory; no like-for-like figure exists in this data.
- "Cold" evicts the weight file only. The engine binaries, the Python
  packages and the Metal shader cache stayed warm.
- 7–12B class: one model (Qwen3.5 9B). Nothing above 6 GB was run.
- The geist CPU row is context for one model only.

## Raw data

`benchmark/results/raw/mlx-vs-geist-m6-2026-11-22/run-4039.jsonl` and
`run-4041.jsonl`: one record per cell with every sample, the gate readings,
the page-cache state, the process statistics, the native rows of each engine
and the generated tokens (home directory written as `~`).
`python3 tools/probes/mlx_vs_geist.py --report <files>` prints the combined
table.

Left on the Mac mini for a rerun: `~/geist-bench-cache/mlx` (17.0 GB: 16.1 GB
MLX snapshots under `hf/`, 0.39 GB virtual environment, uv and its CPython)
and, in the runner's GGUF cache, `Qwen3.5-0.8B-Q4_0.gguf` (0.51 GB) and
`Qwen3.5-9B-Q4_0.gguf` (5.38 GB).
