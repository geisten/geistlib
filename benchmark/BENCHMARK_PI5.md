# geist Benchmarks — Raspberry Pi 5

The Pi 5 (Cortex-A76, 4 cores, no `i8mm`) is geist's design target and the hard
case: an older ARM core where llama.cpp leans on a decades-tuned OpenBLAS fp32
path. The numbers below were measured on a real, **quiesced** Pi 5.

> ⚠️ **Measure on a quiesced *and thermally settled* board.** Two confounds bite
> on a passively-cooled Pi 5: (1) a stray background process eating one core
> silently halves the 4-thread numbers; (2) **heat** — a 4.6 B prefill drives the
> board to ~78 °C in under a minute and trips the soft temperature limit, so
> whichever engine you benchmark *second* runs throttled. Check `uptime`,
> `vcgencmd measure_temp`, and `vcgencmd get_throttled` first, and let the board
> cool (< ~60 °C) between engines. An earlier revision of this page reported
> llama.cpp at 22 t/s for pp128 — a thermal artifact (llama was measured right
> after geist heat-soaked the board); cool, it is ~37 t/s. The numbers below were
> re-measured with both engines started from a cool baseline.

## Setup

- **Board:** Raspberry Pi 5 Model B Rev 1.1, 4× Cortex-A76, 4 GB RAM,
  64-bit Raspberry Pi OS (kernel 6.18, Debian).
- **geist build:** `make TARGET=pi5 CC=gcc` (gcc 14.2; gcc ≥ 14 required for
  `-std=c23`; OpenBLAS + OpenMP). Builds clean under `-Werror`.
- **Reference:** llama.cpp `d05fe1d`, built with OpenBLAS
  (`cmake -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON`). CPU-only
  (the Pi has no GPU).
- **Model:** Gemma 4 E2B-it, Q4_K_M — the **identical** GGUF for both engines.
- **Weights:** `GEIST_WEIGHT_MMAP=1` (the default) keeps the 1.93 GB PLE table
  mmap-aliased rather than copied resident — important on the 4 GB board.

```sh
make TARGET=pi5 CC=gcc
M=gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
# geist auto-runs prefill on all 4 cores and decode on 3.
# Each point is the MEAN of 10 measured repeats, taken AFTER a discarded
# warm-up run (--warmup) that pages weights resident and spins up the OMP pool:
OMP_WAIT_POLICY=active \
  bin/pi5/release/tests/bench_perf_sweep --gguf $M --seq-lens 128,256,512,1024 --decode-n 16 --warmup 16 --repeats 10
# llama.cpp reference (4 threads, prefill sweep; llama-bench warms up internally):
llama-bench -m $M -p 128,256,512,1024 -n 32 -t 4
```

Both engines **warm up before measuring** — geist discards a `--warmup`-token
prefill+decode run; `llama-bench` runs its own warm-up iterations — so the
figures reflect steady state, not cold caches.

## Measured results (June 2026, quiesced)

**Identical protocol for both engines:** prefill-only, **8 reps**, 4 threads,
each **started from a cool baseline** (<56 °C) and run back-to-back in one session
(geist `bench_perf_sweep --decode-n 0 --repeats 8`; llama `llama-bench -n 0 -r 8`).
Same weights, same quantization, CPU-only. Decode is measured separately (below).

| seq_len | llama.cpp (OpenBLAS, `d05fe1d`) | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | **37.4** | 34.8 | llama 1.07× |
|  256 | **39.4** | 34.2 | llama 1.15× |
|  512 | **37.6** | 32.9 | llama 1.14× |
| 1024 | **35.9** | 31.5 | llama 1.14× |
| **decode** (best, t=3) | 6.8 | 6.9 | ≈ par |

**On the Pi, llama.cpp's OpenBLAS prefill leads geist by ~7–15 % at every length;
both curves are flat (~37–39 vs ~33–35 t/s). Decode is a tie (~6.8 t/s).** This
is the hard case geist was built around: llama leans on a decades-tuned OpenBLAS
fp32 sgemm, and on an A76 without `i8mm` that path is genuinely fast and hard to
beat. geist's native int8 (W4A8) kernel is competitive and flat but does not
overtake it here — closing the remaining ~10–15 % is open work. geist's real wins
are elsewhere: **decode parity**, a **dependency-free static binary**, and a
**clean sweep on Apple AMX** (see [BENCHMARK.md](BENCHMARK.md)).

> **Correction (the numbers above are re-measured).** An earlier revision reported
> llama.cpp at 22 / 30 / 33 / 34 t/s — which made geist *appear* to win short
> context. That was a **thermal artifact**: geist's long prefills ran first and
> drove the passively-cooled board to ~78 °C (soft-limit `0x80000`), so llama,
> benchmarked second, was throttled. Re-measured from a cool start, llama is
> ~37–39 t/s and flat. (The original *pre-session* page already had llama ≈ 37.5;
> the 22 was a regression introduced mid-session and caught on review.)

> **geist's own prefill curve was separately fixed.** It used to *fall* with
> context (32 → 23 t/s at 1024) because the O(n²) SDPA **core was single-threaded**
> (profiling: attention stage 22 %→45 %; thread-scaling FFN ×3.7 vs core ~×1.0).
> Parallelizing it over query positions (`#pragma omp parallel for`, bit-exact)
> flattened geist's curve (pp1024 +35 %, to 31.4). It closed geist's *internal*
> bottleneck but, as the table shows, llama still leads Pi prefill. See
> [BENCHMARKING.md](BENCHMARKING.md) for the profiler.

> **Thermal ceiling.** Both sweeps reach ~77–79 °C and trip the soft temperature
> limit by their longest length on this passively-cooled board, so the absolute
> long-context numbers carry a throttle ceiling. The comparison is fair (both
> start cool, similar trajectories); with active cooling both would be a few % higher.

On Apple AMX the picture is geist-favoured at every length (see
[BENCHMARK.md](BENCHMARK.md), the Apple M1 Max write-up in this folder).

## Re-verification — current llama.cpp + ollama (June 20, 2026)

Re-measured on the same board after the `src/base` / `src/quant` / arch-ops /
GEMM-provider refactors, and against a **current** llama.cpp (the pinned
`d05fe1d` above was a year old — a fair comparison should use today's upstream).
Same GGUF, 4 threads, each engine cold-started; geist is the post-refactor build,
llama.cpp is `acd79d6` (2026-06-14) built two ways.

| engine / build | pp128 | pp256 | pp512 | pp1024 | decode |
| --- | :---: | :---: | :---: | :---: | :---: |
| **geist** (post-refactor) | 33.5 | 33.4 | 32.4 | 31.0 | 6.4 |
| llama.cpp `acd79d6` **OpenBLAS** | 38.3 | 38.0 | 36.8 | 35.5 | 6.90 |
| llama.cpp `acd79d6` **CPU** (no BLAS) | 39.2 | 38.1 | 37.3 | 35.2 | 6.88 |
| llama.cpp `d05fe1d` (the pinned baseline) | 37.4 | 39.4 | 37.6 | 35.9 | 6.8 |

**Two things this settles:**

1. **The refactors did not move geist.** Prefill is within ~1–4 % of the
   pre-refactor numbers (pure measurement/thermal noise) — expected, since they
   were structural (file moves, build seams); the compute kernels are byte-identical.
2. **"Use a current llama.cpp" changes nothing here, and neither does dropping
   BLAS.** Current upstream (`acd79d6`) ≈ the year-old `d05fe1d` ≈ ~38 t/s, and the
   modern ggml CPU backend ties OpenBLAS (~39 vs ~38) — the A76 has no `i8mm`, so
   the repacked int8 path has no SDOT/i8mm edge to exploit over a tuned fp32 sgemm.
   geist stays ~13–15 % behind on prefill and ~par on decode, exactly as before.

### ollama

Added `ollama` 0.30.10 (its bundled llama.cpp, ggml CPU backend) on the same GGUF
via a `Modelfile` (`FROM …/gemma4-e2b-Q4_K_M.gguf`). Harness:
[`ollama_bench_pi5.py`](ollama_bench_pi5.py).

> **ollama OOM-kills on the 4 GB board by default — `use_mmap` is required.**
> Stock ollama loads weights resident; gemma4-e2b's **1.93 GB PLE table** then sits
> in anonymous RAM and `llama-server` is OOM-killed at load (`signal: killed`,
> anon-rss ≈ 1.96 GB), independent of context size. Forcing `use_mmap:true` (plus a
> small `num_batch` so the prefill buffer fits) keeps the weights file-backed and it
> loads. **This is exactly the 4 GB constraint geist solves natively** via
> `GEIST_WEIGHT_MMAP` — geist runs the model here with no tuning.

Decode lands ~par (ollama ~6.2 t/s, measured while the board was already warm vs
the cold llama.cpp/geist runs). **Its prefill is *not* reported here:** ollama is
llama.cpp, yet its REST-API timing fields gave ~44 t/s — ~15 % above the same
engine measured by standalone `llama-bench` (38–39), and that while running
hotter, which is physically backwards. The discrepancy is an artifact of how the
API reports `prompt_eval_duration`, not a real win. For a like-for-like ollama
prefill number, read the **llama.cpp CPU** row above — it is the same runner.

### Total tok/s — the user-facing metric

Prefill t/s alone understates geist: a real generation is **decode-dominated**
(one prefill, many decode steps), so end-to-end throughput is what a user feels.
Measured with the speculative output head **on** (the default since June 2026),
each engine cold-started under a tight ≤50 °C gate (no thermal skew), total =
(P+D)/(t_prefill + t_decode). Harness: [`total_tps.py`](total_tps.py).

**32-token prompt + 128 decode (decode-heavy):**

| engine | prefill | decode | **total t/s** |
| --- | :---: | :---: | :---: |
| **geist** | 30.0 | **7.51** | **8.83** |
| llama.cpp (CPU) | 38.5 | 6.83 | 8.18 |
| llama.cpp (OpenBLAS) | 34.7 | 6.61 | 7.89 |
| ollama | (44.7)\* | 7.09 | (13.0)\* |

**128-token prompt + 128 decode (more prefill weight):**

| engine | prefill | decode | **total t/s** |
| --- | :---: | :---: | :---: |
| **geist** | 33.5 | 6.67 | 11.12 |
| llama.cpp (CPU) | 38.6 | 6.65 | 11.35 |
| llama.cpp (OpenBLAS) | 38.3 | 6.48 | 11.08 |
| ollama | (45.5)\* | 6.95 | (12.8)\* |

**Total tracks decode, not prefill** (decode is the many-token phase). geist has
the lowest prefill but the **fastest decode** (the spec head), so it **leads
end-to-end at a short prompt** (8.83 vs 8.18 / 7.89) and **ties at a longer one**
(11.1 vs 11.1–11.3, where the prefill gap starts to weigh in). So the Q4_K prefill
kernel work matters only for prompt-heavy / time-to-first-token workloads, not for
typical chat throughput.

> \* ollama's prefill (and hence its total) carries the REST-API timing inflation
> documented above; substituting the real engine prefill (≈ llama.cpp CPU) puts
> ollama's total ≈ geist. ollama and geist decode the same (~7.0) — both run the
> spec head's equivalent.

## Thread placement (quiesced)

The two phases want different core counts, and geist sets each automatically:

| threads | geist prefill pp256 | geist decode | llama prefill pp256 | llama decode |
| :---: | :---: | :---: | :---: | :---: |
| 4 | **34.1** | 6.79 | **38.5** | 6.21 |
| 3 | 26.4 | **6.92** | 31.1 | **6.81** |

**Prefill** is compute-bound and scales with all 4 homogeneous A76 cores for
*both* engines (geist 26 → 34, llama 31 → 38 from 3 → 4 threads). **Decode** is
memory-bandwidth-bound and fastest at 3 — the 4th thread just adds LPDDR
contention — for both engines too (geist 6.92 vs 6.79, llama 6.81 vs 6.21). geist
auto-selects 4 for prefill and 3 for decode; override with `GEIST_PREFILL_THREADS`
/ `GEIST_DECODE_THREADS`. (These are cool-start numbers; an earlier revision showed
llama "flat across 3/4 threads" — another thermal artifact.)

## Predecode is correctly off on the Pi

geist's `q4k_predecode` fast path (which speeds prefill on Apple) is gated to
`has_accelerate`, i.e. **off on the Pi.** Forcing it on
(`GEIST_Q4K_PREDECODE=1 GEIST_Q4K_MTILE_PREFILL=1 GEIST_Q4K_NTILE_PREFILL=1`)
makes prefill **slower** — measured clean pp256 30.3 → 21.7 tps (−28 %): the
predecoded block is ~1.9× the bytes of raw Q4_K, and the Pi's LPDDR4X bandwidth
makes that byte-doubling cost more than the saved scale-unpack compute (the same
bandwidth-vs-compute trade-off as the Q8_0 engine on Apple, sharper here). The
gate is correct.

## mmap tuning is a no-op on the Pi (16 KB pages, no THP)

geist applies best-effort `madvise` hints to the weight mapping (Linux
`MADV_HUGEPAGE`; opt-in `MADV_WILLNEED` via `GEIST_MMAP_PREFETCH=1`). On a 4 KB-
page Linux server transparent huge pages cut TLB misses on the big weight
tables — a real win. **On the Pi 5 it does nothing:** the kernel already uses
**16 KB base pages** (4× fewer than 4 KB → TLB pressure is already low) and ships
**no THP** (`/sys/kernel/mm/transparent_hugepage` is absent), so `MADV_HUGEPAGE`
is literally a no-op. Measured pp256: default 34.0, `GEIST_MMAP_PREFETCH=1` 34.3,
`GEIST_NO_HUGEPAGE=1` 34.3 — all within noise. The hints are correct and harmless
here; the lever lives on 4 KB-page hosts.

## Quality

The Pi 5 build is numerically sound: greedy output matches the Apple build
bit-for-bit on the structured-output probes, so `-ffast-math` + Cortex-A76 NEON
do not perturb decoding on these tasks.

### MMLU (accuracy parity with llama.cpp)

`tools/eval_mmlu.py` runs the standard lm-eval cloze (5-shot, log-prob of
` A`/` B`/` C`/` D`, argmax) on `cais/mmlu`, driving `eval_geist` with the
model's own GGUF tokenizer. Quality is hardware-independent (same GGUF → same
logits modulo kernel rounding), so this was run on the Mac; the Pi build shares
the kernels. Same 500 questions (`--shuffle` seed 1234), same prompts.

| engine    | MMLU (Gemma 4 E2B Q4_K_M, 5-shot) |
|-----------|-----------------------------------|
| llama.cpp | **54.0%** (270/500)               |
| geist     | **52.8%** (264/500)               |

The 1.2-point gap is 6 questions — inside the ±4.4% binomial CI at n=500 — and
comes from Q4_K kernel rounding flipping a few close-call argmaxes; per-subject
scores track llama.cpp closely. geist is quality-faithful to the reference on
identical weights. Reproduce: `llama-server -m model.gguf -c 4096` then
`python3 tools/eval_mmlu_llama.py --hf --shuffle --limit 500` for the reference,
and `tools/eval_mmlu.py --hf --shuffle --limit 500` for geist.

> Gotcha baked into the harness: Gemma needs `<bos>` prepended (`--bos 2`,
> default on). Without it the model runs out-of-distribution and predicts a
> newline after `Answer:`, collapsing accuracy to ~37%. llama-server's
> `/completion` adds BOS itself, so the comparison stays apples-to-apples.

> Measured on one Pi 5 in June 2026, not in CI (CI has no Pi 5 hardware). If you
> reproduce or refute these, please open a PR with your board, OS, thread count,
> and the llama.cpp commit.
