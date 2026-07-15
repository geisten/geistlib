# geist Benchmarks

This file documents **how** geist is benchmarked and records reproducible
results. The auto-recorded table at the bottom is written by
`make bench-small` / `make bench-detailed`; everything above the
`<!-- BENCH:AUTO -->` marker is hand-maintained prose and is preserved across
runs.

> **Reproducibility over headline numbers.** Every recorded row is tagged with
> host, OS, target, mode, thread count, and model, so results from different
> machines never silently overwrite each other. Throughput varies a lot with
> core count and `OMP_NUM_THREADS`/`OMP_WAIT_POLICY` — always read a row with
> its tags.

## Methodology

Perf is measured by the `bench_session_throughput` binary (built by
`make bench`), driven through `tools/bench_quality_perf.py`:

- **Model:** Gemma 4 E2B-it, Q4_K_M (`make fetch-model`).
- **Warm-up:** 64-token prefill, then reset (excludes cold caches / page-ins).
- **Prefill:** time to prefill 200 tokens → tok/s.
- **Decode:** time to autoregressively decode 50 tokens → tok/s.
- **Best-of-N:** `small` takes the best of 2 runs, `detailed` the best of 5,
  to suppress scheduler noise.

```sh
make fetch-model                        # one-time, ~3.1 GB
OMP_WAIT_POLICY=active make bench-small  # records a row below
# tune threads:
BENCH_THREADS=6 OMP_WAIT_POLICY=active make bench-detailed
```

`OMP_WAIT_POLICY=active` (or `KMP_BLOCKTIME=infinite`) matters on the `mac-omp`
target — passive wait adds large per-matmul thread-pool wake-up overhead. Thread
*placement* is handled automatically: geist pins prefill to the performance
cores and decode to P-cores−1 (see the comparison below). Override the workload
size with `GEIST_BENCH_PP` / `GEIST_BENCH_TG` to match an external reference
(e.g. `llama-bench -p 512 -n 128`).

## Comparison vs llama.cpp (Apple M1 Max, CPU, measured June 2026)

Same machine, same `gemma4-e2b-Q4_K_M.gguf`, both CPU-only. llama.cpp build
`d05fe1d` run with `-ngl 0` (BLAS/Accelerate, no GPU offload). geist auto-pins to
the 8 performance cores; llama at `-t 8`. Full prefill sweep 128 → 1024 tokens,
**best-of** (peak uncontended throughput — see the methodology note below):

| seq_len | llama.cpp `-ngl 0`, t=8 | geist (P-core pinned) | winner |
| ---: | :---: | :---: | :--- |
|  128 | 141 | **164** | **geist 1.16×** |
|  256 | 147 | **161** | **geist 1.10×** |
|  512 | 128 | **150** | **geist 1.17×** |
| 1024 |  97 | **144** | **geist 1.48×** |
| **decode** (tg32) | ~26 | ~26–32 | ≈ par (jitter-bound here — see Pi) |

**geist leads prefill at every length and the lead *widens* with context** —
geist's dense-fp32 path here is **Accelerate/AMX** (Apple's matrix coprocessor),
which holds ~flat from 128 to 1024 tokens (164 → 144), while llama.cpp's CPU-only
path degrades sharply past 256 (147 → 97). Decode is at parity and memory-
bandwidth-bound for both; on this live desktop the 16-token decode window is too
jitter-prone to rank — the **controlled decode comparison is on the quiesced Pi 5**
([BENCHMARK_PI5.md](BENCHMARK_PI5.md)), where geist edges llama 1.03×.

> **Methodology — why best-of here, mean-of-10 on the Pi.** This M1 Max is a
> *developer workstation* that cannot be quiesced while in use (WindowServer,
> browser, IDE all compete for the P-cores). On a contended box the **mean** of N
> runs is dominated by interference spikes (we measured the same prefill swing
> ±20 % run-to-run), so we report the **best of 10 repeats** — the least-interrupted
> run, which approximates the uncontended ceiling and is stable across independent
> campaigns (pp512 best 3394/3416 ms across two runs; pp1024 7119/7116 ms). Both
> engines use best-of (geist: `bench_perf_sweep --repeats 10`; llama: max over 3
> `llama-bench` passes), so the comparison is apples-to-apples. The **Raspberry
> Pi 5 is a dedicated headless box**, genuinely quiesced (load 0.0), so there we
> report the clean **mean of 10** with <2 % spread. The prior single-point table
> (pp512 152/156) predates this rebuild.*

**Decode-kernel investigation (negative results, for whoever picks this up).**
The single-row Q4_K decode GEMV measures ~17 GB/s/core single-threaded — well
below M1 single-core memory bandwidth, so it is **compute-bound per thread**;
the full 8-thread decode then runs ~93 GB/s aggregate vs llama.cpp's ~113, i.e.
roughly memory-bandwidth-bound at the top. Things tried that did **not** close
the gap: (1) four independent int32 accumulators to break the per-super-block
`vmlaq` dependency chain — bit-exact but throughput-neutral, so the kernel is
*not* latency-bound; (2) routing decode through the predecoded-block layout —
slower for m=1 (re-quantize + per-call alloc, GEMM-shaped kernel); (3) more
decode threads (7≈8≈parity); (4) vectorizing the eight 6-bit scale/min unpacks
(get_scale_min_k4 ×8) into ~14 NEON ops — bit-exact (verified over 2M random
inputs) but throughput-neutral, so the kernel is *not* scalar-unpack-bound
either. `fp16_to_fp32` is already a hardware `vcvt`; gate/up are already fused
via the pair kernel.

Both the ILP and the scalar-vectorization experiments being neutral points to
the same conclusion: the kernel is **SIMD-throughput-bound at the `vdotq` floor**
(256 4-bit weights / 16 int8-MACs-per-`vdotq` = 16 `vdotq`/super-block, the
hard minimum — llama.cpp's `vec_dot_q4_K_q8_K` has the same floor). The decode
GEMV is therefore already near-optimal for the NEON ISA.

**`i8mm`/SMMLA is not an option on M1.** The one ISA-level lever that could beat
the SDOT floor — `SMMLA` int8 matrix-multiply (~2× `SDOT` throughput) — requires
`FEAT_I8MM` (ARMv8.6). M1/M1 Max report `hw.optional.arm.FEAT_I8MM = 0` (it
arrived with M2). And SMMLA only helps m≥2 anyway, i.e. batched/speculative
decode, not single-token `tg`. Apple AMX (via Accelerate) is fp32/fp16-only, so
using it for a quantized weight would mean dequantizing — fatal for a
bandwidth-bound decode. So on M1 there is no verifiable path past the SDOT floor.

**The 8-bit (Q8_0 W8A8) engine is *slower* on Mac, not faster.** The Q8_0 decode
kernel is much simpler than Q4_K (one fp16 scale + two `SDOT` per 32-block; no
nibble unpack, no 8 sub-scale extractions, no per-dot `vmlaq`), and indeed runs
at ~31 GB/s/core vs Q4_K's ~17 single-threaded. But decode is
**memory-bandwidth-bound**, and Q8_0 stores ~1.06 B/weight vs Q4_K's ~0.56
(~1.9×). Measured on the 1536×262144 lm_head GEMV at 7 threads: Q4_K 2.4 ms
(95 GB/s) vs Q8_0 4.0 ms (108 GB/s) → **Q8_0 is 1.66× slower**. What matters at
the bandwidth ceiling is weights/s = GB/s ÷ bytes-per-weight: Q4_K 169 G/s vs
Q8_0 102 G/s. Fewer bits wins. The W8A8 engine exists for *natively* Q8_0 models
(where you want 8-bit accuracy), not as an accelerator for a Q4_K model — and on
the lower-bandwidth Pi 5 (LPDDR4X) the byte penalty is even worse. **Q4_K is
already the bandwidth-optimal format for CPU decode.**

Reproduce:

```sh
# geist (auto-pins prefill→P-cores, decode→P-cores−1):
GEIST_BENCH_PP=512 GEIST_BENCH_TG=128 OMP_WAIT_POLICY=active \
  bin/$(mk/detect-target.sh)/release/tests/bench_session_throughput model.gguf
# llama.cpp (CPU-only, matched workload):
llama-bench -m model.gguf -ngl 0 -t 8 -p 512 -n 128
```

**Key finding — thread placement dominates on heterogeneous cores.** On Apple
Silicon the efficiency ("E") cores stall a static OpenMP partition: defaulting
to `omp_get_num_procs()` (all 10 cores) gave pp512 ≈ 91 t/s, while pinning to
the 8 performance cores gives ≈ 143. geist now reads `hw.perflevel0.physicalcpu`
and pins prefill to the P-cores and decode to P-cores−1 (decode fires ~210 tiny
matmuls/token and contends when every core is saturated). Override with
`GEIST_PREFILL_THREADS` / `GEIST_DECODE_THREADS`.

To reproduce a head-to-head on *your* hardware with the llama.cpp commit
pinned, see [BENCHMARKING.md](BENCHMARKING.md).

## Quality — MMLU (`make bench-mmlu`)

`tools/eval_mmlu.py` measures MMLU accuracy self-contained: it drives the
`eval_geist` REPL and tokenizes with the model's **own** GGUF tokenizer (no HF
tokenizer, no tokenizer-mismatch), using the standard 5-shot log-likelihood
cloze (score " A"/" B"/" C"/" D" after `Answer:`, take the argmax). Being a
base-completion eval it sidesteps the chat-template parity question entirely.

The harness is verified: on the embedded smoke sample it scores 0/5 at
`--shots 0` (small models collapse to a position bias — always "A") and **5/5**
at `--shots 5`, confirming the scorer is correct and that few-shot is what
matters.

**Measured:** Gemma 4 E2B-it Q4_K_M on a 200-question shuffled cross-subject
sample (50 subjects, 5-shot, seed 1234) scores **MMLU 0.445** (89/200) — well
above the 0.25 random-chance baseline, in line with expectations for a model
this size. Reproduce with `make bench-mmlu` (or `MMLU_LIMIT=0` for the full
set). A representative cross-subject run:

```sh
pip install datasets
make bench-mmlu                 # 200 shuffled questions, 5-shot (seed-fixed)
make bench-mmlu MMLU_LIMIT=0    # full ~14k set
```

*(Run `make bench-mmlu` to record geist's accuracy on your build; the
deterministic `--shuffle` seed makes the sample reproducible.)*

## Batched / serving throughput (decode amortization)

Single-token decode is memory-bandwidth-bound: it streams the whole model per
token. A **batched** forward (m sequences, or m speculative candidates) reads
each weight **once** and computes m token-positions, so the bandwidth cost is
amortized across the batch and the work shifts onto the compute-bound prefill
kernels. Measured throughput of one forward at batch m (M1 Max, Q4_K_M),
`GEIST_BENCH_PP=m GEIST_BENCH_TG=1`:

| batch m | forward t/s | vs m=1 |
| :---: | :---: | :---: |
| 1 (= single-stream decode) | 37 | 1.0× |
| 8  | 94  | 2.6× |
| 16 | 120 | 3.3× |
| 64 | **155** | **4.2×** |

At m≥64 batched decode reaches the compute-bound prefill ceiling (~155 t/s). So
for **serving multiple concurrent streams**, aggregate decode throughput is
~4× single-stream — the right lever when the workload is throughput-bound rather
than single-stream-latency-bound. (Caveat: this is the linear-layer ceiling;
real multi-sequence decode also pays per-sequence attention, which the bench's
single-sequence prefill does not model.)

Quality (perplexity / KL-divergence vs the reference, sampled MMLU/GSM8K) is
likewise documented in [BENCHMARKING.md](BENCHMARKING.md); it needs
the HF tokenizer and datasets and is not part of the hermetic `make` flow.

<!-- BENCH:AUTO -->

| Date | Model | Host | OS | Target/Mode | Threads | Prefill tok/s | Decode tok/s |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 2026-06-10 | gemma4-e2b-Q4_K_M.gguf | MBP-Germar.local/arm64 | Darwin 25.5.0 | mac-omp/release | default | 77.2 | 10.2 |
