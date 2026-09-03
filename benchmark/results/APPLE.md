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

Perf is measured by `bench_perf_sweep`, driven through
`tools/bench_quality_perf.py`. The machine-readable source of truth is
[`benchmark/apple_cpu_protocol.json`](../apple_cpu_protocol.json); the driver
loads that file rather than duplicating these constants:

- **Model:** Gemma 4 E2B-it, Q4_K_M (`make fetch-model`).
- **Workload** — the two suites in `SWEEP_WORKLOAD`
  (`tools/bench_quality_perf.py`), which is the single source of truth:

  | suite | prefill | decode | warm-up | repeats |
  | :-- | ---: | ---: | ---: | ---: |
  | `small` (`make bench-small`) | 128 tok | 32 tok | 8 tok | 3 |
  | `detailed` (`make bench-detailed`) | 512 tok | 64 tok | 64 tok | 10 |

- **Warm-up:** discarded, not measured — it pages the weights resident,
  resolves the backend kernels and spins up the OMP pool, so the repeats
  measure steady state rather than cold start.
- **Aggregation:** the sweep reports the **mean** over the repeats as the
  headline `*_tps`, and carries `*_best` / `*_worst` in the same record. The
  recorded table below is the mean; the comparison tables further down quote
  the **best**, for the reason in the methodology note.

| suite | prefill sequence | decode | discarded warm-up | measured repeats | aggregation |
| :-- | --: | --: | --: | --: | :-- |
| `small` | 128 | 32 | 8 | 3 | mean |
| `detailed` | 512 | 64 | 64 | 10 | mean |

The model is Gemma 4 E2B-it Q4_K_M (`make fetch-model`), SHA-256
`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`.
The discarded warm-up pages weights resident and starts the OpenMP pool.
`bench_perf_sweep` retains every measured sample in execution order, then adds
mean/best/worst summaries. The wrapper writes a timestamped raw JSONL envelope
with the full geist commit, model and binary SHA-256, compiler, linked libraries,
host/OS, thread controls, exact protocol and diagnostics. Runs on the same day
therefore no longer overwrite one another.

```sh
make fetch-model                         # one-time, ~3.1 GB
OMP_WAIT_POLICY=active make bench-small  # records a row below
OMP_WAIT_POLICY=active make bench-detailed
```

**Do not set `BENCH_THREADS` on Apple silicon.** Left unset, the backend sizes
the OMP pool to the performance-core count (`hw.perflevel0.physicalcpu`, 8 on
an M1 Max) — an explicit value always wins, and every other value is worse.
OMP's own default is `num_procs`, which includes the two efficiency cores, and
a static partition then waits on the E-core chunks; the sizing comment in
`src/backends/cpu_neon/backend.c` records that one as pp512 91 tps on 10 cores
vs 145 on the 8 P-cores. Fewer threads than P-cores is worse *and* noisier —
measured 2026-08-31 on this M1 Max, three back-to-back runs of one binary at
`--seq-lens 512 --decode-n 32 --warmup 64 --repeats 5`:

| `OMP_NUM_THREADS` | pp512 best-of-5, three runs | spread |
| ---: | :-- | ---: |
| 8 (= P-cores, the default) | 140.7 / 141.9 / 140.1 | **1.3 %** |
| 6 | 141.1 / 122.4 / 126.4 | 15 % |
| 4 | 126.7 / 98.0 / 99.3 | 29 % |

Below the P-core count the desktop's own load lands on the cores geist is not
using *and* on the ones it is, so the spread grows faster than the mean falls.
A number quoted without its thread count is not comparable to anything.

For code-revision comparisons, use the stricter interleaved protocol:

```sh
python3 tools/bench_mac_ab.py \
  --gguf gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \
  --variant parent=/path/to/parent/bench_perf_sweep \
  --variant candidate=/path/to/candidate/bench_perf_sweep
```

It runs three order-rotated cycles over sequences 128/256/512/1024, with a
64-token warm-up, 64 decode tokens and three ordered samples per cycle. Before
every run it requires 30 continuously quiet seconds below load 0.2/core, then a
60-second cooldown. Results use the median and median absolute deviation over
all nine samples; a repeatable regression beyond 5% fails the command. Raw
JSONL and the Markdown report are written outside the checkout under
`~/bench-geistlib/apple-ab/`. Every run records `pmset` thermal warnings and
power source both immediately before and after measurement. Package temperature
is recorded as unavailable when macOS does not expose it without privilege.
Interrupted runs retain a `.jsonl.partial` suffix and cannot be mistaken for a
complete baseline. Resume a long campaign without discarding already accepted
runs by repeating the identical model and `--variant` arguments and adding
`--resume /path/to/run.jsonl.partial`; the runner rejects a changed binary hash,
model, protocol, environment, baseline or non-prefix run order.

To prove which resolved CPU weight paths actually execute, build the opt-in
profile (the release build compiles the counters out completely):

```sh
make MODE=perf EXTRA_CFLAGS=-DGEIST_PROFILE_WEIGHT_PATHS \
  BACKENDS="cpu_neon cpu_scalar" bin/mac-omp/perf/tests/bench_perf_sweep
bin/mac-omp/perf/tests/bench_perf_sweep --gguf model.gguf \
  --seq-lens 128 --decode-n 8 --warmup 8 --repeats 1
```

The once-per-process `[weight-paths]` JSON reports M=1 and M>1 calls for every
dtype, including explicit zeroes. The scheduled
[`apple-perf.yml`](../../.github/workflows/apple-perf.yml) job runs the detailed
suite weekly on Apple hardware, uploads raw provenance and enforces conservative
60 prefill / 10 decode tok/s floors. Those are cliff guards for scalar fallback,
`-O0` or lost parallelism; the interleaved local run remains the arbiter for
single-digit drift.

### Q4_0 regression attribution

An opt-in path-profile run of the pinned Gemma model at pp128/tg8 on commit
`02cb4fc` counted `q4_K` m1=2186/mN=543, `q6_K` m1=288/mN=72 and F32
m1=852/mN=213. Every Q4_0 counter was exactly zero. The recent Q4_0 x8 changes
therefore cannot directly affect this model's linear kernels; only a shared
runtime/workspace change or host interference could explain a measured delta.
The profiler is compiled out of release builds: the release `linear.o` before
and after instrumentation was byte-identical (SHA-256
`3395ffabdb746e1ab33d58619963a59c6a35b811c0c2b5e447d4acab73a3152c`).
Throughput from the profiling run is intentionally not used as a baseline.

`OMP_WAIT_POLICY=active` (or `KMP_BLOCKTIME=infinite`) matters on the `mac-omp`
target — passive wait adds large per-matmul thread-pool wake-up overhead. The
backend sets it itself at create time (`setenv` with `overwrite=0`, so an
explicit policy still wins); the command lines above state it because the
recorded rows should say what they ran under. Thread *counts* are handled
automatically: geist sizes the prefill pool to the performance-core count and
the decode pool to P-cores−1 (see the comparison below). This is a thread-count
policy, not CPU-ID affinity; macOS still schedules the workers.

**Which kernels a number actually measures.** `GEIST_LOG_KERNELS=1` makes the
cpu_neon resolver count its decisions and print them per catalog row at
backend destroy — so "does this model even run the code I changed?" is
answered by the binary rather than by reading the GGUF's dtype histogram:

```
$ GEIST_LOG_KERNELS=1 bin/mac-omp/release/tests/bench_perf_sweep ... 2>&1 >/dev/null
[geist] resolved kernels (tensors per catalog row):
[geist]   q4_K                        182
[geist]   q6_K                         24
[geist]   f32                          71
```

The reference model is Q4_K/Q6_K/F32 and nothing else. A change confined to,
say, the Q4_0 paths cannot move this number, however suggestive the timings
look — check here before attributing a swing to a commit (#327).

To match an external reference workload (e.g. `llama-bench -p 512 -n 128`),
drive the sweep binary directly — it takes the shape on the command line:

```sh
GEIST_GGUF_PATH=... OMP_WAIT_POLICY=active \
  bin/mac-omp/release/tests/bench_perf_sweep \
  --seq-lens 512 --decode-n 128 --warmup 64 --repeats 10
```

### A/B-ing two commits on a desktop that cannot be quiesced

Comparing two builds here needs a **paired** design, not two campaigns: the
absolute level moves ~27 % between windows on this machine (WindowServer, a
browser, an indexer), which is five times the effect sizes anyone is chasing.
Run the two binaries 15 s apart with the order alternating per pair, then read
the **within-pair ratio** — the drift is common to both members and cancels.
Freeze the binaries outside the build tree first; a `make` during the series
silently swaps one side.

Worked example — `c1e74ad` (before the Q4_0 x8 series) vs `0792e55` (main),
2026-08-31, M1 Max / Darwin 25.5.0 (macOS 26.5.1) / Apple clang 21.0.0 /
libomp 22.1.8,
`OMP_NUM_THREADS=8`, `OMP_WAIT_POLICY=active`, gemma4-e2b-Q4_K_M
(`9378bc47…`), 8 pairs at `--seq-lens 512 --decode-n 32 --warmup 64
--repeats 5`:

| | median paired Δ | per-pair range | absolute level swing |
| :-- | ---: | :-- | ---: |
| prefill pp512 | **−1.1 %** | −10.6 % … +12.1 % | 101.7 … 129.1 (27 %) |
| decode tg32 | **−1.0 %** | −11.5 % … +28.9 % | 16.4 … 22.9 (39 %) |

Both directions appear (3 of 8 pairs positive on each axis; sign test
p = 0.73), so the two trees are indistinguishable here. Note what that costs:
16 runs over ~25 minutes resolve nothing finer than about ±5 %. **A number
below that threshold cannot be established on this machine at all** — take it
to the quiesced Pi 5 ([PI5.md](PI5.md)), or expect the answer to be "noise".

What this does *not* explain is why the June comparison table below reads
pp512 150 while a calm window on 2026-08-31 topped out at 140.7. The paired
series rules out the code between `c1e74ad` and main as the cause, and the OS
is not it either (the recorded row's `Darwin 25.5.0` is still this host's
kernel release — the macOS *product* version moved to 26.x on the same
Darwin major, which is a different numbering, not a different kernel). The
residual is either machine state that the tags do not capture, or a change
older than `c1e74ad`. Either way it is not a regression this ticket can
attribute, and a June best-of number is not a baseline anyone can re-derive.

## Comparison vs llama.cpp (Apple M1 Max, CPU, measured June 2026)

Same machine, same `gemma4-e2b-Q4_K_M.gguf`, both CPU-only. llama.cpp build
`d05fe1d` run with `-ngl 0` (BLAS/Accelerate, no GPU offload). geist uses an
8-thread prefill pool; llama runs at `-t 8`. Neither engine is given explicit
CPU-ID affinity. Full prefill sweep 128 → 1024 tokens,
**best-of** (peak uncontended throughput — see the methodology note below):

This is a frozen historical cross-engine campaign, not output from the current
`small`/`detailed` suites above. Its explicit best-of-10 aggregation remains
documented so the numbers are not silently reinterpreted; new revision A/B
decisions use the interleaved median/MAD protocol.

| seq_len | llama.cpp `-ngl 0`, t=8 | geist (8-thread pool) | winner |
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
([PI5.md](PI5.md)), where geist edges llama 1.03×.

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
# geist (auto-sizes prefill→P-core count, decode→P-core count−1):
OMP_WAIT_POLICY=active \
  bin/$(mk/detect-target.sh)/release/tests/bench_perf_sweep \
    --gguf model.gguf --seq-lens 512 --decode-n 128 --warmup 64 --repeats 10 --emit-jsonl
# llama.cpp (CPU-only, matched workload):
llama-bench -m model.gguf -ngl 0 -t 8 -p 512 -n 128
```

**Key finding — thread-pool size dominates on heterogeneous cores.** On Apple
Silicon the efficiency ("E") cores stall a static OpenMP partition: defaulting
to `omp_get_num_procs()` (all 10 cores) gave pp512 ≈ 91 t/s, while limiting the
pool to 8 threads gives ≈ 143. geist now reads `hw.perflevel0.physicalcpu` and
uses that count for prefill and one fewer thread for decode (decode fires ~210
tiny matmuls/token and contends when every core is saturated). This does not set
CPU affinity. Override the counts with
`GEIST_PREFILL_THREADS` / `GEIST_DECODE_THREADS`.

To reproduce a head-to-head on *your* hardware with the llama.cpp commit
pinned, see [../METHODOLOGY.md](../METHODOLOGY.md).

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
likewise documented in [../METHODOLOGY.md](../METHODOLOGY.md); it needs
the HF tokenizer and datasets and is not part of the hermetic `make` flow.

<!-- BENCH:AUTO -->

| Date | Model | Host | OS | Target/Mode | Threads | Prefill tok/s | Decode tok/s | Spread | TTFT ms | Commit | Model SHA256 | Suite |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2026-06-10 | gemma4-e2b-Q4_K_M.gguf | MBP-Germar.local/arm64 | Darwin 25.5.0 | mac-omp/release | default | 77.2 | 10.2 | — | — | — | — | — |

---

## Apple M1 Max (8 P-cores) — prefill (tokens/s, higher is better)

| seq_len | llama.cpp `-ngl 0` | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | 141 | **164** | **geist 1.16×** |
|  256 | 147 | **161** | **geist 1.10×** |
|  512 | 128 | **150** | **geist 1.17×** |
| 1024 |  97 | **144** | **geist 1.48×** |

```
prefill t/s   (each █ ≈ 10 t/s)            geist stays flat ·· llama drops off
 geist  128 ████████████████ 164    llama  128 ██████████████ 141
        256 ████████████████ 161           256 ███████████████ 147
        512 ███████████████ 150            512 █████████████ 128
       1024 ██████████████ 144            1024 ██████████ 97
```

**Decode:** ≈ par (~26 t/s both). On a live desktop the 16-token decode window is
too jitter-prone to rank — see the Pi for the controlled decode result.

geist's dense-fp32 path here is **Accelerate/AMX** (Apple's matrix coprocessor),
which holds ~flat from 128 → 1024 tokens (164 → 144), while llama.cpp's CPU-only
path degrades sharply past 256 (147 → 97).
