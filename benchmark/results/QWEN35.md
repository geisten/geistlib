# geist Benchmarks — qwen35 hybrid family (Qwen3.5 / 3.8)

The qwen35 models are hybrids: 3 of every 4 layers are gated-DeltaNet linear
attention (recurrent conv + delta-rule state), the rest softmax attention.
**Decode** is recurrence-friendly and competitive with llama.cpp; the
original 3–4× **prefill** gap turned out to be the mN linear kernels, not
the recurrence (#287 falsified that attribution) — #288 closed most of it
on Mac by unserializing the dequant+SGEMM prefill path.

> ⚠️ Same measurement rules as [PI5.md](PI5.md): quiesced machine, warmed
> caches, and on the Pi a **thermally settled** board — the per-sweep
> temperatures are logged below. All numbers CPU-only, identical GGUF files
> for both engines.

## Setup

- **Mac:** Apple M-series (8 P-cores used), macOS. geist `make` (clang,
  Accelerate GEMM, OpenMP). Reference: llama.cpp `3fc4e10` (b9820, Homebrew),
  `llama-bench -ngl 0 -t 8` (CPU-only; its Metal numbers are noted separately).
- **Pi 5:** Model B Rev 1.1, 4× Cortex-A76, 4 GB, 64-bit RPi OS. geist
  `make TARGET=pi5 CC=gcc`. Reference: llama.cpp `acd79d6` on-board build (OpenBLAS, `-t 4`).
- **Models:** Qwen3.5-0.8B **Q8_0** (`make fetch-qwen35-model`, SHA-pinned;
  the CI fixture) and Qwen3.5-4B **Q4_0** (unsloth). The 27B (Mac only,
  15 GiB) is Qwen3.8-27B **Q4_0**.
- **geist protocol:** `bench_perf_sweep --gguf <M> --seq-lens ... --decode-n ...
  --warmup ... --repeats ...` — mean over repeats after a discarded warm-up.
  Thread pool + wait policy are set by the backend since #286 — **no env
  variables needed** (that alone was 6.1 → 22.3 t/s on the 4B; see the PR).
- **llama protocol:** `llama-bench -ngl 0 -p 128[,512] -n 16|32 -r 2..3`.

## Mac (M-series, 8 threads, CPU-only) — measured 2026-08-27, geist #295 branch

Full #287 + #288 + #289 + #291 + #295 stack (chunked delta-rule
prefill, parallel dequant+SGEMM prefill, Q6_K x8 lm_head GEMV, Q4_0
x8 decode GEMV, int8 mN GEMM on the x8 layout). Pre-stack reference
(2026-08-24): prefill 0.8B 139.1, 4B 44.1/40.5, 27B 6.6; decode
0.8B 91.8, 4B 22.4, 27B 4.3.

| model | metric | geist | llama.cpp CPU | ratio |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 467.3 | 496.7 | llama 1.06× |
| 0.8B Q8_0 | prefill pp512 | 448.8 | — | |
| 0.8B Q8_0 | **decode** | **99.9** | 77.9 | **geist 1.28×** |
| 4B Q4_0 | prefill pp128 | 114.0 | 142.4 | llama 1.25× |
| 4B Q4_0 | prefill pp512 | 113.3 | 144.9 | llama 1.28× |
| 4B Q4_0 | **decode** | **27.6** | 27.8 | ~parity |
| 27B Q4_0 | prefill pp128 | 18.4 | 26.5 | llama 1.44× |
| 27B Q4_0 | **decode** | **5.7** | 5.8 | ~parity |

Decode is measured after a 128/512-token prefill (real KV + recurrent
state); 4B decode at 512 context is 25.5. RSS includes the packed x8
copies (#289 lm_head + #291 all Q4_0 projections): 4B 5.7 GB,
27B 27.7 GB — `GEIST_Q4_0_X8_GEMV=0` / `GEIST_Q6K_X8_GEMV=0` trade
the speed back for the memory. #291 A/B, same run: 4B decode
27.1 vs 25.4 (+7 %), 27B 5.6 vs 4.9 (+14 %); prefill untouched.

### Transactional speculative-state baseline (#296 PR 2) — 2026-08-28

Before wiring the MTP head, `bench_speculative` was run with the existing
n-gram drafter and `GEIST_SPEC_MIN_L=1`, deliberately forcing the new
DeltaNet checkpoint/restore path. This is a transaction-cost baseline, not
an MTP result. Greedy output equivalence is covered separately by
`test_speculative_primitives_int`, including byte-identical conv and S state
for every accepted prefix (0..4 on 0.8B, 0..2 on 27B).

| model | sequential | transactional spec | speedup | tokens/spec call |
| :-- | ---: | ---: | ---: | ---: |
| 0.8B Q8_0 | 31.7 tok/s | 19.3 tok/s | 0.61× | 4.63 |
| 27B Q4_0 | 1.7 tok/s | 1.1 tok/s | 0.67× | 1.74 |

The forced n-gram path is a loss on both sizes. On 27B, a lazy checkpoint is
about 150 MiB and low draft acceptance compounds that cost. PR 2 therefore
establishes correctness and bounded memory; later MTP PRs must beat this
baseline through higher acceptance and should avoid replay on the common
full-accept path.

### Isolated MTP forward gate (#296 PR 3) — 2026-08-28

PR 3 executes the separately loaded 27B MTP block but deliberately does not
enable it in the speculative engine yet. The real-weight integration test
feeds a deterministic two-row hidden fixture through token/hidden RMSNorm,
`eh_proj`, the gated full-attention + SwiGLU block, shared-head RMSNorm and
the tied output head. Reset/repeat is byte-identical (`tokens=34,4180`), all
10,240 returned hidden values are finite, the MTP KV length advances from 0
to 2, and non-zero sentinels prove the target KV/recurrent/logit state remains
untouched. The 0.8B fixture (which has no MTP block) is the negative
allocation/API gate.

The normal target path was also A/B checked on the mandatory 27B Q4_0 model
(`pp128`, eight decode tokens, two discarded warm-up tokens, two repeats,
eight threads). The short sweep is intentionally a regression smoke test,
not a new headline benchmark:

| revision | prefill | decode | RSS |
| :-- | --: | --: | --: |
| PR 2 base | 13.70 tok/s | 3.61 tok/s | 29,405 MiB |
| PR 3 | 14.69 tok/s | 4.02 tok/s | 29,414 MiB |

There is no target-path slowdown within run-to-run noise; the small RSS delta
comes from the session-sized MTP KV/scratch at this benchmark's short context.
Draft acceptance and end-to-end quality are intentionally not claimed here:
those require PR 4 to feed real target hidden rows and compare the resulting
draft/verify sequence directly with llama.cpp.

### Native MTP speculative decode (#296 PR 4) — 2026-08-28

PR 4 wires the isolated head into the engine. After every target batch, the
raw target hidden rows are shifted by one position and mirrored into the
MTP-only cache. The native drafter starts with the target model's free greedy
seed, recursively proposes up to `k_max` tokens, and leaves provisional cache
writes at the target boundary. Verification then overwrites those positions
with authoritative target hidden rows. Partial accepts restore the MTP hidden
carry/cache boundary together with the DeltaNet transaction and replay only
the accepted prefix.

MTP is opt-in via `GEIST_MTP=1`. Without it, ordinary decode does not execute
the additional block and speculative decode keeps the existing n-gram
fallback. The 27B real-weight test now covers target catch-up, equal target/MTP
cache lengths, a native greedy draft chain, provisional-write rollback, and
reset. The isolated known result remains `tokens=34,4180`,
`max|hidden|=55.2256`; therefore engine synchronization did not change the PR 3
primitive numerically.

Greedy correctness was measured end-to-end on the mandatory 27B Q4_0 model:
30 sequential tokens and 30 MTP-speculative tokens were identical. The run
needed 7 speculative calls (4.29 emitted tokens/call), including partial-accept
rollback. The no-MTP 0.8B path also remains identical and ASan-clean.

The five-prompt `bench_speculative` sweep used 50 tokens per prompt,
`k_max=4`, greedy sampling, CPU NEON, and `GEIST_MTP=1`:

| model | sequential | MTP speculative | speedup | tokens/spec call |
| :-- | ---: | ---: | ---: | ---: |
| 27B Q4_0 | 4.5 tok/s | 3.5 tok/s | 0.78× | 3.38 |

Acceptance is high enough to reduce target invocations, but the current CPU
implementation loses 22% overall because every generated draft token performs
an unbatched MTP block and full tied-vocabulary head. This makes MTP useful as
a correctness-complete experimental path, not a default CPU optimization.
Batching/accelerating the draft head is the next performance gate.

For an external reference, llama.cpp `1fd6dfe` was built CPU-only against the
same 27B Q4_0 file and run greedy with `draft-mtp`, `n_max=4`. On the exact
first benchmark input it decoded 51 tokens at 4.76 tok/s with 40/40 accepted
drafts; geist reached 2.84 tok/s on that prompt with 3.33 emitted tokens/call.
The absolute comparison is conservative for llama.cpp because its MTP tool
maps the same GGUF twice and the local build lacked OpenMP, but it still shows
that geist's bottleneck is draft execution rather than target acceptance.
Quality remains target-exact by construction and by the 30-token equality
test: MTP only proposes tokens; the target model verifies every committed
token.

### Experimental MTP sketch head (#296 PR 5) — 2026-08-28

PR 5 evaluates the existing host i8-sketch output head inside the recursive
single-row MTP draft. It is gated independently by `GEIST_MTP_SPEC_HEAD=1`;
the default remains the dense MTP head. The fast path preserves the target
head's dense/sparse metadata and falls back to the dense projection whenever
the model, backend, or sampling mode is ineligible.

The mandatory 27B Q4_0 test exercises both the dense fallback and enabled
sketch path. Fast drafts are valid and deterministic, and the end-to-end
greedy loop remains target-exact: 30/30 tokens match sequential decoding. On
the repetitive integration prompt it emitted 31 tokens in 8 verification
calls (3.88 tokens/call).

The full five-prompt, 50-token sweep measured both configurations on the same
revision:

| 27B Q4_0 draft head | sequential | speculative | speedup | tokens/spec call |
| :-- | ---: | ---: | ---: | ---: |
| dense (default) | 3.2 tok/s | 2.4 tok/s | 0.76× | 3.38 |
| i8 sketch (opt-in) | 3.4 tok/s | 2.2 tok/s | 0.65× | 3.16 |

This is a measured rejection for the default path. Compared with PR 4's dense
MTP head, acceptance falls from 3.38 to 3.16 tokens/call and the end-to-end
speedup factor from 0.76× to 0.65×. Approximate finalist ranking changes enough
draft tokens to require more expensive target verification, overwhelming the
cheaper head. The default result reproduces PR 4's 0.78×/3.38 within run noise.
The opt-in remains useful for profiling and future sketch tuning, but it is not
enabled automatically; therefore PR 5 introduces no default performance or
quality regression.

### MTP stage profile and retained verification result (#296 PR 6) — 2026-08-28

PR 6 adds the MTP path to the existing `GEIST_PROFILE_FORWARD=1` diagnostic.
On the mandatory 27B Q4_0 model, 102 MTP input rows from the integration loop
split the measured MTP time as follows:

| MTP stage | time | share |
| :-- | --: | --: |
| gated attention + SwiGLU block | 423.59 ms | 46.2% |
| tied dense vocabulary head | 322.50 ms | 35.2% |
| `eh_proj` | 168.31 ms | 18.4% |
| input, norms, concat, copies | 1.54 ms | 0.2% |

The profile rules out host-side input preparation as the next useful target:
81.4% is in the MTP block and tied head, with another 18.4% in the input
projection. A material CPU speedup therefore needs cheaper/batched model math,
not another memcpy-level optimization.

Verification also already computes the correction/bonus token and its logits.
The transformer primitives now retain that exact pending result across a full
accept, or reconstruct it while replaying a partially accepted DeltaNet
prefix. `GEIST_SPEC_RETAIN_PENDING=1` lets the engine defer emitting that token
until the next call, avoiding the old immediate single-token correction
prefill. The switch is deliberately experimental: changing the call boundary
also changes the n-gram history available to the drafter.

Both modes are greedy-exact on the 0.8B and mandatory 27B fixtures (30/30 tokens
match sequential decode), and the rollback test covers every accepted prefix.
The 0.8B paths are ASan-clean. The controlled 27B five-prompt sweep was:

| 27B Q4_0 cadence | sequential | MTP speculative | speedup | tokens/spec call |
| :-- | ---: | ---: | ---: | ---: |
| established/default | 3.7 tok/s | 2.5 tok/s | 0.67× | 3.38 |
| retain pending (opt-in) | 2.5 tok/s | 1.3 tok/s | 0.54× | 2.60 |
| default after `origin/main` `5cf2b10` | 4.0 tok/s | 2.9 tok/s | 0.72× | 3.38 |

Absolute rates varied with machine state versus PR 5, but the within-run
result is unambiguous. Saving the redundant prefill does not compensate for
the lower n-gram acceptance caused by deferring the correction token. The
default cadence is therefore unchanged. llama.cpp's earlier controlled
4.76 tok/s result remains ahead of both geist modes; this PR narrows the next
optimization target to the three model-math stages above rather than token
bookkeeping.

The post-merge isolated real-weight check also exercises Main's untied
`output.weight` in the MTP head. It is reset/repeat deterministic at
`tokens=34,375`, `max|hidden|=67.5937`, preserves the target recurrent
sentinels, and leaves target KV/logits untouched.

(For the GPU story — geist Metal now ahead of llama.cpp Metal on the
27B — see the Metal section below; the CPU numbers here stand on their
own.)

**Reading:** the 0.8B beats llama.cpp on decode by 1.28× at prefill
parity, and the 4B/27B reach decode parity (0.96–0.97×) with #291's
interleaved Q4_0 GEMV (kernel-level +61–72 % GB/s, end-to-end +7–14 %
— the rest of the budget is the Q6_K head near its compute bound and
the DN recurrence). Prefill went
from 3–4× behind to 1.06–2× via two findings: chunking the delta-rule
recurrence (#287) moved it only ~0–7 % in isolation — falsifying the
original recurrence attribution — while the real culprit was the
*serial* Mac dequant+SGEMM tile loop, fixed in #288 (the chunked
recurrence became load-bearing once #288 landed: both together give
the 0.8B its 467 t/s). #289 then took the 4B/27B lm_head (Q6_K,
a third of 4B decode) from 33 to ~45 GB/s. Remaining levers, in
diminishing order: mN quantized-GEMM throughput on the 4B/27B
prefill (per-tile dequant+SGEMM vs llama.cpp's repacked int8 GEMM)
and Q4_0 m1 GEMV interleaving for the last ~20 % of large-model
decode.

### Chunked delta-rule prefill (#287) — measured 2026-08-25, pre-#288

A/B on the same build, `GEIST_DN_SEQ_PREFILL=1` forcing the sequential
path (Mac, 8 threads, load < 2 at start):

| model | metric | chunked | sequential | Δ |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 145.3 | 138.3 | +5 % |
| 0.8B Q8_0 | prefill pp512 | 146.6 | 136.6 | +7 % |
| 4B Q4_0 | prefill pp128 | 40.0 | 42.3 | −5 % (noisy run) |
| 4B Q4_0 | prefill pp512 | 42.8 | 41.8 | +2.5 % |

The chunked recurrence is kept: correct (pinned by
test_deltanet_chunk_unit at f32 precision), never slower outside
noise, and load-bearing now that #288 stopped the mN kernels from
dominating prefill time.

## Mac Metal backend (M1 Max GPU) — measured 2026-08-28, main (post #309/#312)

The #300–#312 stack: simdgroup GEMV/GEMM kernels for
Q4_0/Q4_1/Q8_0/Q5_K/IQ4_XS/IQ4_NL/Q3_K/IQ3_S (llama mul_mv / mul_mm
structures), chunked DeltaNet prefill (port of the CPU chunk recipe;
its serial predecessor was 72 % of prefill wall), and the IQ4 loader
that makes the UD mixed quant run at all. Starting point on this
branch was 5.3 pp / 2.7 tg on the 27B (correctness-first kernels).

**Protocol:** cool state = 240 s GPU cooldown **and** a resident model —
the two fight each other when several 15 GiB files rotate through the
page cache, so every run is preceded by a full `cat model > /dev/null`
pre-touch. An eviction shows up as RSS below the model size and drags
both engines equally (measured: UD at 35 pp with 9 GB resident vs 79 pp
at 33 GB). `bench_perf_sweep` pp512/tg64, mean of 2 repeats after a
discarded 64-token warmup. Reference: llama.cpp Metal `3fc4e10`
(b9820, Homebrew), same GGUF, back-to-back on the same protocol.

| model | metric | geist Metal | llama.cpp Metal | ratio |
| :-- | :-- | ---: | ---: | :-- |
| 27B Q4_0 | prefill pp512 | **104.4** | 93.1 ±15 | **geist 1.12×** |
| 27B Q4_0 | **decode tg64** | **11.6** | 8.2 | **geist 1.41×** |
| 27B UD-Q4_K_M | prefill pp512 | 94.2 | 105.2 ±7 | llama 1.12× |
| 27B UD-Q4_K_M | **decode tg64** | 7.5–8.4 | 8.1 | ~parity |
| 4B Q4_0 | prefill pp512 | 733.8 | 926 (warm) | llama 1.26× |
| 4B Q4_0 | decode tg64 | 52.3 | 61.8 (warm) | llama 1.18× |
| gemma4-e2b Q4_K_M | prefill pp512 | 992.4 | 1540 (2026-07 ref) | llama 1.55× |
| gemma4-e2b Q4_K_M | decode tg64 | 79.3 | 92.8 (2026-07 ref) | llama 1.17× |

**Reading:** the 27B — the model this stack is for — now beats
llama.cpp Metal on **both** axes (1.12× prefill, 1.41× decode). The
UD mixed quant reaches decode parity; its remaining 1.12× prefill
gap spreads across the K-quant + IQ4 mm kernels (IQ4_XS is only
29 % of the UD's weight bytes) — all on the shared mm_sg template
at its ~6–7 TF plateau, so the recoverable share is small (#323).
The 4B row is the #322 program end state (626 → 734 via DN
sub-chunking #340, batched embed #345 and the fused silu_mul #347;
the remaining gap is the same GEMM plateau plus a
dependency-chained small-dispatch floor — analysis and priced-out
options in the issue). `GEIST_M_MAX` 64-vs-256 is a wash on the
27B — the #340 occupancy win is a small-model effect. gemma4
numbers are the old 2026-07 program state restored (the PLE probe
regression had silently zeroed them) on the new kernel stack.

Attribution highlights, for whoever continues: prefill wall = max of
overlapping GPU chains, not their sum — single-category skips of the
subtractive profiler show nothing, combinations do
(`GEIST_METAL_PROFILE=1`, repeats=1 only). The whole 1.4 s DeltaNet
chain on the 4B was 2·C `mem_device` threadgroup barriers at ~54 µs
each, not compute. Half-staged GEMM weights make unpinned-scale parity
noise of √n·ulp size — the test pins are deliberate.

## Raspberry Pi 5 (4 GB, quiesced, thermally gated) — measured 2026-08-24

geist `9b36c1d`, `TARGET=pi5`, gcc 14.2. Sweep temperatures: 0.8B started
at 48.5 °C, ended 60.6 °C; the 4B sweep started after cooling to 51.3 °C,
ended 65.0 °C (soft limit not reached; low throttle nibble 0 throughout).

| model | metric | geist | llama.cpp (on-board) | notes |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 61.8 | 90.6 | RSS 1.0 GB |
| 0.8B Q8_0 | prefill pp256 | 62.3 | — | |
| 0.8B Q8_0 | **decode** | **13.0** | 9.3 | |
| 0.8B IQ4_XS | prefill pp256 | 62.3 | — | measured 2026-08-28, #314 |
| 0.8B IQ4_XS | **decode** | **18.5** | — | was 4.4 via the dequant trampoline |
| 4B Q4_0 | prefill pp128 | 8.2 | 24.8 | RSS 2.9 GB — fits the 4 GB board |
| 4B Q4_0 | **decode** | 3.8 | 3.3 | |

**Reading:** on the design target, **geist wins decode on both models** —
0.8B 13.0 vs 9.3 t/s (**1.39×**), 4B 3.8 vs 3.3 t/s (**1.14×**) — the same
pattern as the BitNet results: specialized int8 kernels plus no per-layer
graph overhead. Prefill stays llama's (1.5× / 3.0×); per the #287
findings above, closing it is an mN-kernel problem, not a recurrence
problem (Pi A/B of the chunked prefill pending — the balance may
differ on 4 cores without Accelerate). The 0.8B hybrid is fully usable on the Pi
(13 t/s decode Q8_0, 18.5 t/s IQ4_XS via the #314 native W4A8 GEMVs);
the 4B fits the 4 GB board and runs at reading speed. The 27B (15 GiB)
is not attempted on the Pi.

### Pi 5 4B six-way A/B (2026-08-29, main `93ef84f`) — the x8 policy flip

pp128/pp256 + 16-token decode, 3 repeats, thermally gated 46–57 °C;
"3.8-4B" is the community `empero-ai/Qwen3.8-4B-Distill` (qwen35 arch,
32+1 NextN blocks — the extra block is never traversed, and it measures
identically to the 3.5-4B). llama.cpp on-board reference: 24.8 pp / 3.3 tg.

| model | variant | prefill pp256 | decode | RSS |
| :-- | :-- | --: | --: | --: |
| 3.5-4B | Q4_0 default (pre-flip) | 9.1 | 4.0 | 2.84 GB |
| 3.5-4B | Q4_0 + x8 int8 GEMM | **16.5** | 4.1 | 3.44 GB |
| 3.5-4B | IQ4_XS | 8.7 | 3.9 | 2.96 GB |
| 3.8-4B | Q4_0 default (pre-flip) | 9.2 | 4.2 | 2.94 GB |
| 3.8-4B | Q4_0 + x8 int8 GEMM | **18.4** | 4.3 | 3.34 GB |
| 3.8-4B | IQ4_XS | 8.6 | 4.1 | 3.00 GB |

Post-A/B addendum (2026-08-29, #321): the IQ4_XS mN trampoline is
replaced by a native 4-token tile kernel — 4B IQ4_XS prefill 8.6 →
15.3 t/s, 0.8B 62.3 → 86.3, decode/RSS unchanged — so finding (2)'s
"IQ4 prefill rides the trampoline" no longer holds; the bpw argument
for decode still does.

Findings: (1) the #295 x8 int8 GEMM doubles Pi prefill and costs only
~0.5 GB net RSS — the cold Q4_0 mmap pages get evicted, so the feared
+1× packed-copy residency never materializes; it is default-on wherever
SDOT exists since this A/B (`GEIST_Q4_0_X8_GEMV=0` opts out). (2)
IQ4_XS buys the 4B nothing: Q4_0 and IQ4_XS are both ~4.25 bpw, so the
bandwidth-bound decode ties, and IQ4 prefill still rides the dequant
trampoline — the 0.8B's 13→18.5 decode jump came from the Q8_0→IQ4
byte halving, not the format. (3) The remaining ~1.35× prefill gap to
llama.cpp was the per-(row,token) `vaddvq` horizontal adds in the x8
inner loop vs llama's lane-SDOT accumulation. Both halves of that
lever are executed: in-kernel transposes + `vdotq_laneq_s32` (#318,
16.5 → 19.4 t/s), then the v2 layout that bakes the transpose into
the pack and migrates the m1 decode GEMV to lane-SDOT as well (#324,
19.4 → **21.1 t/s**, decode 4.2 → 4.36). Every step was gated
bit-identical on-board (logits `cmp` + greedy-generation diff across
builds). End state: 1.17× behind llama.cpp's 24.8 on prefill, ahead
1.32× on decode.

Quality gate (2026-08-28, on-board): the `bench_quality` battery ran
for every Pi-resident model (qwen3.5 0.8B Q8_0/IQ4_XS, 4B Q4_0,
qwen3-0.6B, gemma4-e2b, bitnet-2b4t) with coherent greedy output on
all of them, and the IQ4_XS NEON kernel matches the f32 scalar
reference to top-4 logit rank (top-1 identical; the residual is
gcc `-ffast-math` reassociation, not the kernel).

## Reproduce

```sh
# Mac
make && make fetch-qwen35-model
bin/mac-omp/release/tests/bench_perf_sweep \
  --gguf gguf_artifacts/qwen3.5-0.8b-q8_0.gguf \
  --seq-lens 128 --decode-n 32 --warmup 16 --repeats 5
llama-bench -m gguf_artifacts/qwen3.5-0.8b-q8_0.gguf -ngl 0 -t 8 -p 128 -n 32 -r 3

# Pi 5 (quiesced, < 60 °C between engines; see PI5.md)
make TARGET=pi5 CC=gcc && make fetch-qwen35-model
bin/pi5/release/tests/bench_perf_sweep \
  --gguf gguf_artifacts/qwen3.5-0.8b-q8_0.gguf \
  --seq-lens 128,256 --decode-n 16 --warmup 8 --repeats 5
```
