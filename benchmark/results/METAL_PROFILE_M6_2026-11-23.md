# Where geist's Metal time goes against mlx-lm's on the Mac mini M6 (2026-11-23)

Follow-up to [`MLX_VS_GEIST_M6_2026-11-22.md`](MLX_VS_GEIST_M6_2026-11-22.md),
which measured mlx-lm 1.2–1.4× faster in prefill and 1.1–1.3× in decode and
profiled nothing (#590). Two pairs were profiled here on the same machine
(Apple M6 Mac mini, Mac18,5, 12 cores, 32 GB, macOS 27.2; runner
`macmini-m6`): Qwen3.5 0.8B at 8 bit, where both engines hold the same
amount of weight data and generate the same tokens, and Llama 3.2 3B at
4 bit. Workload as before: 512 prompt tokens, 128 greedy tokens.

**Result.**

- **Decode is limited by memory on both engines, and geist's GPU sits idle
  between steps.** On the 8-bit pair a decode step costs geist 6.32 ms of GPU
  time and mlx-lm 6.24 ms (1.3 % apart); geist's step takes 6.60 ms because
  the GPU has no command buffer for 0.28 ms of it, mlx-lm's 6.25 ms because
  it queues the next step before the current one ends. That idle time is the
  same 0.26–0.29 ms per step on the 3B model, where it is 2 % of a step.
- **The rest of the 4-bit decode gap is weight bytes.** Llama 3.2 3B
  Q4_K_M holds 12 % more bytes than the MLX 4-bit file and spends 13.6 %
  more GPU time per step; 0.81 ms of the 1.76 ms is the output head, which
  Q4_K_M keeps at 6.56 bits per weight.
- **Prefill is GPU-bound in geist and the GPU work is 1.45–1.47× mlx-lm's.**
  The 512 tokens take geist 71.0 ms (Qwen) and 249.5 ms (Llama) of GPU time
  with no gaps; mlx-lm's batched pass over 511 tokens takes 48.3 ms and
  171.6 ms. mlx-lm then spends 12–22 ms of its own on the host and on a
  single-token step before the first token, which is why the end-to-end
  ratio is smaller than the kernels'. Where geist's extra GPU time sits is
  known only in part: the chunked DeltaNet (Qwen), the serial order of the
  dispatches, and GEMM kernels that are slower than mlx-lm's on the 4-bit
  pair. The rows per pass, the first suspect, are not it.
- **#590's changes** (concurrent encoders with barriers only where a
  dispatch depends on an earlier one; a decode step's first command buffer
  committed early) bring the 8-bit pair from 152.4 to 156.7 tok/s in decode
  and from 7151 to 7367 tok/s in prefill, with the same logits bit for bit;
  mlx-lm decodes at 160.4. Details under [Levers](#levers).

Every geist number in the profile is of the engine as it was before #590's
changes (`GEIST_METAL_CONCURRENT=0 GEIST_METAL_PIPELINE_FIRST=0`, the same
command stream as `main`).

## How it was measured

`tools/probes/metal_trace.m` is a small library that replaces the
implementations of a handful of methods on the Metal classes of the default
device: `commit`, `waitUntilCompleted`, the encoder constructors,
`setComputePipelineState:`, the dispatch calls, the memory barriers and
`newComputePipelineStateWith…`. For every command buffer it records the
commit time, `GPUStartTime` and `GPUEndTime`, the time a thread spent waiting
on it, and its dispatches per kernel (the `MTLFunction` name of the bound
pipeline). It loads into geist with `DYLD_INSERT_LIBRARIES` and into mlx-lm
with `ctypes` before `import mlx`, so both engines are measured by the same
instrument; `bench_perf_sweep` and the mlx-lm runner stamp the start of a
run, the first token and the end into the trace. The traced rates equal the
untraced ones (geist 7189 against 7201 tok/s prefill and 152.9 against 152.9
decode; mlx-lm 7265 against 7278 and 159.9 against 160.1).

What is **measured**:

- wall time of a phase, and the process's CPU time in it (`getrusage`);
- GPU time: the union of the command buffers' `GPUStartTime`–`GPUEndTime`
  intervals ("GPU busy"), and the wall time not covered by it ("no buffer on
  the GPU");
- command buffers, dispatches and memory barriers per phase, dispatches per
  kernel;
- for geist's decode, where the idle part of a step goes (its buffers carry
  the wait).

What is **inferred**:

- GPU time per kernel. Metal reports times per command buffer, not per
  dispatch, so the per-kernel shares come from separate runs with one op per
  command buffer (`GEIST_METAL_PIPELINE=1 GEIST_METAL_PIPELINE_PREFILL=1`;
  `MLX_MAX_OPS_PER_BUFFER=1`, which still puts two dispatches in most
  buffers). Such runs are slower than the default ones in decode (geist 147.5
  against 152.9 tok/s, mlx-lm 139.7 against 160.1) because every buffer adds
  a fixed cost, which inflates the small kernels. The shares say which class
  dominates; they are not a budget of the default run.
- That a decode step is memory-bound, from the equal GPU time of the two
  engines on equal weight bytes, and from the 4-bit pair's GPU time following
  its byte ratio.

What could **not** be had for mlx-lm: the GPU time of a single kernel in its
default mode, and the split of its host time (graph construction, encoding,
Python). Its CPU time is a total.

## Decode

Per forward step, microseconds, mean of three repeats of 127 steps (run
4092). mlx-lm's window holds 128 steps of GPU work, since the step after the
last yielded token is already queued; its step time is the one its token
stamps give.

| model | engine | step | GPU busy | no buffer on the GPU | CPU | command buffers | dispatches | barriers |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | geist | 6596 | 6320 | 275 | 285 | 2 | 361 | – (serial encoder) |
| | mlx-lm | 6252 | 6237 | 15 | 5279 | 17 | 618 | 396 |
| Llama 3.2 3B, 4-bit | geist | 14986 | 14701 | 285 | 266 | 2 | 260 | – (serial encoder) |
| | mlx-lm | 12962 | 12938 | 24 | 2076 | 14 | 488 | 332 |

- **GPU time per step is the same on the 8-bit pair** (6320 against 6237 µs,
  1.3 %), although mlx-lm runs 1.7× as many dispatches. Both files hold
  8.5 bits per weight. The step is a walk over the weights.
- **geist's GPU is idle 275–285 µs per step**, mlx-lm's 15–24 µs. geist
  waits for a step, reads the token, encodes the next step and commits it;
  mlx-lm builds and submits the next step while the current one runs
  (`mx.async_eval`), so its buffers wait in the queue for 3–10 ms and the
  GPU never runs dry. It pays for that with a busy core: 5.3 ms of CPU per
  6.3 ms step on the small model, against geist's 0.29 ms.
- geist's idle time per step splits into (medians): from the GPU finishing
  to `waitUntilCompleted` returning, 72 µs (Qwen) and 69 µs (Llama); from
  there to the first commit of the next step, i.e. reading the token and
  encoding the first 192 dispatches, 93 and 105 µs; from that commit to the
  GPU starting, 113 and 111 µs.
- **On Llama 3.2 3B the GPU time differs by 1763 µs per step (13.6 %).**
  The GGUF is 2.02 GB at 5.03 bits per weight, the MLX file 1.81 GB at 4.50
  (ratio 1.12). In the one-op runs the output head alone takes 2239 µs in
  geist (`matvec_q6k_n4`: Q4_K_M stores `token_embd` as Q6_K, 6.56 bits per
  weight) and 1427 µs in mlx-lm, a difference of 812 µs; the remaining
  ~950 µs is 8 % on the body, whose bytes differ by about 7 %. On the 8-bit
  pair the heads cost the same, 1757 and 1718 µs.

GPU time by kernel class, one op per command buffer (inferred shares, see
above):

| class | Qwen geist | Qwen mlx-lm | Llama geist | Llama mlx-lm |
| :-- | --: | --: | --: | --: |
| quantized matmul, with the output head | 90.4 % | 61.3 % | 94.2 % | 67.0 % |
| of it the output head | 23.5 % | 25.7 % | 15.0 % | 10.8 % |
| DeltaNet mixer | 5.1 % | 5.3 % | – | – |
| attention (with RoPE, KV append) | 1.4 % | 2.8 % | 4.2 % | 8.1 % |
| norms | 1.6 % | 11.0 % | 1.5 % | 6.5 % |
| elementwise and copies | 1.3 % | 18.3 % | 0.0 % | 18.1 % |
| sampling (argmax, logsumexp) | 0.2 % | 1.0 % | 0.0 % | 0.3 % |

geist's kernels are fused further (a gate, an up projection and the SiLU in
one dispatch; q, k and v in one), so its time sits in the matmuls; mlx-lm's
many small kernels each pay the per-buffer cost of this mode. In geist the
lm head is one `matvec` per step and the readback of the token is not a
kernel: it is part of the 93–105 µs above.

GPU state over a 768-token decode, `macmon` at 100 ms, samples with the GPU
more than half active:

| model | engine | GPU frequency MHz | GPU active | GPU W | CPU W | tok/s |
| :-- | :-- | --: | --: | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | geist | 1679 | 97 % | 5.8 | 0.6 | 151.2 |
| | mlx-lm | 1295 | 75 % | 3.6 | 2.8 | 160.7 |
| Llama 3.2 3B, 4-bit | geist | 1551 | 89 % | 10.5 | 0.4 | 65.7 |
| | mlx-lm | 1285 | 74 % | 5.6 | 1.4 | 75.5 |

geist's GPU runs at a higher frequency and draws 1.6–1.9× the power of
mlx-lm's for no more tokens: the governor clocks the GPU up for geist's
kernels, and the step stays bound by memory. This was not followed further
here; it says geist's decode kernels do more arithmetic per byte read than
mlx-lm's, which costs energy, not time.

## Prefill

Milliseconds, mean of three repeats (run 4092). geist: the `prefill_tokens`
call for 512 tokens. mlx-lm: from the start of `stream_generate` to the
first yielded token, which is a batched pass over 511 tokens, then one
single-token step for the last prompt token and the sampling.

| model | engine | wall | GPU busy | no buffer on the GPU | CPU | command buffers | dispatches | barriers |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | geist | 71.3 | 71.0 | 0.3 | 4.2 | 61 | 981 | – |
| | mlx-lm | 70.4 | 56.7 | 13.7 | 46.0 | 46 | 1718 | 1038 |
| Llama 3.2 3B, 4-bit | geist | 249.8 | 249.5 | 0.2 | 3.0 | 43 | 683 | – |
| | mlx-lm | 194.0 | 190.4 | 3.5 | 19.6 | 50 | 1301 | 797 |

mlx-lm's GPU time splits into the command buffers that carry a weight GEMM,
i.e. the batched pass, and the rest:

| model | batched pass, 511 tokens | single-token step and sampling | longest GPU gap |
| :-- | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | 48.3 ms | 8.5 ms | 7.9 ms |
| Llama 3.2 3B, 4-bit | 171.6 ms | 19.5 ms | 2.2 ms |

- **geist's prefill is GPU time and nothing else** (99.6–99.9 % busy, 3–4 ms
  of CPU). Its 512 tokens take 71.0 and 249.5 ms on the GPU; mlx-lm's 511
  take 48.3 and 171.6 ms. The kernels' ratio is 1.47 (Qwen) and 1.45
  (Llama).
- mlx-lm loses part of that lead outside the batched pass: 0.7–2.5 ms
  before its first commit, a gap of 7.9 ms between the pass and the
  single-token step on the Qwen pair (host work; 2.2 ms on Llama), the step
  itself, and 2–5 ms between the step's end and the token reaching the
  caller. On
  2026-11-22 its Qwen prefill took 60.3 ms where it took 70.4 ms today, with
  the same script, file and package versions; the GPU work measured today
  leaves room for that only in the host part. Its Llama prefill is the same
  on both days (2634 tok/s).

GPU time by kernel class, one op per command buffer (inferred shares):

| class | Qwen geist | Qwen mlx-lm | Llama geist | Llama mlx-lm |
| :-- | --: | --: | --: | --: |
| quantized matmul | 64.7 % | 49.7 % | 87.2 % | 61.8 % |
| DeltaNet mixer | 21.9 % | 6.2 % | – | – |
| attention | 2.6 % | 2.8 % | 5.9 % | 5.5 % |
| norms | 2.4 % | 10.8 % | 3.6 % | 7.0 % |
| elementwise and copies | 8.4 % | 30.4 % | 3.3 % | 25.7 % |

What the classes say about geist's 1.45–1.47×, and what the levers below
add:

- **The chunked DeltaNet costs 15.3 ms of Qwen's 69.9 ms** in this mode,
  7 kernels per layer and pass (252 dispatches); mlx-lm's recurrence kernel
  and convolution take 4.2 ms. That is about half of the 22.7 ms between
  geist's GPU time and mlx-lm's batched pass (inferred: one-op shares).
- **The serial order of the dispatches costs at most 16 % (Qwen) and 5 %
  (Llama)**: with the same dispatches in a concurrent encoder without any
  barrier, which computes garbage, the prefill ran at 8376 and 2150 tok/s
  (measured bound). #590 recovers 3 % of it on Qwen and nothing on Llama.
- **The weight GEMMs are not slower on the 8-bit pair and are on the 4-bit
  pair.** In the one-op runs geist's `matmul_q80_mm_tc` dispatches take
  43.5 ms and mlx-lm's command buffers with a quantized GEMM 39.5 ms (those
  buffers hold a second kernel each). On Llama it is 215.8 ms
  (`matmul_q4k_mm_tc`, `matmul_q6k_mm_tc`) against 177.5 ms: the GGUF's
  K-quants hold 12 % more bytes and take more arithmetic to unpack than
  MLX's affine 4-bit groups, and the kernels differ; this measurement does
  not separate the two.
- **The rows per pass are not it.** geist prefills in passes of 256 rows
  (`preferred_m_max`), so 512 tokens are two passes and every weight GEMM is
  dispatched twice (372 on Qwen, 392 on Llama, against mlx-lm's 169 and 191
  in one pass). One pass of 512 rows (`GEIST_M_MAX=512`) is 2.6 % faster on
  Qwen and no faster on Llama; 128 rows are 11 % slower on Qwen. The logits
  are the same bit for bit at 128, 256 and 512 rows on both models.
- What remains after the DeltaNet and the GEMMs, about 8 ms on Qwen and
  40 ms on Llama, is not attributed; the serial order explains at most 10
  and 11 ms of it. mlx-lm's kernels can overlap inside its concurrent
  encoders (it puts a barrier before 60 % of its dispatches), which a
  one-op run does not show.

## Levers

Run 4094 (geistlib `a93ea444`), same workload, three repeats per cell,
median (minimum–maximum); "plain" is geist as on `main`. Every geist cell
produced the same 128 tokens as plain.

Qwen3.5 0.8B, 8-bit (mlx-lm in this run: 7295 tok/s prefill, 160.4 decode):

| geist cell | prefill tok/s | against plain | decode tok/s | against plain |
| :-- | --: | --: | --: | --: |
| plain (two cells) | 7175 (7147–7184) | | 152.6 (152.4–152.6) | |
| concurrent encoders | 7420 (7394–7431) | +3.4 % | 154.7 (154.6–154.8) | +1.4 % |
| first buffer after 8 dispatches | 7181 (7158–7181) | +0.1 % | 154.6 (154.6–154.7) | +1.3 % |
| first buffer after 16 | 7186 (7153–7188) | +0.1 % | 154.3 (154.2–154.4) | +1.1 % |
| first buffer after 32 | 7191 (7159–7192) | +0.2 % | 153.9 (153.9–154.2) | +0.9 % |
| a spinning wait (probe only) | 7187 (7154–7188) | +0.2 % | 153.3 (153.2–153.4) | +0.5 % |
| **#590's defaults: concurrent, first buffer after 16** | 7431 (7390–7431) | +3.6 % | 157.0 (156.8–157.1) | +2.9 % |
| the same with a spinning wait | 7433 (7410–7440) | +3.6 % | 156.9 (156.9–157.0) | +2.8 % |
| 128 rows per prefill pass | 6381 (6351–6384) | −11.1 % | 152.6 (152.5–152.6) | 0.0 % |
| 512 rows per prefill pass | 7360 (7300–7376) | +2.6 % | 152.6 (152.4–152.6) | 0.0 % |
| 512 rows per pass with #590's defaults | 7478 (7383–7486) | +4.2 % | 156.9 (156.6–157.0) | +2.8 % |

From run 4092 (geistlib `df61dd1c`; plain 7201 and 152.9): a concurrent
encoder with a barrier before every dispatch 7203 and 151.9 (−0.6 %); with
no barrier at all, which computes garbage and bounds what any ordering can
reach, 8376 (+16.3 %) and 171.4 (+12.1 %); barriers placed by bind offset
instead of by whole buffer, which is not safe in general (a handle can be
bound at an offset inside its buffer), 7442 and 155.7; command buffers
every 48 dispatches in decode 154.0 (+0.7 %); no rotation 151.0 (−1.2 %);
the tensor API off 3462 in prefill (−52 %).

A decode step of geist after #590's changes, microseconds (run 4094): 6419
against 6606, of which the GPU is busy 6230 against 6331 and has no buffer
for 193 against 279: 74 from the GPU finishing to the wait returning
(unchanged), 42 from there to the first commit (was 93), 77 from the commit
to the GPU starting (was 113). mlx-lm's step is 6234.

Llama 3.2 3B, 4-bit (mlx-lm: 2635 and 77.1). From the cell "first buffer
after 32" on, something else used 13–20 % of the host's CPU (idle 80–87 %
at the cells' gates, 99 % before) and geist ran 4–7 % slower in every cell,
plain included; those cells are compared with the plain cell measured in
that state. mlx-lm's last cell came after it (idle 97 %) and reads 0.5–0.8 %
low.

| geist cell | prefill tok/s | decode tok/s | against |
| :-- | --: | --: | :-- |
| plain, first | 2058 (2058–2059) | 67.5 (67.5–67.5) | |
| concurrent encoders | 2054 (2038–2056) | 67.4 (67.0–67.4) | first plain: −0.2 %, −0.1 % |
| first buffer after 8 | 2057 (2057–2058) | 67.9 (67.8–67.9) | first plain: 0.0 %, +0.6 % |
| first buffer after 16 | 2059 (2057–2061) | 67.7 (67.7–67.8) | first plain: 0.0 %, +0.3 % |
| plain, last (disturbed) | 1960 (1957–2036) | 62.8 (62.5–63.9) | |
| #590's defaults (disturbed) | 1955 (1953–1957) | 62.4 (62.3–62.5) | last plain: −0.3 %, −0.6 % |
| 512 rows per prefill pass (disturbed) | 1952 (1947–1955) | 62.4 (62.4–62.6) | last plain: −0.4 %, −0.6 % |
| 128 rows per prefill pass (disturbed) | 1892 (1850–1905) | 61.4 (58.0–62.4) | last plain: −3.5 %, −2.2 % |

Llama's decode is one chain of dispatches: a concurrent encoder spares 0.8 %
of its barriers, and in run 4092 it was 0.5 % slower that way (66.9 against
67.3 tok/s). #590 therefore lets a kind of sequence that spares fewer than
one barrier in twenty return to serial encoders after 4096 dispatches,
which is what the "concurrent encoders" row above ran (4 % of its decode
encoders were concurrent). Single cells in the disturbed state scatter by
about 1 %; the interleaved A/B below is the comparison of #590 with `main`.

### A/B against main

`tools/bench_revision_ab.py`, `main` (`a1751c84`) against `a93ea444`, each
built from scratch, 10 interleaved cycles with a control copy of the
baseline, in run 4094:

| model | row | main ms | #590 ms | change | 95 % interval | faster | control |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| Qwen3.5 0.8B Q8_0 | prefill @ 512 | 71.6 | 69.5 | −2.9 % | [−3.0 %, −2.8 %] | 10/10 | −0.1 % [−0.2 %, +0.1 %] |
| | decode, 128 tokens | 839.6 | 816.8 | −2.7 % | [−2.8 %, −2.7 %] | 10/10 | 0.0 % [−0.1 %, +0.1 %] |
| Llama 3.2 3B Q4_K_M | prefill @ 512 | 262.3 | 262.0 | −0.2 % | [−0.6 %, +0.5 %] | 7/10 | 0.0 % [−0.6 %, +0.9 %] |
| | decode, 128 tokens | 2038.1 | 2027.4 | −0.6 % | [−0.8 %, −0.2 %] | 9/10 | 0.0 % [−0.3 %, +0.5 %] |

Llama's A/B ran in the disturbed state described above, all three arms
alike. In tokens per second: Qwen 7151 to 7367 prefill and 152.4 to 156.7
decode; Llama 1952 to 1954 and 62.8 to 63.1.

The same logits: `bench_nll`'s hash over every scored logits vector
(`logits_fnv`), decode path (95 tokens one at a time) and prefill path (511
tokens in one call, then 127 scored steps), is equal for `main`'s build,
for this build with the serial encoder and for this build with its defaults,
on both models on the M6 (runs 4092 and 4094), and on eight models on an
M1 Max, where the prefill runs the simdgroup kernels instead of the tensor
ones (Qwen3.5 0.8B and 4B, Qwen3 0.6B, Llama 3.2 3B, Gemma 4 E2B, BitNet
b1.58 large, OLMoE 1B-7B, Huihui-MoE 1.2B).

What the changes cost: the first sequence of a backend takes longer with
concurrent encoders. Process start to first token on a one-token prompt, ten
alternating runs each: 56.8 ms against 58.6 ms (Qwen) and 97.7 ms against
100.0 ms (Llama).

### Where the gap comes from, and what each part would take

Decode, per step:

| part | Qwen3.5 0.8B 8-bit | Llama 3.2 3B 4-bit | what it takes |
| :-- | --: | --: | :-- |
| gap to mlx-lm before #590 | 344 µs (5.5 %) | 2024 µs (15.6 %) | |
| GPU idle between steps (measured) | 260 µs | 261 µs | #590's early first buffer takes 86 µs of it on Qwen and 73 µs on Llama. The remaining 190 µs go only by queueing the next step before the current one has finished, as mlx-lm does: the argmax and the embedding lookup are already on the GPU, but the session would have to take back a step the caller does not continue with (KV position, DeltaNet state). A spinning wait adds nothing on top of #590's defaults (156.9 against 157.0 tok/s) and keeps a core busy |
| serial order of the dispatches (measured) | 83 µs more GPU time than mlx-lm | none: one chain | concurrent encoders: −101 µs on Qwen (15.5 % of the barriers spared), geist's GPU time per step is now mlx-lm's. Barriers by handle instead of by `MTLBuffer` spare 22.2 % (+0.4 % decode in run 4092's probe); the binds would have to carry the handle's extent |
| weight bytes (inferred) | none: 8.5 bits on both sides | 1763 µs: the output head at Q6_K about 810, the body about 950 | not an engine matter. A GGUF at 4.5 bits throughout would read like the MLX file |

Prefill, GPU time for 512 tokens:

| part | Qwen3.5 0.8B 8-bit | Llama 3.2 3B 4-bit | what it takes |
| :-- | --: | --: | :-- |
| geist over mlx-lm's batched pass | 22.7 ms (71.0 against 48.3) | 77.9 ms (249.5 against 171.6) | |
| chunked DeltaNet (inferred, one-op shares) | about 11 ms | – | a recurrence kernel in place of the seven per layer and pass; mlx-lm's is one custom kernel |
| serial order (measured bound) | at most 10 ms; #590 takes 2.3 | at most 11 ms; #590 takes 0 | more independent dispatches or fewer of them. Barriers by handle spare 30 % instead of 6 % on Llama's prefill (+1.3 % in run 4092's probe) |
| weight GEMMs (inferred, one-op) | about 4 ms | about 38 ms | a comparison of `matmul_q4k_mm_tc` with mlx-lm's `affine_qmm_t_nax` kernel on the same shapes; part of it is the K-quants' format |
| not attributed | up to 8 ms | about 30–40 ms | per-dispatch GPU times in the default mode, which neither engine exposes today |


## Limits

- One machine, one day, two pairs, one prompt length. Part of run 4094's
  Llama cells and its Llama A/B ran while something else used the host's
  CPU (see Levers). The idle time per
  decode step is a constant of the engine and should carry over; the prefill
  ratios were measured at 512 tokens only.
- The per-kernel shares come from a mode that changes the run (see above).
  No per-dispatch GPU time exists for either engine in its default mode.
- mlx-lm's host time is not split. What its 7.9 ms prefill gap does was not
  investigated.
- The trace library hooks the classes of the default Metal device as they
  are named on macOS 27; it is a probe, not an API.

## Raw data

`benchmark/results/raw/metal-profile-m6-2026-11-23/`: per run the cells
(`run-<n>-cells.jsonl`: every sample, the gate readings, the tokens), the
job's summary with the trace reports, and run 4094's two A/B reports. Run
4092 is the profile (`metal_profile.sh`), run 4094 the levers
(`metal_profile.sh --set levers`). The traces themselves (about 20 MB per
run, gzip) are the runs' CI artifacts, kept for 14 days.
`python3 tools/probes/metal_trace_report.py LABEL=trace.jsonl.gz …` prints
the tables from them.
