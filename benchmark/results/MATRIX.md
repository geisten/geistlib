# Comparable benchmark matrix

This is the entry point for comparisons across models and systems. A number is
eligible for the matrix only when its raw artifact uses one frozen workload,
names the exact model and engine hashes, retains every sample, and records the
host gate and thread policy. Missing cells stay missing; historical numbers are
not silently normalized into a comparison they cannot support.
Coverage work is tracked in [#364](https://github.com/geisten/geistlib/issues/364).

## Current coverage

| Model / quantization | Apple CPU | Pi 5 CPU | AMD AVX-512 CPU | Apple Metal | NVIDIA Vulkan |
| :-- | :--: | :--: | :--: | :--: | :--: |
| Gemma 4 E2B-it Q4_K_M | [pp ~−47% / tg ~−28%](CROSS-ENGINE-APPLE-M1MAX.md) | [pp ~-13% / tg ~+11%](CROSS-ENGINE-PI5.md) | [pp ~−16% / tg ~+19%](raw/matrix-2026-10-09-llama060/2026-10-09T170023Z_geist_llama_cpu_gemma4-e2b-q4km.md) | [pp ~−36% / tg ~−18%](CROSS-ENGINE-APPLE-M1MAX-METAL.md) | [pp ~+12% / tg ~−4%](raw/matrix-2026-10-09-llama060/2026-10-09T152755Z_geist_llama_gpu_gemma4-e2b-q4km.md) |
| Gemma 4 E4B-it Q4_K_M | not measured | does not fit 4 GB reference host | [pp ~−19% / tg ~+16%](raw/matrix-2026-10-09-llama060/2026-10-09T121522Z_geist_llama_cpu_gemma4-e4b-q4km.md) | not measured | [pp ~−3% / tg ~−6%](raw/matrix-2026-10-09-llama060/2026-10-09T154148Z_geist_llama_gpu_gemma4-e4b-q4km.md) |
| Llama 3.2 3B Q4_K_M | not measured | not measured | [pp ~−26% / tg ~+8%](raw/matrix-2026-10-09-llama060/2026-10-09T124041Z_geist_llama_cpu_llama32-3b-q4km.md) | not measured | [pp ~+4% / tg ~−16%](raw/matrix-2026-10-09-llama060/2026-10-09T155527Z_geist_llama_gpu_llama32-3b-q4km.md) |
| BitNet b1.58 2B-4T I2_S | reference engine has no I2_S type | reference engine has no I2_S type | reference engine has no I2_S type | reference engine has no I2_S type | reference engine has no I2_S type |
| BitNet b1.58-large TQ2_0 | not measured | not measured | [pp ~+61% / tg ~+47%](raw/matrix-2026-10-09-llama060/2026-10-09T130112Z_geist_llama_cpu_bitnet-large-tq2.md) | not measured | [pp ~−6% / tg ~−7%](raw/matrix-2026-10-09-llama060/2026-10-09T184857Z_geist_llama_gpu_bitnet-large-tq2.md) |
| Qwen3 0.6B Q8_0 | not measured | not measured | [pp ~+18% / tg ~+9%](raw/matrix-2026-10-09-llama060/2026-10-09T131443Z_geist_llama_cpu_qwen3-0.6b-q8.md) | not measured | [pp ~+12% / tg ~−9%](raw/matrix-2026-10-09-llama060/2026-10-09T162101Z_geist_llama_gpu_qwen3-0.6b-q8.md) |
| Qwen3.5 0.8B Q8_0 † | not measured | not measured | [pp ~+72% / tg ~+17%](raw/matrix-2026-10-09-llama060/2026-10-09T132823Z_geist_llama_cpu_qwen35-0.8b-q8.md) | not measured | [pp ~+30% / tg ~+13%](raw/matrix-2026-10-09-llama060/2026-10-09T163331Z_geist_llama_gpu_qwen35-0.8b-q8.md) |
| Qwen3.5 4B Q4_0 † | not measured | not measured | [pp ~−18% / tg ~+13%](raw/matrix-2026-10-09-llama060/2026-10-09T134243Z_geist_llama_cpu_qwen35-4b-q4.md) | not measured | [pp ~+5% / tg ~−2%](raw/matrix-2026-10-09-llama060/2026-10-09T164607Z_geist_llama_gpu_qwen35-4b-q4.md) |
| Qwen3.8 27B Q4_0 † | not measured | does not fit reference host | [pp ~−28% / tg ~+8%](raw/matrix-2026-10-09-llama060/2026-10-09T171858Z_geist_llama_cpu_qwen38-27b-q4.md) | not measured | does not fit 11 GB device |
| Ternary Bonsai 2 27B PQ2_0 ‡ | not measured | does not fit reference host | [pp ~−19% / tg ~+13%](MAIN-VS-V011-AMD9950X-2026-10-07.md) | not measured | [pp ~−12% / tg ~+24%](MAIN-VS-V011-AMD9950X-2026-10-07.md) |

A filled cell states the geist-vs-llama.cpp ratio at pp512 and tg64 at depth
512, rounded, and links the full sweep. Rounded on purpose: two runs of the
identical protocol on the same host drift by more than the within-run MAD, so a
cell quoted to a decimal would claim a precision the protocol does not deliver
(see the reproducibility section in the linked report). Ratios are comparable within a column; absolute
tokens/s across columns describe hardware, not engine efficiency.

‡ Reference engine: the PrismML llama.cpp fork (upstream llama.cpp has no
PQ2_0 type); the row is shape-parity only like the qwen35 rows it is built on.

The AMD and NVIDIA columns were re-run on 2026-10-09 at geist `8489cb9`
against llama.cpp **v0.6.0** (`d8123504`), the new reference pin
(`raw/matrix-2026-10-09-llama060/`, same protocol hashes and model files).
Every cell came from the protocol; a cell whose cycles were disturbed by other
load on the host (spread over 10 % between cycles at 512) was re-measured.
Two things moved at once, so read a change against an older cell with both in
mind:

- llama.cpp v0.6.0 is 15–20 % faster in CPU prefill than the previous pin
  `2d8d612e` (Gemma 4 E2B pp512 529 → 627 t/s); geist's own CPU prefill held
  or rose (E2B 527 → 528, Qwen3.5 4B 182 → 257, Qwen3.5 0.8B 866 → 1756).
- geist's Vulkan prefill gained the coopmat GEMMs and attention of
  #681/#683/#686/#712/#719/#720 and the DeltaNet kernels of #708: the NVIDIA
  column went from −70…−91 % to −3…+30 %. TQ2_0 (BitNet b1.58-large) followed
  with #741, which added head_dim 96 tensor-core attention and a 512-token chunk
  for host-visible pools: −56 → −6 %, re-measured at geist `db7f140` (#741's head) the same
  evening.

The Apple and Pi 5 columns and the Bonsai row (PrismML fork) are older
measurements against `2d8d612e`; their reports name the pin.

† Qwen3.5/3.8 rows are shape-parity only: DeltaNet state depends on token
identity and the engines use their native synthetic streams (see the token
note below).

“Not measured” means “no result under the current common protocol”, not “the
engine or model is unsupported”. The result documents beside this file contain
valuable older measurements, but they mix sequence lengths, sample counts,
aggregation rules, engine revisions and thermal conditions. They are historical
evidence, not cells in this matrix.

## Frozen comparison contract

The current head-to-head is defined by
[`cross_engine_cpu_protocol.json`](../cross_engine_cpu_protocol.json) for CPU
columns and [`cross_engine_gpu_protocol.json`](../cross_engine_gpu_protocol.json)
for GPU columns — same shapes, pairing and aggregation; only the backend
contract differs — and run by
[`bench_cross_engine.py`](../../tools/bench_cross_engine.py):

- byte-identical model, verified by SHA-256;
- geist and llama.cpp commits plus binary hashes recorded;
- CPU-only execution verified from llama.cpp's JSON output; GPU cells instead
  verify full offload and record the exact device and backend;
- pp128/256/512/1024 and tg64 at matching context depths;
- one thread profile per host from `host_profiles`, identical for both engines
  within a run (M1 Max 8/7, 9950X 16/15);
- four alternating A/B cycles with three ordered samples per cell and cycle;
- quiet-host gate and cooldown before every engine run;
- median throughput and median absolute deviation, never best-of;
- model loading excluded and one discarded engine-native warmup;
- quality parity established separately before a speedup is promoted.

A cell is a statement about a thread profile, not about the engines in
general: on the 9950X geist's prefill lead runs from −4.6% at 4 threads to
+13.8% at 16 ([THREAD-SWEEP-AMD9950X.md](THREAD-SWEEP-AMD9950X.md), which is
also why that profile's 16/15 is now measured rather than assumed). Decode is
the stable one — geist leads across the whole 4-to-32 range.
On the M1 Max the same protocol reverses the ranking: llama.cpp's Accelerate
path leads at every shape ([CROSS-ENGINE-APPLE-M1MAX.md](CROSS-ENGINE-APPLE-M1MAX.md)).
The matrix records that as it stands; a column is not a verdict on the engine,
and a cell measured is worth more than a cell argued about.

Thread counts are part of a system profile, not a universal constant. A Pi 5
campaign, for example, must declare its own fixed 4/3-thread profile while
keeping the workload, pairing and aggregation contract unchanged. Engine ratios
are comparable within a hardware cell; absolute tokens/s across unlike systems
describe hardware capacity rather than engine efficiency.

Both tools currently use their native deterministic synthetic token streams.
The graph shapes are matched, but token IDs are not claimed to be identical.
For token-dependent routing models, a matrix row additionally requires a shared
token fixture or must carry an explicit “shape parity only” qualification.
