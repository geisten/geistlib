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
| Gemma 4 E2B-it Q4_K_M | not measured | not measured | [pp +18% / tg +15%](CROSS-ENGINE-AMD9950X.md) | not measured | not measured |
| Gemma 4 E4B-it Q4_K_M | not measured | does not fit 4 GB reference host | not measured | not measured | not measured |
| Llama 3.2 3B Q4_K_M | not measured | not measured | not measured | not measured | not measured |
| BitNet b1.58 2B-4T I2_S | not measured | not measured | not measured | not measured | not measured |
| BitNet b1.58-large TQ2_0 | not measured | not measured | not measured | not measured | not measured |
| Qwen3 0.6B Q8_0 | not measured | not measured | not measured | not measured | not measured |
| Qwen3.5 0.8B Q8_0 | not measured | not measured | not measured | not measured | not measured |
| Qwen3.5 4B Q4_0 | not measured | not measured | not measured | not measured | not measured |
| Qwen3.8 27B Q4_0 | not measured | does not fit reference host | not measured | not measured | not measured |

A filled cell states the geist-vs-llama.cpp ratio at pp512 and tg64 at depth
512 and links the full sweep. Ratios are comparable within a column; absolute
tokens/s across columns describe hardware, not engine efficiency.

“Not measured” means “no result under the current common protocol”, not “the
engine or model is unsupported”. The result documents beside this file contain
valuable older measurements, but they mix sequence lengths, sample counts,
aggregation rules, engine revisions and thermal conditions. They are historical
evidence, not cells in this matrix.

## Frozen comparison contract

The current CPU head-to-head is defined by
[`cross_engine_cpu_protocol.json`](../cross_engine_cpu_protocol.json) and run by
[`bench_cross_engine.py`](../../tools/bench_cross_engine.py):

- byte-identical model, verified by SHA-256;
- geist and llama.cpp commits plus binary hashes recorded;
- CPU-only execution verified from llama.cpp's JSON output;
- pp128/256/512/1024 and tg64 at matching context depths;
- one thread profile per host from `host_profiles`, identical for both engines
  within a run (M1 Max 8/7, 9950X 16/15);
- four alternating A/B cycles with three ordered samples per cell and cycle;
- quiet-host gate and cooldown before every engine run;
- median throughput and median absolute deviation, never best-of;
- model loading excluded and one discarded engine-native warmup;
- quality parity established separately before a speedup is promoted.

Thread counts are part of a system profile, not a universal constant. A Pi 5
campaign, for example, must declare its own fixed 4/3-thread profile while
keeping the workload, pairing and aggregation contract unchanged. Engine ratios
are comparable within a hardware cell; absolute tokens/s across unlike systems
describe hardware capacity rather than engine efficiency.

Both tools currently use their native deterministic synthetic token streams.
The graph shapes are matched, but token IDs are not claimed to be identical.
For token-dependent routing models, a matrix row additionally requires a shared
token fixture or must carry an explicit “shape parity only” qualification.
