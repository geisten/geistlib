# C23 review, Phase 0 baselines (#341)

The numbers #333's performance gates compare against: `.text` per library, clean build time and the benchmark matrix, all for main at `6642260d55a1` (main on 2026-10-31, before the first #333 code change (a073758b, GEIST_CLEANUP, 2026-11-02)). Data: [`benchmark/c23_baseline.json`](../c23_baseline.json). A PR is still gated by its own interleaved A/B (`tools/bench_revision_ab.py`, mac-ab.yml & co.); these are the reference points for the end-of-phase runs and for the Phase 5 report.

## `.text` per library

`python3 tools/text_size.py measure build/<target>/release lib/<target>/release/libgeist.<so|dylib>`; compare two such files with `text_size.py compare base.json new.json` (fails above +1 %).

| | `macos-arm64` | `linux-arm64` | `linux-x86_64` |
|---|--:|--:|--:|
| `libgeist.a` (objects) | 896,972 (130) | 806,000 (123) | 1,630,606 (142) |
| shared library | 896,972 | 807,796 | — |
| `src/archs/audio_conformer` | 50,928 | 33,824 | 41,899 |
| `src/archs/transformer` | 230,196 | 239,736 | 395,525 |
| `src/archs/vision_siglip` | 17,068 | 13,808 | 18,499 |
| `src/backends/common` | 37,876 | 37,284 | 55,131 |
| `src/backends/cpu_neon` | 213,364 | 192,952 | — |
| `src/backends/cpu_scalar` | 7,332 | 6,560 | 7,893 |
| `src/backends/cpu_x86` | — | — | 762,912 |
| `src/backends/metal` | 100,368 | — | — |
| `src/base` | 7,628 | 10,388 | 9,852 |
| `src/engine` | 53,888 | 54,760 | 64,598 |
| `src/formats/gguf` | 19,344 | 13,552 | 12,492 |
| `src/formats/ptqtp` | 1,132 | 1,384 | 1,393 |
| `src/io` | 13,044 | 14,092 | 14,786 |
| `third_party/stb` | 144,804 | 187,660 | 245,626 |

- `macos-arm64`: Apple clang 21.0.0, make TARGET=mac-omp (cpu_neon cpu_scalar metal, Accelerate).
- `linux-arm64`: gcc-14 14.2.0 (Debian 13 image geistlib-gcc14:arm64), make TARGET=linux (cpu_neon cpu_scalar, OpenBLAS).
- `linux-x86_64`: x86_64-linux-gnu-gcc-14 14.2.0 (cross, geistlib-gcc14:x86cross), make TARGET=linux LINUX_ARCH=x86_64 GEMM_PROVIDER=native (cpu_x86 cpu_scalar); libgeist.so not linked in the cross build (its TLS relocations need -fPIC there).

## Clean build time

make clean, then make -j8 lib, on the M1 Max (10 cores) under with-mac-lock, 1-min load 10-15 (other agents' unit tests do not take the lock). cpu_s = user+sys of the build, less sensitive to load than real_s.

| build | real s (runs) | user+sys s (runs) |
|---|---|---|
| `make TARGET=mac-omp -j8 lib` | 4.23, 4, 4.01 | 17.8, 17.28, 17.26 |
| `make TARGET=linux CC=gcc-14 -j8 lib (docker, 10 vCPU)` | 7, 7 | 30.05, 29.28 |

An earlier attempt at 1-min load ~430 took 17.7-21.8 s real and 19.4-19.5 s user+sys on the Mac: wall time on this host means nothing without the load next to it.

## Benchmark matrix (bench-matrix.yml cells)

newest geisten/bench-results run per cell whose geist binary is 6642260d (the Sunday 2026-11-01 weekly set and nightly runs), else the newest before 2026-11-02; frozen cross-engine protocol, medians of 4 alternating A/B cycles; pp = prefill tok/s at that prompt length, tg64 = decode tok/s at that depth; vs llama.cpp in percent.

pp512 / tg64 at depth 512, tok/s:

| model | `apple_m6_mini` | `apple_m6_mini_metal` | `apple_m1max` | `amd_9950x` | `nvidia_2080ti_vulkan` | `pi5` |
|---|--:|--:|--:|--:|--:|--:|
| bitnet-large-tq2 | — | — | — | 2388 / 271.3 | 13267 / 340.9 | 129 / 30.7 |
| bonsai2-27b-pq2 | 33 / 12.9 | 225 / 19.0 | — | 83 / 9.6 | 633 / 40.0 | — |
| gemma4-e2b-q4km | 341 / 66.0 | 2699 / 89.4 | 325 / 56.0 ¹ | 952 / 51.7 | 5742 / 158.3 | 28 / 7.1 |
| gemma4-e4b-q4km | 172 / 35.5 | 1356 / 45.7 | 81 / 21.1 ¹ | 500 / 26.1 | 2944 / 96.8 | — |
| llama32-3b-q4km | 236 / 53.1 | 2067 / 69.0 | — | 735 / 37.2 | 4877 / 150.9 | 23 / 5.3 |
| qwen3-0.6b-q8 | 967 / 141.4 | 9336 / 193.7 | — | 1996 / 111.5 | 18380 / 323.4 | 158 / 14.3 |
| qwen35-0.8b-q8 | 967 / 137.4 | 7182 / 153.4 | — | 1743 / 82.1 | 15257 / 327.5 | 106 / 13.3 |
| qwen35-4b-q4 | 147 / 40.8 | 1533 / 49.4 | 164 / 33.8 ¹ | 561 / 26.2 | 3495 / 122.4 | 17 / 4.3 |
| qwen38-27b-q4 | — | 220 / 9.0 | — | 98 / 4.8 | — | — |

¹ not at 6642260d: `apple_m1max` cells are from 2026-10-18/25 (c993379b0, af6689407).

Every cell's other lengths (128, 256, 1024), its ratio to llama.cpp and the archive file are in the JSON.

**Not measured:**

- apple_m1max (CPU): cells from 2026-10-18/25 only; the dispatched nightly set at 6642260d (run 2818) failed 'host did not become quiet' (load/core 45, 0 % idle from other agents).
- apple_m1max_metal: no run ever archived; run 2818's two nightly cells failed the same way.

## MoE rows of #333 (Qwen3-30B-A3B, Qwen3.6-35B-A3B)

moe-baseline.yml on geist-vulkan (Ryzen 9 9950X + RTX 2080 Ti) at 6642260d, 2 interleaved cycles, bench_perf_sweep --seq-lens 32,128,512,1024 --decode-n 64 --warmup 8 --repeats 3; medians; llama.cpp d81235049 llama-bench pp512/tg32; desktop not bench-locked (quiet input off), host load 13-16 of 32 threads. Runs 2819 and 2820.

| model | configuration | pp32 | pp128 | pp512 | pp1024 | tg64@32 | tg64@512 | tg64@1024 | RSS MB |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen3-30b-a3b-q4km | geist-cpu | 152.9 | 270.4 | 271.2 | 259.2 | 36.5 | 35.2 | 33.9 | 17538 |
| qwen3-30b-a3b-q4km | geist-hot | 151.6 | 266.3 | 268.7 | 257.1 | 46.8 | 34.4 | 35.6 | 24973 |
| qwen3-30b-a3b-q4km | llama-cpu (pp512; tg32 at depth 0) | | | 254.5 | | 35.0 | | | |
| qwen36-35b-a3b-ud-q4kxl | geist-cpu | 163.6 | 206.5 | 209.5 | 207.2 | 23.7 | 23.6 | 23.5 | 20784 |
| qwen36-35b-a3b-ud-q4kxl | geist-gpu | 281.6 | 490.2 | 501.1 | 501.9 | 76.0 | 75.9 | 75.9 | 12665 |
| qwen36-35b-a3b-ud-q4kxl | llama-cpu (pp512; tg32 at depth 0) | | | 219.1 | | 20.5 | | | |

`geist-hot` = the model on cpu_x86 with the hottest experts on the GPU (#48 profile); `geist-gpu` = the model on Vulkan with its experts on a CPU backend and the hottest copied to the GPU.

## Synthetic Llama 3.2 1B Q4_K (hermetic row)

gguf_artifacts/synth/llama32-1b-q4_k.gguf (tools/gen_synth_gguf.py), bench_perf_sweep --seq-lens 32,128,512,1024 --decode-n 64 --warmup 32 --repeats 5 on the M1 Max under with-mac-lock at 1-min load ~15: indicative only, repeat spread 3-30 % (prefill/decode best-to-worst).

| backend | seq | prefill tok/s | decode tok/s | spread pp / tg % | RSS MB |
|---|--:|--:|--:|---|--:|
| M1 Max cpu_neon | 32 | 640.9 | 123.2 | 11.4 / 6.8 | 2113 |
| M1 Max cpu_neon | 128 | 645.2 | 119.1 | 11.4 / 15.0 | 2113 |
| M1 Max cpu_neon | 512 | 591.2 | 105.4 | 7.0 / 30.5 | 2113 |
| M1 Max cpu_neon | 1024 | 530.8 | 97.3 | 12.1 / 20.3 | 2113 |
| M1 Max Metal | 32 | 1409.4 | 248.1 | 7.2 / 5.5 | 250 |
| M1 Max Metal | 128 | 2701.1 | 241.8 | 20.9 / 15.1 | 250 |
| M1 Max Metal | 512 | 2310.6 | 217.6 | 10.4 / 3.0 | 251 |
| M1 Max Metal | 1024 | 2361.1 | 219.7 | 14.2 / 3.4 | 251 |

## Gaps

- M1 Max CPU at 6642260d and M1 Max Metal at all. First attempt (run 2818, 2026-11-04 night): the host was saturated by other agents' unit tests (0 % idle) and bench-matrix's quiet gate refused every cell. Second attempt (2026-11-04 evening, other agents idle): run 2990 failed the 80 % idle gate at 73-79 % idle (the workstation's browser, Spotlight); the four nightly cells run by hand under `with-mac-lock` with the idle gate at 70 % then failed the unchanged 5 % repeat-spread gate on all four tries each (spreads 8.8-49.7 %, corespotlightd/hybridsearchd/mobileassetd at 50-85 % CPU). Nothing was recorded; the spread gate is not relaxed. The M6 columns cover Apple CPU and Metal at 6642260d; the M1 Max needs a night with Spotlight idle.
- Third M1 Max attempt (#444), 2026-11-10 01:31–06:00: same binary, protocol and unchanged gates, one `with-mac-lock` lease per cell, each started only after Spotlight was idle and the host ≥ 85 % idle (pre-run idle 84–93 %). Every cell failed the 5 % repeat-spread gate on every try. Spreads of the last measurement per try: CPU qwen35-0.8b-q8 22.4 / 13.7 / 12.2 %, CPU gemma4-e2b-q4km 5.4 / 7.2 / 9.6 %, Metal gemma4-e2b-q4km 14.1 / 6.2 / 6.4 %, Metal bonsai2-27b-pq2 6.0 / 24.0 % (every spread in [`raw/c23-m1max-2026-11-10/`](raw/c23-m1max-2026-11-10/README.md)). Disturbers outside the lock: Chrome helpers (~46 % CPU), other Claude sessions, geist-lab's qwen-math500 runner between the cells, Spotlight until 01:46. An idle gate before the run does not keep this desktop quiet for the 10–30 min of a cell; the M1 Max cells need a window in which the Mac is kept idle on purpose (no browser, no agents, geist-lab paused), or the Apple baseline rests on the M6 columns.
- The synthetic row is indicative (spread up to 30 %); per-PR A/B on the M6 (mac-ab.yml) is the gate for it.
- Thread sweeps (`tools/bench_thread_sweep.py`), the batched decode path and `make bench-quality-small` were not run for Phase 0; the cross-engine protocol fixes the thread counts per host profile.
