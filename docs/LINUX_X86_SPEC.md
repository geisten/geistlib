# Linux / x86_64 backend spec — `feat/cpu-x86`

This document is the working specification for `geist`'s native Linux/x86_64
compute path. The goal is concrete and measurable: **beat current llama.cpp and
ollama on CPU** on AMD Zen 5 (and equivalent AVX-512+VNNI hardware), under the
quantified win criteria in [Win criteria](#win-criteria) below.

Status today (commit `0ec695a`):

- `src/backends/cpu_x86/` exists as a *policy skeleton* (no kernels).
- `cpu_x86` is **not** in `src/engine/backend_registry.c`.
- `mk/target-linux.mk` **fails the build on x86_64** with guidance pointing at
  this gap.

This spec closes that gap.

## Scope

Build a real `cpu_x86` backend with AVX2 / AVX-512 / AVX-512+VNNI /
AVX-512+BF16 runtime dispatch, phased merges to `main` behind a
`make BACKENDS="cpu_x86 cpu_scalar"` opt-in until the win criteria are
measured and verified on a target host.

The reference host for win-criterion measurement is **AMD Ryzen 9 9950X (Zen 5,
16C/32T, DDR5-6400, 60 GB RAM)**. The same binary is expected to run on every
AVX-512+VNNI host (Zen 4+, Sapphire Rapids+, Ice Lake-X+) and to fall back to
AVX2 on Zen 3 / Haswell+ and to `cpu_scalar` on older CPUs.

## Architecture commits (load-bearing — do not regret-fix later)

These three commitments must hold from the first kernel TU forward, so the
follow-up extension to other quant families (Q5_K, Q3_K, IQ2_S, IQ3_S, TQ2_0)
is additive rather than a rewrite.

1. **Kernel function-pointer signature is quant-family-agnostic.** Inner
   kernels are reached via:

   ```c
   typedef void (*decode_kernel_fn)(
       const void          *weights_predecoded,
       const void          *acts_predecoded,
       float               *out,
       int                  n_blocks,
       const struct kernel_ctx *ctx);
   ```

   `weights_predecoded` / `acts_predecoded` are family-opaque (W4A8 sees int8
   nibbles, IQ-quants see codebook indices, ternary sees a bitmap). Scales,
   codebooks and layout descriptors travel in `ctx`. Do **not** hardcode
   `const int8_t *acts` or `const int32_t *block_scales` into the public
   kernel ABI.

2. **Predecoded weight layout is per-quant via descriptor.** Block-width,
   codebook offset, bitmap stride, super-scale offset all live in a
   `kernel_layout_desc` carried inside `ctx`. No `#define BLOCK_W 32` in
   kernel TUs. The W4A8 path's natural 32-wide layout is one row in the
   layout table; Q5_K's 64-wide layout is another row.

3. **Dispatch through the function pointer even when there is only one
   kernel.** The temptation in Phase 1a — when only the VPDPBUSD path
   exists — is to inline the kernel into the backend hot path and skip the
   indirection. Don't. ICache impact is nil (the same target is hit
   repeatedly per token), and skipping the indirection is what blocks
   Phase-(c) extension later.

## ISA-tier dispatch

Per-translation-unit `-march=` compile flags, one TU per ISA tier per kernel,
runtime dispatch via function-pointer init at backend bind. Same pattern as
llama.cpp / OpenBLAS / FFmpeg.

Mandatory: AVX-512F + AVX-512BW + AVX-512DQ + AVX-512VL at the "AVX-512" tier.
AVX-VNNI (the 256-bit form) is **not** used — pure 512-bit kernels only at the
VNNI tier; AVX2 is the fallback below AVX-512.

`mk/backend-cpu_x86.mk` carries the per-TU patterns:

```make
%_avx2.o:        CFLAGS += -mavx2 -mfma -mbmi2
%_avx512.o:      CFLAGS += -mavx512f -mavx512bw -mavx512dq -mavx512vl
%_avx512_vnni.o: CFLAGS += -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
%_avx512_bf16.o: CFLAGS += -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512bf16
```

The dispatcher TU (`kernel_w4a8.c`, etc.) is compiled at the baseline
`-march=x86-64-v3` so it links into hosts that lack the higher ISAs.

CPUID feature detection comes from the existing `geist_hw_probe` (already
populates `has_avx2`, `has_avx512f`, `has_avx512_vnni` via
`__builtin_cpu_supports` — see `src/base/hw_probe.c:140-143`). The
`cpu_x86_isa_from_probe()` mapper in
`src/backends/cpu_x86/kernel_catalog.c` consumes those bits and selects the
function-pointer table at backend bind.

Override at runtime via `GEIST_FORCE_ISA={avx2,avx512,avx512_vnni,avx512_bf16}`
(also serves CI: every tier is exercised on a single AVX-512 runner by
forcing the lower tiers).

## Kernel matrix (MVP)

| Quant | Decode | Low-M Prefill | High-M Prefill | ISA tiers |
| --- | --- | --- | --- | --- |
| Q4_K_M | W4A8 native VPDPBUSD | W4A8 native VPDPBUSD | W4A8→BF16-SGEMM | AVX2 / AVX-512 / +VNNI / +BF16 |
| Q8_0 | native VPDPBUSD | native VPDPBUSD | BF16-SGEMM | AVX2 / AVX-512 / +VNNI / +BF16 |
| Q6_K | dequant→fp32 GEMV | — | dequant→BF16-SGEMM | AVX-512 only |
| Q5_K, Q3_K, IQ2_S, IQ3_S, TQ2_0 | `cpu_scalar` fallback | `cpu_scalar` | `cpu_scalar` | — |
| KIVI KV-cache | scalar (existing `src/backends/common/kivi.c`) | — | — | — |

Notes:

- **W4A8 is the actual execution kernel.** Q4_K disk-format weights are
  predecoded at model load into the W4A8 layout, so the inner loop is one
  VPDPBUSD-or-equivalent per 32-block — the same shape llama.cpp's
  `vec_dot_q4_K_q8_K` ends up at. The predecoder owns the nibble unpack +
  6-bit super-scale + Q8_K block scale, baked into pre-shuffled buffers.
- **Q6_K reuses the Phase-2 BF16-SGEMM trampoline.** No native W6A8 kernel
  is written. Q6_K is dominantly the tied LM head (decode m=1, bandwidth-
  bound) — dequant→fp32 GEMV is sufficient.
- **`gemma4_kernels.c` (`src/backends/common/`) gets AVX-512 arms** for
  layer-norm / activations / paired FFN gate-up. Currently `#if defined(__ARM_NEON)`
  with scalar fallback; adds `#elif defined(__AVX512F__)` paths.
- **KIVI inherits the existing scalar implementation for free** — it is
  pure C23 today, no NEON intrinsics (see file header in
  `src/backends/common/kivi.c`). AVX-512 specialization is a Phase-3
  optional lever, not in MVP scope.
- **Spec-head (`src/archs/transformer/forward/spec_head.c`) becomes
  eligible** on x86 once the W4A8 sketch GEMV is wired (Phase 1a). The
  arch-layer gate `(non-NEON/dotprod) → dense fallback` flips to "fast
  int8 GEMV available". Expected win on Zen 5 with DDR5-6400 is smaller
  than on Pi 5 LPDDR4X (decode is less bandwidth-starved here) — budget
  +2–3% rather than the +5.7% measured on Pi 5.

## Prefill kernel topology: VPDPBUSD ↔ BF16-SGEMM crossover

Two inner kernels, one outer loop, an M-threshold `M_crit` chosen empirically
at backend bind from a tiny micro-bench (analog to the Pi 5 dequant→cblas
crossover documented at `src/backends/cpu_neon/kernel_catalog.h:79-86`):

- **M < M_crit:** native VPDPBUSD (one VPDPBUSD per row × column-block; scale
  application per 32-block at the tail).
- **M ≥ M_crit:** dequant W4A8 → BF16 tile → BF16-SGEMM. In Phase 1b the
  SGEMM is `cblas_sgemm` against an OpenBLAS-linked path with BF16 staging;
  in Phase 2 it becomes a native AVX-512+BF16 VDPBF16PS kernel that
  delivers ~2× fp32 throughput.

`M_crit` is **machine-measured at first use** and cached in the kernel-policy
catalog; expected ≈ 64 on Zen 5 with DDR5-6400, but will vary on EPYC / Sapphire
Rapids / older hosts. Override via `GEIST_X86_MCRIT=<n>` for benchmark sweeps.

This is the same bifurcation `cpu_neon` already does on Apple Silicon (native
NEON at small M, dequant→AMX-Accelerate-SGEMM at large M — see the trampoline
discussion in `src/backends/cpu_neon/weight_resolve.c:1281`). The x86 design
inherits that pattern.

## Threading and CCD topology

Zen 5 9950X is two CCDs (CCD0: physical cores 0–7, CCD1: 8–15), each with its
own 32 MB L3, joined over Infinity Fabric. SMT is off by default for BLAS-
shaped workloads (5–15% loss empirically).

**Default profile:**

| Workload | Threads | Pinning | Why |
| --- | --- | --- | --- |
| Prefill | 16 | physical cores 0–15, both CCDs | compute-bound; needs aggregate FMA throughput |
| Decode | 8 | physical cores 0–7, CCD0 only | bandwidth-bound; one CCD saturates one memory controller; lower latency variance |

This asymmetry mirrors `cpu_neon`'s existing Pi 5 ("prefill=4, decode=3")
and Apple Silicon ("prefill=P-cores, decode=P-cores−1") patterns. The
*reason* is the same on every platform: decode contention on every saturated
core costs more than the missing core's compute.

CCD auto-detection is part of MVP. `geist_hw_probe` reads
`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list` to discover L3
domains and derives the per-CCD core list. The AMD-specific CPUID leaf
`0x80000026` is the documented fallback when `/sys` is unavailable
(containers, embedded). Portable to EPYC (up to 12 CCDs), Threadripper PRO,
and Intel hybrid (P/E split surfaces as two L3 domains as well).

Override env:

| Variable | Default | Effect |
| --- | --- | --- |
| `GEIST_PREFILL_THREADS` | 16 (auto) | thread count for prefill |
| `GEIST_DECODE_THREADS` | 8 (auto) | thread count for decode |
| `GEIST_PREFILL_CPUSET` | `0-15` (auto) | core list for prefill |
| `GEIST_DECODE_CPUSET` | `0-7` (auto) | core list for decode |
| `GEIST_DISABLE_SMT` | `1` | use physical cores only |

## Build target

`TARGET=linux` on x86_64 stops failing; the existing aarch64 path is
preserved.

`mk/target-linux.mk` branches on `LINUX_ARCH := $(shell uname -m)`:

```make
ifeq ($(LINUX_ARCH),x86_64)
  CC ?= cc
  BACKENDS       ?= cpu_x86 cpu_scalar
  CFLAGS_TARGET  := -march=x86-64-v3 -mtune=generic -O3 -fopenmp -ffast-math
  LDFLAGS_TARGET := -fopenmp
  LDLIBS_TARGET  := -lm
  GEMM_PROVIDER  ?= openblas
else
  # … existing aarch64 path …
endif
```

`-mtune=generic` is intentional (not `znver5`) — the same binary must
perform reasonably on Intel and older AMD. The per-TU `-march=` flags
listed under [ISA-tier dispatch](#isa-tier-dispatch) carry the specific
ISA tuning where it matters.

## Branch and merge strategy

`feat/cpu-x86` is a long-lived branch with weekly rebases on `main`. Phase
merges land on `main` with the new code present in the tree but **gated**:

- `cpu_x86` ships in `backend_registry.c` from Phase 0 forward.
- `mk/target-linux.mk` continues to **fail-fast on x86_64** with the current
  guidance message **until** the final win-criteria pass.
- During the gated phases, x86 users opt in via
  `make BACKENDS="cpu_x86 cpu_scalar"`.
- The final Phase-2 merge flips `target-linux.mk` from "error" to
  "`BACKENDS ?= cpu_x86 cpu_scalar`".

This matches the existing intent in `mk/backend-cpu_x86.mk:5-7` ("*gated
here so it compiles only when explicitly requested … keeping the dead
scaffolding out of the default CPU builds until a real x86 backend is
wired up*").

## Compiler

| Use | Compiler | Notes |
| --- | --- | --- |
| CI | gcc-14 | stability, Distro-default on Debian 13 / Ubuntu 26.04 |
| Bench-box | clang-18 | better AVX-512 scheduler; ~1–3% on hot inner kernels |
| Published numbers | clang-18 | tagged in the benchmark methodology box |

AOCC (AMD's clang fork) is out-of-scope for MVP — useful if every last
percent matters, but the distro-friction and proprietary patching is
not worth it for a baseline win.

## Quality gates (γ)

1. **MMLU ≥ NEON baseline − 0.2 pp.** NEON measures 0.445 on Gemma 4 E2B-it
   Q4_K_M (200-question shuffled 5-shot, see `benchmark/BENCHMARK.md`).
   x86 must score ≥ 0.443.
2. **Function calling + JSON: 14/14 on both suites**, no drop.
3. **Cross-ISA-consistency tests.** `tests/cross_isa_consistency.c` loads
   a committed Golden set (NEON-generated, ~100 random inputs per quant,
   fp32 expected outputs in `tests/goldens/`) and verifies that every
   x86 ISA tier (AVX2 / AVX-512 / +VNNI / +BF16) and `cpu_scalar` produces
   output within ±1e-3 of the Golden. Goldens are regenerated by
   `make gen-cross-isa-goldens` on a NEON host (Pi 5 or M1); regeneration
   is gated by a Makefile target so they don't drift silently.

## Win criteria

MVP is "done" — i.e. `target-linux.mk` flips x86_64 from "error" to default —
when all of the following hold on the reference host (AMD Ryzen 9 9950X) in
**two independent best-of-10 campaigns** (different day, different boot):

| Metric | Threshold |
| --- | --- |
| pp512 | ≥ 1.10× llama.cpp |
| tg32 | ≥ 1.05× llama.cpp |
| total tok/s (pp200 / tg50) | ≥ 1.10× llama.cpp |
| total tok/s vs ollama | ≥ 1.10× ollama (HTTP-overhead-bereinigt) |

If a campaign fails, the rule is **do not merge**: implement the Phase-2
VDPBF16PS native SGEMM lever (or the next-most-promising kernel improvement)
and re-measure.

If after Phase 2 the margin is not reached: publish an honest Pi-5-style page
("geist competitive within X% on pp512, leads decode by Y%, native BF16 SGEMM
in progress") and keep the gate up. **Do not publish wins that don't measure.**

## Methodology

Workstation mode, best-of-10, two independent campaigns. The 9950X is a
developer-class box; quiescing it perfectly is impractical. Best-of-10
approximates the uncontended ceiling per the existing M1 Max methodology
(`benchmark/BENCHMARK.md:64-75`).

**Bench-box prep — mandatory and documented in the methodology box of
`benchmark/BENCHMARK_ZEN5.md`:**

- BIOS: SMT off (or Linux `echo off > /sys/devices/system/cpu/smt/control`).
- Kernel cmdline: `mitigations=off transparent_hugepage=always amd_pstate=passive`.
- `cpupower frequency-set -g performance`.
- `cpupower idle-set -d 2` (block deeper idle states).
- Bench session is ssh-only (no X server, no browser, no IDE).
- `make bench-zen5-clean` macro performs setup, runs the sweep, restores.

## Reference pin (global refresh)

The win-criteria measurement uses a **fresh** llama.cpp pin captured at the
start of the campaign — not the existing `d05fe1d` carried in the Pi 5 / M1
pages. The new pin replaces `d05fe1d` across all three benchmark pages
(`BENCHMARK.md`, `BENCHMARK_PI5.md`, `BENCHMARK_ZEN5.md`) and the Pi 5 +
M1 numbers are re-run against the new reference as part of the bench-wrap
phase. This is the "*refresh vs current llama.cpp*" cadence already in the
project (commit `5510abd`).

Ollama is pinned by `ollama --version`. The GGUF that ollama serves is
verified to be hash-identical to `gguf_artifacts/gemma4-e2b-Q4_K_M.gguf`
via `ollama show <model> --modelfile`; if it differs (ollama ships a slightly
different Q4_K mixture), the bench is annotated.

## Workload

`bench_perf_sweep` on Gemma 4 E2B-it Q4_K_M:

- Prefill sweep: 128, 256, 512, 1024.
- Decode: tg32, tg64.
- Batched decode: m = 1, 8, 16, 64 (matches the M1-page table at
  `benchmark/BENCHMARK.md:204-210`).
- Total tok/s: pp200 / tg50 (the existing user-facing default from
  commit `c234a96`).

Three tables in `benchmark/BENCHMARK_ZEN5.md`: prefill sweep / decode /
batched, plus a methodology box (compiler, BIOS settings, kernel cmdline,
campaign dates, llama.cpp + ollama pins).

## Ollama integration

Adapter cloned from `benchmark/ollama_bench_pi5.py`. Measures via
`POST /api/generate` with `stream: false` and reads `prompt_eval_count`,
`prompt_eval_duration`, `eval_count`, `eval_duration` (nanoseconds, ollama-
internal). HTTP roundtrip overhead is bypassed.

Cold-start is discarded (first generate after `ollama serve` start). Model
identity is verified by hashing the file referenced in
`ollama show <model> --modelfile` against the project's pinned GGUF before
the run starts.

## CI

GitHub Actions Linux x86_64 job added on top of the existing matrix. The
default Linux runner is exercised across **all four ISA tiers** via
`GEIST_FORCE_ISA={avx2,avx512,avx512_vnni,avx512_bf16}` — even on a runner
that supports only AVX-512, the lower tiers are exercised by forcing-down
the dispatch.

The cross-ISA-consistency suite (Quality gate γ.3) runs in CI on the same
job: NEON-generated Goldens are checked into `tests/goldens/`, the x86 job
verifies all four x86 tiers + `cpu_scalar` against them.

Self-hosted 9950X CI runner (for native VNNI + BF16 verification) is a
Phase-3 follow-up, not MVP.

## Phase plan

| Phase | Duration | Deliverable | Merge trigger |
| --- | --- | --- | --- |
| **0 — Setup** | 3 d | branch; `target-linux.mk` x86 branch (still gated); `backend-cpu_x86.mk` per-ISA `-march=` patterns; `hw_probe` CCD detection; empty `cpu_x86` backend descriptor + registry slot; skeleton TUs (`kernel_w4a8.c` + per-tier TUs) | builds + tests pass on x86 with `cpu_scalar` fallback |
| **1a — VPDPBUSD W4A8 + Q8_0** | 2 w | native int8 decode + low-M prefill kernels, all 4 ISA tiers; predecoder Q4_K → W4A8; AVX-512 arms in `gemma4_kernels.c`; spec-head gate flip | cross-ISA-consistency tests pass; first end-to-end bench measurement |
| **1b — BF16-SGEMM trampoline** | 1.5 w | W4A8 → BF16 dequant; `cblas_sgemm` BF16 trampoline; `M_crit` auto-pick in `kernel_catalog`; Q6_K predecoder | pp512 measurement shows the M_crit crossover win |
| **2 — Native VDPBF16PS-SGEMM** | 2 w | native AVX-512 BF16 SGEMM (replaces `cblas_sgemm` for M ≥ M_crit); musl-static x86 artifact | win criteria met on campaign 1 |
| **Bench-wrap** | 5 d | fresh llama.cpp pin; re-run Pi 5 + M1 with the new pin; `BENCHMARK_ZEN5.md`; README update; campaign 2 on 9950X; flip `target-linux.mk` x86_64 default | both campaigns pass; all three benchmark pages updated |

Total: ≈ 6.5 weeks for User + Claude Code.

## Out of scope (deferred to Phase 3+)

- Q5_K, Q3_K, IQ2_S, IQ3_S, TQ2_0 native x86 kernels (the architecture above
  is additive — adding them later is "new predecoders + new kernel TUs",
  no dispatcher rewrite).
- AVX-512 specialization of KIVI.
- AOCC-compiled build variant.
- CCD-aware per-layer placement (the "Profile D" / NUMA-aware option from
  the design discussion).
- Long-context benchmarks (pp4k / 8k / 16k).
- Gemma 4 27B workload (60 GB RAM allows it; Pi 5 cross-comparison falls
  out, so it's a follow-up page rather than an MVP table).

## Risks (acknowledged, not in MVP path)

- **Zen 5 AVX-512 throughput verification.** Zen 5 is documented as full
  512-bit AVX-512 (vs Zen 4's 256-bit-doubled). The first Phase-1a
  measurement must verify that VPDPBUSD delivers near the theoretical
  64 int8 MAC / cycle / core before sinking time into further tuning.
- **Best-of-10 variance.** ±3–5% campaign-to-campaign is realistic on
  workstation hardware; the two-campaign rule above is the mitigation.
- **DDR5 frequency asymmetry.** DDR5-6400 vs DDR5-5600 changes the
  decode bandwidth ceiling by ~14%; the campaign documents the
  installed DIMM speed in the methodology box.
- **AVX-512 signal-frame size** (XSAVE ~2 KB) is larger than NEON's. Bench
  Python harnesses doing many short-lived calls pay a small context-switch
  cost; cosmetic only, not a workload concern.
