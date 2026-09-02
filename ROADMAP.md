# geist Roadmap

geist exists to make capable AI run on hardware people already own — CPUs, small
GPUs, down to a Raspberry Pi — and to see how far small, heavily quantized models
go when the *whole* stack is built around them. It began as one developer's way of
understanding how these models work, and it's still an experiment: everything here
is open to help. The mission in full is in the
[README](README.md#where-this-is-going).

## Tracks

| Track | Goal | Status |
| :-- | :-- | :-- |
| **Max quantization** | Ternary (1.58-bit) & binary as first-class citizens, not a bolt-on | ✅ BitNet `I2_S`/`TQ2_0`, integer-only kernels; beats bitnet.cpp on Pi 5 & x86 |
| **Fast per platform** | The fastest path on *each* target, not a lowest common denominator | ✅ ARM64 NEON · macOS Accelerate/AMX · x86-64 AVX-512/VNNI |
| **One-file install** | Engine + model in one dependency-free binary | ✅ `geist_model_load_from_memory` aliases weights zero-copy out of a binary's read-only data |
| **GPU where it helps** | Optional accelerators, never a requirement | 🚧 Metal wins on Qwen3.8-27B Q4_0; Gemma prefill remains ~36 % behind · Vulkan decode at ~86 %, prefill open |
| **Production hardening** | Memory-safe loaders, deterministic execution and allocation-free hot paths in modern C23 | 🚧 Core review findings #325–#336 are complete except the bounded C23 migration in #328; performance and quality remain merge gates |
| **Models that adapt** | Dynamic specialization, learning, self-organization | 🔬 research |

<sub>✅ shipped · 🚧 in progress · 🔬 exploring</sub>

## Scope

geistlib is an inference engine and nothing else: it loads models and produces
tokens, and has no opinion about chat templates, tool use, authorization or
whether a model may act. Its extension boundary is the C API in `include/geist.h`
([API_CONTRACT.md](docs/API_CONTRACT.md)) — agent loops, resident services,
product UX and domain evaluations belong to whoever links it.

That split is deliberate. An engine that stays application-neutral can be
embedded by anyone; one that grows a product opinion can only be embedded by
people who share it.

## Near-term: hardening without regressions

The next engineering pass turns the whole-codebase review into production
invariants. Correctness and memory safety land before broader API cleanup;
performance work remains incremental so every change can be measured against its
parent rather than hidden inside a large rewrite.

| Priority | Workstream | Issues | Done when |
| :-- | :-- | :-- | :-- |
| **Done** | Review hardening: size safety, correctness, lifecycle, concurrency and allocation-free hot paths | [#325](https://github.com/geisten/geistlib/issues/325), [#326](https://github.com/geisten/geistlib/issues/326), [#327](https://github.com/geisten/geistlib/issues/327), [#329](https://github.com/geisten/geistlib/issues/329), [#330](https://github.com/geisten/geistlib/issues/330), [#331](https://github.com/geisten/geistlib/issues/331), [#332](https://github.com/geisten/geistlib/issues/332), [#333](https://github.com/geisten/geistlib/issues/333), [#334](https://github.com/geisten/geistlib/issues/334), [#335](https://github.com/geisten/geistlib/issues/335), [#336](https://github.com/geisten/geistlib/issues/336) | ✅ The reviewed invariants are covered by release, sanitizer and performance gates |
| **P1** | Complete the bounded C23 array-contract migration | [#328](https://github.com/geisten/geistlib/issues/328) | The remaining hot `linear_*` family is migrated independently with exact-output, optimized-assembly and parent/patch benchmark evidence |
| **P1** | Metal throughput for 4B-class models | [#322](https://github.com/geisten/geistlib/issues/322) | The small-shape GEMM path closes the measured prefill gap without decode, quality or memory regressions |
| **P1** | Metal model residency | [#357](https://github.com/geisten/geistlib/issues/357) | Device upload no longer keeps an unnecessary second model-sized mapping resident, with loader and inference parity retained |
| **P1** | Optimization-independent PTQTP parity gate | [#376](https://github.com/geisten/geistlib/issues/376) | GEMM/GEMV parity has a proved invariant and passes release plus ASan/UBSan without weakening quality coverage |
| **Cross-cutting** | Same-protocol performance matrix | [#364](https://github.com/geisten/geistlib/issues/364) | Current models and systems publish reproducible results with pinned revisions, model hashes, settings and raw samples |
| **P2** | BitNet x86 decode follow-ups | [#212](https://github.com/geisten/geistlib/issues/212) | A measured threading, placement, LM-head or kernel change improves decode without harming prefill or correctness |

### Definition of done

Every item above carries the same non-negotiable gates:

- **C23 contracts:** array lengths/capacities precede the arrays they describe;
  real minimum-size contracts use VLA parameters. Code uses `bool`, `nullptr`,
  typed `constexpr` constants and `[[nodiscard]]` where applicable; `restrict` is
  added only when non-aliasing is proven. `typeof` and `auto` may simplify local
  expressions without hiding ownership or changing generated code.
- **Safe cursor arithmetic:** validate `len <= (size_t)(end - p)` before advancing.
  Neither `p + len` nor `&p[len]` is a valid bounds check for an untrusted length.
- **Benchmark each relevant change:** record a parent baseline, then rerun the
  affected microbenchmark and end-to-end workload after every performance-relevant
  change on identical hardware, model, compiler flags, thread placement and power
  settings. Keep the raw samples with the PR.
- **No regressions:** a repeatable throughput, latency, allocation, peak-memory or
  quality regression blocks completion. Safety/refactor changes preserve exact
  output where possible; existing logit, accuracy, perplexity and WER tolerances
  may not be widened to make a change pass.

## Distribution: one static binary per platform

**Decided (June 2026): ship per-platform static binaries via a CI matrix — not a
single Cosmopolitan/APE binary.** An APE would trade away the whole point (per-
platform SIMD + the platform's matrix accelerator) for OS reach that edge
inference doesn't need — and it can't link geist's OpenMP + BLAS/FFT fast paths
anyway. The llama.cpp model — a small static binary per target — gives the same
"runs anywhere" feel with no performance sacrifice.

The enabler is one `geist_gemm`/`geist_gemv` abstraction that every dense-fp32 call
site routes through, so BLAS/FFT become *optional, per platform*:

| Platform | Quant path | Dense fp32 | Binary |
| :-- | :-- | :-- | :-- |
| **macOS-ARM** | native int8 | Accelerate / AMX | system-self-contained |
| **linux-arm64** | native int8 | native NEON | musl-static, BLAS-free, tiny |
| **x86-64 Linux** | native int8 (AVX-512) | AVX-512 / OpenBLAS | prebuilt musl-static |

Native int8 is the bulk of text inference and already wins on ARM (≈30 vs ≈13 t/s
on a Pi 5 against the dequant→OpenBLAS path), so dense fp32 — measured at ~2.6 % of
text inference — no longer forces a BLAS dependency. All of the above ships
CI-green on `main`; a `v*` tag builds and attaches the artifacts.

## Deferred (non-goals for now)

- **Cosmopolitan / APE** — rejected, see above.
- **Windows & Intel-Mac binaries** — the x86-64 AVX-512 backend ships for Linux;
  Windows (MinGW vs MSVC) and Intel-Mac remain deferred.
