# CI test-coverage matrix

What each shipped environment is tested with, and the deliberate gaps. Source of
truth: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) (per-push /
per-PR) and [`release.yml`](../.gitea/workflows/release.yml) (per-tag build +
smoke). Goal: **every environment we ship a release artifact for is exercised by
the test suite**, not just built.

## Matrix

| Environment | Build | Unit | Int + e2e (real model) | musl-static | ASan/UBSan | CI job |
| :-- | :--: | :--: | :--: | :--: | :--: | :-- |
| **macOS arm64** (Accelerate/AMX) | ✅ | ✅ | ⚪ skip¹ | — | ✅⁴ | `build-test` |
| **macOS x86_64** (cpu_x86 AVX2, Accelerate) | ✅ | ✅ | ⚪ skip¹ | — | ✅⁴ | `build-test` |
| **Linux arm64** (cpu_neon, glibc) | ✅ | ✅ | ✅ | ✅ | ✅ | `build-test`, `build-test-musl`, `asan` |
| **Linux x86_64** (cpu_x86 AVX-512/VNNI, glibc) | ✅ | ✅ | ✅³ | ✅ | ✅² | `build-test-x86_64`, `build-test-musl-x86_64`, `asan-x86_64` |
| **Linux x86_64** (cpu_scalar, no SIMD) | ✅ | ✅ | — | — | — | `build-test-x86_64-scalar` |
| **Linux x86_64** (cpu_x86, clang) | ✅ | ✅ | — | — | — | `build-test-x86_64-clang`⁵ |

Every environment in [`release.yml`](../.gitea/workflows/release.yml)
(macos-arm64, linux-arm64, linux-x86_64) has build **and** test coverage. On
top of the matrix, dedicated legs gate every PR: TSan multi-session (x86_64),
the coverage ratchet (arm64), AVX-512 under Intel SDE, Vulkan on lavapipe, the
Metal GPU step inside the macOS arm64 leg, and `check-headers` (every public
header compiled standalone as C23 and C++17, inside `build-test`).

No self-hosted runner is attached to this repository: on a public repo, any
pull request can edit a workflow to target it. The hardware tier (Pi 5, native
AVX-512, Vulkan on a physical GPU, A/B and cross-engine benchmarks) lives in
the private `git.geisten.net/geisten-hw/geistlib-hw` (self-hosted Gitea),
which checks out this repo at a given ref — nightly on `main`, and on demand
for a branch or a reviewed PR.

## Caveats and deliberate gaps

1. **macOS int/e2e — skipped on purpose.** The real-model path (forward pass,
   tokenizer, KV, chat loops) runs on both Linux arches; macOS runners are the
   slowest and the model download dominates. macOS runs the full unit suite.
2. **x86_64 ASan/UBSan (`asan-x86_64`).** Model-free unit suite with the
   release binary's backends and GEMM, UBSan halting, leak detection on.
3. **x86_64 int/e2e — required.** Includes a `GEIST_FORCE_ISA=avx2` pass, so
   the non-AVX-512 dispatch is exercised even on runners that have AVX-512 and
   an unguarded AVX-512 kernel (SIGILL on x86-64-v3) fails the PR.
4. **macOS ASan/UBSan (`build-test`, both arches).** Catches clang-only
   breakage at -O1 that the gcc sanitizer legs cannot see, e.g. an `omp simd`
   the instrumentation cannot satisfy (`-Wpass-failed=transform-warning` under
   `-Werror`). Model-free unit suite.
5. **x86_64 clang (`build-test-x86_64-clang`).** Every other x86_64 leg uses
   gcc. clang outlines OpenMP `parallel for` bodies into functions that do not
   inherit a kernel's `target` attribute, which can abort instruction
   selection (#415); this leg builds and runs the model-free unit suite.

### AVX-512 is exercised *opportunistically*, not guaranteed

The x86_64 release binary is **x86-64-v3 baseline (AVX2/FMA) with the AVX-512/VNNI
kernels compiled per translation unit and runtime-dispatched via `hw_probe`
(`__builtin_cpu_supports`)** — one binary runs on any x86-64-v3 CPU and uses
AVX-512 only where present, **no SIGILL**. GitHub-hosted runners do **not**
guarantee an AVX-512 CPU, so:

- `build-test-x86_64` **reports** whether the runner CPU has AVX-512 (see its
  "Report CPU features" step). When present, the AVX-512 kernels are exercised;
  when absent, runtime dispatch falls back to AVX2/scalar and the AVX-512 kernels
  are *built but not run* on that leg.
- `build-test-x86_64-scalar` **guarantees** the SIMD-free portable reference
  builds and passes — the baseline every SIMD kernel is checked against, runner
  CPU notwithstanding.

The `avx512-sde` job **guarantees** AVX-512/VNNI execution on every PR: the shipped x86_64 binaries run under Intel SDE's Sapphire Rapids
emulation, `test_x86_isa_dispatch_unit` hard-fails unless the dispatcher
actually selects the VNNI tier (a silent downgrade cannot pass), and the
targeted W4A8/W8A8/Q4Kx8/Q6K/i2s and INT8-KV attention kernel tests execute
with their built-in scalar-parity checks. Coverage tiers for x86 SIMD are therefore: **built**
(all legs) → **opportunistically executed** (`build-test-x86_64`, CPU
permitting, reported per run) → **guaranteed executed under emulation**
(`avx512-sde`) → **real hardware** (`native-avx512` in
`git.geisten.net/geisten-hw/geistlib-hw`, nightly; mandatory
AVX-512F/BW/DQ/VL+VNNI enforced by the same dispatch test, BF16 reported
explicitly either way).

## Model fixtures — what the int/e2e legs actually load

| Fixture | Family it proves | Source (pinned) | Mandatory where |
| :-- | :-- | :-- | :-- |
| `gemma4-e2b-Q4_K_M.gguf` (~3.1 GB) | gemma (primary reference) | `unsloth/gemma-4-E2B-it-GGUF` | Linux arm64 + x86_64 int/e2e |
| `smollm2-360m-instruct-q8_0.gguf` (~369 MB) | llama populator + GPT-2-BPE tokenizer mode | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`, revision- and SHA-256-pinned in the Makefile (`LLAMA_MODEL_SHA256`), Apache-2.0 | Linux arm64 + x86_64 int/e2e |
| `qwen3-0.6b-q8_0.gguf` (~609 MB) | qwen3 geometry, per-head Q/K norm and tokenizer mode | `Qwen/Qwen3-0.6B-GGUF`, revision- and SHA-256-pinned in the Makefile | Linux arm64 + x86_64 int/e2e |
| `qwen3.5-0.8b-q8_0.gguf` (~780 MB) | qwen35 hybrid DeltaNet/attention family | `unsloth/Qwen3.5-0.8B-GGUF`, revision- and SHA-256-pinned in the Makefile | Linux arm64 + x86_64 int/e2e |
| `audio_tower.safetensors` (~614 MB) | Conformer audio encoder, W8A8 attention and LConv | Range-extracted and SHA-256-pinned (`make fetch-audio-tower`) | `audio-smoke` (arm64, NEON) + `audio-parity-x86_64` (AVX-512/VNNI) |

The llama tests (`test_llama_load_int`, `test_llama_e2e_int`) are **executed,
not merely built**: both Linux int/e2e legs fetch the model
(`make fetch-llama-model`, which verifies the SHA-256 on every run and fails
loudly on a truncated download, a corrupted cache, or a changed upstream) and
run with `GEIST_STRICT_FIXTURES=gguf`, which turns a "model not found" skip
into a failure. The CI cache key embeds the content pin, so re-pinning the
model rotates the cache. Local `make test` without the model skips cleanly.

### Metal is built and executed on a real GPU in every PR

Hosted `macos-15` runners expose a usable Metal device. The macOS arm64 leg
builds `BACKENDS="metal cpu_neon cpu_scalar"` and runs a mandatory GPU step:

- `test_backend_metal_probe` — registration, backend lifecycle, buffer
  round-trip (host↔device);
- `test_backend_metal_parity_unit` — `linear_m1`/`linear_mN` numerical
  parity against `cpu_scalar` on identical weight bytes for every dtype the
  metal resolver covers (Q4_0/Q4_1/Q8_0, the K-quants, IQ4_XS/IQ4_NL,
  Q3_K/IQ3_S, F32), with x/w/y allocated through the backend's buffer API so
  the GPU path runs by construction. Tolerance 1e-3 relative; several
  formats pin their block scales small because the simdgroup GEMM stages
  weights in **half**, whose integer lattice ends at 2048 (a staging-precision
  property, not a bug; unpinned scales produce √n·ulp noise).

In this step a SKIP (exit 77) **fails**: on `macos-15` a device is expected,
and a skipped gate must not read as a green one. On Linux legs metal is not
built and both tests skip cleanly in the unit suite.

Model e2e on Metal is a separate tier. Every PR builds the fixture-gated
`test_qwen35_metal_e2e_int`, but the macOS PR leg does not run the real-model
suite. The weekly `gemma4-metal-smoke` workflow generates end to end with the
E2B reference on a hosted macOS runner (#305–#307); Qwen35 Metal e2e is a
manual, fixture-provisioned gate.

## Vulkan: software tier on every PR, hardware tier nightly

The `vulkan-lavapipe` job builds `BACKENDS="vulkan cpu_x86 cpu_scalar"` and
executes `test_backend_vulkan_registry_unit`, the buffer round-trip test and
`test_backend_vulkan_linear_parity` on Mesa's **lavapipe** — a real Vulkan
implementation (loader → ICD → SPIR-V pipelines), so the full dispatch chain
runs on a hosted runner. The tests are invoked directly: exit 77 (missing
loader/device) fails the job. `vulkaninfo --summary` is logged per run.
Tolerances live in the parity test: 1e-3 relative (f32 paths), 2e-2 for the
f16 coopmat path where exposed. Minimum Vulkan: 1.2.

Physical-GPU validation is the `vulkan-gpu` workflow in the private
`git.geisten.net/geisten-hw/geistlib-hw`, nightly on `main` and on demand for
a ref: the same tests on a desktop with a discrete GPU (RTX 2080 Ti). A
missing device is a failure, not a skip, same as the lavapipe leg. It does not
gate PRs — a PR that touches the Vulkan backend is dispatched there by hand
before merging. Runner setup and operational notes: that repo's README.

On top of the three, this tier runs model e2e the emulated tiers cannot
afford: `test_known_answer_e2e` (five cloze prompts, floor 4/5) and
`test_prefill_determinism_int`, against a GGUF on the runner host named by the
repo variable `GEIST_GGUF_PATH`. Kernel parity does not imply a working forward
pass.

### Diagnostics are kept as artifacts

Both software-emulated tiers upload a `ci-diagnostics/` artifact on every run,
pass or fail:

| job | artifact | holds |
| :-- | :-- | :-- |
| `vulkan-lavapipe` | `vulkan-lavapipe-diagnostics` | `vulkaninfo --summary`, mesa/loader package versions, per-test output |
| `avx512-sde` | `avx512-sde-diagnostics` | host `lscpu`, per-test output under SDE |

A Mesa or driver update can land under the job without a change in this
repository, so the first question after a red run is what the environment
was; hence `if: always()`.

## Coverage ratchet

The `coverage` job runs on every push to `main` (not on PRs: ~65 min), and
pushes to `main` are never cancelled by a later merge, so every main commit
is measured. It builds `MODE=cov` (gcc-14, `-O1 --coverage`, Linux arm64)
and runs the model-free unit suite plus the real-model int suite with
`GEIST_STRICT_FIXTURES=gguf` — a fixture skip would hollow out the
measurement, so it fails instead. (e2e is not measured: it re-drives the same
engine paths for another ~30 min and moves the numbers by noise.) `gcovr`
produces JSON + Cobertura XML + HTML (uploaded as the `coverage-report` artifact), and
`scripts/coverage_gate.py` gates per-subsystem **line and branch** coverage
against the versioned baselines in `benchmark/coverage_baselines.json`, with
the overall figure and per-subsystem table published to the job summary.

- **Scope**: `src/base`, `src/engine`, `src/io`, `src/formats`, `src/quant`,
  `src/archs/transformer`, `src/archs/audio_conformer`,
  `src/archs/vision_siglip`, `src/backends/common`, `src/backends/cpu_scalar`.
  `vision_siglip` is barely reached by unit + int (~12 % line); its baseline
  only keeps it from falling.
- **Exclusions, each deliberate**: `src/backends/cpu_x86`, `metal/`,
  `vulkan/` cannot execute on this runner (their correctness legs live in
  the x86/metal/vulkan jobs); `cpu_neon` is measured but not gated (SIMD
  leg, one runner family); `third_party/` and generated `*_spv.h` are not
  our code; tests and benches are the instruments, not the subject.
- **Ratchet**: measured on `main`, rounded DOWN to whole percent, tolerance
  0.5 pp per metric. Raising after an improvement is routine; lowering is a
  reviewed, justified edit of the baselines file. An unset (`null`) baseline
  fails the gate and prints the measured value, so new subsystems enter by
  deliberate commit, not by silent adoption.
- **Security floor**: `src/io` (the malformed-GGUF parser surface) holds a
  hard 85 % line-coverage floor regardless of the ratchet. Raise it with the
  coverage; never let it drift down.
- **Self-test**: `tests/test_coverage_gate_py.py` (hermetic, runs in
  `make test-py`) proves the gate fires on regression, empty scope, unset
  baseline and floor violation — and passes when on-baseline.

## Fuzzing (parsers)

Two harnesses in `tests/`, ordered by attack surface — both need neither a
model nor the network, because the input *is* the format:

- `fuzz_gguf.c` → `gguf_open_memory` plus every accessor on the result
  (tensor list, dtype names, metadata getters, one index past the end) and it
  reads each tensor's payload bytes, so a length that escapes the mapping is a
  read ASan reports. The whole file is untrusted input.
- `fuzz_tokenizer.c` → `gguf_tokenizer_load_copy` out of the same file, then
  encode/decode/`id_for_text`. The GGUF is closed and freed right after the
  load: copy mode promises independence from the mapping, so a surviving
  pointer becomes a use-after-free instead of a silent success. Vocab, scores,
  `token_type` and merges are all attacker-controlled arrays.

Both are entry points for libFuzzer *and* carry a deterministic PRNG driver
behind `GEIST_FUZZ_STANDALONE`, which is what lets them run in a gcc job:

| Target | Job | Cost |
| :-- | :-- | :-- |
| `make MODE=asan fuzz` (3000 runs/harness, fixed seed) | `asan-x86_64` | < 1 s, reuses that job's sanitizer tree |
| `make fuzz-libfuzzer FUZZ_SECONDS=30` (coverage-guided) | `build-test-x86_64-clang` | ~1 min run, ~7 M execs per target |

`MODE=fuzz` (asan + ubsan + `-fsanitize=fuzzer-no-link`) instruments the
library in its own build tree. Do not substitute
`MODE=asan EXTRA_CFLAGS=-fsanitize=fuzzer-no-link`: the build system does not
track flags, an existing asan tree is reused as-is, and the fuzzer runs blind.

The corpus is generated by the harness (`--seed`), not checked in, and not
persisted between runs. A nightly long run is not worth it until the corpus is
cached (`actions/cache`, one entry per target): from the same seed, 10 minutes
explore little beyond what 30 s at ~230 k exec/s already reach.

## Non-goals

- **Windows** — no supported toolchain, CI leg or release artifact exists; it is
  not a target.
- **x86 below x86-64-v3** — the shipped baseline is AVX2/FMA. AVX-512/VNNI is
  runtime-dispatched where present, with per-kernel AVX2/scalar fallbacks; no
  separate SIMD tier below x86-64-v3 is maintained.
