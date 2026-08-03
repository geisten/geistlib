# CI test-coverage matrix

What each shipped environment is tested with, and the deliberate gaps. Source of
truth: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) (per-push /
per-PR) and [`release.yml`](../.github/workflows/release.yml) (per-tag build +
smoke). Goal: **every environment we ship a release artifact for is exercised by
the test suite**, not just built.

## Matrix

| Environment | Build | Unit | Int + e2e (real model) | musl-static | ASan/UBSan | CI job |
| :-- | :--: | :--: | :--: | :--: | :--: | :-- |
| **macOS arm64** (Accelerate/AMX) | ✅ | ✅ | ⚪ skip¹ | — | — | `build-test` |
| **Linux arm64** (cpu_neon, glibc) | ✅ | ✅ | ✅ | ✅ | ✅ | `build-test`, `build-test-musl`, `asan` |
| **Linux x86_64** (cpu_x86 AVX-512/VNNI, glibc) | ✅ | ✅ | ⚠️³ non-blocking | ✅ | ⚪² | `build-test-x86_64`, `build-test-musl-x86_64` |
| **Linux x86_64** (cpu_scalar, no SIMD) | ✅ | ✅ | — | — | — | `build-test-x86_64-scalar` |

Every environment in [`release.yml`](../.github/workflows/release.yml)
(macos-arm64, linux-arm64, linux-x86_64) now has build **and** test coverage
here — previously x86_64 was built + smoke-run only, never tested.

## Caveats and deliberate gaps

1. **macOS int/e2e — skipped on purpose.** The real-model product path (forward
   pass, tokenizer, KV, agent/chat loops) is exercised on **both** Linux arches;
   macOS runners are the slowest/costliest and the model download dominates.
   Revisit if a macOS-specific product-path bug ever appears. macOS still runs
   the full unit suite.
2. **x86_64 ASan/UBSan — not yet.** The sanitizer job runs on arm64; it catches
   memory/UB bugs in the shared C engine + kernels regardless of SIMD path. An
   x86-specific ASan leg is a reasonable follow-up if an x86-only UB is suspected.
3. **x86_64 int/e2e — non-blocking, tracked in #96.** This leg immediately did
   its job: it caught a real shipping bug — the `cpu_x86` backend has AVX-512
   kernels in the forward path that lack a runtime CPU guard, so the real-model
   prefill SIGILLs on an AVX-512-less x86-64-v3 runner. `build-test-x86_64`'s
   **build + unit stay REQUIRED**; the int/e2e **step** is `continue-on-error`
   (still runs and logs the failures, but doesn't gate) until #96 guards those
   kernels. Flip it back to required when #96 lands. See #96 for the audit scope.

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

Since #184 the `avx512-sde` job **guarantees** AVX-512/VNNI execution on every
PR: the shipped x86_64 binaries run under Intel SDE's Sapphire Rapids
emulation, `test_x86_isa_dispatch_unit` hard-fails unless the dispatcher
actually selects the VNNI tier (a silent downgrade cannot pass), and the
targeted W4A8/W8A8/Q4Kx8/Q6K/i2s kernel tests execute with their built-in
scalar-parity checks. Coverage tiers for x86 SIMD are therefore: **built**
(all legs) → **opportunistically executed** (`build-test-x86_64`, CPU
permitting, reported per run) → **guaranteed executed under emulation**
(`avx512-sde`) → **real-hardware perf/e2e** (self-hosted only; runner
contract: labels `[self-hosted, linux, x64, geist-avx512]`, mandatory
AVX-512F/BW/DQ/VL+VNNI enforced by the same dispatch test, BF16 reported
explicitly either way).

## Model fixtures — what the int/e2e legs actually load

| Fixture | Family it proves | Source (pinned) | Mandatory where |
| :-- | :-- | :-- | :-- |
| `gemma4-e2b-Q4_K_M.gguf` (~2.9 GB) | gemma (primary reference) | `unsloth/gemma-4-E2B-it-GGUF` | Linux arm64 + x86_64 int/e2e |
| `smollm2-360m-instruct-q8_0.gguf` (~369 MB) | llama populator + GPT-2-BPE tokenizer mode | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`, SHA-256-pinned in the Makefile (`LLAMA_MODEL_SHA256`), Apache-2.0 | Linux arm64 + x86_64 int/e2e |

The llama tests (`test_llama_load_int`, `test_llama_e2e_int`) are **executed,
not merely built**: both Linux int/e2e legs fetch the model
(`make fetch-llama-model`, which verifies the SHA-256 on every run and fails
loudly on a truncated download, a corrupted cache, or a changed upstream) and
run with `GEIST_STRICT_FIXTURES=gguf`, which turns a "model not found" skip
into a failure. The CI cache key embeds the content-pin prefix, so re-pinning
the model rotates the cache. Local `make test` without the model keeps
skipping cleanly (#180).

### Metal is built AND executed on a real GPU in every PR (#181)

The manual probe spike answered its question: **hosted `macos-15` runners
expose a usable Metal device**. The macOS matrix leg therefore builds
`BACKENDS="metal cpu_neon cpu_scalar"` and runs a mandatory GPU step:

- `test_backend_metal_probe` — registration, backend lifecycle, buffer
  round-trip (host↔device);
- `test_backend_metal_parity_unit` — Q4_K/Q6_K/F32 `linear_m1`/`linear_mN`
  numerical parity against `cpu_scalar` on identical weight bytes, with x/w/y
  allocated through the backend's buffer API so the GPU path runs by
  construction (unregistered pointers would take the host fallback and gate
  the CPU). Tolerance 1e-3 relative; the Q6_K test data pins block scales
  small because the simdgroup GEMM stages weights in **half**, whose integer
  lattice ends at 2048 — a documented staging-precision property, not a bug.

In this step a SKIP (exit 77) **fails**: on `macos-15` a device is expected,
and a skipped gate must not read as a green one. On Linux legs metal is not
built and both tests skip cleanly in the unit suite.

Not yet covered: a full GGUF e2e **on** metal. The metal resolver covers
Q4_K/Q6_K/F32 only (the SmolLM2 fixture is Q8_0), and the gemma-on-metal
path currently fails at head resolution (`output.weight` on a
tied-embedding model) — tracked in #181/#186 as backend feature work, not
CI wiring.

## Vulkan: software tier on every PR, hardware tier self-hosted (#182)

The `vulkan-lavapipe` job builds `BACKENDS="vulkan cpu_x86 cpu_scalar"` and
executes `test_backend_vulkan_registry_unit`, the buffer round-trip test and
`test_backend_vulkan_linear_parity` on Mesa's **lavapipe** — a real Vulkan
implementation (loader → ICD → SPIR-V pipelines), so the full dispatch chain
runs on a hosted runner. The tests are invoked directly: exit 77 (missing
loader/device) fails the job. `vulkaninfo --summary` is logged per run.
Tolerances live in the parity test: 1e-3 relative (f32 paths), 2e-2 for the
f16 coopmat path where exposed. Minimum Vulkan: 1.2.

Physical-GPU validation (model load, prefill, decode) is self-hosted —
runner contract: labels `[self-hosted, linux, x64, geist-vulkan]`, working
loader + ICD for the physical device; there the same three tests plus a
model e2e run, and a missing device is a failure, not a skip.

## Coverage ratchet (#185)

The `coverage` job builds `MODE=cov` (gcc-14, `-O1 --coverage`, Linux arm64)
and runs the model-free unit suite plus the real-model int suite with
`GEIST_STRICT_FIXTURES=gguf` — a fixture skip would hollow out the
measurement, so it fails instead. (e2e is deliberately not measured: on an
instrumented -O1 build it re-drives the same engine paths through the CLI
wrappers for another ~half hour and changes the numbers by noise.) `gcovr` produces JSON + Cobertura XML +
HTML (uploaded as the `coverage-report` artifact), and
`scripts/coverage_gate.py` gates per-subsystem **line and branch** coverage
against the versioned baselines in `benchmark/coverage_baselines.json`, with
the overall figure and per-subsystem table published to the job summary.

- **Scope**: `src/base`, `src/engine`, `src/io`, `src/formats`,
  `src/archs/transformer`, `src/backends/common`, `src/backends/cpu_scalar`.
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
  hard line-coverage floor regardless of the ratchet — pinned at the
  measured 34 % for now (the safetensors half of `src/io` is fixture-gated
  and skips on CI, which keeps the intended 35 % just out of reach); raise
  it with the coverage, never let it drift down.
- **Self-test**: `tests/test_coverage_gate_py.py` (hermetic, runs in
  `make test-py`) proves the gate fires on regression, empty scope, unset
  baseline and floor violation — and passes when on-baseline.

## Non-goals

- **Windows** — geist is POSIX (fork/execvp, Unix-domain sockets, mmap); the
  installer is POSIX `sh`. Not a target.
- **x86 without AVX-512** — runs via the runtime-dispatched `cpu_scalar` fallback;
  no separate SIMD tier below AVX2/x86-64-v3 is maintained.
