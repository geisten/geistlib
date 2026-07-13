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

To *guarantee* AVX-512 execution, run the suite on an AVX-512-capable
self-hosted or larger runner; tracked as a possible follow-up under the CI epic.

## Non-goals

- **Windows** — geist is POSIX (fork/execvp, Unix-domain sockets, mmap); the
  installer is POSIX `sh`. Not a target.
- **x86 without AVX-512** — runs via the runtime-dispatched `cpu_scalar` fallback;
  no separate SIMD tier below AVX2/x86-64-v3 is maintained.
