# geist Roadmap

## Multi-platform distribution & the GEMM/FFT backend abstraction

**Decision (June 2026): ship per-platform static binaries via a CI matrix — *not*
a single Cosmopolitan/APE binary.** Cosmopolitan would trade away exactly what
geist exists for (per-platform SIMD + the platform's matrix accelerator) to gain
OS reach that edge inference does not need. It is also a hard blocker today:
geist's threading is OpenMP (no libgomp under Cosmopolitan) and its fast paths
use OpenBLAS / Accelerate / FFTW3, none of which link into an APE. The
industry-standard alternative (llama.cpp model) delivers the same "self-contained,
runs anywhere" experience without sacrificing a single performance path.

### Target architecture — BLAS/FFT optional, chosen per platform

The enabling refactor is **one `geist_gemm` / `geist_gemv` abstraction** that all
dense-fp32 call sites route through, with swappable backends. The same pattern
applies to the audio FFT (vDSP | pocketfft). BLAS/FFT become *optional, per
platform* — not a global "with or without".

| Platform        | Quant path  | Dense fp32             | FFT             | Binary                                  |
| :-------------- | :---------- | :--------------------- | :-------------- | :-------------------------------------- |
| **macOS-ARM**   | native int8 | Accelerate / **AMX**   | vDSP            | system-self-contained (framework always present) |
| **linux-arm64** | native int8 | **native NEON fp32**   | vendored pocketfft | **musl-static, BLAS-free, tiny**     |
| **x86-64 Linux** *(✅)* | native int8 (AVX-512) | AVX-512 / OpenBLAS | pocketfft   | **prebuilt musl-static** (+ from source) |

Why this maps to "fastest per platform": the quant matmuls (the bulk of text
inference) already win natively on ARM (measured: native int8 ≈ 30 t/s vs the
dequant→OpenBLAS-sgemm path ≈ 13 t/s on Pi 5). Accelerate/AMX is a real,
Apple-only hardware win for dense fp32 and is always present on macOS, so the Mac
binary keeps it for free. x86-64 now has a native **AVX-512 / VNNI** backend — it
matches-to-beats llama.cpp on a Ryzen 9 9950X (Zen 5); OpenBLAS remains only for
the vision/audio dense fp32 path.

### Work sequence

**Status (June 2026): Steps 1–4 complete and CI-green on `main` (macOS-arm64 +
linux-arm64: build + unit tests + clang-format gate). Tagging a `v*` tag
(`git tag v0.2.0`) builds and attaches the ARM64 artifacts via `release.yml`.**

1. ✅ **`geist_gemm` / `geist_gemv` abstraction** — all dense-fp32 cblas call
   sites route through one interface; backend selectable at build time
   (`GEMM_PROVIDER=accelerate|openblas|native`, via `mk/gemm-<provider>.mk`).
   A future MKL/BLIS/ARMPL provider slots in as another fragment.
2. ✅ **Measured** — dense fp32 is **~2.6 %** of text inference (Pi 5), and
   native int8 beats the OpenBLAS dequant→sgemm path **2.3×** for the quant
   matmuls. Well under the 10 % gate → linux-arm64 goes BLAS-free.
3. ✅ **Native NEON fp32 GEMM/GEMV (4×4 register-blocked) + vendored radix-2
   FFT** — the BLAS-free, FFTW-free build (`GEMM_PROVIDER=native`) depends only
   on libc/libm/libgomp; quality gate unchanged (28/28). Audio keeps vDSP on macOS.
4. ✅ **CI matrix v0.1 = ARM only** — `release.yml` builds `linux-arm64` (fully
   static ELF, no deps) + `macos-arm64` (static libomp + Accelerate, system
   frameworks only). Both validated; a `geist` CLI is the entry point.
5. ✅ **v0.2 — AVX backend** — x86-64 **Linux**: native AVX-512 / VNNI, matches-to-
   beats llama.cpp on a Ryzen 9 9950X (Zen 5); **prebuilt x86-64 binaries now ship**
   (model-less + embedded `geist-bitnet`). Intel-Mac and Windows: not yet.

### Deliberate non-goals / deferred

- **Cosmopolitan / APE** — rejected (see above).
- **Windows binaries** — the x86-64 AVX-512 backend ships prebuilt (linux-x86_64,
  competitive with llama.cpp); Intel-Mac and **Windows** support remain deferred.

### Open packaging details (not design forks)

- ✅ **Release CLI artifact** — the `geist` CLI (`tools/geist.c`); the static
  binaries package it. `examples/simple_generate` stays as the embedding example.
- **Windows toolchain** — MinGW vs MSVC; deferred (see non-goals).
