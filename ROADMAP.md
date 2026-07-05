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
| **One-file install** | Engine + model in one dependency-free binary | ✅ prebuilt binaries + `make EMBED_MODEL=…` |
| **Small-model agents** | A tight harness so a 2 B model rivals a bigger one on a narrow task | 🚧 whitelist tool loop, PMI routing + forced calls ([agent.md](docs/agent.md)) |
| **Memory for small models** | Recall sized to what these models can hold — no vector store | 🚧 file-based memory palace ([agent.md](docs/agent.md#the-memory-palace--mindh)) |
| **Models that adapt** | Dynamic specialization, learning, self-organization | 🔬 research |

<sub>✅ shipped · 🚧 in progress · 🔬 exploring</sub>

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
