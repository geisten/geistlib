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
| **Small-model agents** | A tight harness so a 2 B model rivals a bigger one on a narrow task | ✅ bounded static + per-request dynamic tools, typed forced calls, host round-trips ([agent.md](docs/agent.md)) |
| **Memory for small models** | Recall sized to what these models can hold — no vector store | 🚧 file-based memory palace, in the [geistwissen](https://github.com/geisten/geistwissen) consumer (the engine ships the tool-use interface) |
| **Models that adapt** | Dynamic specialization, learning, self-organization | 🔬 research |

<sub>✅ shipped · 🚧 in progress · 🔬 exploring</sub>

## Adapter ecosystem

Geist remains application-neutral. Its resident service and dynamic-tools
protocol are the stable extension boundary; authorization, execution, product
UX, distribution, and domain evaluations belong to adapter repositories.
The first adapter is maintained in
[`geisten/geist-home-assistant`](https://github.com/geisten/geist-home-assistant),
including its independent roadmap and release phases.

### Phase 3: host-neutral dynamic tools — complete

The server accepts an immutable `tools` array per request. Tool names are
offered capabilities rather than compiled assumptions; arguments use a bounded,
documented JSON-Schema subset and are checked before the host boundary. The host
executes, returns correlated results, and Geist continues the conversation.
Multiple calls, global budgets, one bounded retry, cancellation, typed forced
arguments and low-confidence clarification are covered by deterministic tests.

`make dynamic-example-host` builds a separate C
calculator/profile host with no HA or model-runtime dependency, proving that
other applications can implement and compile against the same contract. See
[Dynamic tools v1](docs/proposals/dynamic-tools-v1.md).

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
