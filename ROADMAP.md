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

## Product track: private Home Assistant edge agent

The first product-shaped use case for the small-model agent track is a local
Home Assistant conversation agent on Raspberry Pi 5 and CPU-only Linux hosts.
Home Assistant remains the authority for entity exposure: the integration
derives a registry only from Assist-exposed entities and pushes it to the
resident geist process. Geist resolves targets deterministically, admits only
its compiled home-tool families, and performs the local HA REST calls.

### Preview exit criteria

The Home Assistant preview is ready to announce when all of these are measured
on a clean installation rather than inferred from engine-level tests:

| criterion | target |
| :-- | :-- |
| Installation | first working request in **10 minutes or less**, without compiling |
| Clear commands | **≥ 90%** correct end-to-end actions on the published HA evaluation set |
| Authorization | **0 actions** against entities not exposed by Home Assistant |
| Simple-command latency | warm p50 **≤ 3 s** on Raspberry Pi 5 |
| Complex tool latency | warm p50 **≤ 10 s** on Raspberry Pi 5 |
| Languages | the same published core suite passes in **German and English** |
| Reliability | **24 h** resident soak without crash, model reload, or unbounded RSS growth |

Latency is a product budget, not an engine benchmark: it includes request
parsing, model routing/generation, the Home Assistant tool round-trip, and the
final response. If a small model cannot meet the simple-command budget, those
commands take a deterministic Assist-intent fast path and the LLM handles only
ambiguous or multi-step work.

### Delivery stages

1. ✅ Resident Unix-socket daemon: `geist-home --serve /path/geist.sock`; the
   model stays warm and the socket is created mode `0600`.
2. ✅ Home Assistant Conversation preview: Assist utterances, exposed-entity
   registry synchronization, deterministic device resolution, and bounded home
   actions are running end to end on Raspberry Pi 5.
3. 🚧 Reproducible installation: versioned component package, service installer,
   diagnostics, upgrade/rollback instructions, and clean-host acceptance test.
   Pi 5 staging plus an isolated internal-model round-trip complete in 9 s with
   a preinstalled binary, but the smoke prompt answered incorrectly; clean-image
   setup and semantic acceptance therefore remain open.
4. ARM64/x86-64 add-on with model storage, checksums, health checks and updates;
   keep the Unix socket for same-host deployments.
5. Published German/English HA evaluation corpus, security cases, 24 h soak,
   and reproducible Pi 5 latency report. The deterministic stub-backed corpus
   passes 56/56 on Pi 5 with Gemma 4 E2B Q4_K_M. A disposable real-HA test
   proves an exposed light can change while an existing unexposed light remains
   untouched. Ten warm real-HA simple actions measure p50 2.095 s/p95 2.122 s;
   ten relative climate read-modify-write actions measure p50 2.216 s/p95
   2.256 s on Pi 5. Both latency budgets pass; the 24 h soak remains open.

A general HTTP inference API is a separate interoperability feature, not a
dependency of the Home Assistant product track.

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
