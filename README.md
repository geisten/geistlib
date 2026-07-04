<p align="center">
  <img src="assets/neuron.png" alt="geist" width="100%">
</p>

# geist 👻

> **Your LLM should be a file you own, not a service you rent.** geist runs capable
> LLMs **entirely on the CPU — private, offline, dependency-free** — on hardware you
> already have, down to a Raspberry Pi. No cloud, no Python, no CUDA, nothing to
> install.
>
> The proof is `geist-bitnet`: **one binary with Microsoft's ternary BitNet 2B-4T
> baked in.** Copy it to a Pi and it generates text, **drives tools**, and searches
> the web — all locally, and it decodes **~2× faster than Microsoft's own
> bitnet.cpp**. Need more? The same engine runs **Gemma 4 with vision + audio** from
> one model file.

<p align="center">
  <strong>2.1×</strong> BitNet decode vs bitnet.cpp <sub>(Pi 5)</sub> &nbsp;·&nbsp;
  <strong>1.5×</strong> prefill vs llama.cpp <sub>(M1 Max)</sub> &nbsp;·&nbsp;
  <strong>1.4×</strong> BitNet decode vs bitnet.cpp <sub>(x86)</sub> &nbsp;·&nbsp;
  <strong>&lt; 1 MB</strong> binary, zero deps
  <br>
  <sub><a href="#faster-where-it-counts">↓ full scoreboard — 12 measurements across Linux &amp; macOS</a></sub>
</p>

[![CI](https://github.com/geisten/geisten/actions/workflows/ci.yml/badge.svg)](https://github.com/geisten/geisten/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C23-orange.svg)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20(ARM64%20%2B%20x86--64)-lightgrey.svg)](#getting-started)
[![Status](https://img.shields.io/badge/status-experimental%20(v0.3.3)-yellow.svg)](#status)
[![Discussions](https://img.shields.io/badge/Discussions-ask%20%26%20share-5865F2.svg)](https://github.com/geisten/geisten/discussions)
[![Good first issues](https://img.shields.io/github/issues/geisten/geisten/good%20first%20issue?label=good%20first%20issue&color=7057ff)](https://github.com/geisten/geisten/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

**Questions, ideas, or stuck?** → [GitHub Discussions](https://github.com/geisten/geisten/discussions) · **Found a bug?** → [open an issue](https://github.com/geisten/geisten/issues/new) · **Want to build?** → [good first issues](https://github.com/geisten/geisten/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

<p align="center">
  <img src="assets/demo-bitnet-trio.gif" alt="One geist-bitnet binary doing three things in a row on a Mac: generate text, then drive tools to list a folder and search the web — model baked in, no model file" width="100%">
</p>

*One self-contained `geist-bitnet` (BitNet b1.58 2B-4T baked in, **no model file**)
doing three things back-to-back: generate, then **drive tools** — list a folder,
search the web live. Same binary runs real-time on a
[Raspberry Pi 5](#faster-where-it-counts).*

---

## Run it now — model baked in

Download one file, run it — **no model file, no model argument**. The model lives
inside the binary.

| Platform | Single-file download (model included) |
| :-- | :-- |
| **Raspberry Pi / Linux** · ARM64 | [⬇ geist-bitnet-linux-arm64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-bitnet-linux-arm64.tar.gz) |
| **macOS** · Apple Silicon | [⬇ geist-bitnet-macos-arm64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-bitnet-macos-arm64.tar.gz) |
| **Linux** · x86-64 (AVX-512) | [⬇ geist-bitnet-linux-x86_64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-bitnet-linux-x86_64.tar.gz) |

```bash
./geist-bitnet "The capital of France is"     # generate — no model argument
./geist-bitnet agent "Summarize report.md"    # one-shot tool-use agent
./geist-bitnet chat                           # multi-turn chat + memory
```

---

## Run it now — your own model

Two downloads: the **engine** (pick your platform) and a **model** (one GGUF runs
on every platform). Then point the engine at the model.

**Step 1 — the engine** (< 1 MB, model-less):

| Platform | Engine download |
| :-- | :-- |
| **macOS** · Apple Silicon | [⬇ geist-macos-arm64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-macos-arm64.tar.gz) |
| **Raspberry Pi / Linux** · ARM64 | [⬇ geist-linux-arm64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-linux-arm64.tar.gz) |
| **Linux** · x86-64 (AVX-512) | [⬇ geist-linux-x86_64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-linux-x86_64.tar.gz) |

**Step 2 — a model** (one file, any platform):

| Model | Size | Direct download |
| :-- | --: | :-- |
| **Gemma 4 E2B-it** · `Q4_K_M` — text · vision · audio | 2.9 GB | [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf) |
| **Gemma 4 E4B-it** · `Q4_K_M` — bigger, text · vision · audio | 4.6 GB | [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) |
| **BitNet b1.58 2B-4T** · `i2_s` — ternary, beats bitnet.cpp (Pi 5 & x86) | 1.1 GB | [⬇ gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf) |

**Step 3 — run** (the model path is the only difference from the baked-in binary):

```bash
./geist       model.gguf "The capital of France is"   # generate text
./geist agent model.gguf "Summarize report.md"        # one-shot tool-use agent
./geist chat  model.gguf                               # multi-turn chat + memory
```

<sub>Prebuilt for macOS · ARM64, Linux · ARM64 and Linux · x86-64 (AVX-512, runs on any x86-64-v3 CPU). Windows is still pending. BitNet is a **base model with no tool training** — geist forces a valid tool call from outside the sampler, so it routes and calls anyway.</sub>

---

## Why geist?

Four design choices that make *a file you own* real:

### One binary, zero dependencies
Static musl on Linux ARM (< 1 MB), Apple frameworks only on macOS. Fold the model
in too (`make EMBED_MODEL=…`) and deployment is *literally one file*.

### Faster where it counts
Same GGUF, greedy decode. geist leads **end-to-end throughput** on a Pi 5,
**prefill** on Apple's matrix unit, **matches-to-beats llama.cpp on AMD x86**
(AVX-512), and **beats Microsoft's bitnet.cpp on ternary BitNet on both Pi 5 and
x86** (9950X: prefill +30 %, decode +38 %) — across edge and desktop:

<p align="center">
  <img src="assets/headline_benchmarks.svg" alt="Horizontal scoreboard of geist's throughput as a ratio of the baseline engine, grouped by system and tagged with its OS. Raspberry Pi 5 (Linux): BitNet decode 2.1x bitnet.cpp; Gemma decode 1.1x and total 1.1x (short prompt) llama.cpp, dropping to ~1.0x total at a longer prompt and 0.9x on prefill. Apple M1 Max (macOS): Gemma prefill 1.5x llama.cpp. AMD Ryzen 9 9950X (Linux): BitNet decode 1.4x and prefill 1.3x bitnet.cpp; Gemma decode 1.1x and prefill 1.0x, Llama 3.2 prefill 1.0x and decode 1.0x llama.cpp. Each row is a different metric and baseline; sub-parity rows are shown too." width="100%">
</p>

*All 12 measurements, grouped by system and tagged with its **OS** (🐧 Linux / 🍎
macOS) — each bar is geist ÷ its own baseline engine, on its own metric. geist
**leads on the metric that defines each platform** (Pi decode & total, M1 prefill,
x86 BitNet), and we show the near-parity and sub-parity rows too (Pi prefill, the
long-prompt total). Below, the one that matters most for chat: **end-to-end
total** throughput.*

<p align="center">
  <img src="assets/demo-pi5-bitnet.gif" alt="On a Raspberry Pi 5: real-time BitNet b1.58 2B-4T text generation from a single dependency-free binary" width="100%">
</p>

*Real-time on a **Raspberry Pi 5** — ternary BitNet b1.58 2B-4T (`i2_s`), no GPU,
no driver stack.*

### Ternary (1.58-bit) as a first-class citizen
geist runs Microsoft's BitNet b1.58 (`TQ2_0` and canonical `I2_S`) with integer-only
dot products — ARM **SDOT** (add/subtract, no multiplies) and x86 **AVX-512 VNNI**.
It beats Microsoft's own bitnet.cpp on **both**: a Pi 5 decodes **~2×** (17.4 vs
8.2 t/s), and an AMD 9950X does prefill **+30 %** (884 vs 679) and decode **+38 %**
(77.9 vs 56.5 t/s).

### On-device agent for small models
A bounded, whitelist-gated tool loop lets a 2 B model *do* things — all in the same
process, nothing leaving the machine except an explicit web request:

| capability | tool | notes |
| :-- | :-- | :-- |
| List a directory | `list_dir` | `opendir`, no shell |
| Read & summarize a file | `summarize_file` | local — **no embeddings, no cloud** |
| Search local documents | `doc_search` | keyword scan (local RAG) |
| Search the web | `web_search` | DuckDuckGo or self-hosted **SearXNG** |
| Fetch & read a web page | `web_fetch` | `curl` → tag-stripped text |

*This toolset is expanding steadily.*

**Response time per task** — warm (model resident), greedy, via `geist agent`. A
light task's cost is the model **deciding + forming the call** (a few forward
passes); the tool's own I/O is milliseconds. **Summarize** runs the whole document
through the model, so it scales with length:

| task | Mac · Gemma 4 E2B | Pi 5 · BitNet 2B-4T |
| :-- | --: | --: |
| list a dir · fetch · search¹ | ~4–5 s | ~15–16 s |
| summarize a short note (~1 ¶) | ~5 s | ~18 s |
| summarize an 8 KB article (~4 chunks) | ~80 s | ~3.4 min |

<sub>¹ web tasks add the network round-trip. One-time model load is separate (~3 s
eager on macOS; the Pi `mmap`s). Single-run wall-clock on live machines — ballpark,
not a gate. The Pi figures include the cached router baseline ([#39](https://github.com/geisten/geisten/pull/39)).</sub>

<details>
<summary><strong>Why C?</strong> (the substrate choice, in full)</summary>

Not because it is the fastest (a systems language like Rust ties on raw
performance) and certainly not because it is the safest (it is the opposite).
The core reason is **reach, not speed**:

> **C is the substrate with maximal reach and minimal assumptions — the universal
> ABI and the everywhere-available, transparent compiler that every platform and
> every embedding language already speaks. We knowingly pay for that reach with
> memory safety.**

This maps directly onto promise #1 — *one file, runs anywhere, embeds anywhere*:
the header **is** the ABI (any language FFIs in with no shim), every
architecture/OS/accelerator toolchain speaks C first, and the source maps almost
1:1 to the emitted instructions — which matters when you reason about NEON kernels
by the cycle. Performance is table-stakes here, shared with the alternatives; what
picks C is ubiquity + zero-ceremony interop + transparency.

The honest counter-position: **if memory safety outweighed ubiquity and
simplicity for you, Rust would be the better choice.** We deliberately weighed it
the other way, and offset the safety cost with strict warnings
(`-Werror -Wshadow -Wundef`), ASan/UBSan CI (`make MODE=asan`), bit-exact golden
tests, and a small auditable core (the stable text path is ~70 lines).

</details>

---

## Models that run today

Two models are first-class and one-download-and-go. Everything below runs on the
same `./geist` binary — pick by your hardware and what you need.

| Model | Modality | Quant | ~Size | RAM | Best on | Get it |
| :-- | :-- | :-- | --: | --: | :-- | :-- |
| **Gemma 4 E2B-it** | text · vision · audio | `Q4_K_M` | 2.9 GB | ≥ 4 GB | Mac / Pi 5 | `make fetch-model` · [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf) |
| Gemma 4 E4B-it | text · vision · audio | `Q4_K_M` | 4.6 GB | ≥ 6 GB | Mac | [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) |
| **BitNet b1.58 2B-4T** | text (ternary) | `i2_s` | 1.1 GB | ≥ 4 GB | **Pi 5 · x86** | [⬇ gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf) |
| BitNet b1.58-large | text (ternary) | `TQ2_0` | 207 MB | ≥ 1 GB | smallest footprint | convert from [1bitLLM ↗](https://huggingface.co/1bitLLM/bitnet_b1_58-large) |

```bash
# Gemma 4 E2B-it (text + vision + audio towers, all on one binary)
make fetch-model

# BitNet b1.58 2B-4T — the ~2× decode win on a Pi 5
curl -L -o bitnet-2b4t.i2_s.gguf \
  https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf
```

> **Vision & audio** ride on the Gemma 4 model — the engine has SigLIP (vision) and
> a Conformer (audio) tower built in; see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> for attaching image/audio inputs. **TQ2_0** has no canonical GGUF yet — convert
> the 1bitLLM base (see [`benchmark/TERNARY_BITNET.md`](benchmark/TERNARY_BITNET.md)).

---

## Getting Started

> **Just want to run it?** Prebuilt binaries (macOS · ARM64, Linux · ARM64,
> Linux · x86-64) are at the [top](#run-it-now--model-baked-in). This section builds from source —
> any platform with a C23 compiler, the path for a custom target or Windows.

### Prerequisites
- A C23 compiler: **gcc ≥ 14**, or Apple-clang ≥ 16 (Xcode 16 / macOS 15).
- `make`.
- **macOS:** Homebrew `libomp` recommended for multi-threading.

### 1. Build
`make` auto-detects your target and drops a `./geist` symlink in the repo root:

```bash
git clone https://github.com/geisten/geisten && cd geisten
make                       # or: make TARGET=mac-omp | pi5 | linux
```

### 2. Get a model
```bash
make fetch-model           # Gemma 4 E2B-it Q4_K_M (~3.1 GB) — optional helper
```

---

## Usage

### Embed the library (C)

The stable text path is ~15 lines: `geist_backend_create` → `geist_model_load` →
`geist_session_create` → loop `geist_session_decode_step`. The header **is** the ABI
— any language FFIs in with no shim. Runnable example: `make -C examples`; full
walkthrough in [`docs/QUICKSTART.md`](docs/QUICKSTART.md) and the API in
[`include/geist.h`](include/geist.h).

### Ship one file (model baked in)

**Prebuilt:** every [release](https://github.com/geisten/geisten/releases/latest)
ships a `geist-bitnet-<platform>.tar.gz` — BitNet 2B-4T already baked in, no model
file, no path argument. Download, extract, `./geist-bitnet "your prompt"` — or just
`curl … install.sh | sh` ([top](#run-it-now--model-baked-in)). That's the whole app.

**Build your own** with any GGUF. The plain `make` build gives you a `geist` that
**takes a model path** (you bring the GGUF); a separate **`make EMBED_MODEL=…`** build
*bakes the model in*, so that binary needs **no model argument**.

Give it its own name with `EMBED_NAME` so it's never confused with the
model-needing `geist`:

```bash
make EMBED_MODEL=bitnet-2b4t.i2_s.gguf EMBED_NAME=geist-bitnet   # GGUF baked in (zero-copy)
./geist-bitnet "The capital of France is"            # generate — no model path
./geist-bitnet agent "Summarize the file report.md"  # tools too — no model path
```

(Agent + chat work on the baked-in model. To ship it, just copy the binary —
`bin/<target>/release/tools/geist` — under whatever name you like.)

The weights are aliased zero-copy from the binary's read-only data (no extra RAM),
so this suits **small** models — the binary grows by the model size, and >~1.5 GB
exceeds the 2 GB GitHub-release limit. (Runs real-time on a Pi 5 —
[see above](#faster-where-it-counts).)

---

## Documentation

| Document | What it covers |
| :-- | :-- |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Run the CLI and embed the library in two minutes. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The three layers, load-time kernel binding, the pipeline. |
| [`docs/agent.md`](docs/agent.md) | The tool-use agent, bundled tools, routing & forced calls, security model. |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Single-binary builds, server/embedded deployment. |
| [`benchmark/`](benchmark/README.md) | Methodology & full results ([Apple/Pi 5](benchmark/BENCHMARK.md), [ternary BitNet](benchmark/TERNARY_BITNET.md)). |
| [`include/geist.h`](include/geist.h) | The public C API, with `STABLE` / `EXPERIMENTAL` stability tags. |

<details>
<summary><strong>Full benchmark numbers</strong> — exact t/s per system (the <a href="#faster-where-it-counts">scoreboard</a> above, as a table)</summary>

| model | platform | metric | **geist** | baseline |
| :-- | :-- | :-- | --: | --: |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | total t/s (32p+128d) | **8.8** | 8.2 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | decode t/s | **7.5** | 6.8 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max** | prefill t/s (pp1024) | **144** | 97 *(llama.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **Pi 5** | decode t/s | **17.4** | 8.2 *(bitnet.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **AMD 9950X** | prefill t/s (pp128) | **884** | 679 *(bitnet.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **AMD 9950X** | decode t/s (tg128) | **77.9** | 56.5 *(bitnet.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **AMD 9950X** | prefill t/s | **512** | 495 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **AMD 9950X** | decode t/s | **48.6** | 44.1 *(llama.cpp)* |
| Llama 3.2 3B (Q4_K_M) | **AMD 9950X** | prefill t/s | **351** | 346 *(llama.cpp)* |
| Llama 3.2 3B (Q4_K_M) | **AMD 9950X** | decode t/s | 34.1 | 34.5 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal, experimental)* | prefill t/s (pp512) | 762 | 1551 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal, experimental)* | decode t/s (tg64) | 55.2 | 92.7 *(llama.cpp Metal)* |

<sub>**Baseline versions:** llama.cpp `d05fe1d` (Pi 5, M1 Max) · `b9827` (x86) — bitnet.cpp = [microsoft/BitNet](https://github.com/microsoft/BitNet) `master` (its bundled llama.cpp fork, unpinned `--depth 1` clone). Full methodology: [`benchmark/`](benchmark/README.md).</sub>

<sub>**Metal backend status** (`BACKENDS="… metal"`): experimental — greedy-decode tokens verified identical to the `cpu_scalar` reference at every optimization step. Ops batch into one Metal command buffer per prefill batch / decode step and submit only when the host touches GPU-referenced memory. Prefill runs the simdgroup flash-attention kernels and decode the split-KV flash kernels on a native f16 KV cache (half the KV memory, zero per-call conversion), with fused per-layer blocks (q/k/v norm+RoPE+KV-append, gate+up GeGLU matvec, k/v pair, PLE), llama-style pipelined command buffers, and a device greedy argmax (4-byte token readback); the remaining gap to llama.cpp is residual serialization and kernel-level GEMM work — the roadmap with measured targets lives in `docs/proposals/metal-beat-llamacpp-plan.md`. Cool-state protocol (240 s idle), geist (`GEIST_M_MAX=128`) and llama.cpp measured back-to-back (Homebrew llama.cpp, `BLAS,MTL`), M1 Max 32-core.</sub>
</details>

---

## Status

`geist` is **v0.3.3 — experimental**. It runs Gemma 4 (text + vision + audio) end
to end on the CPU backends and has a broad C test suite (`make test`). The
`STABLE` core (load → session → decode → tokenize) is the part to build on;
`EXPERIMENTAL`-tagged surfaces (KV-cache modes, speculative decode, multimodal
attach) may still change between minor versions.

---

## Contributing

The interesting work is wide open — low-level kernels and quantization research,
not yet-another-wrapper. **From clone to green tests in 30 seconds:**

```bash
git clone https://github.com/geisten/geisten && cd geisten
make && make test          # builds ./geist, runs the full C suite
```

---

## Citation

Using geist in research? A "Cite this repository" button is on the repo sidebar
(from [`CITATION.cff`](CITATION.cff)), or use:

```bibtex
@software{schlegel_geist_2026,
  author  = {Schlegel, Germar},
  title   = {geist: a dependency-free CPU inference engine and on-device agent for small LLMs},
  year    = {2026},
  version = {0.3.3},
  url     = {https://github.com/geisten/geisten}
}
```

---

## License

Licensed under the **Apache License 2.0** — permissive, with an explicit patent
grant. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

📄 [Impressum](https://geisten.net/impressum.html) · © 2026 geisten Holding UG (haftungsbeschränkt)

*"The future of AI is local, private, and embedded."* 👻
