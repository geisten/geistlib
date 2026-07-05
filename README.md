<p align="center">
  <img src="assets/neuron.png" alt="geist" width="100%">
</p>

# geist 👻

> **We want AI to belong to everyone.** Not rented from a data center, but running on
> the hardware you already have — your laptop, a Raspberry Pi, an old CPU with no GPU
> in sight. geist runs capable LLMs **entirely on the CPU — private, offline,
> dependency-free**, with nothing to install. It began as one developer's attempt to
> actually *understand* how these models work, by building the engine from scratch.
> It's still that: an experiment, and an open invitation to join in.

<p align="center">
  <strong>▶ <a href="#run-it-now--model-baked-in">Download &amp; run in 30 seconds</a></strong>
  &nbsp;·&nbsp; one file, zero deps &nbsp;·&nbsp; runs on a Raspberry Pi
  &nbsp;·&nbsp; <strong>~2×</strong> faster than Microsoft's bitnet.cpp
</p>

---

## Run it now — model baked in

**One file. No model to download, no arguments, no dependencies.** `geist-bitnet` has
Microsoft's ternary BitNet 2B-4T **baked into the binary** — grab your platform and run:

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

Need vision + audio? The same engine runs **Gemma 4** from one model file — see
[Run your own model](#run-your-own-model).

[![CI](https://github.com/geisten/geisten/actions/workflows/ci.yml/badge.svg)](https://github.com/geisten/geisten/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C23-orange.svg)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20(ARM64%20%2B%20x86--64)-lightgrey.svg)](#build-from-source)
[![Status](https://img.shields.io/badge/status-experimental%20(v0.3.3)-yellow.svg)](#status)
[![Discussions](https://img.shields.io/badge/Discussions-ask%20%26%20share-5865F2.svg)](https://github.com/geisten/geisten/discussions)

**Questions or stuck?** → [Discussions](https://github.com/geisten/geisten/discussions) · **Bug?** → [open an issue](https://github.com/geisten/geisten/issues/new) · **Want to build?** → [good first issues](https://github.com/geisten/geisten/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

<p align="center">
  <img src="assets/demo-bitnet-trio.gif" alt="One geist-bitnet binary doing three things in a row on a Mac: generate text, then drive tools to list a folder and search the web — model baked in, no model file" width="100%">
</p>

*One self-contained `geist-bitnet` (BitNet b1.58 2B-4T baked in, **no model file**):
generate, then **drive tools** — list a folder, search the web live. Same binary runs
real-time on a [Raspberry Pi 5](#faster-where-it-counts).*

---

## What a small model can do — on-device agents

> **Experimental & actively evolving.** This is the frontier we're pushing: getting a
> 2 B model to *do* things that normally need a big one. See the
> [roadmap](ROADMAP.md) and [`docs/agent.md`](docs/agent.md).

A bounded, whitelist-gated tool loop lets a small model act — all in the same process,
nothing leaving the machine except an explicit web request:

| capability | tool | notes |
| :-- | :-- | :-- |
| List a directory | `list_dir` | `opendir`, no shell |
| Read & summarize a file | `summarize_file` | local — **no embeddings, no cloud** |
| Search local documents | `doc_search` | keyword scan (local RAG) |
| Search the web | `web_search` | DuckDuckGo or self-hosted **SearXNG** |
| Fetch & read a web page | `web_fetch` | `curl` → tag-stripped text |

The idea: a tight harness (routing, forced tool calls, a small-model memory) so a 2 B
model reaches big-model quality on a **narrow, well-defined task** — offline. BitNet is
a base model with **no tool training**; geist forces a valid tool call from outside the
sampler, so it routes and calls anyway. This toolset is expanding steadily.

<details>
<summary><strong>Response time per task</strong> — warm, greedy, via <code>geist agent</code></summary>

A light task's cost is the model **deciding + forming the call** (a few forward passes);
the tool's own I/O is milliseconds. **Summarize** runs the whole document through the
model, so it scales with length:

| task | Mac · Gemma 4 E2B | Pi 5 · BitNet 2B-4T |
| :-- | --: | --: |
| list a dir · fetch · search¹ | ~4–5 s | ~15–16 s |
| summarize a short note (~1 ¶) | ~5 s | ~18 s |
| summarize an 8 KB article (~4 chunks) | ~80 s | ~3.4 min |

<sub>¹ web tasks add the network round-trip. One-time model load is separate (~3 s eager
on macOS; the Pi `mmap`s). Single-run wall-clock on live machines — ballpark, not a gate.
The Pi figures include the cached router baseline
([#39](https://github.com/geisten/geisten/pull/39)).</sub>
</details>

---

## Faster where it counts

> **A snapshot, not a leaderboard.** Measured **July 2026** · geist **v0.3.3** vs
> llama.cpp `d05fe1d` / bitnet.cpp `master` · same GGUF, greedy decode, cool-state.
> Reproduce with `make bench-small`. Numbers move as both engines change.

Generation speed — **decode tokens/s**, the pace you actually feel:

| System | Model | **geist** | baseline |
| :-- | :-- | --: | --: |
| **Raspberry Pi 5** | BitNet 2B-4T (`i2_s`) | **17.4** | 8.2 · bitnet.cpp |
| **Raspberry Pi 5** | Gemma 4 E2B (`Q4_K_M`) | **7.5** | 6.8 · llama.cpp |
| **AMD 9950X** | BitNet 2B-4T (`i2_s`) | **77.9** | 56.5 · bitnet.cpp |
| **AMD 9950X** | Gemma 4 E2B (`Q4_K_M`) | **48.6** | 44.1 · llama.cpp |

On an **M1 Max**, prefill is **1.5×** llama.cpp (144 vs 97 t/s). geist leads on the metric
that defines each platform — and we publish the near- and sub-parity rows too.

<p align="center">
  <img src="assets/demo-pi5-bitnet.gif" alt="On a Raspberry Pi 5: real-time BitNet b1.58 2B-4T text generation from a single dependency-free binary" width="100%">
</p>

*Real-time on a **Raspberry Pi 5** — ternary BitNet b1.58 2B-4T (`i2_s`), no GPU, no driver stack.*

<details>
<summary><strong>Full numbers</strong> — every metric (prefill · decode · total), all 12 measurements</summary>

<p align="center">
  <img src="assets/headline_benchmarks.svg" alt="Horizontal scoreboard of geist's throughput as a ratio of the baseline engine, grouped by system and tagged with its OS. Raspberry Pi 5 (Linux): BitNet decode 2.1x bitnet.cpp; Gemma decode 1.1x and total 1.1x (short prompt) llama.cpp, dropping to ~1.0x total at a longer prompt and 0.9x on prefill. Apple M1 Max (macOS): Gemma prefill 1.5x llama.cpp. AMD Ryzen 9 9950X (Linux): BitNet decode 1.4x and prefill 1.3x bitnet.cpp; Gemma decode 1.1x and prefill 1.0x, Llama 3.2 prefill 1.0x and decode 1.0x llama.cpp. Each row is a different metric and baseline; sub-parity rows are shown too." width="100%">
</p>

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
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal, experimental)* | prefill t/s (pp512) | 987 | 1542 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal, experimental)* | decode t/s (tg64) | 81.2 | 91.3 *(llama.cpp Metal)* |

<sub>**Baseline versions:** llama.cpp `d05fe1d` (Pi 5, M1 Max) · `b9827` (x86) — bitnet.cpp = [microsoft/BitNet](https://github.com/microsoft/BitNet) `master` (its bundled llama.cpp fork, unpinned `--depth 1` clone). Cool-state protocol (240 s idle), geist and llama.cpp measured back-to-back. Full methodology: [`benchmark/`](benchmark/README.md).</sub>

<sub>**Metal backend** (`BACKENDS="… metal"`, experimental): greedy-decode tokens verified identical to the `cpu_scalar` reference; decode within **12 %** of llama.cpp (81.2 vs 91.3 t/s), holding 73 t/s at kv≈2100. Simdgroup flash attention on a native f16 KV cache, fused per-layer blocks, pipelined command buffers, device greedy argmax. Long context works past the 4096 default (the session window sizes from prompt + decode budget). Full ledger: `docs/proposals/metal-beat-llamacpp-plan.md`.</sub>
</details>

---

## Why geist?

- **One binary, zero dependencies.** Static musl on Linux ARM (< 1 MB), Apple
  frameworks only on macOS. Fold the model in (`make EMBED_MODEL=…`) and deployment is
  *literally one file*.
- **Ternary (1.58-bit) as a first-class citizen.** geist runs BitNet b1.58 (`TQ2_0`
  and canonical `I2_S`) with integer-only dot products — ARM **SDOT** (add/subtract, no
  multiplies) and x86 **AVX-512 VNNI** — which is why it beats bitnet.cpp on both Pi 5
  and x86.
- **Runs where you are.** ARM64 (NEON), macOS (Accelerate/AMX), x86-64 (AVX-512/VNNI) —
  the fastest path on each target, no CUDA, no Python.

<details>
<summary><strong>Why C?</strong> (the substrate choice, in full)</summary>

Not because it is the fastest (a systems language like Rust ties on raw performance) and
certainly not because it is the safest (it is the opposite). The core reason is **reach,
not speed**:

> **C is the substrate with maximal reach and minimal assumptions — the universal ABI
> and the everywhere-available, transparent compiler that every platform and every
> embedding language already speaks. We knowingly pay for that reach with memory safety.**

The header **is** the ABI (any language FFIs in with no shim), every
architecture/OS/accelerator toolchain speaks C first, and the source maps almost 1:1 to
the emitted instructions — which matters when you reason about NEON kernels by the cycle.

The honest counter-position: **if memory safety outweighed ubiquity and simplicity for
you, Rust would be the better choice.** We weighed it the other way, and offset the safety
cost with strict warnings (`-Werror -Wshadow -Wundef`), ASan/UBSan CI (`make MODE=asan`),
bit-exact golden tests, and a small auditable core (the stable text path is ~70 lines).

</details>

---

## Run your own model

Two downloads: the **engine** (pick your platform) and a **model** (one GGUF runs
everywhere). Then point the engine at the model.

**Step 1 — the engine** (< 1 MB, model-less):

| Platform | Engine download |
| :-- | :-- |
| **macOS** · Apple Silicon | [⬇ geist-macos-arm64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-macos-arm64.tar.gz) |
| **Raspberry Pi / Linux** · ARM64 | [⬇ geist-linux-arm64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-linux-arm64.tar.gz) |
| **Linux** · x86-64 (AVX-512) | [⬇ geist-linux-x86_64.tar.gz](https://github.com/geisten/geisten/releases/latest/download/geist-linux-x86_64.tar.gz) |

**Step 2 — a model** (all run on the same `./geist` binary — pick by hardware):

| Model | Modality | Quant | ~Size | RAM | Best on | Get it |
| :-- | :-- | :-- | --: | --: | :-- | :-- |
| **Gemma 4 E2B-it** | text · vision · audio | `Q4_K_M` | 2.9 GB | ≥ 4 GB | Mac / Pi 5 | `make fetch-model` · [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf) |
| Gemma 4 E4B-it | text · vision · audio | `Q4_K_M` | 4.6 GB | ≥ 6 GB | Mac | [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) |
| **BitNet b1.58 2B-4T** | text (ternary) | `i2_s` | 1.1 GB | ≥ 4 GB | **Pi 5 · x86** | [⬇ gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf) |
| BitNet b1.58-large | text (ternary) | `TQ2_0` | 207 MB | ≥ 1 GB | smallest footprint | convert from [1bitLLM ↗](https://huggingface.co/1bitLLM/bitnet_b1_58-large) |

**Step 3 — run** (the model path is the only difference from the baked-in binary):

```bash
./geist       model.gguf "The capital of France is"   # generate text
./geist agent model.gguf "Summarize report.md"        # one-shot tool-use agent
./geist chat  model.gguf                               # multi-turn chat + memory
```

> **Vision & audio** ride on the Gemma 4 model — the engine has SigLIP (vision) and a
> Conformer (audio) tower built in; see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
> **TQ2_0** has no canonical GGUF yet — convert the 1bitLLM base (see
> [`benchmark/TERNARY_BITNET.md`](benchmark/TERNARY_BITNET.md)).

<sub>Prebuilt for macOS · ARM64, Linux · ARM64 and Linux · x86-64 (AVX-512, runs on any x86-64-v3 CPU). Windows is still pending.</sub>

---

## Build from source

> **Just want to run it?** Prebuilt binaries are at the [top](#run-it-now--model-baked-in).
> This section builds from source — any platform with a C23 compiler, and the path for a
> custom target or Windows.

**Prerequisites:** a C23 compiler (**gcc ≥ 14**, or Apple-clang ≥ 16), `make`, and on
macOS Homebrew `libomp` for multi-threading.

```bash
git clone https://github.com/geisten/geisten && cd geisten
make                       # auto-detects target; drops a ./geist symlink
                           # or: make TARGET=mac-omp | pi5 | linux
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

The plain `make` build gives a `geist` that **takes a model path**; a separate
**`make EMBED_MODEL=…`** build *bakes the model in*, so that binary needs **no model
argument**. Give it its own name with `EMBED_NAME`:

```bash
make EMBED_MODEL=bitnet-2b4t.i2_s.gguf EMBED_NAME=geist-bitnet   # GGUF baked in (zero-copy)
./geist-bitnet "The capital of France is"            # generate — no model path
./geist-bitnet agent "Summarize the file report.md"  # tools too — no model path
```

Weights are aliased zero-copy from the binary's read-only data (no extra RAM), so this
suits **small** models — the binary grows by the model size, and >~1.5 GB exceeds the
2 GB GitHub-release limit. Every [release](https://github.com/geisten/geisten/releases/latest)
already ships a prebuilt `geist-bitnet-<platform>.tar.gz`. Full guide:
[`docs/DEPLOY.md`](docs/DEPLOY.md).

---

## Documentation

| Document | What it covers |
| :-- | :-- |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Run the CLI and embed the library in two minutes. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The three layers, load-time kernel binding, the pipeline. |
| [`docs/agent.md`](docs/agent.md) | The tool-use agent, bundled tools, routing & forced calls, security model. |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Single-binary builds, server/embedded deployment. |
| [`ROADMAP.md`](ROADMAP.md) | Where this is going — the agenda tracks and their status. |
| [`benchmark/`](benchmark/README.md) | Methodology & full results ([Apple/Pi 5](benchmark/BENCHMARK.md), [ternary BitNet](benchmark/TERNARY_BITNET.md)). |
| [`include/geist.h`](include/geist.h) | The public C API, with `STABLE` / `EXPERIMENTAL` stability tags. |

---

## Status

`geist` is **v0.3.3 — experimental**. It runs Gemma 4 (text + vision + audio) end to end
on the CPU backends and has a broad C test suite (`make test`). The `STABLE` core
(load → session → decode → tokenize) is the part to build on; `EXPERIMENTAL`-tagged
surfaces (KV-cache modes, speculative decode, multimodal attach, the agent layer) may
still change between minor versions.

---

## Where this is going

geist isn't trying to out-benchmark llama.cpp or replace anyone's toolchain. It started
as one developer's way of understanding how these models actually work — by building the
engine, kernel by kernel, from scratch. That spirit still drives it: we'd rather open new
doors than race down someone else's track.

The throughline is one belief — **small, heavily quantized models can do far more than
people assume, if the whole stack is built around them.** So that's what we're building:

- **Squeeze the model, not the user** — quantize aggressively and run the most extreme
  quantizations (ternary and binary) as first-class citizens, not afterthoughts.
- **Research ternary / binary models** — [BitNet](benchmark/TERNARY_BITNET.md) is just
  the start; 1.58-bit is where the interesting math lives.
- **Optimized for what people actually own** — CPUs and small GPUs, down to a Pi 5.
- **One-step install** — engine plus model, nothing else to set up.
- **Agents built for small models** — a tight harness, fixed algorithms where they beat
  a forward pass, conversion to markdown, and more, so a 2 B model can match a much
  bigger one on a narrow task.
- **A memory that fits a small model** — recall shaped for what these models can hold,
  not a bolt-on vector store.
- **Models that adapt** — dynamically specializing to a task, learning, self-organizing.

Most of this is barely started. That's the point — [come build it with us](#contributing).
The full agenda with status is in the [roadmap](ROADMAP.md).

---

## Contributing

The interesting work is wide open — low-level kernels and quantization research, not
yet-another-wrapper. **From clone to green tests in 30 seconds:**

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

Licensed under the **Apache License 2.0** — permissive, with an explicit patent grant.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

📄 [Impressum](https://geisten.net/impressum.html) · © 2026 geisten Holding UG (haftungsbeschränkt)

*"The future of AI is local, private, and embedded."* 👻
