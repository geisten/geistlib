<p align="center">
  <img src="assets/neuron.png" alt="geistlib" width="100%">
</p>

# geistlib 👻

> **A tiny, dependency-free inference engine for small LLMs.** Run capable models
> on the CPU you already own — especially a Raspberry Pi — private, offline, no
> cloud. A GPU speeds it up but is never required.

<p align="center">
  <strong>local by default</strong> &nbsp;·&nbsp;
  <strong>CPU-first, GPU-optional</strong> &nbsp;·&nbsp;
  <strong>one small runtime</strong> &nbsp;·&nbsp;
  <strong>embeddable C library</strong>
</p>

<p align="center">
  <strong>1.96×</strong> BitNet decode vs bitnet.cpp <sub>(Pi 5, <code>make bench</code>)</sub> &nbsp;·&nbsp;
  <strong>1.5×</strong> prefill vs llama.cpp <sub>(M1 Max)</sub> &nbsp;·&nbsp;
  <strong>1.9×</strong> BitNet decode vs bitnet.cpp <sub>(x86)</sub> &nbsp;·&nbsp;
  <strong>&lt; 1 MB</strong> binary, zero deps
  <br>
  <sub><a href="#faster-where-it-counts">↓ full scoreboard — decode t/s on every system, one metric</a></sub>
</p>

[![CI](https://github.com/geisten/geistlib/actions/workflows/ci.yml/badge.svg)](https://github.com/geisten/geistlib/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C23-orange.svg)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20(ARM64%20%2B%20x86--64)-lightgrey.svg)](#quick-start)
[![Status](https://img.shields.io/badge/status-experimental%20(v0.7.0)-yellow.svg)](#status)
[![Discussions](https://img.shields.io/badge/Discussions-ask%20%26%20share-5865F2.svg)](https://github.com/geisten/geistlib/discussions)
[![Good first issues](https://img.shields.io/github/issues/geisten/geistlib/good%20first%20issue?label=good%20first%20issue&color=7057ff)](https://github.com/geisten/geistlib/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

**Questions, ideas, or stuck?** → [GitHub Discussions](https://github.com/geisten/geistlib/discussions) · **Found a bug?** → [open an issue](https://github.com/geisten/geistlib/issues/new) · **Want to build?** → [good first issues](https://github.com/geisten/geistlib/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

<p align="center">
  <img src="assets/demo-pi5-bitnet.gif" alt="On a Raspberry Pi 5: real-time BitNet b1.58 2B-4T text generation from a single dependency-free binary" width="100%">
</p>

*Real-time on a Raspberry Pi 5 — ternary BitNet b1.58 2B-4T (`i2_s`) generating
text from a single dependency-free binary, no GPU, no driver stack.*

geistlib is just the engine and a small CLI to run models — build it plain and
bring your own GGUF, or bake a model in for a single self-contained binary, on
macOS and Linux (ARM64 + x86-64). It squeezes the most out of small models so
they run where you are: a laptop, a Raspberry Pi, an old CPU with no GPU in
sight. It began as one developer's attempt to *understand* how these models work
by building the engine from scratch, kernel by kernel. It still is: an
experiment, and an open invitation to join in.

---

## Quick start

geistlib is a **library**. It has no CLI of its own: the tool-use runtime, the
resident daemon and the single-file binary with a model baked in live in
[geisten/geistagent](https://github.com/geisten/geistagent), which links this
engine. If you came here to *run* something, that is the repository you want.

### Use the prebuilt SDK

Every [release](https://github.com/geisten/geistlib/releases/latest) ships
`libgeist-<platform>.tar.gz` — `libgeist.a`, `include/*.h` and `LICENSE`.
Platforms: `macos-arm64`, `linux-arm64`, `linux-x86_64`. Verify it against
`SHA256SUMS`, then link:

```bash
cc -std=c23 -I libgeist-linux-arm64/include my_app.c \
   libgeist-linux-arm64/libgeist.a -fopenmp -lm -o my_app
```

<sub>The static archive is an OpenMP build, so a consumer supplies the OpenMP
runtime: `-fopenmp` on Linux, `-framework Accelerate <libomp>/lib/libomp.a` on
macOS.</sub>

### Build from source

Any platform with a C23 compiler. Prerequisites: **gcc ≥ 14** or Apple-clang
≥ 16 (Xcode 16 / macOS 15) and `make`; on macOS, Homebrew `libomp`.

```bash
git clone https://github.com/geisten/geistlib && cd geistlib
make lib                   # auto-detects target; or: make TARGET=mac-omp | pi5 | linux
make fetch-model           # optional: pull Gemma 4 E2B-it Q4_K_M (~3.1 GB)
make run ARGS='gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"'
```

`make run` builds and runs `examples/simple_generate.c` — the smallest useful
program against the STABLE core, and the thing to copy when embedding.

---

## Models

Any GGUF that carries its own tokenizer works. Two models are first-class and
one-download-and-go.

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
> for attaching image/audio inputs. TQ2_0 has no canonical GGUF yet — convert
> the 1bitLLM base (see [`benchmark/results/TERNARY.md`](benchmark/results/TERNARY.md)).

---

## Why geistlib?

geistlib is not trying to be a universal model catalog or a drop-in replacement for
every GPU inference server. It is deliberately optimized for a narrower job:

- **Small models, done right:** aggressive quantization and ternary BitNet as
  first-class citizens, where memory bandwidth matters most.
- **Constrained hardware:** platform-specific kernels for Raspberry Pi and
  CPU-only hosts; a GPU (experimental Metal/Vulkan) is optional, never required.
- **One auditable C runtime:** no Python environment, no container — a single
  small binary you can embed anywhere.

### One binary, zero dependencies
Static musl on Linux ARM (< 1 MB), Apple frameworks only on macOS. Fold the model
in too and deployment is *literally one file* — see
[geistagent](https://github.com/geisten/geistagent).

### Faster where it counts
Same GGUF, greedy decode. geistlib leads end-to-end throughput on a Pi 5,
prefill on Apple's matrix unit, matches-to-beats [llama.cpp](https://github.com/ggml-org/llama.cpp)
on AMD x86 (AVX-512), and beats Microsoft's bitnet.cpp on ternary BitNet on both
Pi 5 and x86 — across edge and desktop:

<p align="center">
  <img src="assets/headline_benchmarks.svg" alt="Decode-throughput scoreboard: geistlib divided by its baseline engine, decode tokens/s, grouped by system. Every bar is the same metric so they are directly comparable. Raspberry Pi 5 (Linux): BitNet decode 1.96x bitnet.cpp, BitNet prefill 0.99x bitnet.cpp, Gemma decode 1.1x llama.cpp. AMD Ryzen 9 9950X (Linux): BitNet decode 1.9x bitnet.cpp, Gemma decode 1.1x llama.cpp, Llama 3.2 decode 1.0x llama.cpp. Sub-parity rows are shown too." width="100%">
</p>

*One metric on every bar: decode t/s (tokens/s while generating) — geistlib ÷ its
own baseline engine, so the bars are directly comparable. Decode is the number you
feel in a chat. Prefill and total for every system (incl. the M1 Max prefill
win and the Pi long-prompt total) are in the [full numbers table](#documentation)
below; sub-parity rows (Llama 3.2 on x86) are shown here too — nothing cherry-picked.*

Reproduce it on your own hardware:

```bash
make lib && make fetch-model                # build libgeist.a + pull the Gemma GGUF
OMP_WAIT_POLICY=active make bench-small      # records decode t/s to benchmark/results/APPLE.md
```

<sub>Cross-engine comparison vs a pinned llama.cpp, quality (MMLU) and full
methodology: [`benchmark/METHODOLOGY.md`](benchmark/METHODOLOGY.md).</sub>

### Ternary (1.58-bit) as a first-class citizen
geistlib runs Microsoft's BitNet b1.58 (`TQ2_0` and canonical `I2_S`) with integer-only
dot products — ARM SDOT (add/subtract, no multiplies) and x86 AVX-512 VNNI. It beats
Microsoft's own bitnet.cpp on both a Pi 5 (**~2×** decode) and an AMD 9950X — exact
t/s in the [full numbers](#documentation) below, 1.6-bpw base-3 decode packing
included (#104). BitNet is ternary, so the whole 2B-4T model is just 1.2 GB — about
a third the footprint of a comparable general model, small enough to run on a 4 GB Pi.

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

## For developers

### Embed the library (C)

The stable text path is ~15 lines: `geist_backend_create` → `geist_model_load` →
`geist_session_create` → loop `geist_session_decode_step`. The header **is** the ABI
— any language FFIs in with no shim. Runnable example: `make -C examples`; full
walkthrough in [`docs/QUICKSTART.md`](docs/QUICKSTART.md) and the API in
[`include/geist.h`](include/geist.h).

### Ship engine and model as one file

Baking a GGUF into a binary is a property of an executable, and this repository
builds none. [geistagent](https://github.com/geisten/geistagent) does it —
`make bin EMBED_MODEL=model.gguf` there — and publishes signed single-file
runtimes with BitNet b1.58 2B-4T already inside. The weights alias zero-copy
from the binary's read-only data, so the cost is disk, not RAM.

---

## Documentation

Repository ownership and the complete map are in
[`docs/README.md`](docs/README.md).

| Document | What it covers |
| :-- | :-- |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Embed the library in two minutes. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The three layers, load-time kernel binding, the pipeline. |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Building the library and consuming the packaged SDK. |
| [`benchmark/`](benchmark/README.md) | Methodology & full results ([Apple/Pi 5](benchmark/results/APPLE.md), [ternary BitNet](benchmark/results/TERNARY.md), [Vulkan GPU](benchmark/results/VULKAN.md)). |
| [`include/geist.h`](include/geist.h) | The public C API, with `STABLE` / `EXPERIMENTAL` stability tags. |

<details>
<summary><strong>Full benchmark numbers</strong> — exact t/s per system (the <a href="#faster-where-it-counts">scoreboard</a> above, as a table)</summary>

| model | platform | metric | **geistlib** | baseline |
| :-- | :-- | :-- | --: | --: |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | total t/s (32p+128d) | **8.8** | 8.2 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | decode t/s | **7.5** | 6.8 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max** | prefill t/s (pp1024) | **144** | 97 *(llama.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **Pi 5** | decode t/s | **17.4** | 8.2 *(bitnet.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **AMD 9950X** | prefill t/s (pp128) | **1098** | 679.9 *(bitnet.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **AMD 9950X** | decode t/s (tg128) | **103.1** | 54.3 *(bitnet.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **AMD 9950X** | prefill t/s | **512** | 495 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **AMD 9950X** | decode t/s | **48.6** | 44.1 *(llama.cpp)* |
| Llama 3.2 3B (Q4_K_M) | **AMD 9950X** | prefill t/s | **351** | 346 *(llama.cpp)* |
| Llama 3.2 3B (Q4_K_M) | **AMD 9950X** | decode t/s | 34.1 | 34.5 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal, experimental)* | prefill t/s (pp512) | 987 | 1542 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal, experimental)* | decode t/s (tg64) | 81.2 | 91.3 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **RTX 2080 Ti** *(Vulkan, experimental)* | prefill t/s (pp512) | 1150 | 4639 *(llama.cpp Vulkan)* |
| Gemma 4 E2B-it (Q4_K_M) | **RTX 2080 Ti** *(Vulkan, experimental)* | decode t/s (tg128) | 132.3 | 154 *(llama.cpp Vulkan)* |

<sub>**Baseline versions:** llama.cpp `d05fe1d` (Pi 5, M1 Max) · `b9827` (x86) · `d0f9d2e` (Vulkan) — bitnet.cpp = [microsoft/BitNet](https://github.com/microsoft/BitNet) `master` (its bundled llama.cpp fork, unpinned `--depth 1` clone). Full methodology: [`benchmark/`](benchmark/README.md).</sub>

<sub>**Metal (Apple GPU) backend** (`BACKENDS="… metal"`) is experimental: greedy decode is bit-exact vs the `cpu_scalar` reference and within **12 %** of llama.cpp Metal (81.2 vs 91.3 t/s decode), holding up at long context past the 4096 default. Full kernel notes and the measurement ledger: [`benchmark/results/METAL.md`](benchmark/results/METAL.md).</sub>

<sub>**Vulkan backend** (`BACKENDS="… vulkan"`) is experimental: the first non-Apple GPU path (NVIDIA Turing tested; libvulkan is dlopen'd, no link-time dependency). Quality gate passed (MMLU-200 0.520 vs 0.490 on the CPU path, 14/14 tool-calling) and decode reaches **~86 %** of llama.cpp Vulkan (132.3 vs 154 t/s tg128); prefill is the open front. The phase-by-phase lab log lives in [`benchmark/results/VULKAN.md`](benchmark/results/VULKAN.md).</sub>
</details>

---

## Status

`geistlib` is **v0.7.0 — experimental**. It runs Gemma 4 (text + vision + audio) end
to end on the CPU backends — plus experimental GPU backends (Metal on Apple,
Vulkan on Linux/NVIDIA) — and
has a broad C test suite (`make test`). The
`STABLE` core (load → session → decode → tokenize) is the part to build on;
`EXPERIMENTAL`-tagged surfaces (KV-cache modes, speculative decode, multimodal
attach) may still change between minor versions.

---

## Roadmap

geistlib isn't trying to out-benchmark llama.cpp or replace anyone's toolchain. It
started as one developer's way of understanding how these models actually work — by
building the engine, kernel by kernel, from scratch. That spirit still drives it:
we'd rather open new doors than race down someone else's track.

The throughline is one belief — **small, heavily quantized models can do far more
than people assume, if the whole stack is built around them.** So that's what we're
building:

- **Squeeze the model, not the user** — quantize aggressively and run the most
  extreme quantizations (ternary and binary) as first-class citizens, not
  afterthoughts.
- **Research ternary / binary models** — [BitNet](benchmark/results/TERNARY.md) is
  just the start; 1.58-bit is where the interesting math lives.
- **Optimized for what people actually own** — CPUs and small GPUs, all the way down
  to a Raspberry Pi 5.
- **One-step install** — engine plus model, nothing else to set up.
- **Models that adapt** — dynamically specializing to a task, learning,
  self-organizing over time.

Most of this is barely started. That's the point — [come build it with
us](#contributing).

---

## Contributing

The interesting work is wide open — low-level kernels and quantization research,
not yet-another-wrapper. **From clone to green tests in 30 seconds:**

```bash
git clone https://github.com/geisten/geistlib && cd geistlib
make lib && make test      # builds libgeist.a, runs the full C suite
```

---

## Citation

Using geistlib in research? A "Cite this repository" button is on the repo sidebar
(from [`CITATION.cff`](CITATION.cff)), or use:

```bibtex
@software{schlegel_geistlib_2026,
  author  = {Schlegel, Germar},
  title   = {geistlib: a dependency-free inference engine for small LLMs},
  year    = {2026},
  version = {0.7.0},
  url     = {https://github.com/geisten/geistlib}
}
```

---

## License

Licensed under the **Apache License 2.0** — permissive, with an explicit patent
grant. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

📄 [Impressum](https://geisten.net/impressum.html) · © 2026 geisten Holding UG (haftungsbeschränkt)

*"The future of AI is local, private, and embedded."* 👻
