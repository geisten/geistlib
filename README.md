<p align="center">
  <img src="assets/neuron.png" alt="geist" width="100%">
</p>

# geist 👻

> Run a real LLM **and an on-device agent** from one tiny CPU binary — no BLAS, no Python, no CUDA, nothing to install.

[![CI](https://github.com/geisten/geistlib/actions/workflows/ci.yml/badge.svg)](https://github.com/geisten/geistlib/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C23-orange.svg)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20(ARM64)-lightgrey.svg)](#-getting-started)
[![Status](https://img.shields.io/badge/status-experimental%20(v0.3.0)-yellow.svg)](#-status)

**geist** is a high-performance inference engine that runs small LLMs **on the CPU
with zero dependencies**. One small static binary. Copy it to a machine and it
runs — it generates text, reads your local files, and searches the web, all
locally.

> 🚀 **In a hurry?** Jump to [Getting Started](#-getting-started) — build and chat
> in two minutes.

---

## 🤔 Why geist?

The universal engines (llama.cpp & co.) run *every* model on *every* backend. That
generality costs you: a dispatch loop in the hot path, a pile of optional
dependencies, and a runtime to install before anything happens.

geist makes the opposite bet — **a few small models, done excellently, on cheap
hardware you already own.**

- **Nothing to install.** Deploy by copying one file. The Linux ARM build is a
  fully static musl binary (**< 1 MB**, `ldd` → *not a dynamic executable*); the
  macOS build links only Apple's own frameworks. No BLAS, no Python, no CUDA.
- **Embed it anywhere.** The C header *is* the ABI — any language FFIs in with no
  shim. The whole stable text-generation core is **~70 lines of C**.
- **Built for weak models.** A 2 B model rarely emits a clean tool call. geist
  doesn't trust it to: routing and tool-call structure are *forced from outside
  the sampler*, so even an untrained model drives the tools reliably.
- **Edge-first.** A 4.6 B model fits in **4 GB of RAM** on a $50–$100 board — no
  GPU, no driver stack, on-device audio built in.

---

## ✨ Features

### One binary, zero dependencies
Static musl on Linux ARM (< 1 MB), Apple frameworks only on macOS. Fold the model
in too (`make EMBED_MODEL=…`) and deployment is *literally one file*.

### Faster where it counts on the edge
Same GGUF, greedy decode. geist leads **end-to-end throughput** on a Pi 5 and
**prefill** on Apple's matrix unit:

| model | platform | metric | **geist** | baseline |
| :-- | :-- | :-- | --: | --: |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | total t/s (32p+128d) | **8.8** | 8.2 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | decode t/s | **7.5** | 6.8 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max** | prefill t/s (pp1024) | **144** | 97 *(llama.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **Pi 5** | decode t/s | **17.4** | 8.2 *(bitnet.cpp)* |

<img src="assets/pi5_pp_decode_total.svg" alt="Prefill, decode and total tokens/s for geist vs llama.cpp on a Pi 5: total tracks decode; geist has the lowest prefill but the highest decode, leading total at the short prompt." width="100%">

What you *feel* when you run a model is end-to-end throughput, and that's
decode-dominated — which is exactly where geist wins. Full methodology and the
complete sweep: [`benchmark/`](benchmark/README.md).

**Honest take — when to pick which:**

| Pick **geist** when… | Pick **llama.cpp** when… |
| :-- | :-- |
| You want the fastest end-to-end tokens on a Pi 5 / edge CPU | You need raw **prefill** on no-`i8mm` ARM (its OpenBLAS sgemm still edges geist ~10–15 %) |
| Deployment must be **one dependency-free binary** (no BLAS/Python) | You need a model or backend geist doesn't ship (GPU, x86, Llama/Qwen/Mistral, …) |
| You're embedding an engine through a plain **C ABI** | You want the broadest format & sampler coverage today |
| You run **ternary BitNet** (~2× bitnet.cpp) | — |

### Ternary (1.58-bit) as a first-class citizen
geist runs Microsoft's BitNet b1.58 (`TQ2_0` and canonical `I2_S`) with ARM
**SDOT** — integer add/subtract only, no multiplies. On a Pi 5 that's **~2×
Microsoft's own bitnet.cpp** (decode **17.4** vs 8.2 t/s).

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

**Response time per task** — warm (model resident), greedy, via `geist_shell`. A
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

### Native multimodal audio
A built-in Conformer audio tower — the LLM "hears" audio directly via embedding
prefixes, skipping the slow *Whisper → text → LLM* cascade. (Engine-level today;
agent tool wiring is next.)

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

## 📦 Models that run today

Two models are first-class and one-download-and-go. Everything below runs on the
same `./geist` binary — pick by your hardware and what you need.

| Model | Modality | Quant | ~Size | RAM | Best on | Get it |
| :-- | :-- | :-- | --: | --: | :-- | :-- |
| **Gemma 4 E2B-it** | text · vision · audio | `Q4_K_M` | 2.9 GB | ≥ 4 GB | Mac / Pi 5 | `make fetch-model` · [HF ↗](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) |
| **BitNet b1.58 2B-4T** | text (ternary) | `i2_s` | 1.1 GB | ≥ 4 GB | **Pi 5 / edge** | curl ↓ · [HF ↗](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) |
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

## 🚀 Getting Started

### Install (prebuilt, the impatient path)

Grab the latest dependency-free binary — no toolchain, no build:

```bash
# macOS (Apple Silicon)
curl -L https://github.com/geisten/geistlib/releases/latest/download/geist-macos-arm64.tar.gz | tar xz
# Linux (ARM64, fully static musl)
curl -L https://github.com/geisten/geistlib/releases/latest/download/geist-linux-arm64.tar.gz | tar xz
./geist-*/geist --version
```

> Prebuilt binaries are **ARM64 only** today (Apple Silicon + aarch64 Linux) — the
> platforms geist is fast on. x86-64 / Windows land with the AVX backend. On other
> platforms, build from source below.

### Prerequisites
- A C23 compiler: **gcc ≥ 14**, or Apple-clang ≥ 16 (Xcode 16 / macOS 15).
- `make`.
- **macOS:** Homebrew `libomp` recommended for multi-threading.

### 1. Build
`make` auto-detects your target and drops a `./geist` symlink in the repo root:

```bash
git clone https://github.com/geisten/geistlib && cd geistlib
make                       # or: make TARGET=mac-omp | pi5 | linux
```

### 2. Get a model
```bash
make fetch-model           # Gemma 4 E2B-it Q4_K_M (~3.1 GB) — optional helper
```

### 3. Run
```bash
OMP_WAIT_POLICY=active ./geist gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"
```

```console
loaded gemma4-e2b-Q4_K_M.gguf (arch: transformer)
The capital of France is Paris.
```

> `make run ARGS='…'` sets `OMP_WAIT_POLICY=active` for you (it matters for
> multi-thread perf).

---

## 💡 Usage

### Generate from the CLI

```console
$ OMP_WAIT_POLICY=active ./geist gemma4-e2b-Q4_K_M.gguf "Write a haiku about the ocean:" -n 40
Write a haiku about the ocean:

Blue waves crash on sand,
Salt spray kisses the warm air,
Ocean's deep secrets.
```

<p align="center">
  <img src="assets/demo-cli.gif" alt="geist CLI streaming a haiku from Gemma 4 E2B-it on the CPU" width="100%">
</p>

### Drive the agent

`geist_shell` is the demo agent CLI (`make bin` → `bin/<target>/release/tools/geist_shell`).
`GEIST_FORCE_CALL=1` forces the tool call so an untrained model still drives the tools:

```console
$ GEIST_FORCE_CALL=1 ./geist_shell model.gguf "Show me the contents of this folder"
notes.txt   report.md   config.toml   src

$ GEIST_FORCE_CALL=1 ./geist_shell model.gguf "Summarize the file report.md"
The Q3 plan migrates the billing system to the new ledger service, aiming for 40%
lower reconciliation latency and a single source of truth for invoices …

$ GEIST_FORCE_CALL=1 ./geist_shell model.gguf "Search the web for FIFA World Cup 2026"
1. 2026 FIFA World Cup - Wikipedia
   https://en.wikipedia.org/wiki/2026_FIFA_World_Cup
…
```

Set `GEIST_AGENT_TRACE=1` to watch each step live (printed to **stderr**, so the
answer on stdout stays clean) — useful while a request thinks for a few seconds:

```console
$ GEIST_FORCE_CALL=1 GEIST_AGENT_TRACE=1 ./geist_shell model.gguf "Summarize the file report.md"
· routing summarize_file: selected
→ calling summarize_file: {"path":"report.md"}
⚙ running summarize_file
✓ observed summarize_file: The Q3 plan migrates the billing system …
● answering: The Q3 plan migrates the billing system …
```

The same steps are a structured **output type** (`struct geist_agent_event`) your
own host can consume — render a spinner, log it, or stream it to a UI as JSON.
See [`docs/agent.md`](docs/agent.md#progress-events).

<p align="center">
  <img src="assets/demo-agent.gif" alt="geist on-device agent: a 2B model lists a directory, summarizes a local file, and searches the web — all on the CPU" width="100%">
</p>

*Real `geist_shell` run on Gemma 4 E2B-it (Mac, idle time trimmed): `list_dir` →
`summarize_file` → live `web_search`, all in one process.*

### Embed the library (C)

The whole stable text path is this small:

```c
#include <geist.h>
#include <stdio.h>

int main(void) {
    struct geist_backend *be = nullptr;
    geist_backend_create("auto", nullptr, nullptr, &be);

    struct geist_model *model = nullptr;
    geist_model_load("gemma4.gguf", be, &model);

    struct geist_session *sess = nullptr;
    struct geist_session_opts opts = {0};
    geist_session_create(model, be, &opts, &sess);
    geist_session_set_prompt(sess, "The capital of France is");

    geist_token_t tok = 0;
    while (geist_session_decode_step(sess, &tok) == GEIST_OK) {
        const char *piece = geist_session_token_to_str(sess, tok);
        if (piece == nullptr) break;
        printf("%s", piece);
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
}
```

Build a runnable copy with `make -C examples` — full walkthrough in
[`docs/QUICKSTART.md`](docs/QUICKSTART.md).

### Ship one file (model baked in)

```bash
make EMBED_MODEL=path/to/model.gguf   # bakes the GGUF into ./geist (zero-copy aliased)
./geist "The capital of France is"    # the CLI now takes only a prompt
```

---

## 📚 Documentation

| Document | What it covers |
| :-- | :-- |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Run the CLI and embed the library in two minutes. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The three layers, load-time kernel binding, the pipeline. |
| [`docs/agent.md`](docs/agent.md) | The tool-use agent, bundled tools, routing & forced calls, security model. |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Single-binary builds, server/embedded deployment. |
| [`benchmark/`](benchmark/README.md) | Methodology & full results ([Apple/Pi 5](benchmark/BENCHMARK.md), [ternary BitNet](benchmark/TERNARY_BITNET.md)). |
| [`include/geist.h`](include/geist.h) | The public C API, with `STABLE` / `EXPERIMENTAL` stability tags. |

---

## 🧭 Status

`geist` is **v0.3.0 — experimental**. It runs Gemma 4 (text + vision + audio) end
to end on the CPU backends and has a broad C test suite (`make test`). The
`STABLE` core (load → session → decode → tokenize) is the part to build on;
`EXPERIMENTAL`-tagged surfaces (KV-cache modes, speculative decode, multimodal
attach) may still change between minor versions.

---

## 🤝 Contributing

Contributions are welcome — especially **NEON/AMX microkernels** and **low-bit
quantization research**, where most of the interesting work lives. Open an issue,
pick a [roadmap](ROADMAP.md) item, or send a PR. Start with
[CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 🎓 Citation

Using geist in research? A "Cite this repository" button is on the repo sidebar
(from [`CITATION.cff`](CITATION.cff)), or use:

```bibtex
@software{schlegel_geist_2026,
  author  = {Schlegel, Germar},
  title   = {geist: a dependency-free CPU inference engine and on-device agent for small LLMs},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/geisten/geistlib}
}
```

---

## 📜 License

Licensed under the **Apache License 2.0** — permissive, with an explicit patent
grant. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

📄 [Impressum](https://geisten.net/impressum.html) · © 2026 geisten Holding UG (haftungsbeschränkt)

*"The future of AI is local, private, and embedded."* 👻
