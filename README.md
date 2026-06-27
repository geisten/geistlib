<p align="center">
  <img src="assets/neuron.png" alt="geist" width="100%">
</p>

# geist 👻

[![CI](https://github.com/geisten/geistlib/actions/workflows/ci.yml/badge.svg)](https://github.com/geisten/geistlib/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C23-orange.svg)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20(ARM64)-lightgrey.svg)](#-build--usage)
[![Status](https://img.shields.io/badge/status-experimental%20(v0.2.1)-yellow.svg)](#-status)

**geist** is a high-performance inference engine — and an **on-device agent** —
that runs small LLMs **on the CPU with zero dependencies**: one small static
binary, no BLAS, no Python, no CUDA, no runtime to install. Copy it to the machine
and it runs, reads your local files, and searches the web — all locally.

That is the bet, and it is a different one from the universal engines:

- **Dependency-free, CPU-only.** Nothing to install — deploy by copying one file,
  embed it through the C ABI. The Linux ARM build is a single **fully static musl**
  binary (<1 MB, `ldd` → *not a dynamic executable*) that runs on any aarch64 Linux
  with no libc at all; the macOS build links **only the OS's own frameworks**
  (Accelerate + libSystem — Apple ships no static libc), so there's nothing to
  install there either.
- **Focused on small models, done excellently.** Where llama.cpp runs *every*
  model on *every* backend, geist does **a few small ones excellently** (Gemma 4
  E2B-it and ternary 1.58-bit BitNet today) — every tensor bound to a hand-picked
  kernel at load time, not a generic dispatch loop. On the edge that focus
  **beats** the universal engine: end-to-end throughput above llama.cpp on a Pi 5,
  and ~2× Microsoft's bitnet.cpp on ternary ([benchmarks](#-performance-both-models-macos--pi-5)).
- **Agentic, built for weak models.** A bounded, whitelist-gated tool loop lets a
  2 B model **read and summarize local documents** (no embeddings, no cloud) and
  **search the web** — reliably, because routing and tool-call structure are
  *forced* from outside the sampler, not left to a model that was never tool-trained
  ([how](#-on-device-agent)).
- **Edge-first (Raspberry Pi 5).** Tuned for \$50–\$100 CPUs: a 4.6 B model fits in
  4 GB of RAM, no GPU or driver stack, decode at parity, on-device audio built in.
  The win is deploying and running *simply* on cheap hardware — not topping the
  prefill chart.

> **Status: experimental (v0.2.1).** The public API in [`include/geist.h`](include/geist.h)
> carries per-symbol stability tags (`STABLE` / `EXPERIMENTAL`). Expect churn in
> `EXPERIMENTAL` surfaces until 1.0. Issues and PRs welcome — see
> [CONTRIBUTING.md](CONTRIBUTING.md).

> 🚀 **Get started in two minutes:** [`docs/QUICKSTART.md`](docs/QUICKSTART.md) —
> run the CLI and embed the library (with a complete copy-paste C program).

---

## ✨ Demo

Build, then point the `geist` CLI at a GGUF:

`make` builds the engine and drops a `./geist` symlink in the repo root:

```console
$ make
$ OMP_WAIT_POLICY=active ./geist gemma4-e2b-Q4_K_M.gguf "The capital of France is"
loaded gemma4-e2b-Q4_K_M.gguf (arch: transformer)
The capital of France is Paris.

$ OMP_WAIT_POLICY=active ./geist gemma4-e2b-Q4_K_M.gguf "Write a haiku about the ocean:" -n 40
Write a haiku about the ocean:

Blue waves crash on sand,
Salt spray kisses the warm air,
Ocean's deep secrets.
```

(`OMP_WAIT_POLICY=active` matters for multi-thread perf; `make run ARGS='…'` sets
it for you.)

*Real output from the `geist` CLI on Gemma 4 E2B-it (Q4_K_M), greedy decode.
Reproduce with `make fetch-model` then the commands above. The whole stable
text-generation core is ~70 lines of C — see
[`examples/simple_generate.c`](examples/simple_generate.c) to embed it.*

---

## 🚀 Performance: both models, macOS & Pi 5

**At a glance** — identical GGUF for geist and the baseline, greedy decode. geist
leads where it counts on the edge (end-to-end throughput) and on Apple's matrix
unit (prefill); on the A76 it trades raw prefill for the fastest decode. Full
methodology and the complete sweep live in [`benchmark/`](benchmark/README.md).

| model | platform | metric | **geist** | baseline | source |
| :-- | :-- | :-- | --: | --: | :-- |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | total t/s (32p+128d) | **8.8** | 8.2 *(llama.cpp)* | [↓](#-performance-both-models-macos--pi-5) |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | decode t/s | **7.5** | 6.8 *(llama.cpp)* | [↓](#-performance-both-models-macos--pi-5) |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max** | prefill t/s (pp1024) | **144** | 97 *(llama.cpp)* | [BENCHMARK.md](benchmark/BENCHMARK.md) |
| BitNet b1.58 2B-4T (`i2_s`) | **Pi 5** | decode t/s | **17.4** | 8.2 *(bitnet.cpp)* | [↓](#bitnet-b158-2b-4t-on-a-raspberry-pi-5) |

> macOS is **prefill-led** here (decode is memory-bound and less interesting on a
> desktop); the Pi 5 is the edge target where the full prefill+decode+total story
> is measured. BitNet is the Pi/edge play — there is no macOS BitNet baseline to
> compare against.

What you actually feel when you run a model is **end-to-end throughput**: type a
short prompt, watch tokens stream out. That's decode-dominated. Same GGUF on both
engines, greedy, on a Pi 5 (Cortex-A76, 2.4 GHz, best thread count).

**Gemma 4 E2B-it (Q4_K_M)** — end-to-end, 32-token prompt + 128 generated, each
engine cold-started:

| engine | **total tok/s** | decode tok/s | prefill tok/s |
| :-- | --: | --: | --: |
| **geist** | **8.8** | **7.5** | 30.0 |
| llama.cpp (CPU) | 8.2 | 6.8 | 38.5 |
| llama.cpp (OpenBLAS) | 7.9 | 6.6 | 34.7 |

geist **leads end-to-end** here: generation is decode-dominated, and geist's
speculative output head (on by default; [how it works](#bitnet-b158-2b-4t-on-a-raspberry-pi-5))
gives it the fastest decode — enough to overcome its lower prefill. It also ships
as a single **< 1 MB static binary** (no OpenBLAS, no Python, no runtime): copy it
onto the Pi and it runs. The chart shows why — total tracks decode, not prefill:

<img src="assets/pi5_pp_decode_total.svg" alt="Prefill, decode and total tokens/s for geist vs llama.cpp on a Pi 5 at a 32- and 128-token prompt: total sits just above decode for every engine; geist has the lowest prefill but the highest decode, leading total at the short prompt and tying at the long one." width="100%">

At a longer 128-token prompt, prefill weighs more and it's a tie (geist 11.1 vs
llama 11.1–11.3) — full sweep and the per-phase split in
[benchmark/BENCHMARK_PI5.md](benchmark/BENCHMARK_PI5.md).

**BitNet b1.58 2B-4T (`i2_s`)** — where 1.58-bit ternary pays off:

| engine | **decode tok/s** |
| :-- | --: |
| **geist** | **17.4** |
| bitnet.cpp (Microsoft's reference) | 8.2 |

**~2× the reference engine** on the same Pi — from a speculative int8 output head
([how it works](#bitnet-b158-2b-4t-on-a-raspberry-pi-5)). This is the model to
pick if you want the fastest tokens-per-second on cheap hardware.

<details>
<summary>Prefill throughput sweep (Gemma 4, 128 → 1024 tokens; Apple M1 Max + Pi 5)</summary>

Prefill barely moves the end-to-end number above (decode dominates generation),
but for prompt-heavy workloads here is the full sweep, identical GGUF both engines:

**Apple M1 Max** — prefill t/s (best-of-10):

| seq_len | 128 | 256 | 512 | 1024 |
| :-- | :---: | :---: | :---: | :---: |
| llama.cpp `-ngl 0` | 141 | 147 | 128 | 97 |
| **geist** | **164** | **161** | **150** | **144** |
| | geist 1.16× | geist 1.10× | geist 1.17× | **geist 1.48×** |

**Raspberry Pi 5** — prefill t/s (8 reps, cool start):

| seq_len | 128 | 256 | 512 | 1024 |
| :-- | :---: | :---: | :---: | :---: |
| **llama.cpp** (OpenBLAS) | **37.4** | **39.4** | **37.6** | **35.9** |
| geist | 34.8 | 34.2 | 32.9 | 31.5 |

geist's dense path uses **Accelerate/AMX** on Apple (wins, lead widens to 1.48× at
1024); on the A76 (no `i8mm`) llama.cpp's decades-tuned OpenBLAS sgemm leads
prefill by ~10–15 % — the hard case geist is built around. Charts, per-phase
analysis and methodology: [`benchmark/`](benchmark/README.md).
</details>

---

## 🤖 On-device agent

geist is not only an engine — it ships a **bounded, whitelist-gated tool-use
agent** (header-only, in `tools/`) so a small local model can *do* things: read
your files, summarize them, and search the web. All of it runs in the **same
process** as the model, with **no extra dependencies** and nothing leaving the
machine except an explicit web request.

(`geist_shell` is the demo agent CLI — `make bin` builds it to
`bin/<target>/release/tools/geist_shell`; `GEIST_FORCE_CALL=1` forces the tool
call so an untrained model still drives the tools.)

```console
$ GEIST_FORCE_CALL=1 ./geist_shell model.gguf "Fasse die Datei bericht.md zusammen"
The report proposes a Q3 migration to the new billing system, …

$ GEIST_FORCE_CALL=1 ./geist_shell model.gguf "Zeige mir den Inhalt des aktuellen Ordners"
src   build   assets   README.md

$ GEIST_FORCE_CALL=1 ./geist_shell model.gguf "Suche im Web nach FIFA World Cup 2026"
1. 2026 FIFA World Cup - Wikipedia
   https://en.wikipedia.org/wiki/2026_FIFA_World_Cup
…
```

Bundled tools (each a no-shell, host-gated `*_tool()` you opt into):

| tool | what it does |
| :-- | :-- |
| `list_dir` | list a directory (`opendir`, no shell) |
| `summarize_file` | read a text file under a fixed root and refine-summarize it — **local, no embeddings, no cloud** |
| `doc_search` | keyword search across a directory of documents (local RAG) |
| `web_search` | search the web — DuckDuckGo, or a self-hosted **SearXNG** (JSON) instance |
| `web_fetch` | fetch a URL and return tag-stripped text |

**Why it works on a model that was never tool-trained.** A 2 B model rarely emits
a clean tool call on its own. geist does not rely on it: the host decides what can
run (a fixed whitelist), the right tool is **chosen by scoring tool names**
(PMI-calibrated, so a frequent token doesn't always win), and the JSON call
**structure is forced** token-by-token with the argument lifted from the request —
all reconstructed from the public `peek_logits`/`prefill_tokens` API, with **no
in-engine sampler change**. The model picks the tool; geist guarantees the call.

The security boundary is the **host, not the model**: fixed scope, whitelist gate,
a `max_steps` budget, and per-tool input validation (e.g. `web_fetch` runs `curl`
via `fork`+`execvp` — no shell, scheme gate, host allowlist). Full design,
including how to embed it in an iOS/Android host with platform-native tools:
**[`docs/agent.md`](docs/agent.md)**.

---

## 🛠 Under the hood

The pitch above (dependency-free, focused, edge-first) is delivered by a few
deliberate engineering choices — the *how* behind the *why*:

### Zero-Dispatch Architecture
Unlike generic engines that use complex layer-dispatch loops, `geist` uses **Kernel Binding**. At load time, every tensor is bound directly to a specialized kernel pointer. This eliminates vtable overhead and management logic during the hot path—critical for single-core-heavy edge CPUs, and it is only practical because geist targets a *focused* set of models rather than every architecture.

### BLAS/FFT optional per platform
The `geist_gemm` abstraction (and the same per-platform pattern for the audio FFT) lets each platform pick the fastest path *and* the leanest dependency set: ARM ships fully self-contained (native NEON fp32 + a vendored radix-2 FFT, no OpenBLAS/FFTW), while macOS keeps Accelerate/AMX and vDSP because the framework is always present. This is what makes the "copy one file" deployment above possible without giving up the platform's matrix accelerator.

### Ternary (1.58-bit) as a First-Class Citizen
We don't treat low-bit formats as an afterthought. Our backend is built for a **multiplication-free future**. `geist` includes native paths for BitNet b1.58 — both `TQ2_0` and Microsoft's canonical `I2_S` — using ARM **SDOT** (`vdotq_s32`) so the matmuls are integer add/subtract only, no multiplies, maximizing performance on hardware without powerful NPUs.

#### BitNet b1.58 2B-4T on a Raspberry Pi 5

Microsoft's `bitnet-b1.58-2B-4T` (`ggml-model-i2_s.gguf`), measured with
`tests/bench_perf_sweep` on a Pi 5 (Cortex-A76, **no `i8mm`**), 2 threads,
2.4 GHz, mean-of-5 after a discarded warm-up:

| context | prefill t/s | **decode t/s** | end-to-end t/s |
| --: | --: | --: | --: |
| 32  | 46.4 | **17.4** | 22.0 |
| 128 | 48.5 | **16.4** | 29.3 |
| 256 | 47.0 | **15.0** | 33.0 |

Versus **bitnet.cpp** (Microsoft's reference, the same `i2_s` model on the same
Pi): geist decode **17.4 t/s vs 8.2–8.7** — roughly **2× faster**. Both peak at
2 threads.

**Run it on your own Pi 5** — from nothing to a chatting BitNet in three steps
(Raspberry Pi OS / Debian; needs a C23 compiler, `gcc ≥ 14`):

```bash
# 1. Build geist (auto-detects the Pi 5 target; drops a ./geist symlink)
sudo apt install -y git build-essential libopenblas-dev
git clone https://github.com/geisten/geistlib && cd geistlib && make

# 2. Download Microsoft's BitNet b1.58 2B-4T ternary model (~1.1 GB, fits in 4 GB RAM)
curl -L -o bitnet-2b4t.i2_s.gguf \
  https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf

# 3. Generate — the speculative head is on by default (GEIST_SPEC_HEAD=0 disables)
OMP_WAIT_POLICY=active OMP_NUM_THREADS=2 \
  ./geist bitnet-2b4t.i2_s.gguf "The capital of France is" -n 64
```

The decode win comes from a **speculative output head** (default on for greedy;
`GEIST_SPEC_HEAD=0` forces the exact dense head):
on this model the lm_head is a tied **F16** embedding (656 MB read *per token*,
~50 % of decode). geist keeps a stride-subsampled int8 "sketch" of the table
(~82 MB), rough-ranks the whole 128 K vocabulary with one SDOT pass, then
computes **exact** f16 logits for only the top-512 candidates. Greedy output is
byte-identical to the dense head. Full method and what *didn't* work:
[`benchmark/TERNARY_BITNET.md`](benchmark/TERNARY_BITNET.md).

The same head also works on **Gemma 4 E2B** (tied Q6_K lm_head over a 256 K
vocab, ~32 % of decode). There phase 3 reuses the dense **W6A8** kernel on a
one-row view, so finalist logits are bit-exact; the only knob is how many
finalists the sketch must keep for the argmax to be among them — 4096 on the
256 K vocab (vs 512 for BitNet), which makes greedy **byte-identical** to the
dense head for **+5 %** decode (or +14 % if you trade exactness back via a
smaller `GEIST_SPEC_TOPK`).

### Native Multimodal Audio
`geist` features a built-in Conformer-based audio tower. Instead of a slow "Whisper → Text → LLM" cascade, we support direct audio-embedding prefixes. The LLM "hears" the audio directly, reducing latency and preserving prosody.

### Why C?
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

---

## 📂 Directory Layout

- [`src/`](src/) — Core inference engine and platform-specific kernels.
- [`include/`](include/) — Public C API headers.
- [`examples/`](examples/) — Minimal self-contained integration examples.
- [`benchmark/`](benchmark/) — Benchmark scripts and performance evaluations.
- [`docs/`](docs/) — Architectural notes and guides.

---

## 📚 Documentation

| Document | What it covers |
| :-- | :-- |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Run the CLI and embed the library in two minutes (copy-paste C program). |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The three layers, load-time kernel binding, the processing pipeline, and the agent above the ABI. |
| [`docs/agent.md`](docs/agent.md) | The tool-use **agent**, the bundled tools, tool routing & forced calls, the security model, and embedding it. |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Single-binary builds, server/embedded deployment, Unix-socket transport. |
| [`benchmark/`](benchmark/README.md) | Methodology and full results — [Apple/Pi 5](benchmark/BENCHMARK.md), [Pi 5 deep-dive](benchmark/BENCHMARK_PI5.md), [ternary BitNet](benchmark/TERNARY_BITNET.md). |
| [`include/geist.h`](include/geist.h) | The public C API, with per-symbol `STABLE` / `EXPERIMENTAL` stability tags. |

---

## 📦 Build & Usage

> **In a hurry?** [`docs/QUICKSTART.md`](docs/QUICKSTART.md) walks through running
> the CLI and embedding the library (with a complete copy-paste C program) in two
> minutes.

### Requirements
- C compiler with `-std=c23` support: gcc ≥ 14, or Apple-clang ≥ 16 (Xcode 16 / macOS 15).
- `make`.
- **Mac:** Homebrew `libomp` recommended for multi-threading.

### Quick Start
```bash
# Build (target auto-detected: mac-omp / mac / pi5 / linux). Drops a ./geist symlink.
make                       # or: make TARGET=mac-omp | pi5 | linux

# Grab a reference model (Gemma 4 E2B-it Q4_K_M, ~3.1 GB) — optional helper.
make fetch-model

# Generate (the symlink saves you the bin/<target>/<mode> path):
OMP_WAIT_POLICY=active ./geist gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"
make run ARGS='gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "Write a haiku" -n 40'   # same, OMP set for you

# Or the interactive evaluation REPL (full build dir; eval_geist has no symlink):
OMP_WAIT_POLICY=active bin/`mk/detect-target.sh`/release/tools/eval_geist gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

A minimal C program using the public API lives in
[`examples/`](examples/) — build it with `make -C examples`.

### Public headers

The API is split by audience (since 0.2.0), so an app that only generates text
includes a small surface:

| Header | For | Holds |
| :-- | :-- | :-- |
| **`geist.h`** | running a model | backend → model → session → `set_prompt` → `decode_step` → `token_to_str` |
| **`geist_util.h`** | chat / advanced apps | `geist_model_eos_token` & special tokens, tokenize, multimodal `attach_*`, speculative decode, telemetry |
| `geist_types.h` | backend authors | low-level tensor / op / dtype types |
| `geist_backend.h` | backend authors | the backend vtable + descriptor |

### Minimal C API Usage Example

Below is the stable API pattern to load a model and stream generation:

```c
#include <geist.h>
#include <stdio.h>

int main() {
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

### Single-binary builds (model included)

geist already ships as one dependency-free binary; you can fold the **model** in
too, so deployment is *literally one file* — no GGUF to ship alongside:

```bash
make EMBED_MODEL=path/to/model.gguf       # bakes the GGUF into ./geist
./geist "The capital of France is"        # the CLI now takes only a prompt
```

The weights are aliased **zero-copy** from the binary's read-only data (no extra
RAM), so this is for **small models** — the binary grows by the model size, and
binaries larger than ~1.5 GB exceed the 2 GB GitHub-release limit. The model
must carry its own tokenizer. Your own app gets the same superpower via the public API:

```c
extern const unsigned char model_start[], model_end[];   // your embedded blob
geist_model_load_from_memory(model_start, model_end - model_start, be, &model);
```

(Start `make EMBED_MODEL=...` from a clean tree — `make clean` first — since it
recompiles the CLI with the model baked in.)

---

## 🗺 Roadmap

- [x] **Flatten Pi 5 prefill:** FFN-streaming, lm-head argmax, and a multi-threaded
  O(n²) attention core — the Pi prefill curve is now flat (pp1024 +35 %), though
  llama.cpp's OpenBLAS still edges raw prefill on the A76.
- [x] **BitNet Optimization:** 2B-4T `I2_S` on the Pi 5 now decodes at **17.4 t/s**
  via a speculative int8 output head — ~2× bitnet.cpp on the same box; see
  `benchmark/TERNARY_BITNET.md`.
- [ ] **Dynamic Quantization:** Release the first mixed-low-bit recipe for Gemma 4.
- [ ] **Dynamic runtime threading:** choose the thread count per phase, and back off
  under thermal/load pressure, at runtime — instead of the fixed prefill=4 / decode=3.
- [x] **Single-file app + model:** fuse the weights into the executable so a
  deployment is literally *one* binary — engine and model, nothing else to ship.
  Shipped in v0.2.1 (`make EMBED_MODEL=…` + `geist_model_load_from_memory`,
  zero-copy aliased). Practical for **small** models — the binary grows by the
  model size; large GGUFs (a streamed `geist pack` format) remain future work.
- [ ] **Realtime Audio Demo:** A standalone VAD-to-Instruction voice assistant on Pi 5.

---

## 🧭 Status

`geist` is **v0.2.1 — experimental**. It runs Gemma 4 (text + vision + audio) end
to end on the CPU backends and has a broad C test suite (`make test`), but the
`EXPERIMENTAL`-tagged parts of the API (KV-cache modes, speculative decode, AWQ,
multimodal attach) may still change between minor versions. The `STABLE` core
(load → session → decode → tokenize) is the part to build on.

## 📜 License & Contribution

`geist` is licensed under the **Apache License 2.0** — permissive, with an
explicit patent grant. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

We welcome technical contributions, especially in the area of **NEON/AMX
microkernels** and **low-bit quantization research**. Start with
[CONTRIBUTING.md](CONTRIBUTING.md).

---

📄 [Impressum](https://geisten.net/impressum.html) · © 2026 geisten Holding UG (haftungsbeschränkt)

*“The future of AI is local, private, and embedded.”* 👻
