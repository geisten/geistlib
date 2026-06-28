# Changelog

All notable changes to geist are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0. While in 0.x, `EXPERIMENTAL`-tagged API may change in any
minor release.

## [Unreleased]

### Changed — clearer benchmark chart (total tok/s, geist vs llama.cpp)

- Replaced the prefill/decode/total matplotlib chart with a focused grouped-bar
  SVG of **total tok/s** (the user-facing metric) — geist vs llama.cpp (CPU and
  OpenBLAS) on a Pi 5, labelled with model + OS, honest about the long-prompt tie.
  Generated dependency-free by [`benchmark/chart_total_tps.py`](benchmark/chart_total_tps.py)
  straight from `pi5_results.json` (no matplotlib).

### Changed — generation stops on a sentence by default (no `-n` needed)

- `geist <model> "prompt"` treats the token budget as a **soft target**: it rounds
  up to the next sentence end (capped at 2×) instead of cutting mid-word. A base
  model on a bare completion prompt emits no end token, so the old hard 64-token
  default ended like "…Paris is also known". `-n N` is still an exact **hard** cap.
  So you never *need* to pass `-n` for a clean result.

### Fixed — decode HTML entities in fetched/searched text

- `webfetch_strip_html` now decodes HTML entities (`&amp;` → `&`, `&lt;`/`&gt;`,
  `&quot;`, `&#39;`, `&nbsp;`, numeric `&#NN;`/`&#xHH;`), so `web_search` titles and
  `web_fetch` page text read cleanly instead of showing raw `&amp;`. Bare `&` in
  running text is left untouched. Covered by `test_webfetch_unit`.

## [0.3.1] — 2026-06-28

### Added — one-command install + prebuilt single-file binaries

- **`install.sh`** — `curl -fsSL …/install.sh | sh` detects your platform,
  downloads the single-file `geist-bitnet` (BitNet 2B-4T baked in), and drops it on
  your PATH. One command, one file, nothing else to fetch.
- Releases now ship a `geist-bitnet-<platform>.tar.gz` by default — a self-contained
  binary that runs with **no model argument**. The release workflow's
  `EMBED_MODEL_URL` defaults to the BitNet GGUF; clear it to stop shipping them.
- README "Run it now" is now genuinely copy-paste runnable and split into two clear
  paths: ① the single baked-in binary (no model file), ② the < 1 MB engine + a GGUF
  (one model file runs on every platform). All download links are direct.

### Changed — `geist agent` sensible defaults (force + trace on)

- `geist agent` now **forces the tool call by default** (`GEIST_FORCE_CALL=0` to
  opt out). The bundled models (BitNet 2B-4T, Gemma 4 E2B) aren't tool-trained, so
  without forcing the agent would never run a tool — making it the default means
  `geist agent "<request>"` just works, no env prefix. (`geist chat` never forces.)
- The per-step trace (`· routing → calling → running → observed → answering`) also
  prints by default for `geist agent` — it goes to stderr, so the answer on stdout
  stays clean and piping is unaffected. Silence it with `GEIST_AGENT_TRACE=0`.
  `geist chat` is the opposite: trace stays **opt-in** (`GEIST_AGENT_TRACE=1`) so a
  conversation is quiet by default.

### Added — name the embedded binary (`EMBED_NAME`)

- `make EMBED_MODEL=... EMBED_NAME=geist-bitnet` names the self-contained binary
  distinctly. An embedded binary takes **no model-path argument** (the model is
  baked in), unlike the plain `geist` — giving it its own name avoids the "which
  one needs a model?" confusion. Defaults to `geist` (unchanged for normal builds).

### Added — single-file builds get the agent + chat

- A `make EMBED_MODEL=...` build is no longer text-only: `geist agent <request>`
  and `geist chat` now drive the baked-in model (no model-path argument). One
  self-contained binary generates text *and* runs tools — demoed with BitNet
  b1.58 2B-4T embedded, generating and summarizing a file on a Raspberry Pi 5.
  `geist_agent_main` takes the embedded GGUF bounds; `agent_main_parse_args`
  gained `want_model` to drop the model positional when it is baked in.

### Changed — memory tools are opt-in (`GEIST_MIND_DIR`)

- The agent's default toolset dropped from 7 to 5: `remember`/`recall` are
  included only when a palace is configured (`GEIST_MIND_DIR`). On weak models the
  router scores tool names, and the two memory tools made common requests (e.g.
  "summarize report.md") mis-route to `recall` on some CPU backends (BitNet/NEON).
  Fewer default tools → robust routing across backends. `geist chat`'s
  `/remember`,`/recall` slash commands are unaffected (they call `mind.h` directly).

### Changed — bounded chat context (sliding window)

- Multi-turn `geist chat` now evicts the oldest turns once the transcript passes
  a budget (`agent_compact`): it keeps the protected system prompt and the most
  recent whole turns, down to a target size. This bounds per-turn re-prefill (a
  long chat stays O(n) instead of O(n²)) and replaces the old hard "context full"
  stop — the model forgets the evicted turns. The eviction point is a documented
  hook for folding the dropped span into a running summary later (the summarizer
  already exists). Tunable via `GEIST_AGENT_CTX_BUDGET` / `_TARGET`.

### Added — `geist chat` + memory tools

- The interactive chat is now the **`geist chat`** subcommand, rebuilt on the
  agent engine: `geist_agent_run` gained a `conversation` flag that keeps the
  transcript across turns. It carries the full toolset and the memory palace, and
  inherits the engine's chat-template handling (the old hand-rolled inline framing
  and its stop-marker leak are gone). Removed the `geist_chat` binary.
- The memory palace is now model-callable via two tools (`tools/agent_memory.h`):
  `remember(text)` (title auto-derived from the first line — single-arg so it
  works under a forced call) and `recall(slug)`. "Search my notes" reuses
  `doc_search` over `$GEIST_MIND_DIR`. Both tools are in `geist agent` and `geist
  chat`; when memory is present the notes index is injected so `recall` is usable
  one-shot. The `/remember`, `/recall`, `/notes` slash commands stay as the
  reliable manual path on un-tool-trained models.

### Changed — one agent CLI, folded into `geist`

- The tool-use agent is now the **`geist agent`** subcommand of the main CLI, not
  a separate binary. `geist <model> <prompt>` generates text; `geist agent <model>
  <request>` runs the whitelist-gated tool loop (list_dir, summarize_file,
  doc_search, web_search, web_fetch). Both honour `GEIST_FORCE_CALL=1` and
  `GEIST_AGENT_TRACE=1`. This removes the "which binary?" footgun — `./geist` no
  longer silently ignores the agent env vars.
- Removed the `geist_agent` and `geist_shell` demo binaries (merged into the
  subcommand). The reusable engine `agent_main.h` gained a tool-builder callback
  (so a tool's ctx can reference the loaded model) and now owns the force-call +
  trace env knobs, so every CLI built on it behaves identically.

## [0.3.0] — 2026-06-23

### Added — on-device tool-use agent

- A bounded, whitelist-gated tool-use loop lets a small local model read files and
  search the web **in-process**: `list_dir`, `summarize_file`, `doc_search` (local
  keyword RAG, paragraph-granular + overlap-scored), and `web_fetch` (curl, no
  shell, scheme + host gated). Tool routing and the JSON call structure are forced
  from outside the sampler, so even untrained 2 B models drive the tools reliably.
  Full design and security model in `docs/agent.md`.
- A reusable agent CLI engine (`tools/agent_main.h`) with the `geist_agent`
  reference CLI, plus an interactive chat mode with a file-based memory palace.

### Changed — CI hardening

- New jobs: ASan + UBSan unit tests, a musl/Alpine build (tests what we ship),
  real-model integration + e2e tests, and ccache-cached compilation.

### Changed — speculative output head is now on by default

- The speculative int8-sketch lm_head (below) now defaults **on** for greedy
  decode on an eligible tied head; `GEIST_SPEC_HEAD=0` forces the exact dense
  head. Verified byte-identical greedy output on Gemma 4 (Q6_K head, 256 K
  vocab) and BitNet (F16, 128 K) for ~+5 % Pi 5 decode. Non-greedy sampling,
  ineligible dtypes, and non-NEON/dotprod hosts always fall back to the dense
  head, so the default change is a no-op there.

### Added — speculative int8 output head for BitNet 2B-4T decode (Pi 5)

- `GEIST_SPEC_HEAD=1` enables a speculative lm_head for large **tied F16**
  embeddings (`src/archs/transformer/forward/spec_head.c`). On Microsoft's
  BitNet-b1.58-2B-4T `I2_S` model the F16 lm_head is ~656 MB read per token —
  ~50 % of decode. The spec head keeps a stride-4 int8 **sketch** of the
  embedding (`[vocab, hidden/4]`, ~82 MB), rough-ranks the whole 128 K vocab with
  one SDOT pass, takes the top-512, and computes **exact f16 logits** for only
  those. Greedy output is byte-identical to the dense head (the deciding logits
  are unquantized); opt-in, non-greedy sampling falls back automatically.
- Result on a Raspberry Pi 5 (A76, `tests/bench_perf_sweep`, 2 t, 2.4 GHz):
  BitNet 2B-4T `I2_S` **decode 9.83 → 17.4 tok/s** — ahead of both other engines
  built and run on the same box: Cougar (Rust + `ea` SIMD) **12.3** and bitnet.cpp
  **8.2** (~2×). See `benchmark/TERNARY_BITNET.md` for the same-box three-engine
  comparison and the layer-matmul kernel shapes (4-row, fused gate+up) that were
  tried and reverted as A76 regressions.
- The spec head also covers **block-quantized** tied lm_heads (Q3_K/Q4_K/Q5_K/
  Q6_K/Q8_0). Phase 3 builds a one-row view of the embedding and calls the *same*
  `linear_m1` the dense head uses (W6A8 for Q6_K), so finalist logits are
  **bit-exact** — no f32-dequant approximation. The only approximation is sketch
  recall (which rows become finalists), so `GEIST_SPEC_TOPK` is now vocab-aware
  (512 for ≤200 K, 4096 above) and tunable along with `GEIST_SPEC_STRIDE`. On
  **Gemma 4 E2B** (tied Q6_K, 256 K vocab) greedy is byte-identical to the dense
  head at TOP_K 4096 for **+5 % decode** (6.94 → 7.29 t/s, 4 t; or +14 % if a
  smaller TOP_K is allowed to diverge). Opt-in, greedy only.

## [0.2.1]

### Added — embed a model into the binary (single-file deploy)

- `geist_model_load_from_memory(data, size, be, &model)` (`geist.h`): load a GGUF
  that is already in memory. Weights are aliased zero-copy from the buffer (the
  caller keeps it alive); the GGUF must carry its own tokenizer; text-only. Backed
  by a new internal `gguf_open_memory`.
- `make EMBED_MODEL=path/to/model.gguf` bakes a GGUF into the `geist` CLI via an
  `.incbin` stub (portable ELF + Mach-O), so the engine *and* the model ship as
  one binary; the CLI then takes only a prompt. For small models — the binary
  grows by the model size (build warns past ~1.5 GB).
- This **completes the "Single-file app + model" roadmap item** — deployment can
  be literally one binary. Scoped to small models on purpose: beyond the build
  cost, the default `-mcmodel=small` PC-relative addressing range caps an
  embedded blob (~2 GB on x86-64, ~4 GB on AArch64), and a standalone GGUF is
  already `mmap`-aliased zero-copy at runtime — so for large models the
  separate-file form keeps engine/model decoupling with no RAM penalty. A
  streamed `geist pack` format for large models remains future work.

### Added — per-platform mmap hints for large models

- The weight `mmap` now applies best-effort `madvise` hints. Linux:
  `MADV_HUGEPAGE` (transparent huge pages → fewer TLB misses on the big weight
  tables — a real win on **4 KB-page Linux servers**; `GEIST_NO_HUGEPAGE=1` to
  disable). All platforms: opt-in `MADV_WILLNEED` prefault via
  `GEIST_MMAP_PREFETCH=1` (steadier first-token latency, bigger upfront read).
  Honestly measured: **no effect on the Raspberry Pi 5** — it already uses 16 KB
  base pages and has no THP, so the TLB win is moot there; the lever is for
  4 KB-page Linux. No regression (Pi pp256 unchanged within noise).

### Added — docs
- `docs/QUICKSTART.md`: a two-minute guide to running the CLI and embedding the
  library (complete copy-paste C program, clean EOS stop, single-file build,
  performance knobs), linked from the README.

## [0.2.0]

### Changed — public API split by audience (source-compatibility break)

`geist.h` is now the **minimal surface to run a model** (backend → model →
session → `set_prompt` → `decode_step` → `token_to_str`). Helpers and advanced
features moved to new headers. **No signatures changed**; declarations moved.

- `geist_util.h` *(new)* — chat / advanced apps: special tokens
  (`geist_model_eos_token` / `_bos_token` / `_token_by_text`),
  `geist_session_tokenize` / `_prefill_tokens`, multimodal `attach_audio/image/video`,
  `pin_prefix`, `peek_logits`, `decode_speculative`, telemetry, and
  `geist_backend_supports_op`.
- `geist_types.h` *(new)* — backend-author territory: low-level tensor / op /
  dtype / buffer / quant types (previously in `geist.h`). Pulled in by
  `geist_backend.h` and `geist_weight.h`.

**Migration:** pure text generation needs no change (`#include <geist.h>` still
compiles; `examples/simple_generate.c` and `tools/geist.c` are untouched). Apps
using special tokens / multimodal / speculative / telemetry add
`#include <geist_util.h>`. Backend/tensor-type code adds `#include <geist_types.h>`.

### Added
- `geist_model_eos_token`, `geist_model_bos_token`, `geist_model_token_by_text`
  (`geist_util.h`): special-token ids from GGUF metadata, so a chat app stops on
  `tok == eos` instead of string-matching decoded output.

## [0.1.3]

### Changed
- Linux release artifact is a fully static **musl** binary (was glibc): ~40 %
  smaller download, portable across any aarch64 Linux with no libc dependency.
  Both release binaries are stripped.

## [0.1.2]

### Changed
- Parallelized the O(n²) int8 SDPA prefill core (bit-exact) — flat Pi 5 prefill
  curve (pp1024 +35 %). Benchmark re-measured honestly (Pi thermal-throttling
  artifact corrected; matched cool-start protocol; llama.cpp leads Pi prefill).

## [0.1.1]

### Added
- `geist_gemm` abstraction + BLAS-free native NEON fp32 + vendored FFT; fully
  dependency-free static ARM build; CI release matrix; int8-kernel tuning; CLI.

## [0.1.0]

First public release.

### Added
- C23 inference runtime with a stable C ABI (`include/geist.h`), per-symbol
  `STABLE` / `EXPERIMENTAL` stability tags.
- Backends: `cpu_neon` (Apple Silicon + ARM64, OpenMP-parallel kernels) and
  `cpu_scalar` (portable reference). `cpu_x86` is a policy skeleton.
- Quantization: GGUF `Q4_0/Q8_0`, k-quants `Q3_K/Q4_K/Q5_K/Q6_K`, IQ-quants
  `IQ2_S/IQ3_S`, and ternary `TQ2_0` for 1.58-bit models. Zero-dispatch kernel
  binding: every tensor is bound to a specialized kernel at load time.
- Transformer architecture (Gemma 4 family) with RoPE, GQA attention, KV cache,
  and per-session sampler (greedy / top-k / top-p / temperature).
- KV-cache quantization modes (INT8, KIVI), AWQ scale loading, and an n-gram
  speculative-decode path (all `EXPERIMENTAL`).
- Native multimodal: Conformer audio tower (`attach_audio`) and SigLIP vision
  tower for image/video soft-token prefixes (`attach_image` / `attach_video`).
- Build system with per-target/per-mode segregation (`mac`, `mac-omp`, `pi5`,
  `linux`/generic ARM64), `debug`/`asan`/`perf` modes, and on-demand reference
  model fetch (`make fetch-model`).
- Test suite (exit-code contract, `_unit`/`_int`/`_e2e` tiers) and a
  reproducible perf benchmark harness (`make bench-small`).
- `examples/simple_generate` demonstrating the stable text-generation core.

[Unreleased]: https://github.com/geisten/geistlib/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/geisten/geistlib/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/geisten/geistlib/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/geisten/geistlib/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/geisten/geistlib/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/geisten/geistlib/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/geisten/geistlib/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/geisten/geistlib/releases/tag/v0.1.0
