# Changelog

All notable changes to geist are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0. While in 0.x, `EXPERIMENTAL`-tagged API may change in any
minor release.

## [Unreleased]

### Changed — engine core slimmed to inference + `--serve`

- The `geist` CLI is now inference-only plus the resident dynamic-tools daemon:
  `geist -m model "prompt"` (instruct/`--raw` generation) and
  `geist -m model --serve <socket>`. The `agent` and `chat` subcommands and the
  interactive REPL are removed — a served request supplies its own toolset over
  `dynamic-tools-v1`, and one-shot/REPL tool use belongs in a consumer that links
  libgeist.
- The concrete reference tools (`doc_search`, `summarize_file`, `list_dir`,
  `web_search`, `web_fetch`, `remember`/`recall` + the `mind` palace) and the
  `stock_movers` demo tool moved **out of the engine** into the
  [geist-wissen](https://github.com/geisten/geist-wissen) consumer. geist keeps the
  tool-use **interface** (`agent.h`, `agent_main.h`, the `dynamic_*_v1.h` set),
  which continues to ship in the libgeist SDK. See geist#110.
- Removed the `make bench-tooling` / `bench-agent` quality benches and their CI
  quality gate, the `tests/bench_agent_eval` harness + `tests/data/agent_eval`
  corpus, the per-tool unit/integration tests, and `benchmark/AGENT_EVAL.md`
  (agent-layer reliability is now measured in the consumer). `docs/agent.md` is
  slimmed to the interface.

## [0.4.0] — 2026-07-13

First release of the host-neutral `dynamic-tools-v1` runtime artifact. A consumer
pins the protocol id `dynamic-tools-v1` + a binary SHA-256 (published as
`SHA256SUMS`) and verifies with the startup health handshake
(`{"type":"health"}` → `…"status":"ready"`). Start forms: `geist --serve SOCKET`
runs the internal (baked-in) model; `geist -m MODEL --serve SOCKET` an external
one. Compatibility contract: `docs/proposals/dynamic-tools-v1.md`.

### Added — host-neutral dynamic tools v1

- `geist agent --serve` accepts an immutable per-request `tools` array and
  performs correlated `tool.call` / `tool.result` round trips over the local
  socket. The server accepts only the dynamic JSON protocol.
- Removed the Home Assistant REST/token client, registry-push and line-protocol
  adapters, unused protocol-v2 stack, and their migration-only tests/config.
- Added a fixed-memory JSON parser and documented Schema-v1 subset with strict
  name, type, required/optional field, enum, array, bound and duplicate-key
  validation. Unsupported keywords fail request compilation.
- Typed forced calls now cover multiple arguments, numbers, booleans, scalar
  enums and enum arrays. Low-confidence routes clarify; invalid/off-list calls
  never cross the host boundary.
- Added global call/retry budgets, correlated cancellation, HA-owned dynamic
  execution, and the independent `make dynamic-example-host` reference build.
- Added deterministic security/HA suites and a real BitNet + Unix-socket + C-host
  end-to-end transcript under `docs/benchmarks/`.
- Added a model-free dynamic-tools-v1 health handshake and a UI-only Home
  Assistant Config/Reconfigure Flow with validated socket and DE/EN errors.
- Added a polling HA Health entity, automatically recovering Repairs, and
  config-entry diagnostics that expose no paths, addresses or HA content.
- Added zero-queue HA request admission, fresh-socket reconnect semantics,
  at-most-one correlated cancellation, and content-free lifecycle logging.
- Added explicit request language and bounded request context plus HA-owned,
  in-memory conversation history with turn/byte/conversation LRU limits.
- Added the Home Assistant app repository and protected-compatible multi-arch
  scaffold with AppArmor, `/data`-only persistence, protocol healthcheck and a
  non-publishing `aarch64`/`amd64` CI build matrix.
- Added an agent-executable Home Assistant implementation plan for phases 0–8;
  HTTP/REST server requirements are explicitly out of scope.

## [0.3.3] — 2026-07-01

### Fixed — release embedded-build download resilience

- The embedded release builds fetch the ~1.1 GB BitNet GGUF 3× per release; on
  v0.3.3 HuggingFace rate-limited (HTTP 429) and the `--retry-delay 2` was too
  short, failing the x86 embedded job. Bumped to `--retry 5 --retry-delay 15
  --retry-all-errors` so a transient 429 is ridden out.

### Added — BitNet b1.58 2B-4T on x86 (AVX-512): beats bitnet.cpp

- The `cpu_x86` backend now runs **BitNet-2B-4T (I2_S ternary)** end-to-end and
  **beats Microsoft's bitnet.cpp on both metrics** (AMD Ryzen 9 9950X, Zen 5,
  16T, same `ggml-model-i2_s.gguf`): prefill **pp128 884 vs 679 t/s (+30 %)**,
  decode **tg128 77.9 vs 56.5 t/s (+38 %)**. It was previously non-functional on
  x86 (`cpu_scalar` had no I2_S linear nor an F16 lm_head path → zero tokens).
- **Ternary kernels** (`kernel_i2s*`): biased-u8 `VPDPBUSD` dot over the packed
  0.25 B/wt stream (Zen 5 has no s8×s8 VNNI, so codes {0,1,2} + a per-token
  sum-correction), unpacked in-register. The prefill GEMM uses an **x4
  row-interleaved layout** (`i2s_to_x4`) — 4 output rows packed at 2-bit
  granularity per byte so one activation load feeds 4 rows (4× fewer act loads),
  the decisive prefill win.
- **lm_head**: ported the NEON-only **speculative i8-sketch head** (`spec_head.c`)
  to x86 (AVX2 sketch dot + F16C finalists) — reads a ~82 MB subsampled sketch
  instead of the 657 MB F16 table, greedy output **byte-identical to the exact
  f16 dense head**. A `Q8` lm_head (`f16_to_q8w`) is the sampling /
  `GEIST_SPEC_HEAD=0` fallback (`GEIST_Q8_LMHEAD=0` forces exact F16C). On x86 the
  spec-head is gated to the F16 lm_head; Gemma/Llama keep their exact dense
  Q-decode.
- Cross-validated byte-identical to bitnet.cpp on the packed path; the x4 /
  spec-head paths verified against the scalar oracle / exact f16
  (`test_i2s_gemv_unit`, `test_q8w_gemv_unit`). `cpu_scalar` gained I2_S / F16 /
  BF16 linear (the unblock + test oracle).
- **`GEIST_I2S_PAIR=1`** (opt-in, default off): fuses the gate+up / q+k decode
  GEMVs into one OMP region sharing a single activation quant (5 OMP
  regions/layer → 3). Measured **neutral on the 9950X** — the ternary GEMVs are
  already bandwidth-bound (~73 GB/s aggregate) under both active and passive OMP
  wait — so it stays off by default; exposed for hosts where the caller measures
  a benefit (low memory bandwidth, high core count, costly thread wakeups).
  Bit-identical to the unfused path (`test_i2s_gemv_unit`).

### Added — prebuilt linux-x86_64 release binary (AVX-512)

- `release.yml` now also builds a **`geist-linux-x86_64.tar.gz`** — a dependency-free
  musl-static binary with the native AVX-512/VNNI backend (`BACKENDS="cpu_x86
  cpu_scalar"`, `GEMM_PROVIDER=native`). Baseline `x86-64-v3` (Haswell / Zen+) with
  AVX-512 kernels runtime-dispatched via `hw_probe`, so the one binary runs on any
  x86-64-v3 CPU. Pair it with a Gemma / Llama / BitNet GGUF. Windows still not shipped.
- …and a **`geist-bitnet-linux-x86_64.tar.gz`** single-file (BitNet 2B-4T baked in),
  now that the x86 I2_S ternary + spec-head kernels make BitNet fast on x86 — same
  one-file, no-model-argument deal as the arm64 / macOS embedded builds.

### Changed — docs reflect the landed x86-64 (AVX-512) backend

- README, `install.sh` and ROADMAP no longer say "x86 / Windows wait on the AVX
  backend" — that backend has landed. x86-64 Linux (AVX-512) now builds from source
  and is competitive with llama.cpp; prebuilt binaries stay ARM64 for now, Windows
  is still pending. Platform badge updated to `ARM64 + x86-64`.

### Added — AMD x86 (AVX-512) benchmarks vs llama.cpp

- README now reports the `cpu_x86` backend's measurements on an **AMD Ryzen 9
  9950X** (Zen 5, 16C/32T): Gemma 4 E2B Q4_K_M prefill **512 vs 495** (+3.4 %) and
  decode **48.6 vs 44.1** (+10 %); Llama 3.2 3B prefill **351 vs 346** (+1.4 %),
  decode 34.1 vs 34.5 (parity) — geist matches-to-beats llama.cpp on x86. Added to
  the headline table and the system-grouped scoreboard chart.

### Changed — clearer benchmark charts (geist vs the baseline)

- Replaced the prefill/decode/total matplotlib chart with two focused, dependency-free
  SVGs (pure stdlib, numbers straight from JSON, no matplotlib):
  - **`headline_benchmarks.svg`** — a scoreboard of geist's throughput as a ratio of
    the baseline across model × OS, each row its own headline metric + baseline
    (BitNet decode 2.1× bitnet.cpp; Gemma prefill 1.5× llama.cpp; Gemma decode/total
    ~1.1× llama.cpp). Generated by [`benchmark/chart_headline.py`](benchmark/chart_headline.py).
  - **`pi5_total_tps.svg`** — total tok/s, geist vs llama.cpp (CPU + OpenBLAS) on a
    Pi 5, honest about the long-prompt tie. Generated by
    [`benchmark/chart_total_tps.py`](benchmark/chart_total_tps.py).

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
