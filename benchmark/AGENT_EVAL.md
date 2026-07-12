# Agent-layer reliability eval — recorded results

This file records reproducible runs of `make bench-agent` (the agent-layer
reliability eval: routing / args / chains / answer content — see
[docs/agent.md, "Testing the agent"](../docs/agent.md#testing-the-agent--make-bench-agent)
for what is measured and how). Like [BENCHMARK.md](BENCHMARK.md), every row is
tagged with host, OS, target/mode, model, **git version, and execution date**,
so results from different environments never silently overwrite each other.

## Methodology

- **Harness:** `tests/bench_agent_eval` over `tests/data/agent_eval/cases.jsonl`
  (48 cases: 15 single-tool / 8 chains / 4 ambiguous / 3 negative / 18 e2e,
  incl. the memory roundtrip and two multi-turn conversations).
- **Determinism:** greedy decode, fixture docs, content-sensitive in-process
  web stubs — no network. One process, model loaded once.
- **Modes:** `forced` (`force_call`, the shipped path — this is what the gate
  scores) and `free` (diagnostic column).
- **Gate:** `AGENT_EVAL_MIN` (exit 1 below it). Thresholds are calibrated per
  model on a baseline run and then fixed.
- **Timing:** per-mode wall seconds as printed by the harness; total wall from
  `time`. Timing is workload-dependent (chains and summaries decode more
  tokens than routing-only cases) — read it as "cost of one full eval run in
  this environment", not as a throughput benchmark (that is
  [BENCHMARK.md](BENCHMARK.md)).
- **Raw logs:** filed (untracked) under
  `bench_runs/agent_eval/<date>_<target>_<model>_<commit>.log`.

Reproduce:

```sh
GEIST_GGUF_PATH=gguf_artifacts/bitnet-2b4t-i2_s.gguf make bench-agent
```

## Recorded runs

| date | version | host | OS | target/mode | threads | model | gate | forced pass | forced wall | free pass | free wall | total wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-10 | ecbb8f4 (+realpath fix) | Raspberry Pi 5 (4×A76, 4 GB) | Debian 13.5 (6.18 rpt) | pi5/release (OpenBLAS) | 4 | bitnet-2b4t-i2_s | 43 → **PASS** | **43/48** | 722 s | 4/48 | 1460 s | 2183 s |
| 2026-07-10 | v0.3.3-31+stocks | Apple M1 Max (10c) | macOS 26.5.1 | mac-omp/release | OMP default | bitnet-2b4t-i2_s | 43 → **PASS** | 43/48 | 339 s | 4/48 | 472 s | 811 s |
| 2026-07-09 | v0.3.3-29-gc640d9a | Apple M1 Max (10c) | macOS 26.5.1 | mac-omp/release | OMP default | bitnet-2b4t-i2_s | 41 → **PASS** | 41/46 | 256 s | 6/46 | 419 s | 675 s (446 % CPU) |

Home Assistant product evaluations and live-pipeline measurements moved to the dedicated `geisten/geist-home-assistant` repository. This file retains only engine and general-agent evidence.

**Fixed (same day): the system-prompt pin (f87de25).** BOS + system prompt
are pinned via `geist_session_pin_prefix` on first run; `reset()` rewinds to
the pinned prefix and only the transcript suffix re-prefills. Routing runs on
a DEDICATED second session — the first single-session attempt scored the
selection prompt behind the pinned system prompt and doc_search stole 7
routings incl. "What is 2 plus 2?" (the tool table in the prompt contaminates
the cloze; PMI does not cancel it). On its own session, routing is
bit-identical to the legacy path: main eval restored to exactly 43/48, home
stays 41/41, both gates green at unchanged thresholds. Measured after the
fix: **Pi turn 3.6–4.7 s** steady state (turn 1 pays the one-time
pin+baseline build: 9.3 s), **full Assist pipeline 5.2 s** — ~3× faster.
Mac eval walls: main forced 339→116 s.

The 2026-07-10 runsThe 2026-07-10 runs add the `stock_movers` tool (48 cases, 8-tool menu) and
the routing restructure it forced: evidence BAN mask (rare-name tools do not
compete without their lexical evidence) plus unwindowed evidence-beats-reply.
Full-name routing scores were tried and measured worse (36/48) — see the
`ponytail:` note in `agent_score_names`. Remaining forced fails are the five
known-honest ones: ch-4, ch-7, amb-1, amb-2, conv-4.

**Pi 5 counter-check result:** the forced mode (the shipped, gated path) is
**identical to the mac run** — same per-stage totals, same five failing
cases. The feared NEON-vs-AMX fp-accumulation flips in close PMI routing
races did not materialize: the evidence rules leave enough margin that
backend numerics don't decide routings. The gate passes on the Pi with the
same threshold (43). Free mode differs per-case (long greedy decodes diverge
across backends, sum coincidentally 4/48 on both) — diagnostic only. Cost:
the full eval is ~2.7× mac wall time on a quiesced Pi 5 (forced 722 s vs
339 s), fine for a nightly.

### 2026-07-09 · mac-omp · bitnet-2b4t-i2_s · v0.3.3-29-gc640d9a

Per-category (forced / free):

| category | pass | route | args | chain | answer |
|---|---|---|---|---|---|
| single (15) | 15/15 · 4/15 | 15/15 · 12/15 | 12/12 · 7/12 | 15/15 · 5/15 | — |
| chain (8)   | 6/8 · 0/8    | 7/8 · 3/8    | —            | 6/8 · 0/8   | — |
| ambig (4)   | 2/4 · 0/4    | 2/4 · 0/4    | 2/4 · 0/4    | 2/4 · 0/4   | — |
| neg (3)     | 3/3 · 0/3    | 3/3 · 0/3    | —            | 3/3 · 0/3   | — |
| e2e (16)    | 15/16 · 2/16 | 16/16 · 8/16 | 10/10 · 5/10 | 16/16 · 3/16 | 15/16 · 5/16 |
| **total (46)** | **41/46** · 6/46 | 43/46 · 23/46 | 24/26 · 12/26 | 42/46 · 8/46 | 15/16 · 5/16 |

Known-honest forced fails (documented, part of the fixed gate): `ch-4`
(whole-request query lifting too noisy for doc_search's paragraph scoring),
`ch-7`/`amb-1`/`amb-2` (genuine routing ambiguity), `conv-4` (German
free-form follow-up degenerates on the 2B; English context carry passes).

Live-web smoke (`make bench-agent-live`, manual, not gated) on the same
build/date: `live-1` (example.com fetch, answer-checked) PASS; `live-2/3`
hit DuckDuckGo rate-limiting — the chain degrades as designed (error
observation becomes the answer). Use `GEIST_SEARX_ENDPOINT` for stable runs.

Advisory judge (`make bench-agent-judge`, gemma4:26b via local Ollama,
never gated): **coherent 41/46**, flagged `ch-7`, `amb-1` (the known routing
fails, with correct reasons), `conv-4` (the German degeneration), and
`neg-1/3` (tool-ish string answers to unfulfillable requests — the 2B
ceiling). The judge's first run had additionally flagged `neg-1/2/3` as bare
`{` fragments — a finding the mechanical checks were blind to (no `expect`
on neg cases); that led to the call-proof reply turns (`agent_generate_reply`:
prose-primed, call tokens banned every step, the router's no-tool decision
terminal), after which "What is 2 plus 2?" answers `4` and neg-2 became a
gated regression test. The judge and the substring gate fail in orthogonal
ways — which is why the judge stays advisory and the gate stays mechanical.

## The LLM-half: geist vs Ollama on a Pi 5 (2026-07-11)

The community's voice-assistant split keeps deterministic phrase-matching for
device control and an **LLM for "the rest"** (general Q&A, light reasoning). This
measures that second half on the deployment box itself — can a 4 GB Pi 5 run it
self-contained, and how does geist compare to Ollama there?

**Setup:** 10 general one-shot tasks (facts, unit conversion, one-line reasoning,
DE+EN), warm, greedy, 4 threads, HA
container idling. geist via `geist <model> -c` (chat-templated one-shot); Ollama
via `/api/generate`, `think:false`, `use_mmap:true`, `num_ctx 1024`. Prompts are
unique (no Ollama prompt cache).

| config | model (footprint) | mean s/task | median | quality |
|---|---|--:|--:|---|
| **geist-bitnet** | BitNet 2B-4T i2_s (**1.2 GB**, ternary) | **1.85** | 1.79 | substance correct |
| ollama-gemma | Gemma 4 E2B Q4 (**3.4 GB**) | 3.90 | 3.73 | cleanest (mature wrapper) |
| geist-gemma | Gemma 4 E2B Q4 (2.9 GB) | 5.03 | 5.00 | correct |

**Headline — footprint / self-sufficiency:** BitNet's 1.2 GB runs comfortably
alongside HA; Ollama's 3.4 GB Gemma is **OOM-killed on load without `use_mmap`**
(`llama-server ... signal: killed` after 2 min) and the box's 2 GB swap sat at
**100 % used** throughout the Gemma runs. A ternary model is the comfortable fit
on a 4 GB Pi; a Q4 general model is at the edge.

**geist-BitNet is the fastest (~1.85 s), even handicapped.** geist ran as a *cold
CLI* (reloads per call) against Ollama's *warm resident server*; BitNet's load is
cheap (1.2 GB mmap, page-cache hot) so cold≈warm, and it still beat the warm
Ollama. Same turn on an M1 Max: ~0.8 s.

**Honest caveats.** (1) geist-gemma slower than ollama-gemma (5.03 vs 3.90) is a
**cold-CLI vs warm-server artifact**, not an engine loss — the same-model warm
comparison (geist ≥ llama.cpp on Gemma E2B) is in the throughput table above; a resident `geist --serve` daemon would avoid repeated model startup.
(2) Ollama's API wall-clock is ~15 % prefill-optimistic. (3) The only model gap
was **cosmetic**: the bare `geist <model> "prompt"` CLI leaked template tokens
(`<|im_sep|>`) and stopped poorly — fixed by `geist -c/--chat`, which applies the
model's chat template (the agent/serve layer already had the wrapper). N=10, easy
tasks; this sizes the LLM-half, it is not a general capability eval.
