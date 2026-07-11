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
| 2026-07-11 | home + lock flow | Apple M1 Max (10c) | macOS 26.5.1 | mac-omp/release | OMP default | bitnet-2b4t-i2_s | **home** 41 → **PASS** | **41/41** | 141 s | 3/41 | 330 s | 471 s |
| 2026-07-10 | home nightly (2b4ef81) | Raspberry Pi 5 (4×A76, 4 GB, HA container idling) | Debian 13.5 | pi5/release | 4 | bitnet-2b4t-i2_s | **home** 31 → **PASS** | **31/31** | 252 s | — (nightly runs forced only) | — | 252 s |
| 2026-07-10 | home appliance (51da99b) | Apple M1 Max (10c) | macOS 26.5.1 | mac-omp/release | OMP default | bitnet-2b4t-i2_s | **home** 31 → **PASS** | **31/31** | 114 s | 3/31 | 227 s | 341 s |
| 2026-07-10 | v0.3.3-31+stocks | Apple M1 Max (10c) | macOS 26.5.1 | mac-omp/release | OMP default | bitnet-2b4t-i2_s | 43 → **PASS** | 43/48 | 339 s | 4/48 | 472 s | 811 s |
| 2026-07-09 | v0.3.3-29-gc640d9a | Apple M1 Max (10c) | macOS 26.5.1 | mac-omp/release | OMP default | bitnet-2b4t-i2_s | 41 → **PASS** | 41/46 | 256 s | 6/46 | 419 s | 675 s (446 % CPU) |

**The home appliance eval** (`make bench-agent-home`, `cases_home.jsonl`, 31
cases): the standalone 2-tool domain menu routes **perfectly** — 31/31 forced,
including all four deliberate-ambiguity answers, the lock refusals, and the
pronoun conversation turns against the mutating state stub. Baseline was 21/31;
the fixes were German-compound noun evidence, lock verbs, and the sentence-
shape (imperative/question) tie-breaker. Live-verified end-to-end against a
real Home Assistant container on the Pi 5 (template + demo entities; the
appliance switches a real light, sets the demo thermostat, reads sensors).
Free mode stays diagnostic. This is the per-domain methodology working as
designed: narrow menu, own corpus, own gate, better-than-demo reliability.

**Lock confirmation flow (2026-07-11):** locks moved from hard refusal to a
deterministic two-turn flow — locking runs directly (the safe direction), an
unlock request only ARMS a file-based pending slot and answers with a
challenge; the unlock executes only when the immediately following command
carries the literal confirm word and resolves to the same entity. One-shot
(any other command disarms), 120 s TTL, status queries in between are
allowed. The corpus grew to 41 cases (10 lock cases across three
conversations incl. the disarm proof) — still **41/41 forced**, gate raised
to 41. Live-verified against a real HA template lock on the Pi 5: initial
locked → challenge (state verified UNCHANGED) → confirm → unlocked → cold
confirm refused → relock → locked. The model never decides the security
question — it only ferries the user's words.

**Appliance latency (Pi 5, --serve daemon, measured 2026-07-11):** one
"Schalte das Licht im Flur ein" turn costs **12–13 s** at the daemon socket;
the full Assist pipeline (the phone-app path) adds ~1–2 s → **~14 s
end-to-end**. Stable across turns (not transcript growth at this length):
the cost is the per-turn prefill work — selection prompt + the FULL
transcript re-prefill including the constant system prompt (the documented
O(n²) ceiling). The mac does the same turn in ~3 s.

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
Mac eval walls: main forced 339→116 s, home 114→51 s.

**Home-gate nightly (Pi 5):** `scripts/nightly-home-gate.sh` runs the DEPLOYED
tree's forced home gate every night at 03:30 (crontab), one summary line per
night in `~/geist-nightly/home-gate.log`, full per-date logs kept two weeks,
and a `~/geist-nightly/FAILED` marker while the latest run is red (the morning
check — no notification infra by design). Both paths verified live: a red run
(threshold forced to 32) sets the marker and logs FAIL with the true counts;
the next green run clears it. Deployment is a deliberate manual rsync+make, so
a red nightly always means "this deployed state regressed". ~250 s per night
with the HA container idling alongside (load noted in every log).

The 2026-07-10 runs add the `stock_movers` tool (48 cases, 8-tool menu) and
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
