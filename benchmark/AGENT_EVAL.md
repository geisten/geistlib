# Agent-layer reliability eval — recorded results

This file records reproducible runs of `make bench-agent` (the agent-layer
reliability eval: routing / args / chains / answer content — see
[docs/agent.md, "Testing the agent"](../docs/agent.md#testing-the-agent--make-bench-agent)
for what is measured and how). Like [BENCHMARK.md](BENCHMARK.md), every row is
tagged with host, OS, target/mode, model, **git version, and execution date**,
so results from different environments never silently overwrite each other.

## Methodology

- **Harness:** `tests/bench_agent_eval` over `tests/data/agent_eval/cases.jsonl`
  (46 cases: 15 single-tool / 8 chains / 4 ambiguous / 3 negative / 16 e2e,
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
| 2026-07-09 | v0.3.3-29-gc640d9a | Apple M1 Max (10c) | macOS 26.5.1 | mac-omp/release | OMP default | bitnet-2b4t-i2_s | 41 → **PASS** | 41/46 | 256 s | 6/46 | 419 s | 675 s (446 % CPU) |
| — pending — | | Raspberry Pi 5 (4×A76) | | pi5/release | 4 | bitnet-2b4t-i2_s | 41 | | | | | |

Pi 5 counter-check (once the box is reachable again; quiesce it first — a
stray process halves 4-thread numbers): same command as above on the Pi.
Close PMI routing races can flip on NEON's different fp accumulation order —
a deviation from the mac numbers is itself a finding, not noise.

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
