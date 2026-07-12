# Home Assistant implementation phases

This is the implementation index for the Geist Home Assistant product. It turns
the product direction in `ROADMAP.md` into work packages another agent can pick
up without conversation history. The detailed current beta backlog lives in
`proposals/home-assistant-phase-2.md`; the wire contract lives in
`proposals/dynamic-tools-v1.md`.

## Non-negotiable architecture

- One resident `geist-home` process keeps the model warm.
- One private local transport: Unix socket for Core/Container; a private app
  transport may be added only when HA OS packaging requires it.
- One application protocol: newline-framed dynamic-tools-v1 JSON.
- No HTTP/REST inference server, OpenAI-compatible API, SSE, public listener,
  API key, HA token, registry-push protocol, or compatibility transport.
- Home Assistant derives request-scoped tools from Assist exposure, revalidates
  policy immediately before execution, executes, and returns typed results.
- Every request has immutable capabilities, schema validation, size/deadline
  limits, and a global call budget.

## How agents execute this plan

Start at the first item marked **NEXT** or **pending** whose dependencies are
complete. Keep one PR to one numbered slice. In the same PR, update its status,
list the verification commands and evidence, and move **NEXT**. Do not introduce
a second protocol or migration adapter. A slice is complete only when its exit
gate is reproducible from a clean checkout.

## Phase 0 — product boundary and evidence baseline ✅

Purpose: establish the niche—small, controlled, private CPU edge agents—and a
measured baseline.

Delivered: product positioning, Pi 5 baseline, bounded home actions, public
roadmap, security boundary, and preview acceptance criteria.

Exit gate: roadmap, architecture, benchmark evidence, and supported/non-
supported actions agree.

## Phase 1 — resident local preview ✅

Purpose: load the model once and expose a private same-host serving surface.

Delivered: `geist-home --serve <socket>`, mode-0600 Unix socket, warm model,
bounded sequential requests, systemd/install/setup scripts, clean-host tests.

Current contract: the server accepts dynamic JSON only. A model-free
`{"type":"health"}` control frame reports readiness. There is no network API.

Exit gate: clean build, sequential requests without reload, bounded request
size/time, SIGPIPE-safe client disconnect, private socket, installer rollback.

## Phase 2 — native Home Assistant beta 🚧

Purpose: make the preview installable and diagnosable without SSH knowledge.
Detailed ordered slices and the **NEXT** pointer are in
`proposals/home-assistant-phase-2.md`.

Delivered: HA-owned execution, dynamic exposure/policy, UI Config Flow with
validated health handshake, reconfigure flow, DE/EN setup errors.

Remaining: diagnostics/Repairs, reconnect and busy UX, protected HA app,
multi-architecture distribution, soak and external beta evidence.

Exit gate: fresh HA OS installation in ten minutes without shell/YAML/token;
Core/Container remains supported through the private Unix socket.

## Phase 3 — host-neutral dynamic tools v1 ✅

Purpose: make Home Assistant one adapter of a general controlled-agent contract.

Delivered: immutable per-request tools, bounded Schema-v1, typed multi-argument
forced calls, off-list rejection, confidence clarification, correlated results,
retry/cancel/global budgets, independent C host and deterministic tests.

Normative artifact: `proposals/dynamic-tools-v1.md`.

Exit gate: `make dynamic-example-host`, dynamic unit/security suites, and real
model/socket/host evidence pass.

## Phase 4 — Home Assistant integration productization 🚧

Purpose: make Geist a polished Assist Conversation Agent distributed through
HACS before considering Home Assistant Core.

Slices:

1. ✅ UI Config Flow, health validation, reconfigure, DE/EN setup strings.
2. ✅ Polling health entity, Repairs with automatic recovery, and fully redacted
   config-entry diagnostics.
3. ✅ Fresh-socket reconnect, at-most-one cancel, zero-queue busy handling and
   structured status/duration logs without request content.
4. ✅ Explicit HA-language metadata and in-memory history bounded by 32
   conversation ids, four turns and 2048 context bytes with deterministic LRU.
5. Pending: HACS layout, Hassfest/HACS validation, tagged compatibility table.

Exit gate: install, configure, break, diagnose, repair, reconfigure and unload
through HA UI; no YAML or daemon logs required.

## Phase 5 — Home Assistant app 🚧 **NEXT**

Purpose: install and supervise the resident runtime on HA OS.

Slices: ✅ protected-compatible `aarch64`/`amd64` scaffold, AppArmor, `/data`
boundary, healthcheck and build-only CI; **NEXT:** pinned model/runtime inputs,
signed image publishing, checksums, provenance and SBOM; then persistent cache,
watchdog, RAM/architecture checks;
AppArmor; no privileged mode, host network, Docker socket, public port or HA
config mount; signed images, SBOM, backup/restore and rollback tests.

Exit gate: add repository, install app, select model, start, add integration and
run the first correct request without SSH.

## Phase 6 — HA evaluation and security evidence 🚧 pending

Purpose: ship safety and quality evidence as a product artifact.

Corpus: at least 150 cases covering state/action, brightness/temperature, areas,
ambiguity, multiple calls, follow-ups, German compounds/colloquialisms,
unexposed/hallucinated entities, injection in names/state, high-impact actions,
tool failures and cancellation.

Metrics: tool and argument accuracy, final-action accuracy, denied-action rate,
clarification rate, TTFT/latency, RSS and reproducible energy where available,
split by model/language/hardware.

Exit gate: zero exposure-boundary violations, published corpus/runner/results,
24-hour Pi 5 soak and reproducible resource bounds.

## Phase 7 — fast path for simple intents 🚧 pending

Purpose: keep routine voice control fast without weakening the single
Conversation Entity or security boundary.

Scope: deterministic handling for simple light/switch/status requests; Geist for
ambiguity, complex questions, summaries, multi-step work and natural follow-ups.
Both paths must use the same exposure/policy executor.

Targets: warm Pi 5 simple request under 3 s; complex tool task under 10 s.

Exit gate: routing corpus proves no policy divergence and latency targets hold
without reducing complex-task quality.

## Phase 8 — public preview launch 🚧 pending

Purpose: launch only after installation, demo, safety and measurements are
reproducible.

Artifacts: signed preview release; 90-second offline Pi 5 demo; honest comparison
with alternatives on size/RAM/startup/latency/tool success; architecture post;
benchmark script; HACS listing; HA Forum and local-AI community launch; known
limitations and compatibility matrix.

Exit gate: Definition of Done in `ROADMAP.md` and Phase-2 scorecard are green,
with at least five external installations across both architectures.

## Deferred until the HA path is complete

Universal RAG, more general tools, broad GPU work, new experimental formats,
Windows GUI, large web UI, autonomous long-running agents, and direct Matter or
MQTT stacks. They do not improve the current product exit gates.
