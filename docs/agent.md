# Tool-use interface — `agent.h`, routing & `dynamic-tools-v1`

> **Scope.** This documents the tool-use **interface** geist ships as part of the
> libgeist SDK: the bounded loop (`agent.h`), routing / forced calls that extract
> a valid action from an *untrained* model, the reusable `--serve` daemon
> (`agent_main.h`), and the host-neutral `dynamic-tools-v1` protocol. geist stays
> the neutral engine + interface; **concrete tools** (document search, a memory
> palace, web fetch, file summarize, …) live in **consumer** projects that link
> libgeist — the reference knowledge assistant is
> [geisten/geist-wissen](https://github.com/geisten/geist-wissen); device control is
> [geisten/geist-home-assistant](https://github.com/geisten/geist-home-assistant).

**Why this layer exists.** A 2 B model won't reliably drive tools on its own.
geist's answer isn't a bigger model — it's a tight harness around a small one: a
bounded tool loop, routing and forced calls that lift a valid action from an
*untrained* model. The goal is big-model usefulness on a narrow, well-defined
task — entirely on-device.

The interaction layer sits **above** the public ABI (`include/geist.h`).
Everything here is **header-only** (`static inline`) in the SDK's headers
(`agent.h`, `agent_main.h`, the `dynamic_*_v1.h` / `json_schema_v1.h` set), so a
desktop binary and an embedded host (iOS/Android) compile the same code
in-process — no `libgeist` surface change and nothing extra to link.

The agent runs in the **same process** as the model: a request is an in-process
function call over a resident `geist_session`. "Resident" therefore means
different things per platform — a long-lived daemon on a server, a live session
object inside an app — but the core is identical.

- [The agent — `agent.h`](#the-agent--agenth)
- [Tools](#tools)
  - [Tool selection & forced calls](#tool-selection--forced-calls)
  - [Per-request dynamic tools](#per-request-dynamic-tools)
  - [Progress events](#progress-events)
- [The `--serve` daemon & dynamic adapters](#the-serve-daemon--dynamic-adapters)
- [Security model](#security-model)
- [Embedding the agent](#embedding-the-agent)

---

## The agent — `agent.h`

A bounded, whitelist-gated tool-use loop over a resident session. A request may
take several internal steps (search → read → search again → answer), so the
public entry is `geist_agent_run` (one request, many steps), not a single step.

```c
/* A tool is host-supplied — the concrete constructors live in your app; here is
 * an inline one to show the shape. */
static enum geist_status echo_invoke(void *ctx, size_t alen, const char a[static alen],
                                     size_t cap, char out[static cap], size_t *olen) {
    (void) ctx;
    *olen = (size_t) snprintf(out, cap, "%.*s", (int) alen, a);
    return GEIST_OK;
}
struct geist_agent agent;                                   /* caller storage (large) */
struct geist_tool  tools[] = {{ .name = "echo", .description = "echo the args",
                                .args_schema = "{\"text\": string}", .invoke = echo_invoke }};
geist_agent_init(&agent, model, session,
                 /*n_tools=*/1, tools, /*max_steps=*/8,
                 /*system_prompt=*/"You are a helpful assistant.");
agent.force_call = true;   /* optional: see "Tool selection & forced calls" below */

char   resp[2048];
size_t n = 0;
geist_agent_run(&agent, strlen(req), req, sizeof resp, resp, &n);   /* -> enum geist_status */
```

`run()` loops, each step:

1. generate one assistant turn,
2. parse the first `{"tool":"<name>","args":{…}}` (tolerates prose / ```json fences),
3. **gate** `<name>` against the whitelist — unknown/forbidden ⇒ never runs, the
   step gets an `error: tool … not allowed` observation instead,
4. run the matching tool, append its result as an observation, reopen the turn,
5. when the model replies in plain text (no tool call) ⇒ that is the final answer.

`max_steps` (default 8) bounds how many tool calls one request can trigger — a
runaway and cost guard on constrained hardware, and a hard cap on actions per
request.

Signatures follow the project C conventions ([CONTRIBUTING.md](../CONTRIBUTING.md)):
count precedes its buffer, caller-provided buffers, `enum geist_status`, no
`assert()`, no hidden heap (the transcript is fixed inside the caller-owned
`struct geist_agent`).

Ceiling (`ponytail:`): the transcript suffix is reprefilled per tool step. The
system/menu prefixes are pinned where the architecture supports it; a future
incremental suffix cursor can remove the remaining repeated prefill.

## Tools

A tool is a host-supplied entry in the whitelist:

```c
struct geist_tool {
    const char *name;          /* whitelist key; must match the emitted "tool" */
    const char *description;   /* one line, used by the router (see below) */
    const char *args_schema;   /* shown to the model, e.g. {"query": string} */
    const char *parameters_schema; /* optional dynamic-tools-v1 JSON Schema */
    enum geist_status (*invoke)(void *ctx,
                                size_t args_len, const char args[static args_len],
                                size_t out_cap,  char       out[static out_cap],
                                size_t *out_len);
    void *ctx;                 /* host state: doc index, HTTP client, home bridge */
};
```

Use `agent_json_str(args, "key", cap, out)` to pull a flat string field from the
validated args. geist ships **no concrete tools** — a consumer supplies them as
one-call `*_tool(...)` constructors (see geist-wissen for the reference
`doc_search` / `summarize_file` / `web_*` / memory-palace set, and
geist-home-assistant for device tools over `dynamic-tools-v1`).

### Tool selection & forced calls

Two mechanisms make tools work on **small, non-tool-trained** models, both driven
entirely over the public `peek_logits` / `prefill_tokens` / `tokenize` API — no
in-engine sampler change:

- **Routing** (`agent_select_tool`) — with more than one tool, the raw
  `{"tool":"` logit often picks a valid-but-wrong tool. Instead the request + a
  tool menu (`name: description`) are framed as a question and each tool **name**
  is scored by its first-token log-prob (an MMLU-style cloze). The score is
  **PMI-calibrated** — the model's prior for each name (given the menu but a
  content-free request) is subtracted, removing token-frequency bias. The menu
  also carries a **`reply` pseudo-entry** — when it wins, no tool is
  forced and the model answers directly ("What is 2 plus 2?"). A request opening
  with a destructive verb no whitelisted tool covers ("Delete report.md",
  "Lösche …") routes to `reply` before any scoring — a refusal instead of a
  forced-but-wrong call, at zero model cost.
  Deterministic tie-breakers settle close races: a request naming a file
  (`note.txt`) prefers a file tool, a literal `http(s)://` URL prefers the
  `url`-arg tool (first-token scoring cannot tell a search from a fetch tool
  — same first token), and a docs word (`docs`, `Dokumente`) prefers the doc
  tool.
- **Forcing** (`agent.force_call = true`) — grammar-*forces* turn 0 to be a valid
  call to the routed tool. Legacy single-key values are lifted from the request.
  A dynamic tool instead builds a typed object from `parameters_schema`:
  required and detected optional fields, integer/number, boolean, scalar enum
  and enum-array values are supported. The complete object is validated before
  the host callback. An unclear required value produces a clarification. The
  legacy forced call is single-shot — the tool's observation is returned as the
  answer — unless a **recipe** continues it: a known 2-step chain fires when the
  request carries a cue word for the second step ("… and *read* it", "…
  *fasse* es zusammen"). Step 1's locator is lifted from step 0's observation
  and invoked directly — zero extra model time. This is why an *untrained* model
  can still drive tools reliably: structure and value are forced, only the
  routing decision is the model's.

For dynamic routing, a calibrated winner must also clear the runner-up/direct
reply by `route_min_margin`; otherwise Geist asks which action the user means.
An invalid follow-up call after a valid result cannot replace that result or run
host code: Geist switches to call-free answer generation over the last valid
observation.

### Per-request dynamic tools

The `--serve` daemon accepts one newline-delimited JSON request per connection:

```json
{"input":"Add 5 and 7","max_tool_steps":4,"tools":[{"name":"CalculatorAdd","description":"Add two integers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}]}
```

Hosts may add bounded `language` and `context` strings. They become
request-local system context; `input` remains the sole source for current forced
tool arguments.

The request compiler copies up to 16 unique tools into immutable storage and
rejects the entire request if a schema or unsupported keyword is invalid. The
documented subset supports object/string/number/integer/boolean/array,
`properties`, `required`, `additionalProperties`, scalar `enum`, numeric bounds,
`items`, `minItems` and `maxItems`, with fixed depth/count limits.

For dynamic tools Geist never owns the action. It emits a correlated
`tool.call`; the host revalidates and executes, returns `tool.result`, and Geist
continues. Calls, one retry, cancellation and final results share the same global
budget. See [dynamic-tools-v1.md](proposals/dynamic-tools-v1.md). Build the
reference host with `make dynamic-example-host`.

### Progress events

`geist_agent_run` is otherwise a black box that returns only the final answer.
Set an optional callback to watch each step live (routing → calling → running →
observed → answering):

```c
a.on_event     = agent_event_print;   /* the bundled printer, or your own */
a.on_event_ctx = stderr;
```

The callback receives a `struct geist_agent_event { phase; step; tool; detail; }`
at every phase boundary (`enum geist_agent_phase`). `nullptr` (the default) = no
events, zero overhead. The bundled `agent_event_print` writes one friendly line
per step to a `FILE*` — `geist_agent_main` wires it to `stderr` **by default**
(`GEIST_AGENT_TRACE=0` silences it), keeping the answer on `stdout` clean:

```
· routing summarize_file: selected
→ calling summarize_file: {"path":"report.md"}
⚙ running summarize_file
✓ observed summarize_file: The report proposes …
● answering: The report proposes …
```

`tool`/`detail` point into agent buffers and are valid **only during the
callback** — copy them if you retain them. A server host can serialize the same
struct to JSON and stream it to a UI.

## The `--serve` daemon & dynamic adapters

`agent_main.h` is a reusable agent CLI engine: an app supplies a system prompt
and a tool builder, and it does arg parsing, model load, one-shot / REPL /
`--serve`, and the force-call + trace env knobs. The `geist` CLI wires it up for
`--serve` only (with an **empty** tool builder — the host supplies tools per
request):

```sh
geist -m model.gguf --serve /path/geist.sock   # or omit -m for an embedded model
make dynamic-example-host
bin/$(mk/detect-target.sh)/release/examples/dynamic_tools_host \
  /path/geist.sock "Add 5 and 7"
```

The model stays warm; each request supplies its immutable tool set and step
budget over a `chmod 600` Unix socket; the host executes validated calls and
returns correlated results. Domain policy and credentials stay outside Geist.
Product adapters (Home Assistant, a knowledge assistant, …) live in separate
repositories and test against the versioned wire contract. A health handshake
(`{"type":"health"}` → `{"type":"health.result",…,"status":"ready"}`) lets a host
probe readiness.

Your own app gets the same engine by linking libgeist and writing a ~15-line
`main` that calls `geist_agent_main` with its own tool builder — see the header
comment in `agent_main.h`.

## Security model

A small model jailbreaks easily as free chat, so **the host — not the model —
decides what can happen**:

1. **Immutable scope** — a short system prompt plus a compiled static whitelist
   or a copied per-request toolset (no model-created capabilities).
2. **Whitelist gate** — the model may emit any text, but only a tool in `tools[]`
   can run; an off-list name is rejected before any side effect.
3. **Global call budget** — `max_steps` caps calls per request; a retry consumes
   the same budget.
4. **Per-tool input validation at the trust boundary** — the tool owns this. A
   `web_fetch`-style tool, for example, should run `curl` via `fork`+`execvp`
   (no shell, so `http://x;rm -rf ~` cannot inject a command), gate the scheme to
   literal `http`/`https`, enforce a host allowlist (exact host or a dot-bounded
   subdomain), and cap size + time. These live in the consumer's tool, not the
   engine.

The transport choice (stdin pipe, Unix socket, in-process) does **not** affect
jailbreak resistance — that lives in steps 1–4. It only affects attack surface:
prefer a Unix-domain socket (filesystem-permission gated) or in-process over an
HTTP listener for a same-host agent.

Hardest guarantee — **grammar masking + typed validation** (shipped): a free
turn that opens a call (`{` or a ```-fence) is decoded *along the call grammar*
over the public peek/prefill API — the tool name per-token constrained to the
whitelist, the arg key to the tool's schema. Dynamic forced calls additionally
build typed multi-field values and validate the entire object against Schema-v1.
The model cannot execute an off-whitelist or schema-invalid call. Repeated or
invalid follow-ups after a valid result switch to call-free answer generation.

## Embedding the agent

- **Desktop / server (Unix):** include `agent.h` + your tools, hold a resident
  `geist_session` + `geist_agent` for the process lifetime, and call
  `geist_agent_run` per request. For repeated requests on a warm model, wrap it
  in a small `accept()` loop on a Unix-domain socket (`chmod 600` = access
  control) — or just use `--serve`. See [DEPLOY.md](DEPLOY.md).
- **iOS / Android:** link `libgeist.a`, include `agent.h` via a bridging header,
  and call `geist_agent_run` directly from Swift/Kotlin — no transport, no socket
  (the OS forbids spawning a second process; a background daemon is suspended).
  Supply platform-native tools (e.g. a `web_fetch` over `URLSession`/`OkHttp`).
