# Agent, CLI & memory palace

**Why this layer exists.** A 2 B model won't reliably drive tools or carry a
long memory on its own. geist's answer isn't a bigger model — it's a tight
harness around a small one: a bounded tool loop, routing and forced calls that
extract a valid action from an *untrained* model, and a file-based memory sized
to what the model can hold. The goal is big-model usefulness on a narrow,
well-defined task — entirely on-device. It's an open, active track (see the
[roadmap](../ROADMAP.md)).

The interaction layer that sits **above** the public ABI (`include/geist.h`).
Everything here lives in `tools/` as **header-only** modules (`static inline`),
so the desktop binaries and an embedded host (iOS/Android) compile the same code
in-process — there is no `libgeist` surface change and nothing new to link.

The agent runs in the **same process** as the model: a request is an in-process
function call over a resident `geist_session`. "Resident" therefore means
different things per platform — a long-lived daemon on a server, a live session
object inside an app — but the core is identical.

- [The CLI — `geist chat`](#the-cli--geist-chat)
- [The memory palace — `mind.h`](#the-memory-palace--mindh)
- [The agent — `agent.h`](#the-agent--agenth)
- [Tools](#tools)
  - [Tool selection & forced calls](#tool-selection--forced-calls)
  - [Progress events](#progress-events)
- [Security model](#security-model)
- [Embedding the agent](#embedding-the-agent)

---

## The CLI — `geist chat`

`geist chat` is an interactive multi-turn conversation, built on the **same agent
engine** as `geist agent` (`geist_agent_run` with `conversation = true`, so the
transcript carries across turns). It has the full toolset plus the memory palace,
and reliable `/slash` control over notes.

```sh
make                                       # builds ./geist
GEIST_MIND_DIR=./mind \
  ./geist chat gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

REPL commands (anything else is a chat turn the model answers — and may call a
tool, including `remember`/`recall`, on its own):

| command | effect |
|---|---|
| `/remember <title> \| <text>` | write a note + index it (explicit title) |
| `/recall <slug>` | queue a note into context for your next message |
| `/notes` | print the index (what's stored) |
| `/help`, `/quit` | help / exit |

The slash commands are the **reliable** manual path (they bypass the model); the
`remember`/`recall` **tools** let a capable model manage memory itself. On the
bundled un-tool-trained models, prefer the slash commands.

`geist chat --selftest` runs a palace round-trip with **no model** (CI-friendly).

Environment: `GEIST_MIND_DIR` (default `./mind`) is the palace directory.

## The memory palace — `mind.h`

A file-based long-term memory: plain Markdown, no database, no embeddings.

```
$GEIST_MIND_DIR/
  INDEX.md          one line per note; loaded into the model's context each session
  <slug>.md         one note per file, with YAML front-matter
```

A note:

```markdown
---
title: Tenancy notice periods
date: 2026-06-21
---
A landlord's ordinary notice is due by the third working day of the month …
```

`INDEX.md` line: `- [<title>](<slug>.md) — <hook> · <date>`

**Retrieval has no vector store.** `INDEX.md` is small and is injected into the
system context at session start, so the model sees the titles/hooks and asks to
`recall(<slug>)`; the full note then enters context. `grep -ri <term> $MIND_DIR`
covers ad-hoc search.

API (all `static inline`, caller-bounded buffers, no hidden heap):

```c
int  mind_remember(const char *title, const char *text);          // write + index; 0 ok
long mind_recall(const char *slug, char *buf, size_t cap);        // bytes, or -1 if absent
long mind_slurp(const char *path, char *buf, size_t cap);         // bounded file read
void mind_slugify(const char *title, char *out, size_t cap);      // "My Note!" -> "my-note"
```

Ceiling (`ponytail:`): the in-context index scales to a few hundred notes; beyond
that, grep the index or shard by tag.

## The agent — `agent.h`

A bounded, whitelist-gated tool-use loop over a resident session. A request may
take several internal steps (search → read → search again → answer), so the
public entry is `geist_agent_run` (one request, many steps), not a single step.

```c
struct geist_agent agent;                                   /* caller storage (large) */
struct geist_tool  tools[] = { docsearch_tool("./docs") };
geist_agent_init(&agent, model, session,
                 /*n_tools=*/1, tools, /*max_steps=*/8,
                 /*system_prompt=*/"You are a document assistant.");
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

Ceilings (`ponytail:`): O(n²) full-transcript reprefill per step; naive JSON
brace-balance (→ grammar-constrained sampling for a hard guarantee, below).

## Tools

A tool is a host-supplied entry in the whitelist:

```c
struct geist_tool {
    const char *name;          /* whitelist key; must match the emitted "tool" */
    const char *description;   /* one line, used by the router (see below) */
    const char *args_schema;   /* shown to the model, e.g. {"query": string} */
    enum geist_status (*invoke)(void *ctx,
                                size_t args_len, const char args[static args_len],
                                size_t out_cap,  char       out[static out_cap],
                                size_t *out_len);
    void *ctx;                 /* host state: doc index, HTTP client, home bridge */
};
```

Use `agent_json_str(args, "key", cap, out)` to pull a flat string field from the
validated args. Shipped tools (each a one-call `*_tool(...)` constructor):

| tool | header | constructor | does |
|---|---|---|---|
| **`list_dir`** | `agent_listdir.h` | `listdir_tool()` | list a directory's entries (`opendir`, no shell) |
| **`summarize_file`** | `agent_summarize.h` | `summarize_file_tool(&ctx)` | read a text file under a root and refine-summarize it (sub-session); `ctx` = `{model, be, root}` |
| **`doc_search`** | `agent_docsearch.h` | `docsearch_tool(dir)` | keyword search over a directory of text files — local RAG, no embeddings |
| **`web_search`** | `agent_websearch.h` | `websearch_tool(endpoint)` | search the web (DuckDuckGo HTML, or SearXNG JSON if `endpoint` points there) → title+URL results |
| **`web_fetch`** | `agent_webfetch.h` | `webfetch_tool(allow_hosts)` | fetch an http(s) URL via `curl`, return tag-stripped text |

`web_search` + `web_fetch` compose: search for pages, then fetch+read one. Both
are Unix/desktop only (`curl` + `fork`); an iOS/Android host supplies its own over
the platform HTTP client. `summarize_file` confines reads to its `root` (rejects
absolute paths and `..`). Ceilings (`ponytail:`): `doc_search` is a linear scan,
plain text only; `web_search` scrapes DuckDuckGo's HTML result anchors (point it
at a SearXNG instance for robustness); HTML entities in titles aren't decoded.

### Tool selection & forced calls

Two mechanisms make tools work on **small, non-tool-trained** models, both driven
entirely over the public `peek_logits` / `prefill_tokens` / `tokenize` API — no
in-engine sampler change:

- **Routing** (`agent_select_tool`) — with more than one tool, the raw
  `{"tool":"` logit often picks a valid-but-wrong tool. Instead the request + a
  tool menu (`name: description`) are framed as a question and each tool **name**
  is scored by its first-token log-prob (an MMLU-style cloze). The score is
  **PMI-calibrated** — the model's prior for each name (given the menu but a
  content-free request) is subtracted, removing token-frequency bias. A
  deterministic tie-breaker prefers a file tool when the request names a file
  (`note.txt`) and the race is close.
- **Forcing** (`agent.force_call = true`) — grammar-*forces* turn 0 to be a valid
  call to the routed tool: the JSON scaffold + the chosen tool name are prefilled
  token-by-token, and a single-key arg's value is **lifted from the request** (the
  path word for a locator arg, else the request itself as the query) rather than
  free-decoded, which a weak model mangles. The forced call is single-shot — the
  tool's observation is returned as the answer. This is why an *untrained* model
  can still drive tools reliably: structure and value are forced, only the routing
  decision is the model's.

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

## Security model

A small model jailbreaks easily as free chat, so **the host — not the model —
decides what can happen**:

1. **Fixed scope** — a short system prompt + a fixed tool whitelist (no "universal
   assistant").
2. **Whitelist gate** — the model may emit any text, but only a tool in `tools[]`
   can run; an off-list name is rejected before any side effect.
3. **Step budget** — `max_steps` caps actions per request.
4. **Per-tool input validation at the trust boundary** — e.g. `web_fetch`:
   - **no shell**: `curl` runs via `fork`+`execvp` with the URL as a separate
     argv element, so a URL like `http://x;rm -rf ~` cannot inject a command;
   - **scheme gate**: only literal `http://` / `https://`;
   - **host allowlist**: exact host or a dot-bounded subdomain (so `example.com`
     allows `www.example.com` but rejects `notexample.com` / `example.com.evil.com`);
   - `curl --proto/--proto-redir =http,https` so a redirect can't change scheme;
     size + time caps.

The transport choice (stdin pipe, Unix socket, in-process) does **not** affect
jailbreak resistance — that lives in steps 1–4. It only affects attack surface:
prefer a Unix-domain socket (filesystem-permission gated) or in-process over an
HTTP listener for a same-host agent.

Hardest guarantee (`ponytail:` upgrade): **in-sampler grammar masking** — mask the
sampler's logits to grammar-valid tokens so the model *cannot* emit anything but a
valid, on-whitelist call. geist already **forces** a valid tool name and re-keys
arguments from outside the sampler (see *Tool selection & forced calls* above);
full in-sampler masking of arbitrary argument grammars is the remaining step.

`web_search` is the lower-risk half of web access — it only talks to a **fixed**
search endpoint (the host is not model-chosen), so the model influences the
*query*, not the destination. But the results it returns are attacker-influenced
text, so a URL the model then hands to `web_fetch` is still untrusted and goes
through `web_fetch`'s scheme + host gate.

Not yet defended: `web_fetch` does not block private/link-local IPs (SSRF) — the
host allowlist is the mitigation for an open fetch.

## Embedding the agent

- **Desktop / server (Unix):** include `agent.h` + your tools, hold a resident
  `geist_session` + `geist_agent` for the process lifetime, and call
  `geist_agent_run` per request. For repeated requests on a warm model, wrap it
  in a small `accept()` loop on a Unix-domain socket (`chmod 600` = access
  control). See [DEPLOY.md](DEPLOY.md).
- **iOS / Android:** link `libgeist.a`, include `agent.h` via a bridging header,
  and call `geist_agent_run` directly from Swift/Kotlin — no transport, no socket
  (the OS forbids spawning a second process; a background daemon is suspended).
  Supply platform-native tools (e.g. a `web_fetch` over `URLSession`/`OkHttp`
  instead of the `curl` one).
