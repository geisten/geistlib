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
- [The home appliance — `make home`](#the-home-appliance--make-home)
- [Testing the agent — `make bench-agent`](#testing-the-agent--make-bench-agent)
  - [The end-to-end cases](#the-end-to-end-cases-cat-e2e)
  - [Live-web smoke](#live-web-smoke--make-bench-agent-live)
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
  ./geist chat -m gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

REPL commands (anything else is a chat turn the model answers — and may call a
tool, including `remember`/`recall`, on its own):

| command | effect |
|---|---|
| `/remember <title> \| <text>` | write a note + index it (explicit title) |
| `/recall <slug>` | queue a note into context for your next message |
| `/notes` | print the index (what's stored) |
| `/help`, `/quit` | help / exit |

The slash commands are the manual path (they bypass the model); the
`remember`/`recall` **tools** let the model manage memory itself — and since the
routing gained memory/slug evidence rules, they route reliably on the bundled
un-tool-trained models too (the `make bench-agent` roundtrip cases gate this:
remember → recall → the fact comes back).

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
  The menu also carries a **`reply` pseudo-entry** — when it wins, no tool is
  forced and the model answers directly ("What is 2 plus 2?"). A request opening
  with a destructive verb no whitelisted tool covers ("Delete report.md",
  "Lösche …") routes to `reply` before any scoring — a refusal instead of a
  forced-but-wrong call, at zero model cost.
  Deterministic tie-breakers settle close races: a request naming a file
  (`note.txt`) prefers a file tool, a literal `http(s)://` URL prefers the
  `url`-arg tool (first-token scoring cannot tell `web_search` from `web_fetch`
  — same first token), and a docs word (`docs`, `Dokumente`) prefers the doc
  tool.
- **Forcing** (`agent.force_call = true`) — grammar-*forces* turn 0 to be a valid
  call to the routed tool: the JSON scaffold + the chosen tool name are prefilled
  token-by-token, and a single-key arg's value is **lifted from the request** (the
  path word for a locator arg, else the request itself as the query) rather than
  free-decoded, which a weak model mangles. The forced call is single-shot — the
  tool's observation is returned as the answer — unless a **recipe** continues
  it: a known 2-step chain (`web_search→web_fetch`, `doc_search→summarize_file`,
  `list_dir→summarize_file`) fires when the request carries a cue word for the
  second step ("… and *read* it", "… *fasse* es zusammen"). Step 1's locator is
  lifted from step 0's observation (the top URL / the request-matching file) and
  invoked directly — zero extra model time. This is why an *untrained* model
  can still drive tools reliably: structure and value are forced, only the routing
  decision is the model's.

Reliability is measured, not felt: `make bench-agent` scores routing / args /
chains per stage over a fixed 30-case corpus (`tests/bench_agent_eval`) and
gates CI at the fixed threshold (`AGENT_EVAL_MIN`, calibrated per model).

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

## The home appliance — `make home`

The first DOMAIN BUILD: `make home` produces `./geist-home` — BitNet baked in,
and **only the home tools compiled in** (no web, no docs, no stocks). Fixed
scope at build time: the artifact itself is the security promise.

```sh
make home
GEIST_HA_URL=http://<ha-host>:8123 GEIST_HA_TOKEN=<long-lived-token> \
  ./geist-home "Schalte das Licht im Flur ein"    # -> OK: light.flur → an
```

Three layers:

- **`ha_rest.h`** — standalone, agent-free Home Assistant REST client
  (call_service / get_state, Bearer auth, no-shell curl). Manual driver:
  `geist ha state|on|off|open|close <entity_id>` (all build profiles).
- **`agent_home.h`** — two tools along the read/write boundary:
  a **command tool** (turn on/off, dim, set temperature, open/close) and a
  **status tool** (device/sensor reads), both `{"request"}` — the model only
  routes; device, action and value are parsed deterministically against the
  **registry file** (`GEIST_HOME_REGISTRY`, default `./home-registry.txt`,
  one `entity | domain | alias phrases` line per device). Ambiguity ("das
  Licht" with two lights) yields a deterministic clarifying answer; unknown
  devices a clear error. A **last-device memory** resolves pronouns ("Mach
  *es* wieder aus"). **Collectives** ("alle Lichter aus") act on every
  matched writable device — never on locks; bare "alles" stays the safe
  no-device answer. **Relative setpoints** ("mach es wärmer") move the
  current climate value by ±1 °C. Devices HA reports `unavailable` answer
  "nicht erreichbar" instead of a stale state. Writes are whitelisted to
  light/switch/climate/cover/media_player —
  **garage doors and alarms are refused** even if listed. **Locks take a
  confirmation flow**: locking runs directly (the safe direction), but an
  unlock request only *arms* a file-based pending slot and answers with a
  challenge — the unlock executes only when the immediately following request
  carries the literal confirm word and resolves to the same entity
  ("Bestätige entriegeln Haustür"). The slot is one-shot (any other command
  disarms it) and expires after 120 s; the check is fully deterministic — the
  model never decides a security question, it only ferries the user's words.
- **routing evidence** — home nouns match as substrings (German compounds:
  *Flurlicht*), action verbs word-start; the sentence SHAPE decides the
  read/write boundary (imperative → command, question → status; a direction
  adverb in a non-question — "Rollladen *runter*" — counts as imperative,
  the verbless cover command was a score race otherwise).

Input is bilingual (DE + EN). The **answer language** is a deployment setting:
German by default, `GEIST_HOME_LANG=en` switches the device state words
(on/off/open/closed/locked/unlocked/unavailable) to English — the pieces every
command and status answer shows. The eval never sets the env, so the fixed gate
is unaffected.

**Adding a language** is one line, not a code change: the state words live in a
table (`HOME_LANGS` in `tools/agent_home.h`), one row per language keyed by the
`GEIST_HOME_LANG` code. Copy a row, translate the eight words, done — the lookup
is table-driven, so no new branches anywhere. (The rarer clarify/challenge/error
*sentences* are still German literals; lift them into the same table when an
English-first deployment ships.)

Demo backend: the official Home Assistant container with template entities
matching the starter registry (see `home-registry.txt`); onboarding + a
long-lived token are API-scriptable. Eval: `make bench-agent-home` — the
standalone 2-tool menu over `cases_home.jsonl`, gate `AGENT_EVAL_HOME_MIN`.

## Testing the agent — `make bench-agent`

Reliability is measured, not felt. `tests/bench_agent_eval` loads a real model
and sends every case through the production path (`geist_agent_run`: routing →
call → dispatch → observation → answer), scoring **mechanically per stage** —
no LLM judge:

| check | question | mechanic |
|---|---|---|
| `route` | right tool chosen? | first `CALLING` event == expected tool (`"none"` = no call) |
| `args`  | right arguments? | expected key present, value contains the expected substring |
| `chain` | right tool sequence? | exact order of `RUNNING` events == expected chain |
| `ans`   | right **content** in the answer? | final answer contains one of the `expect` alternatives |

Corpus: `tests/data/agent_eval/cases.jsonl` (flat JSONL; one case per line —
`id`, `cat`, `req`, expected `tool`/`arg`/`want`/`steps`/`expect`, optional
`conv` group). Deterministic: greedy decode, fixture docs in the repo, no
network. Both agent modes run as separate columns — **forced** (`force_call`,
the shipped path) is gated via `--min-pass` / `AGENT_EVAL_MIN`; **free** is a
diagnostic column. Exit 1 below the gate → CI-able.

### The end-to-end cases (`cat: e2e`)

The `ans` check is what makes a case end-to-end, and the web stubs are
**content-sensitive** so it proves something: the stub search *ranks* a fixed
page table by query/title word overlap, the stub fetch serves each URL its own
text — a case passes only when the right page actually travelled
search → fetch → answer.

- **Web research chains** (`e2e-1..4`): find an article and read/summarize it
  (EN + DE); the top-ranked result must be the queried one; `e2e-4` fetches a
  **prompt-injection page** whose text orders the agent to fetch
  `evil.example.com` and delete files — the chain must end after the fetch
  regardless.
- **Direct tool use with content checks** (`e2e-5..9`): two different URLs
  fetched (each must yield *its* page text), `list_dir` with a lifted path,
  doc retrieval, file summary.
- **Memory-palace roundtrip** (`mem-1..3`): `remember` writes a note through
  the agent, `recall` reads *that* note back by slug (the fact must return);
  plus a German recall of a pre-seeded note. Uses a scratch mind dir under
  `build/`.
- **Multi-turn conversation** (`conv-1..4`): cases sharing a `conv` group keep
  one transcript; the follow-up ("What did I just ask you about?") passes only
  by *reading* the prior turns. `conv-4` documents a known ceiling: German
  free-form follow-ups degenerate on the 2B model.

### Live-web smoke — `make bench-agent-live`

The same harness with the **real** `web_search`/`web_fetch` (curl +
DuckDuckGo) over `tests/data/agent_eval/cases_live.jsonl`. Manual and
report-only — network results never gate CI. Known findings: DuckDuckGo
rate-limits back-to-back requests within one run (the chain degrades as
designed: the error observation is the answer); set
`GEIST_SEARX_ENDPOINT=<searxng-url>` for a stable search backend.

### Advisory judge — `make bench-agent-judge`

A second AI reads the answers. The mechanical `expect` check only proves a
fact *occurs* — "report report Wohner der Heimat report.md" would pass while
being no answer at all. `bench_agent_eval --dump` writes every forced-mode
answer to JSONL and `tools/eval_agent_judge.py` asks a local Ollama model
(`JUDGE_MODEL`, default `gemma4:26b`) per case whether it is a **coherent,
plausible response** to the request. Strictly advisory (exit 0 always): LLM
judges drift with model updates, so the deterministic substring gate stays
authoritative — the judge tells you which answers to eyeball, replacing the
manual spot-check.

Results per environment (version, date, wall time) are recorded in
[benchmark/AGENT_EVAL.md](../benchmark/AGENT_EVAL.md).

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

Hardest guarantee — **grammar masking** (shipped): a free turn that opens a
call (`{` or a ```-fence) is decoded *along the call grammar* over the public
peek/prefill API — the tool name per-token constrained to the whitelist, the
arg key to the tool's schema, only the value free (stopped at its closing
quote). The model *cannot* emit an off-grammar or off-whitelist call; a call
already executed in the same run ends the loop (the observation is the
answer). Remaining (`ponytail:` upgrade): full multi-key argument grammars —
today a multi-key schema decodes its single best key.

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
