# Public API contract

What geistlib promises to consumers that link the library across a release
boundary, and what it deliberately does not.

Every declaration in `include/` carries a `@stability` tag. This page defines
what those tags mean, and pins the subset an out-of-tree **agent runtime**
depends on so the engine cannot break it by accident.

`scripts/check-api-contract.sh` enforces this page against the headers, and
`examples/agent_contract_smoke.c` enforces it against the built library — a
removed symbol fails to link, a changed signature fails to compile. Both run in
CI, so this document cannot quietly become fiction.

## Stability tiers

| Tag | Promise |
|---|---|
| `STABLE since X.Y.Z` | The signature and documented semantics do not change within the same major version. Removal or an incompatible change requires a major bump. |
| `EXPERIMENTAL` | May change or disappear in any release, including a patch. Usable, but pin an exact version and expect to adapt. |

Semver for the library surface:

- **Major** — a `STABLE` symbol is removed or its signature/semantics change
  incompatibly.
- **Minor** — symbols are added, or an `EXPERIMENTAL` symbol is promoted to
  `STABLE`. Additive only; existing callers keep building.
- **Patch** — implementation, performance, and documentation only.

Promoting to `STABLE` is a one-way door: it is a commitment, not a label. A
symbol should only be promoted once a real consumer depends on it and its shape
survives an accelerator/architecture change without a signature break.

### Exceptions taken

The table above is the rule. Where it has been broken, it is recorded here, so
the promise is not quietly weaker than it reads.

| Symbol | Release | What broke | Why |
|---|---|---|---|
| `geist_session_peek_logits` | 0.11.0 | Parameter order changed from `(s, n_logits)` to `(n_logits, s)`. Source-incompatible: callers must swap the arguments. | Aligns the accessor family on the out-size-first order rather than carrying the inconsistency past 1.0. Taken by the maintainer as a deliberate, accepted risk while the user base is small — **not** a precedent, and by the semver rule above it would otherwise have required a major bump. |

## The agent-runtime contract

A tool-use loop — constrained ("masked") decoding, KV-prefix pinning, chat
templating — is implemented out of tree, linking libgeist. That loop is the
security boundary of every product built on it, so the symbols it needs are
contractual.

Constrained decoding is what lets a model that was never trained for tool
calling still emit a well-formed one: the runtime inspects the next-token
logits, masks everything the grammar forbids, and commits token by token. That
requires logit-level access — the three promotions in 0.6.0 below.

Chat templating is out of tree for the same reason the tool-use loop is: the
turn markers a model wants are the runtime's business, and shipping that table
from here as well would put it on two include paths. But the *key* the template
is selected by — the model family — can only come from the model file, so
`geist_model_arch` is contractual as of 0.9.0. Feeding a model another family's
turn tokens pushes it off-distribution and wrecks instruction-following; a
consumer that cannot ask which family it loaded cannot avoid that.

### `include/geist.h`

| Symbol | Used for |
|---|---|
| `geist_backend_create`, `geist_backend_destroy` | runtime lifecycle |
| `geist_model_load`, `geist_model_destroy` | model lifecycle |
| `geist_session_create`, `geist_session_destroy` | resident session |
| `geist_session_set_prompt` | transcript prefill |
| `geist_session_decode_step` | token-by-token generation |
| `geist_session_token_to_str` | detokenizing the emitted surface |
| `geist_session_reset` | rewind to the pinned prefix between turns |
| `geist_model_arch` | selecting the chat template by model family — STABLE since 0.9.0 |

### `include/geist_util.h`

| Symbol | Used for | Status |
|---|---|---|
| `geist_model_eos_token`, `geist_model_bos_token` | turn termination | STABLE since 0.2.0 |
| `geist_model_token_by_text` | chat-template auto-detection | STABLE since 0.2.0 |
| `geist_session_prefill_tokens` | feeding constrained candidates | STABLE since 0.1.0 |
| `geist_session_peek_logits` | **constrained decoding** — the grammar mask | STABLE since 0.6.0 |
| `geist_session_pin_prefix` | amortizing the system prompt across turns | STABLE since 0.6.0 |
| `geist_session_tokenize` | measuring a candidate before committing it | STABLE since 0.6.0 |

**BOS (#263).** One rule on both tokenizer paths (GGUF-embedded and an
external `tokenizer.bin`): `geist_session_tokenize` returns the text's tokens
and adds no BOS (a BOS the text spells out, `<bos>` or `<s>`, encodes as that
id); `geist_model_add_bos` says whether the model expects one; and
`geist_session_set_prompt` starts the prefilled sequence with exactly one BOS
when it does, prepending it unless the prompt's tokens already begin with it.
A consumer that prefills tokens itself (`tokenize` + `prefill_tokens`) adds the
BOS per `geist_model_add_bos`.

`geist_session_peek_logits`'s ownership clause lets an accelerator backend
satisfy it by staging device memory into session storage, so the contract does
not freeze a CPU-only design.

### What `"auto"` picks

`geist_backend_create("auto")` (also `nullptr` or `""`) promises the best
backend compiled in for the host, not a particular one: `metal` on Apple
Silicon when the build has it and a GPU is found (since the release after
0.21.0; `cpu_neon` before), else the preferred CPU backend (`cpu_neon`, `cpu_x86`, `cpu_scalar`); Vulkan is never
picked by `"auto"`. `GEIST_BACKEND=<name>` replaces `"auto"` with that name.

A backend picked by `"auto"` must not fail a model its GPU cannot run. When
`geist_model_load*` on it fails with `GEIST_E_UNSUPPORTED` (a weight format
the GPU has no kernel for) or `GEIST_E_OOM` (the GPU memory budget), or the
model loads but a session of the load's `max_seq_len` does not fit, the load
is repeated on the CPU backend `"auto"` would otherwise pick. That backend is
created with the same opts and allocator and owned by the model (destroyed by
`geist_model_destroy`). The caller keeps passing its `"auto"` handle:
`geist_session_create` and `geist_decision_create` accept it as the model's
backend. `geist_model_backend_name` (EXPERIMENTAL) names the backend the model
runs on; `geist_backend_name` keeps naming the handle's. Explicit names never
fall back, and nothing falls back after load: a session created later that
does not fit fails at create as before. Logits differ in the last bits
between backends, so code that compares bits names its backend.

### Explicitly NOT in the contract

`geist_model_modalities`, `geist_session_attach_audio` / `attach_image` /
`attach_video`, the `geist_session_audio_*` streaming family,
`geist_session_decode_speculative`, `geist_session_verify_drafts`,
`geist_session_checkpoint`, `geist_session_set_draft`, the
`geist_session_snapshot` / `restore`
family, `geist_session_decode_batch` / `geist_model_decode_batch_max`, the `geist_session_stats` family and the layer split
(`geist_session_opts.split_backend` / `split_layers`, #49, with
`GEIST_SPLIT_AUTO` for the planner's choice, #78) remain
`EXPERIMENTAL`. They are useful and supported, but an agent runtime must not
build its core loop on them expecting release-boundary stability.

### Batched decode (experimental, #26)

`geist_session_decode_batch(n, sessions, out_tokens)` is
`geist_session_decode_step` for each of n distinct sessions of one model.
Their forwards run as one batched pass:
- every linear layer and the lm_head read their weights once for the n rows;
- attention, the KV cache and recurrent state stay per session;
- each session samples with its own settings.

The guarantees:
- **Isolation.** For a given n, a session's tokens and logits are
  bit-identical whichever other sessions share the call and in whichever slot.
- **Numerics.** n = 1 is `decode_step` itself. For n > 1 the batched rows
  round differently from a single-row step: logits can differ in the last
  digits, and over long generations a greedy path can diverge.
- **Refusals change nothing.** These are: `nullptr`, a repeated session,
  another model's session, n above `geist_model_decode_batch_max`, KIVI/MTP/
  split members, nothing pending, a full context and cancellation. A failure
  during the pass resets every member.

The decoder vtable carries the two optional `decode_batch` /
`decode_batch_max` slots, appended without moving existing fields.
`geist_model_decode_batch_max` is 0 where a model cannot batch; today
Qwen3.5-family hybrids on Metal and Vulkan can, and the Qwen MoE models (#245)
on Metal where their experts route on the device and on Vulkan with the experts
split with the CPU, up to 8 sessions.

### Optional decision API (experimental)

All declarations in `include/geist_decision.h` are EXPERIMENTAL and outside
the agent-runtime contract. The default-off `DECISION=1` feature adds
independent numeric scoring handles; libraries built without it keep the
symbols as stubs that return unsupported. Generation, embedding and logits
APIs are unaffected. Lifetime, reset, conditional-score semantics,
capabilities and benchmark protocol: [DECISION.md](DECISION.md). The decoder
`logits_vocab_size` hook is appended to the experimental architecture vtable
without changing existing field offsets; the selected-row readout hooks and
`geist_weight` row-tile callbacks are EXPERIMENTAL as well. `GEIST_DECISION_SELECTED_ROWS` requires explicit mode
selection and a supported loaded model/backend pair; its result metadata reports
row work, logical staging bytes and selected-head time. Unsupported modes never
silently fall back to another execution path. Prefix scoring
(`geist_decision_prefix_*`, `geist_decision_score_from_prefix`, #5) needs
`DECISION_CONTEXT=1` as well and is EXPERIMENTAL; its architecture hook
`prefill_geometry` is appended to the end of the decoder vtable.

## The geist-runtime contract

[geist-runtime](https://github.com/geisten/geist-runtime) is the chat layer over
libgeist: templates, streaming text, a conversation it can rewind, a context
window chosen to fit into memory. `examples/runtime_contract_smoke.c` binds
every symbol it calls to a typed function pointer (`make
runtime-contract-smoke`, in CI and in every release job), as the agent gate
does.

### STABLE

The backend, model and session lifecycle above, plus
`geist_backend_name`, `geist_backend_errmsg`, `geist_model_errmsg`,
`geist_session_errmsg` and, promoted for this contract in 0.20.0:

| Symbol | Why the runtime needs it |
| :-- | :-- |
| `geist_model_load_with_opts`, `geist_model_load_from_memory_with_opts` | loading with the window it chose (`max_seq_len`) |
| `geist_model_add_bos`, `geist_model_add_eos` | wrapping the rendered prompt the way the model was trained |

### Explicitly NOT in the contract: EXPERIMENTAL, but gated (#622)

Part of `runtime_contract_smoke.c`, so a change cannot go unnoticed, but not
a promise yet: they become STABLE once geist-runtime has used them through a
release.

| Symbol | Why the runtime needs it |
| :-- | :-- |
| `geist_model_metadata_str` | the chat template and other string metadata, without a second GGUF parser (and for models loaded from memory) |
| `geist_model_context_length` | the trained window, the upper bound of the one it chooses |
| `geist_session_length`, `geist_session_truncate` | rewinding a conversation without processing the kept part again |
| `geist_session_kv_bytes_per_token` | choosing the longest window that fits into memory |
| `geist_model_plan`, `geist_model_plan_from_memory` | the same numbers **before** loading (#625): the window is fixed at load, so a runtime plans, chooses, then loads once |

`geist_session_truncate` refuses (GEIST_E_UNSUPPORTED, the session unchanged)
where the state cannot return to a position: recurrent (DeltaNet) layers, the
compressed KIVI region, MTP drafting. The runtime then resets and prefills
the kept tokens again; for recurrent models, snapshots are the faster route.

## Consuming this contract

Pin a minimum version and check it at compile time:

```c
#include <geist.h>
#if (GEIST_VERSION_MAJOR * 10000 + GEIST_VERSION_MINOR * 100) < 900
#  error "geistlib >= 0.9.0 required for the complete agent-runtime contract"
#endif
```

The SDK artifact (`libgeist-<platform>.tar.gz`) ships `include/` and the static
archive; both contract headers are part of it.

## Including the headers from C++

Every header in `include/` is includable from a C++ translation unit as well
as a C one. The `extern "C"` guards are only half of that: C's
`T arr[static n]` array-parameter form is not C++ grammar, so the headers
spell that contract `T arr[GEIST_AT_LEAST(n)]`, which expands to `static n`
in C and to nothing in C++ (AGENT.md §1). The parameter type is identical
either way — this is a spelling, not an API change, and consumers need no
`#ifdef`.

`make check-headers` compiles each public header standalone as C23 and as
C++17 with `-pedantic-errors`, per PR, so this cannot quietly regress.

## Context length: model cap and session cap (experimental)

Two values bound how many tokens a session can hold:

- **The model cap** is set once, at load. `geist_model_load_with_opts` and
  `geist_model_load_from_memory_with_opts` read `opts->max_seq_len`; with
  nullptr options or 0, the cap is 4096. The model sizes the buffers it owns
  from it (RoPE tables and the default session's scratch), so a larger cap
  costs memory up front. The cap does not follow the GGUF's
  `<arch>.context_length`; `geist_model_context_length` reports that value,
  so a caller can choose a cap up to it (#622).
- **What a position costs**: `geist_session_kv_bytes_per_token` gives a
  session's KV-cache bytes per position for its resolved KV mode, so a
  caller can pick the longest cap that fits into the memory it has.
- **Before loading** (#625): `geist_model_plan` reads only the GGUF's header
  and tensor table (milliseconds, even for a 27B; no weight read, nothing
  allocated on the backend) and reports the trained window, the weight
  bytes, the KV bytes per position a session with the given options would
  get, and the model's own bytes per position of the cap (its RoPE tables).
  One session with cap W needs about weights + W × (KV + model bytes per
  position), plus buffers that do not grow with W.
- **The session cap** is `geist_session_opts.max_seq_len` at
  `geist_session_create`, at most the model cap. A session that asks for more
  is refused. 0 means the model cap. Several sessions on one model may each
  take a smaller cap; their KV caches are sized from it.

`geist_model_load` and `geist_model_load_from_memory` are the nullptr-options
forms and keep the 4096 cap. A consumer that needs a longer context (a server
taking 8k-token prompts, say) passes the same `max_seq_len` at load and at
session create.

## Optional backend resources (experimental)

`geist_backend_resources_snapshot` is an additive, observational API. Its
output is valid only on GEIST_OK. Null handles/outputs are invalid, missing
providers are unsupported, and failures clear all output fields. A destroyed
pointer must never be reused; set it to nullptr and join observers before
backend destruction.

Metal's source is the live backend's `MTLDevice.currentAllocatedSize`: a
provider allocation counter with the provider's alias/heap/driver semantics
(overlapping no-copy wrappers count again), not unique physical residency or
reclaimable system RAM. Do not sum it with RSS. No model, prompt or disk
storage is involved. The experimental backend descriptor has an optional
provider callback; third-party backends recompile with a zero/null callback
to keep explicit unsupported behavior. `test_metal_resources_unit`
(opt-in, `GEIST_TEST_METAL_RESOURCES=1`) checks the counter against direct
device queries, including under concurrent allocate/release.

`geist_backend_memory_info` follows the same rules (null handles/outputs are
invalid, a missing provider is unsupported, failure clears the output). It
answers a planning question, how much device memory is left, not an
accounting one: Vulkan reports the device-local heap and, with
`VK_EXT_memory_budget`, the driver's budget minus usage, which other processes
move between two calls (`device_wide` says which kind of number it is). CPU
backends are unsupported by design; their memory is system RAM.
