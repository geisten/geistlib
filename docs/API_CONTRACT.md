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

The three 0.6.0 entries were `EXPERIMENTAL` until this contract; they are
promoted here and ship in 0.6.0. `geist_session_peek_logits` gained an explicit
ownership clause in the same change so an accelerator backend can satisfy it by
staging device memory into session storage — the promotion does not freeze a
CPU-only design.

### Explicitly NOT in the contract

`geist_session_attach_audio` / `attach_image` / `attach_video`,
`geist_session_decode_speculative`, and the `geist_session_stats` family
remain `EXPERIMENTAL`. They are useful and
supported, but an agent runtime must not build its core loop on them expecting
release-boundary stability.

## Consuming this contract

Pin a minimum version and check it at compile time:

```c
#include <geist.h>
#if (GEIST_VERSION_MAJOR * 10000 + GEIST_VERSION_MINOR * 100) < 600
#  error "geistlib >= 0.6.0 required for the agent-runtime contract"
#endif
```

The SDK artifact (`libgeist-<platform>.tar.gz`) ships `include/` and the static
archive; both contract headers are part of it.
