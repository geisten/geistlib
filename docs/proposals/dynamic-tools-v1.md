# Dynamic tools v1

Status: implemented Phase-3 contract. This interface is host-neutral;
Home Assistant is one adapter, not part of the runtime contract.

## Request and execution model

Before model work, a host may send the exact control frame
`{"type":"health"}` followed by newline. The server answers
`{"type":"health.result","protocol":"dynamic-tools-v1","status":"ready","features":["streaming"]}`.
This exchange performs no generation and is used by host configuration flows.
Other control objects are invalid requests.

`features` is the generic capability mechanism (additive):

1. It is an array of capability strings; the first defined entry is
   `"streaming"` (see [Streaming](#streaming-additive)).
2. A client MUST ignore unknown entries.
3. An absent `features` field means no capabilities (pre-v0.5 engines). The
   health **request** `{"type":"health"}` never changes.
4. A capability never changes engine behavior by itself — activation happens
   exclusively through the matching request field (e.g. `stream:true`). A
   client that has not seen a capability does not send its request field.

A host supplies the available tools on every conversation request:

```json
{
  "input": "Turn on the kitchen light",
  "language": "en-GB",
  "context": "[{\"role\":\"user\",\"content\":\"Earlier question\"}]",
  "max_tool_steps": 4,
  "tools": [
    {
      "name": "HassTurnOn",
      "description": "Turn on an exposed Home Assistant entity",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {"type": "string"}
        },
        "required": ["name"],
        "additionalProperties": false
      }
    }
  ]
}
```

`language` and `context` are optional request metadata. Language is a bounded
BCP-47-like code. Context is a bounded string interpreted as previous
conversation data in the request-local system context; it is never used for
current tool-argument extraction. Hosts own history isolation, retention and
privacy. The HA adapter keeps at most four turns and 2048 context bytes for at
most 32 conversation ids, in memory only.

The offered array is the complete capability set for that request. Geist may
return only a name from that set. Arguments are validated before they cross the
host boundary. Geist never invokes a dynamic tool itself: the host executes it,
returns a `tool.result`, and Geist continues with that observation. A request
has at most 16 tools and 1–16 tool steps; a repeated identical call requires a
new host result and remains subject to the step budget.

Home Assistant derives this array from the current user, Assist pipeline and
exposed entities. Another application can supply database, build-system or
device tools and compile without any Home Assistant source or dependency.

The Unix-socket server uses newline-delimited JSON for the host round trip:

```json
{"type":"tool.call","call_id":"1","name":"HassTurnOn","arguments":{"name":"light.kitchen"}}
{"type":"tool.result","call_id":"1","status":"ok","result":{"state":"on"}}
{"type":"conversation.result","text":"The kitchen light is on."}
```

`call_id` must match exactly. A host may return `status:"retryable"`; Geist
re-emits the same validated call once with a new id. Further retryable results
are delivered to the model as data. Retries consume the request's global call
budget. A correlated `{"type":"cancel",...}` while waiting for a tool result
cancels the session and prevents later calls.

`make dynamic-example-host` builds `examples/dynamic_tools_host.c`, an
independent calculator/profile host. It has no Home Assistant or model/runtime
dependency and revalidates every received name and argument object before its
local dispatch.

## JSON Schema subset

Version 1 deliberately supports a bounded subset rather than claiming general
JSON Schema compatibility:

- the root schema is an object;
- `type`: `object`, `string`, `number`, `integer`, `boolean`, or `array`;
- object keywords: `properties`, `required`, `additionalProperties`;
- `additionalProperties` defaults to `false` for tool safety;
- scalar keyword: `enum` (homogeneous scalar values);
- numeric keywords: `minimum`, `maximum`;
- array keywords: `items`, `minItems`, `maxItems`;
- nested objects and arrays are supported to a maximum schema/value depth of 8;
- at most 64 properties, 32 enum alternatives and 32 array elements;
- duplicate object keys are invalid;
- unsupported schema keywords fail schema compilation rather than being
  silently ignored.

Not in v1: references, unions (`oneOf`/`anyOf`/`allOf`), `null`, tuple arrays,
regular expressions, conditional schemas, dependent fields, default insertion,
or coercion. Hosts must send explicit values of the declared type.

## Safety invariants

1. The per-request offered tool set is immutable for the lifetime of that
   request.
2. Off-list or duplicate tool names fail closed and are never remapped to a
   different action.
3. Arguments must validate against the selected tool's compiled schema before
   a `tool.call` is emitted.
4. The host revalidates the name and arguments before execution; HA additionally
   rechecks user context, exposure and registry version at the action boundary.
5. Tool results are data, not new instructions, and remain bounded in size.
6. Disconnecting stops the request; a correlated cancel while a result is
   pending prevents retries and further calls.
7. Low routing confidence produces a clarification response, not a best-guess
   forced call.

## Compatibility contract

This is what a consumer — Home Assistant, a future gateway, any integration —
pins and how it updates. The runtime is the primary, language-agnostic seam;
the contract must be explicit so consumers build reliably, versioned and
update-capable.

A consumer pins a runtime by **two** things and verifies at startup:

1. **Protocol version** — the literal `dynamic-tools-v1`. This is *the* contract:
   the wire framing, message types, schema subset and safety invariants above are
   fixed for this identifier.
2. **Artifact checksum** — the SHA-256 of a specific release binary (published as
   `SHA256SUMS`). A consumer pins a digest, not a version range.

At startup, before any generation, the consumer sends the health frame and
requires the documented reply:

```
-> {"type":"health"}
<- {"type":"health.result","protocol":"dynamic-tools-v1","status":"ready","features":["streaming"]}
```

A reply whose `protocol` differs from the pinned identifier is an
incompatibility and must fail closed — the consumer does not proceed to
conversation requests.

- **Binary version** (`geist --version`, semver) is **informative**, not the
  contract: patch/minor releases fix bugs or improve performance behind the
  *same* protocol. A consumer never gates on the semver.
- **Update rule:** swap in a newer binary that answers the same protocol id,
  record its new SHA-256, and re-run the startup health handshake. There is no
  in-place self-updater; the consumer owns *when* it swaps.
- **Breaking changes** go **only** through a new protocol identifier
  (`dynamic-tools-v2`) *or* a strictly **additive** field on `dynamic-tools-v1`
  (a new optional request field, or a new optional frame type that pre-existing
  consumers never request and never receive). Any change to an existing frame's
  shape, the schema subset, or a safety invariant requires `-v2`. A protocol
  change updates the golden fixtures in both the runtime and the consumer
  repositories and passes both contract suites before either releases.
- **Integrity** is SHA-256 only; signed provenance / SBOM is deliberately out of
  scope for this contract (a consumer needing supply-chain attestation layers it
  on top).

**In one line:** pin `dynamic-tools-v1` + a binary SHA-256; verify with the
startup health handshake; update by swapping the binary (same protocol) and
re-verifying.

Interop surfaces themselves — an OpenAI-compatible HTTP endpoint, MCP — are
**separate consumer projects** over this protocol, not part of the runtime core,
which stays socket-only and dependency-free.

## Streaming (additive)

The default remains **non-streaming**: one `input` produces a bounded
`tool.call`/`tool.result` exchange and exactly one final `conversation.result`
carrying the complete `text`. A client that does not opt in gets byte-identical
behavior to pre-streaming engines.

Opt-in (strictly additive, no `-v2`):

- A client that has seen `"streaming"` in `health.result.features` MAY set
  `"stream": true` on the conversation request. If present, the field MUST be a
  JSON boolean; anything else is an invalid request.
- With `stream:true` the engine emits **zero or more**
  `{"type":"conversation.delta","text":"…"}` frames, then the unchanged final
  `conversation.result`. Tool-call frames and their ordering are untouched;
  deltas appear only after the last `tool.result`.

Delta semantics:

1. **UTF-8 guarantee.** Every `conversation.delta.text` is valid UTF-8. Token
   pieces that split a multibyte sequence are carried by the engine until the
   code point completes; incomplete bytes never leave a frame. A trailing
   incomplete carry at end of turn is dropped, and a malformed byte (a lone
   continuation or a truncated/invalid lead — possible from a byte-fallback
   tokenizer on degenerate output) is dropped rather than emitted raw.
2. **No prefix guarantee, but no contradiction of a shown result.** The
   concatenation of deltas MAY differ from the final `conversation.result.text`
   (trailing cleanup happens after decode), so the final text alone is
   normative — a client MUST replace its displayed text with the final text
   when the result arrives. As a floor, the engine withholds deltas until the
   answer has produced at least two alphanumeric characters; an answer it then
   discards as degenerate (returning the tool observation instead) therefore
   streams nothing, so a client never renders text the result overrides.
   Deltas are display-only (time-to-first-token).
3. **Scope.** Deltas come only from decoding the user-visible answer. Internal
   decodes (routing, constrained call generation, tool turns) never emit
   deltas. Answers that involve no decode (e.g. a forced call whose
   observation is returned directly) emit zero deltas.

Error behavior:

- **Client gone.** If writing a delta frame fails, the engine aborts the
  in-flight generation, closes the connection and returns to accepting — no
  `conversation.result` follows. The daemon itself is unaffected.
- **Cancellation.** A cancel is handled exactly as without streaming; after it
  is honored, no further `conversation.delta` frames are sent.

## Implemented slices

1. ✅ Generic fixed-memory JSON parser and schema-v1 validator.
2. ✅ Per-request dynamic toolset parser and immutable compiled representation.
3. ✅ Agent request scope: dynamic names/descriptions/schemas, strict dispatch and
   confidence-aware routing.
4. ✅ Typed constrained generation for required/optional fields, enums, arrays and
   multiple arguments.
5. ✅ Host-driven multi-step session with tool results, cancellation and retry.
6. ✅ Home Assistant adapter plus a standalone example host and separate build
   targets proving that neither depends on the other.
7. ✅ Additive streaming: `features` capability discovery, strict `stream`
   opt-in, UTF-8-safe `conversation.delta` frames, abort-on-disconnect.

## Verification map

| Requirement | Primary executable evidence |
|---|---|
| Request tools, dynamic names, immutability and limits | `test_dynamic_request_v1_unit`, `test_dynamic_tools_v1_unit` |
| Schema subset, multi-args, optional fields, enums and arrays | `test_json_schema_v1_unit`, `test_dynamic_arguments_v1_unit` |
| Off-list and invalid arguments fail before execution | `test_agent_unit`, `test_dynamic_tools_v1_unit`, `test_ha_dynamic_tools_v1.py` |
| Multiple calls/results and final conversation result | `test_dynamic_host_v1_unit`, `test_ha_dynamic_tools_v1.py` |
| Strict `stream` field, UTF-8/escape-safe deltas, write-failure latch | `test_dynamic_stream_unit` |
| Global step budget, one retry and correlated cancellation | `test_dynamic_host_v1_unit`, `test_ha_dynamic_tools_v1.py` |
| Low-confidence clarification | routing-margin cases in `test_agent_unit` |
| Adapter-owned exposure/policy/execution | adapter repository contract suite |
| Host-independent integration and build | `make dynamic-example-host` |

The model-dependent quality of selecting and wording calls is evaluated
separately from these deterministic safety contracts; no model output can bypass
the name lookup, schema validator, host policy or global call budget.
