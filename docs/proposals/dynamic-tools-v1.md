# Dynamic tools v1

Status: implementation contract for Phase 3. This interface is host-neutral;
Home Assistant is one adapter, not part of the runtime contract.

## Request and execution model

A host supplies the available tools on every conversation request:

```json
{
  "input": "Turn on the kitchen light",
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
6. Cancellation stops generation and prevents further calls.
7. Low routing confidence produces a clarification response, not a best-guess
   forced call.

## Delivery slices

1. Generic fixed-memory JSON parser and schema-v1 validator.
2. Per-request dynamic toolset parser and immutable compiled representation.
3. Agent request scope: dynamic names/descriptions/schemas, strict dispatch and
   confidence-aware routing.
4. Typed constrained generation for required/optional fields, enums, arrays and
   multiple arguments.
5. Host-driven multi-step session with tool results, cancellation and retry.
6. Home Assistant adapter plus a standalone example host and separate build
   targets proving that neither depends on the other.
