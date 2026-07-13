# geist-bitnet-function-tools

**Function tooling for the [geist](https://github.com/geisten/geisten) inference
engine with the BitNet b1.58 2B-4T model.** The engine is *imported* as a
prebuilt static library — `lib/geistenlib.a` — this project compiles none of
the engine sources itself. On top of it, plain **C functions are exposed as
whitelist-gated agent tools**: the model can route to and call `calculator`,
`clock_now` or `sysinfo`, but can never run anything outside the compiled-in
whitelist.

```
             ┌────────────────────────────────────────────┐
             │  geist-fn (this project)                   │
             │                                            │
 request ──▶ │  agent loop (header-only, docs/agent.md)   │ ──▶ answer
             │    route → force call → validate → invoke  │
             │                                            │
             │  function tools     imported engine        │
             │  ┌──────────────┐   ┌────────────────────┐ │
             │  │ calculator   │   │ lib/geistenlib.a   │ │
             │  │ clock_now    │   │ include/geisten/   │ │
             │  │ sysinfo      │   │ (BitNet b1.58 2B)  │ │
             │  └──────────────┘   └────────────────────┘ │
             └────────────────────────────────────────────┘
```

## Quick start

```sh
make            # imports geistenlib.a (builds geist if needed), builds ./geist-fn
make selftest   # tool registry + schema-validation roundtrip — no model needed
make model      # fetch BitNet b1.58 2B-4T (ternary i2_s GGUF, ~1.2 GB)
make run ARGS='"Add 5 and 7"'
```

Or directly:

```sh
./geist-fn -m models/bitnet-2b4t.i2_s.gguf "What time is it?"
./geist-fn -m models/bitnet-2b4t.i2_s.gguf "Multiply 12 by 34"
./geist-fn -m models/bitnet-2b4t.i2_s.gguf --free "Explain BitNet in one sentence."
```

## How the engine is imported

`make import` produces the project-local engine artifacts (both are
regenerable, therefore not committed):

| artifact | source | contents |
|---|---|---|
| `lib/geistenlib.a` | geist's `lib/<target>/<mode>/libgeist.a` | the complete inference engine (GGUF loader, ternary BitNet kernels, sessions) |
| `include/geisten/` | geist's `include/` | public ABI headers (`geist.h`, `geist_util.h`, …) |
| `include/geisten/tools/` | geist's `tools/` | the header-only agent layer (`agent.h` + dynamic-tools-v1 schema validation) |

Where the engine comes from, in order:

1. `GEISTEN_DIR=/path/to/geisten make` — explicit checkout,
2. `..` — when this project lives inside a geisten checkout as a subproject,
3. `vendor/geisten` — standalone mode: a shallow clone of
   `https://github.com/geisten/geisten.git` (`GEISTEN_REPO`/`GEISTEN_REF`
   override the source).

The build reuses geist's own `mk/target-*.mk` fragments, so `geist-fn` links
with exactly the same compiler flags as the library on every supported target
(`TARGET=mac | mac-omp | pi5 | linux`, auto-detected).

## The function tools

Each tool wraps a native C function (`src/function_tools.h`) into a
`struct geist_tool` with a [dynamic-tools-v1](docs/agent.md) JSON Schema.
Arguments are validated **before** the function runs; an off-whitelist name or
a schema-invalid argument object never reaches host code.

| tool | arguments | does |
|---|---|---|
| `calculator` | `op` ∈ add/subtract/multiply/divide, `a`, `b` | arithmetic on two numbers |
| `clock_now` | `zone` ∈ utc/local (optional) | current date + time |
| `sysinfo` | — | CPU cores, memory, load average |

Forced calls are **on by default** (`--free` disables them): BitNet 2B-4T is
not tool-trained, so PMI-calibrated routing plus grammar-forced, typed calls
are what make tooling reliable on it — the mechanism is documented in
[docs/agent.md](docs/agent.md) (copied from geist, "Tool selection & forced
calls").

### Adding your own function

Append an entry to `FUNCTION_TOOLS[]` in `src/function_tools.h`:

```c
static enum geist_status ft_my_fn(void *ctx, size_t args_len,
                                  const char args[static args_len],
                                  size_t out_cap, char out[static out_cap],
                                  size_t *out_len) {
    double x = 0.0;
    if (!ft_json_num(args, "x", &x))
        return agent_obs(out_cap, out, out_len, "error: my_fn needs \"x\"");
    return agent_obs(out_cap, out, out_len, "%.10g", x * x);
}
```

…and give it a `parameters_schema` so the agent validates the call before your
function runs. `make selftest` exercises every registered tool through the same
validation boundary the agent uses (`agent_tool_invoke_checked`), with no model
file — CI-friendly.

## CLI

```
geist-fn --selftest
geist-fn -m <model.gguf> [--steps N] [--free] "<request>"
```

- `--steps N` — max tool calls per request (default 8, runaway guard)
- `--free`    — disable forced calls (free-form agent loop)
- progress events (`routing → calling → running → observed → answering`)
  stream to stderr; the answer goes to stdout.

## Layout

```
Makefile               imports geistenlib.a + builds ./geist-fn
src/main.c             CLI host: model load → agent loop → answer
src/function_tools.h   the function-tool registry (extend here)
docs/agent.md          agent/tooling documentation, copied from geist
lib/, include/         populated by `make import` (gitignored)
```

## License

Apache-2.0, same as geist ([LICENSE](LICENSE)).
