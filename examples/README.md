# geistlib examples

Four small C programs against what geistlib actually ships: `include/geist.h`
plus the header-only interfaces in `tools/` (`agent.h`, `dynamic_*_v1.h`,
`json_schema_v1.h`). Two are meant to be read and copied. Two are release gates
that run in CI and are the only check of their kind — don't delete them for
looking trivial.

| File | Role | Proves | Built by |
| :-- | :-- | :-- | :-- |
| `simple_generate.c` | example | the STABLE core generates text | `make -C examples` |
| `dynamic_tools_host.c` | example | `dynamic-tools-v1` works with no model and no Home Assistant | `make dynamic-example-host` |
| `embed_smoke.c` | release gate | the packaged `libgeist.a` links against nothing but `-lm` | `.github/workflows/release.yml` |
| `embed_agent_smoke.c` | release gate | the packaged `include/` set is self-contained | `.github/workflows/release.yml` |

The `_smoke` suffix is the marker: a `*_smoke.c` here is compiled against a
*packaged* SDK by the release workflow, never by `make`. They live beside the
examples because they share the one property that defines this directory —
everything here is built from **outside** the library, against the artifact
geistlib publishes. `tests/` is the opposite: `mk/common.mk` globs
`tests/test_*.c` and builds them against the repo tree, which is precisely the
check a packaging gate must not do.

## `simple_generate`

Loads a GGUF, prefills a prompt, greedy-decodes a continuation.

```sh
make                 # libgeist.a for the detected target
make -C examples     # the example against it

OMP_WAIT_POLICY=active examples/simple_generate \
    gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"
# -> The capital of France is Paris.
```

Arguments: `simple_generate <model.gguf> [prompt] [max_new_tokens]`.

It uses only `geist_backend_create` → `geist_model_load` →
`geist_session_create` → `geist_session_set_prompt` → `geist_session_decode_step`
→ `geist_session_token_to_str`. That is the whole stable surface needed to run
text generation; multimodal (`attach_audio` / `attach_image` / `attach_video`)
and the speculative / KV-mode knobs are `EXPERIMENTAL` extensions on top.

`make -C examples` includes `mk/target-$(TARGET).mk`, so the example links with
exactly the compiler, flags and libraries the library itself uses — pass
`TARGET=pi5` / `MODE=debug` to override.

## `dynamic_tools_host`

The host side of the protocol, not a generation sample. It offers calculator and
profile tools to a running Unix-socket agent, validates every returned name and
argument object, executes locally, returns correlated results, and prints the
final conversation result.

It includes `tools/dynamic_tools_v1.h` and **nothing else** — no model, no
runtime library, no Home Assistant. That is the point: it is the executable
proof that the contract holds for any host. Start here for a database, build
system, device or application host.

```sh
make dynamic-example-host
./geist -m model.gguf --serve /tmp/geist.sock
bin/$(mk/detect-target.sh)/release/examples/dynamic_tools_host \
  /tmp/geist.sock "Add 5 and 7"
```

Wire format and schema: [`docs/dynamic-tools-v1.md`](../docs/dynamic-tools-v1.md).

## The two release gates

Both are compiled by `release.yml` against the *packaged* SDK with
`-I <package>/include` — the repo tree is not on the include path. That is what
makes them worth their 74 lines:

- **`embed_smoke.c`** calls only model-free STABLE entry points (version,
  status), so it needs no backend, no GGUF, no OpenMP and no BLAS. If it fails
  to link, the shipped `libgeist.a` is broken for every embedder.
- **`embed_agent_smoke.c`** includes `agent.h` and drives a trivial
  `geist_tool`. `agent.h` is 2777 lines of header-only code that is *not* in
  `libgeist.a`; release.yml copies it and six sibling headers into the package
  by hand. Add an `#include` to one of them without extending that copy list and
  this smoke is the only thing that notices — on all three platforms, before the
  tarball ships.

They run automatically on release. To reproduce one locally, point `-I` at an
unpacked tarball rather than at `include/`.
