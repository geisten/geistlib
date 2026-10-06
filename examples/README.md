# geistlib examples

Small C programs built from **outside** the library, against what geistlib
ships: `include/*.h` and `libgeist.a`. `tests/` is the opposite —
`mk/common.mk` builds `tests/test_*.c` against the repository tree.

| File | Role | Shows / proves | Built by |
| :-- | :-- | :-- | :-- |
| `simple_generate.c` | example | the STABLE core generates text | `make -C examples` |
| `push_to_talk.c` | example | real-time voice loop: mic → VAD → audio attach → answer | `make -C examples` |
| `dictate.c` | example | dictation: one transcript line per utterance on stdout | `make -C examples` |
| `geist_calibrate.c` | example | the caller side of the calibration API: when to measure, where to store, reacting to staleness | `make -C examples` |
| `embed_smoke.c` | release gate | the packaged `libgeist.a` links and runs with no model | `release.yml` |
| `agent_contract_smoke.c` | release gate | the symbols an out-of-tree agent runtime links keep their signatures | `release.yml` + `make agent-contract-smoke` |
| `runtime_contract_smoke.c` | release gate | the same for [geist-runtime](https://github.com/geisten/geist-runtime) | `release.yml` + `make runtime-contract-smoke` |
| [`ffi/`](ffi/README.md) | example | the same integration in Python, Rust, Go and JavaScript | by hand |

`make -C examples` includes `mk/target-$(TARGET).mk`, so the examples link
with exactly the compiler, flags and libraries of the library (override with
`TARGET=pi5`, `MODE=debug`, …).

## `simple_generate`

Loads a GGUF, prefills a prompt, decodes a continuation.

```sh
make lib && make -C examples
OMP_WAIT_POLICY=active examples/simple_generate \
    gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"
# -> The capital of France is Paris.
```

Usage: `simple_generate <model.gguf> [prompt] [max_new_tokens] [-t|--temperature <float>]`.
Temperature 0 (default) is greedy and deterministic.

It uses only `geist_backend_create` → `geist_model_load` →
`geist_session_create` → `geist_session_set_prompt` → `geist_session_decode_step`
→ `geist_session_token_to_str`. Multimodal attach, speculative decode and the
KV modes are `EXPERIMENTAL` extensions on top. The release CLIs
`geist-linux-*` and `geist-bitnet-linux-*` are built from this file.

## `push_to_talk` and `dictate`

Raw 16 kHz mono s16le PCM on stdin (what `arecord` emits), an energy VAD to
segment utterances, one Gemma 4 audio turn per utterance. `push_to_talk`
answers; `dictate` prints a transcript line per utterance, for piping into a
typing tool.

```sh
arecord -f S16_LE -r 16000 -c 1 -t raw | \
    examples/push_to_talk gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

The second argument is the VAD threshold (default 300 RMS);
`GEIST_PTT_PROMPT` / `GEIST_DICTATE_PROMPT` set the instruction. Run from the
repo root (or set `GEIST_AUDIO_MODEL_PATH`) so the audio tower is found. macOS
capture, calibration, prompts and troubleshooting:
[docs/VOICE.md](../docs/VOICE.md).

## `geist_calibrate`

```sh
examples/geist_calibrate <dir> [--force]   # measure, write <dir>/<key> (--force skips the load gate)
examples/geist_calibrate --apply <file>    # validate a stored blob against this machine
```

The library only measures, serializes and validates; this program shows the
policy a consumer owns. Uses the `cpu_neon` backend.

## Release gates

The `*_smoke.c` files are compile-and-link assertions, compiled by
`release.yml` in all three release jobs against the *packaged* SDK, with
`-I <package>/include` and the repo tree off the include path.

- **`embed_smoke.c`** calls only model-free STABLE entry points (version,
  status). If it fails, the shipped `libgeist.a` is unusable for every
  consumer.
- **`agent_contract_smoke.c`** and **`runtime_contract_smoke.c`** bind each
  contracted symbol of [docs/API_CONTRACT.md](../docs/API_CONTRACT.md) to an
  explicitly typed function pointer: a changed signature fails to compile, a
  removed symbol fails to link. Nothing is called, so no model is needed. Both
  also build from the repo tree on every PR (`make agent-contract-smoke`,
  `make runtime-contract-smoke`), so a break fails on the PR that causes it.

To reproduce a gate against a release, point `-I` at an unpacked tarball
instead of `include/`.
