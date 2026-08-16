# geistlib examples

Four small C programs against what geistlib actually ships: `include/geist.h`
and `include/geist_util.h`. One is meant to be read and copied; two are release
gates that run in CI and are the only check of their kind — don't delete them
for looking trivial.

The tool-use programs that used to live here moved out of tree with the agent
layer.

| File | Role | Proves | Built by |
| :-- | :-- | :-- | :-- |
| `simple_generate.c` | example | the STABLE core generates text | `make -C examples` |
| `push_to_talk.c` | example | a real-time voice loop: mic → VAD → audio attach → spoken-to answer | `make -C examples` |
| `embed_smoke.c` | release gate | the packaged `libgeist.a` links and runs with no model | `release.yml` |
| `agent_contract_smoke.c` | release gate | the symbols the out-of-tree agent runtime links still exist, with the same signatures | `release.yml` + `make agent-contract-smoke` |

The `_smoke` suffix is the marker: every `*_smoke.c` here is compiled by
`release.yml` against the *packaged* SDK, with `-I <package>/include` and the
repo tree off the include path. They live beside the example because they share
the one property that defines this directory — everything here is built from
**outside** the library, against the artifact geistlib publishes. `tests/` is
the opposite: `mk/common.mk` globs `tests/test_*.c` and builds them against the
repo tree, which is precisely the check a packaging gate must not do.

`agent_contract_smoke.c` is the exception that proves it: it also builds from
the repo tree on every PR (`make agent-contract-smoke`, wired into `ci.yml`).
That is deliberate — a broken contract should fail on the PR that breaks it, not
weeks later at release. The release run is the one that matters, because only it
proves the *shipped* headers still hold.

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

## `push_to_talk`

The complete real-time voice pattern in ~200 lines of public API: raw
16 kHz mono s16le PCM on stdin (exactly what `arecord` emits), a simple
energy VAD to segment utterances, one Gemma 4 audio turn per utterance.

```sh
make -C examples
arecord -f S16_LE -r 16000 -c 1 -t raw | \
    examples/push_to_talk gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

No microphone? Pipe any 16 kHz WAV plus a second of silence:

```sh
(tail -c +45 clip.wav; dd if=/dev/zero bs=32000 count=1) | \
    examples/push_to_talk model.gguf
```

On macOS there is no `arecord`; ffmpeg's avfoundation input is the
equivalent (verified live — speaker-to-mic loop on a MacBook):

```sh
ffmpeg -f avfoundation -list_devices true -i ""   # find your mic index
ffmpeg -hide_banner -loglevel error -f avfoundation -i ":1" \
       -ar 16000 -ac 1 -f s16le - | \
    examples/push_to_talk model.gguf 1200
```

(`:1` = the built-in mic on a typical MacBook; grant the terminal
microphone permission on first use. `brew install sox` and
`rec -q -t raw -r 16000 -e signed -b 16 -c 1 -` works too.) Calibrate
the threshold against your room: ambient frame RMS on a MacBook mic is
~800, so the default 300 would trigger constantly — measure a few
seconds of silence and set the knob above its peak.

`GEIST_PTT_PROMPT` sets the per-utterance instruction; the second CLI
argument tunes the VAD threshold (default 300 RMS — every room, mic and
gain combination needs the knob). Run from the repo root (or set
`GEIST_AUDIO_MODEL_PATH`) so the audio tower is found; the program uses
`geist_model_modalities()` to fail fast when the model cannot hear.

## The two release gates

Both run in all three release jobs — `linux-arm64`, `linux-x86_64`,
`macos-arm64` — against that platform's packaged SDK. Neither is called for its
output; each is a compile-and-link assertion, which is why 136 lines are worth
keeping:

- **`embed_smoke.c`** (27 lines) calls only model-free STABLE entry points
  (version, status), so it needs no backend and no GGUF. If it fails, the
  shipped `libgeist.a` is unusable for every embedder — the broadest possible
  failure, caught by the smallest possible program.

- **`agent_contract_smoke.c`** (109 lines) binds each symbol of
  [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md) to an explicitly typed
  function pointer. A **changed signature fails to compile**, a **removed symbol
  fails to link**. Nothing is invoked, so no model is needed; taking a
  function's address is enough to force the linker to resolve it. The
  out-of-tree agent runtime links these across a release boundary, so a break
  must surface here rather than in someone else's build.

To reproduce either locally, point `-I` at an unpacked tarball instead of
`include/`. For the contract one, `make agent-contract-smoke` is the quicker
check — it compiles the same assertions against the working tree.
