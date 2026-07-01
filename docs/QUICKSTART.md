# Quickstart

Get geist running as a **command-line tool** in two minutes, then embed it as a
**library** in your own C program. Everything here uses the stable, public API
(`include/geist.h` + `include/geist_util.h`) and the real Makefile targets.

- New to the internals? See [ARCHITECTURE.md](ARCHITECTURE.md).
- Want the numbers? See [../benchmark/BENCHMARK_PI5.md](../benchmark/BENCHMARK_PI5.md).

---

## 1. Run the application (CLI)

### Build

```bash
make                       # target auto-detected: mac-omp / mac / pi5 / linux
                           # → drops a ./geist symlink in the repo root
```

Requirements: a C23 compiler (gcc ≥ 14, or Apple-clang ≥ 16) and `make`. On a
Mac, `brew install libomp` enables multi-threading (the `mac-omp` target).

### Get a model

geist targets **Gemma 4 E2B-it, Q4_K_M**. Fetch the reference GGUF (~3.1 GB):

```bash
make fetch-model           # → gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

Any GGUF that carries its own tokenizer works; `fetch-model` is just a helper.

### Generate

```bash
# The ./geist symlink saves you the bin/<target>/<mode>/ path.
OMP_WAIT_POLICY=active ./geist gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \
    "The capital of France is"
# -> The capital of France is Paris.

# make run sets OMP_WAIT_POLICY for you:
make run ARGS='gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "Write a haiku" -n 40'
```

CLI usage: `geist <model.gguf> [prompt] [-n N]` — `-n` caps new tokens
(default 64). `OMP_WAIT_POLICY=active` keeps the OpenMP threads spinning between
tokens and noticeably improves multi-thread throughput; always set it.

The same binary carries two more subcommands — `geist agent <model> "<request>"`
(one-shot whitelist-gated tool use) and `geist chat <model>` (multi-turn chat +
file-based memory palace). See [agent.md](agent.md).

For an interactive prompt loop, use the evaluation REPL (no symlink — full path):

```bash
OMP_WAIT_POLICY=active bin/`mk/detect-target.sh`/release/tools/eval_geist \
    gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

---

## 2. Use the library (C API)

The whole stable surface to run text generation is six calls:

```
geist_backend_create → geist_model_load → geist_session_create
  → geist_session_set_prompt → geist_session_decode_step → geist_session_token_to_str
```

### Minimal program

```c
#include <stdio.h>
#include <geist.h>        /* core: backend, model, session */
#include <geist_util.h>   /* eos token id, for a clean stop condition */

int main(int argc, char **argv) {
    const char *model_path = argv[1];
    const char *prompt     = argc > 2 ? argv[2] : "The capital of France is";

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend: %s\n", geist_last_create_error());
        return 1;
    }

    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        fprintf(stderr, "load: %s\n", geist_last_create_error());
        return 1;
    }

    struct geist_session_opts opts = {0};   /* all-zero = greedy decoding */
    struct geist_session *sess = nullptr;
    geist_session_create(model, be, &opts, &sess);
    geist_session_set_prompt(sess, prompt);

    /* Stop cleanly on the model's end-of-sequence id (from geist_util.h),
     * instead of string-matching the decoded text. */
    const geist_token_t eos = geist_model_eos_token(model);

    fputs(prompt, stdout);
    for (int i = 0; i < 256; i++) {
        geist_token_t tok;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK) break;
        if (tok == eos) break;
        const char *piece = geist_session_token_to_str(sess, tok);
        if (!piece) break;
        fputs(piece, stdout);
    }
    putchar('\n');

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
```

> `geist_model_eos_token` lives in `geist_util.h` and returns `GEIST_TOKEN_NONE`
> if the model has no tokenizer EOS — so a chat app stops on `tok == eos` rather
> than scanning output for `<end_of_turn>`. The Gemma 4 GGUF reports eos = 106
> (`<turn|>`), which doubles as the end-of-turn marker.

### Build it against `libgeist.a`

The simplest path reuses the repo's own per-target compiler/flags/libs, so your
program links exactly like the library does — no per-platform knowledge to copy:

```bash
make lib                                   # builds lib/<target>/release/libgeist.a
make -C examples                           # builds examples/simple_generate the same way
```

To compile a standalone file by hand, point the compiler at `include/` and the
static archive, then add the platform link libs (Accelerate on macOS, OpenBLAS +
OpenMP on Linux/Pi). The canonical recipe is in
[`examples/Makefile`](../examples/Makefile) — it `include`s
`mk/target-$(TARGET).mk` and uses `$(LDFLAGS_TARGET) $(LDLIBS_TARGET)`, e.g.:

```bash
cc -std=c23 -O2 -Iinclude -o mygen mygen.c \
   lib/$(mk/detect-target.sh)/release/libgeist.a \
   $(LDFLAGS_TARGET) $(LDLIBS_TARGET)   # values from mk/target-<target>.mk
```

See [`examples/simple_generate.c`](../examples/simple_generate.c) for a complete,
buildable version.

### Public headers

| Header | Include when you… | Holds |
| :-- | :-- | :-- |
| **`geist.h`** | run a model | backend → model → session → `set_prompt` → `decode_step` → `token_to_str` |
| **`geist_util.h`** | need EOS/special tokens, multimodal, speculative, telemetry | `geist_model_eos_token` / `_bos_token` / `_token_by_text`, `tokenize`, `attach_audio/image/video`, `decode_speculative`, stats |
| `geist_types.h` | author a backend | low-level tensor / op / dtype types |
| `geist_backend.h` | author a backend | the backend vtable + descriptor |

Pure text generation needs only `geist.h`; add `geist_util.h` for a clean EOS
stop or anything multimodal/advanced.

---

## 3. Ship one file: model embedded in the binary

geist already builds as a single dependency-free executable. You can fold the
**model** in too, so a deployment is *literally one file* — no GGUF alongside:

```bash
make clean                                 # EMBED rebuilds the CLI; start clean
make EMBED_MODEL=path/to/model.gguf        # bakes the GGUF into ./geist
./geist "The capital of France is"         # the CLI now takes only a prompt
```

Weights are aliased **zero-copy** from the binary's read-only data, so this suits
**small** models (the binary grows by the model size). Full single-file &
deployment guide: [DEPLOY.md](DEPLOY.md).

Your own app gets the same capability via the public API — load a GGUF that is
already in memory (e.g. an `.incbin`-embedded blob, or one you `mmap`ed yourself):

```c
extern const unsigned char model_start[], model_end[];   /* your embedded blob */
geist_model_load_from_memory(model_start, model_end - model_start, be, &model);
```

The caller keeps the buffer alive for the model's lifetime (weights are aliased,
not copied).

---

## 4. Performance knobs

| Setting | Effect |
| :-- | :-- |
| `OMP_WAIT_POLICY=active` | keep OMP threads hot between tokens — set it for every run |
| `GEIST_WEIGHT_MMAP=1` | mmap-alias weights instead of copying resident (the default; important on low-RAM boards) |
| `GEIST_PREFILL_THREADS` / `GEIST_DECODE_THREADS` | override the auto thread split (prefill scales with all cores; decode is bandwidth-bound and often fastest one core below) |
| `GEIST_MMAP_PREFETCH=1` | `MADV_WILLNEED` prefault of the weight map — steadier first-token latency on 4 KB-page Linux (no-op on the Pi 5's 16 KB pages) |

Tuning rationale and measured numbers are in
[../benchmark/BENCHMARK_PI5.md](../benchmark/BENCHMARK_PI5.md).
