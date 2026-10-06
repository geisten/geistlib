# Quickstart

Generate text from a source build, then embed geistlib as a library in your
own C program. Everything here uses the public API (`include/geist.h` +
`include/geist_util.h`). The library has no chat templating or tool use; those
belong to the program that links it.

## 1. Generate text

```bash
make lib                   # target auto-detected: mac-omp / mac / pi5 / linux
make fetch-bench-model     # BitNet b1.58 2B-4T, ternary, ~1.2 GB
make fetch-model           # or: Gemma 4 E2B-it Q4_K_M, ~3.1 GB (text weights)
make run ARGS='gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"'
# -> The capital of France is Paris.
```

Requirements: gcc ≥ 14 or Apple clang ≥ 16, and `make`. On macOS,
`brew install libomp` enables multi-threading (the `mac-omp` target). Other
models: [MODELS.md](MODELS.md).

`make run` builds [`examples/simple_generate.c`](../examples/simple_generate.c)
and runs it. For an interactive prompt loop, use the evaluation REPL:

```bash
bin/`mk/detect-target.sh`/release/tools/eval_geist \
    gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

## 2. Use the library

### Get the SDK

- **Prebuilt:** `libgeist-<platform>.tar.gz` from the
  [latest release](https://github.com/geisten/geistlib/releases/latest)
  (`macos-arm64`, `linux-arm64`, `linux-x86_64`) holds `libgeist.a`,
  `include/*.h` and `LICENSE`. Verify it against `SHA256SUMS`.
- **From source:** `make lib` builds `lib/<target>/<mode>/libgeist.a`.

The archive is an OpenMP build, so link `-fopenmp` (Linux) or
`-framework Accelerate <libomp>/lib/libomp.a` (macOS). Details:
[DEPLOY.md](DEPLOY.md).

### Minimal program

Six calls run text generation:
`geist_backend_create` → `geist_model_load` → `geist_session_create` →
`geist_session_set_prompt` → `geist_session_decode_step` →
`geist_session_token_to_str`.

```c
#include <stdio.h>
#include <geist.h>        /* core: backend, model, session */
#include <geist_util.h>   /* eos token id, for a clean stop condition */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [prompt]\n", argv[0]);
        return 2;
    }
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
        geist_backend_destroy(be);
        return 1;
    }

    struct geist_session_opts opts = {0};   /* all-zero = greedy decoding */
    struct geist_session *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK ||
        geist_session_set_prompt(sess, prompt) != GEIST_OK) {
        fprintf(stderr, "session: %s\n",
                sess != nullptr ? geist_session_errmsg(sess) : geist_last_create_error());
        if (sess != nullptr) geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    /* Stop on the model's end-of-sequence id instead of matching text. */
    const geist_token_t eos = geist_model_eos_token(model);

    fputs(prompt, stdout);
    for (int i = 0; i < 256; i++) {
        geist_token_t tok;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK) break;
        if (tok == eos) break;
        const char *piece = geist_session_token_to_str(sess, tok);
        if (piece == nullptr) break;
        fputs(piece, stdout);
    }
    putchar('\n');

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
```

`geist_model_eos_token` returns `GEIST_TOKEN_NONE` if the model has no
tokenizer EOS. Gemma 4 reports eos = 106 (`<turn|>`), which doubles as the
end-of-turn marker.

### Build it

The simplest path reuses the repository's per-target compiler, flags and
libraries:

```bash
make lib                     # lib/<target>/release/libgeist.a
make -C examples             # builds the examples the same way
make -n -C examples simple_generate   # print the expanded command for your target
```

[`examples/Makefile`](../examples/Makefile) is the canonical recipe. Copy the
flags printed by `make -n` rather than pasting Make variables such as
`$(LDFLAGS_TARGET)` into a shell, or add your program to that Makefile.

### Public headers

| Header | Include when you… | Holds |
| :-- | :-- | :-- |
| **`geist.h`** | run a model | backend → model → session → `set_prompt` → `decode_step` → `token_to_str` |
| **`geist_util.h`** | need EOS/special tokens, multimodal, speculative, telemetry | `geist_model_eos_token` / `_bos_token` / `_token_by_text`, `tokenize`, `attach_audio/image/video`, `decode_speculative`, stats |
| `geist_types.h` | author a backend | low-level tensor / op / dtype types |
| `geist_backend.h` | author a backend | the backend vtable + descriptor |

Declarations tagged `@stability STABLE` do not break within a major version;
see [API_CONTRACT.md](API_CONTRACT.md).

## 3. Ship one file

`geist_model_load_from_memory` loads a GGUF that is already in memory, such as
a blob embedded in your executable, so a deployment can be a single file:

```c
extern const unsigned char model_start[], model_end[];   /* your embedded blob */
geist_model_load_from_memory(model_start, model_end - model_start, be, &model);
```

Weights are aliased, not copied: keep the buffer alive for the model's
lifetime. The binary grows by the model size, so this suits small models.
Recipe: [DEPLOY.md](DEPLOY.md).

## 4. Performance knobs

| Setting | Effect |
| :-- | :-- |
| `OMP_WAIT_POLICY=active` | keep OpenMP threads spinning between tokens. On macOS (libomp) the CPU backend sets it for you and bounds the spin: idle workers sleep `GEIST_IDLE_SPIN_MS` (200) after the last token. On Linux (gcc, libgomp) it only takes effect from the environment at process start, and unbounded: set it together with `GOMP_SPINCOUNT=200000000` (~0.2 s on a Pi 5, less on faster cores), or idle workers spin until the process exits |
| `GEIST_IDLE_SPIN_MS` | how long idle OpenMP workers spin after the last parallel region (libomp, default 200 ms); an explicit `KMP_BLOCKTIME` wins |
| `GEIST_PREFILL_BLOCKTIME_MS` | the same, as a per-model value replacing that default (Qwen3.5 from 4 GiB uses 0 for faster prefill) |
| `GEIST_WEIGHT_MMAP=1` | read weights from the file mapping instead of copying them (the default; matters on low-RAM boards) |
| `GEIST_PREFILL_THREADS` / `GEIST_DECODE_THREADS` | override the automatic thread split. Prefill scales with all cores; decode is bandwidth-bound and often fastest one core below. On x86 with SMT, decode defaults to one thread per physical core unless `OMP_NUM_THREADS` is set |
| `GEIST_MMAP_PREFETCH=1` | `MADV_WILLNEED` the weight mapping: steadier first-token latency on 4 KB-page Linux (no-op on the Pi 5's 16 KB pages) |
| `GEIST_BACKEND=<name>` | override the `"auto"` backend choice (e.g. `metal`, `vulkan`) |

Backend-specific knobs: [BACKENDS.md](BACKENDS.md). Measured rationale:
[../benchmark/results/PI5.md](../benchmark/results/PI5.md).
