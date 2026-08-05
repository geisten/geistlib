# FFI examples — the header is the ABI

geistlib ships no language bindings, on purpose: the STABLE core is ~15
plain C calls, a pull-model decode loop (no callbacks), and `NULL` for
every options struct you don't care about. Each file in this directory is
the **complete** integration for one language — no generator, no wrapper
package, no shim. Copy the one you need into your project and own it.

| File | Language | Mechanism | Lines of integration |
| :-- | :-- | :-- | --: |
| `generate.py` | Python | stdlib `ctypes` | ~30 |
| `generate.rs` | Rust | hand-written `extern "C"` (no bindgen) | ~40 |
| `generate.go` | Go | cgo reading `geist.h` directly | ~40 |
| `generate.js` | JavaScript | `bun:ffi` `dlopen` | ~40 |

All four do the same thing as
[`../simple_generate.c`](../simple_generate.c): load a GGUF, prefill a
prompt, stream a greedy continuation, stop at EOS.

## Build the shared library once

Python and JS need a shared object (Rust and Go can also link the static
archive, but using the same `.so`/`.dylib` keeps this page to one build
step). From the repo root, after `make lib`:

```bash
# Linux
cc -shared -o libgeist.so \
  -Wl,--whole-archive lib/linux/release/libgeist.a -Wl,--no-whole-archive \
  -fopenmp -lm

# macOS
cc -dynamiclib -o libgeist.dylib \
  -Wl,-force_load,lib/mac-omp/release/libgeist.a \
  -framework Accelerate "$(brew --prefix libomp)/lib/libomp.a" -lm
```

(`--whole-archive`/`-force_load` for the same reason the SDK smoke uses
them: a shared library must carry *every* object, not just the ones a
particular caller happens to reference.)

## Run

```bash
python3 examples/ffi/generate.py ./libgeist.so model.gguf "The capital of France is"

rustc -O examples/ffi/generate.rs -L . -lgeist -o generate_rs
LD_LIBRARY_PATH=. ./generate_rs model.gguf "The capital of France is"

cd examples/ffi && LD_LIBRARY_PATH=../.. go run generate.go ../../model.gguf "The capital of France is"

bun examples/ffi/generate.js ./libgeist.so model.gguf "The capital of France is"
```

On macOS replace `libgeist.so` with `libgeist.dylib` and
`LD_LIBRARY_PATH` with `DYLD_LIBRARY_PATH`.

## Notes for your own integration

- `geist_session_create(model, backend, NULL, &s)` — `NULL` opts means
  greedy decode and sane defaults; you never have to mirror a C struct
  layout unless you want sampling knobs.
- Errors: every creating call returns `0` on success; the message behind a
  failure is `geist_last_create_error()` / `geist_session_errmsg()`.
- Stability: everything used here is tagged `STABLE` in
  [`include/geist.h`](../../include/geist.h), and CI's contract smoke
  breaks any PR that changes these signatures.
