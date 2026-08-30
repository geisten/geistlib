# Contributing to geist

Thanks for your interest! geist is a lean C23 inference runtime and an open
experiment in how far small, heavily quantized models go on everyday hardware —
the [roadmap](ROADMAP.md) lays out the tracks. The most valuable contributions
right now are in **NEON/AMX microkernels**, **low-bit quantization**
(IQ/TQ/ternary), and **portability** (Windows, wider x86-64 quant coverage, GPU
backends).

## Ground rules

- **Language/standard:** C23. The `src/` tree builds clean under
  `-Wall -Wextra -Wpedantic -Werror -Wshadow -Wundef`. Keep it that way.
- **No new runtime dependencies** without discussion. The point of geist is to
  stay tiny; the only third-party code is the vendored `stb` image headers.
- **Public API discipline:** `include/geist.h` carries per-symbol stability
  tags. Don't break a `STABLE` symbol; new surface starts `EXPERIMENTAL`.
- **License:** by contributing you agree your work is licensed under
  [Apache-2.0](LICENSE).

## Build

```sh
make                      # auto-detect target (mac-omp / mac / pi5 / linux)
make TARGET=pi5           # cross/native Pi 5
make MODE=debug           # -O0 -g
make MODE=asan            # AddressSanitizer + UBSan — run before sending kernel changes
make help                 # all options
```

Toolchain: gcc ≥ 14 or Apple-clang ≥ 16 (`-std=c23`). On macOS, `brew install libomp`
for the multi-threaded `mac-omp` target. On ARM64 Linux you need OpenBLAS
(`libopenblas-dev`); the audio FFT is vendored. For a dependency-free build,
`make GEMM_PROVIDER=native` (native fp32, no OpenBLAS).

> x86-64 has a **native AVX-512 / VNNI backend** (`src/backends/cpu_x86/`) that
> matches-to-beats llama.cpp on a Ryzen 9 9950X. It is **opt-in** for now — build
> with `make TARGET=linux BACKENDS="cpu_x86 cpu_scalar"` (x86_64 still defaults to
> the portable `cpu_scalar` path). Widening its quant coverage and making it the
> default are open, high-value work.

## Tests

There is no test framework — each test is a `main()` and the **exit code** is
the contract (see [tests/README.md](tests/README.md)).

```sh
make test            # unit + integration + python (auto-fetches the model if missing)
make test-unit       # fast, kernel-level, no model needed
make test FILTER=q3k # substring filter
make MODE=asan test  # sanitizer pass
```

| Exit | Meaning |
|------|---------|
| 0    | PASS |
| 77   | SKIPPED (precondition not met — no GGUF, wrong hardware) |
| 99   | ERROR (harness broke) |
| else | FAIL |

New tests follow the `*_unit` / `*_int` / `*_e2e` naming convention and skip
cleanly (exit 77) when their preconditions aren't met. `*_unit` use
`tests/test_helpers.h` (`geist_expect`, `GEIST_REQUIRE_GGUF`).

> **POSIX footgun:** under strict `-std=c23` glibc hides POSIX symbols, so any
> TU using `setenv`/`fork`/`opendir`/… needs `#define _POSIX_C_SOURCE 200809L`
> (or `_GNU_SOURCE` for raw sockets) as its **first line**. Caught by the musl
> CI leg if you forget.

**What CI enforces (not just unit tests).** Every PR is gated on: build + unit
on macOS / Linux-glibc / Linux-musl; **integration + e2e against the real model**
(forward pass, tokenizer, KV, agent/chat loops); a **perf** floor and a
**tool-calling quality** floor; **ASan + UBSan**; and clang-format. Run the heavy
ones locally before pushing:

```sh
make test-int test-e2e            # real-model product path (needs the GGUF)
make MODE=asan test-unit          # AddressSanitizer + UBSan
```

## Benchmarks

```sh
make bench           # raw timing probes (bench_* binaries)
make bench-small     # reproducible perf suite, records to benchmark/results/APPLE.md
```

See [benchmark/METHODOLOGY.md](benchmark/METHODOLOGY.md) for methodology and the
quality/compare-ref procedures. **Never hand-edit recorded benchmark numbers** —
regenerate them on the relevant hardware.

## Formatting

```sh
make format          # rewrite in place (clang-format 22, .clang-format at root)
make format-check    # verify only (CI runs this as a hard gate)
```

## C API style

The project is C23. These are the house rules; they exist so that a
signature tells you its contract without reading the body.

### Lengths before the arrays they describe

```c
void dequant_q4_K_row(size_t n_elems, const void *blocks, float out[static n_elems]);
```

Not style for its own sake: `T arr[static len]` is only expressible when
`len` is already in scope, and it is the only way to state "non-null, and
at least this many elements" to both the reader and the optimizer. Put
every length/capacity parameter before the array it sizes.

Use `[static len]` where the precondition is real. Do not use it on a
parameter that is legitimately null or legitimately shorter — a false
contract is worse than none, because the compiler is entitled to believe it.

### Nullability, constants, attributes

- `nullptr`, `true`, `false`, `bool` — never `NULL` or integer booleans.
- `constexpr` for typed constants instead of object-like macros, where
  compile-time and ABI requirements permit.
- `[[nodiscard]]` on every returned status or size that a caller must check.
- `const` on pointer parameters that are not written through.
- `restrict` **only** when non-aliasing is proven for every call site in
  the tree, not merely expected. Aliasing that the compiler was told could
  not happen is a miscompile, not a slow path.
- `typeof` / `auto` only where they improve clarity; never where they hide
  ownership or change what is generated.

### Bounds and arithmetic

Untrusted lengths are checked by subtraction, before any pointer moves:

```c
if (len > (size_t) (end - p))
    return GEIST_ERR_FORMAT;
const uint8_t *src = p;
p += len;
```

`p + len` and `&p[len]` are the same expression and neither is a valid
check: forming a pointer past one-past-the-end is already undefined, so the
comparison you were about to make may never run. Every size product
reaching an allocation, an index, or an alignment goes through
`src/base/checked.h` (`ckd_mul`, `ckd_add`, `geist_ckd_round_up_pow2`).

### Migrating an existing API

Signature changes land in **bounded batches**, one family at a time, each
independently reviewable and independently benchmarkable:

1. Move the whole family together — including any function-pointer typedef
   it is dispatched through. Make the typedef match the signature exactly
   rather than casting to it; a call through an incompatible function
   pointer is undefined however compatible the representations look, and an
   exact typedef turns a missed call site into a compile error.
2. Rerun the focused microbenchmark for that family plus the standard
   prefill/decode sweep.
3. For hot code, diff the optimized disassembly. A pure reorder should show
   an identical opcode sequence with permuted argument registers; anything
   else needs explaining before it lands.
4. A pure signature change must be bit-identical in output. Do not widen a
   tolerance to absorb one.

Public headers under `include/` additionally need an ABI note in
`CHANGELOG.md` and a version bump — see `docs/API_CONTRACT.md` for what
`STABLE` promises.

## Pull requests

1. Branch from `main`.
2. Keep changes focused; explain *why*, not just *what*.
3. `make MODE=asan test` should pass (or document the skips).
4. For kernel/perf changes, include before/after numbers and the host.
5. One logical change per PR.

Questions or design discussion: open an issue first for anything non-trivial.
