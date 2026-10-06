# Contributing to geist

geist is a lean C23 inference runtime for small, heavily quantized models on
everyday hardware; the [roadmap](ROADMAP.md) lays out the tracks. The most
valuable contributions right now are **NEON / AVX-512 microkernels**,
**low-bit quantization** (IQ/TQ/ternary) and **portability** (Windows, wider
x86-64 quant coverage, GPU backends).

## Ground rules

- **C23.** `src/` builds clean under
  `-Wall -Wextra -Wpedantic -Werror -Wshadow -Wundef`. Keep it that way.
- **Coding rules:** [AGENT.md](AGENT.md) — parameter order, allocation,
  bounds, errors, API migrations. Source comments cite it by section.
- **No new runtime dependencies** without discussion. The only vendored
  third-party code is the `stb` image headers.
- **Public API discipline:** headers in `include/` carry per-symbol
  `@stability` tags. Don't break a `STABLE` symbol; new surface starts
  `EXPERIMENTAL`. Changes to `include/` need a `CHANGELOG.md` note; what
  `STABLE` promises is in [docs/API_CONTRACT.md](docs/API_CONTRACT.md).
- **License:** by contributing you agree your work is licensed under
  [Apache-2.0](LICENSE).

## Build

```sh
make                      # auto-detect target (mac-omp / mac / pi5 / linux)
make TARGET=pi5           # Pi 5 (cross or native)
make MODE=debug           # -O0 -g
make MODE=asan            # AddressSanitizer + UBSan
make help                 # all targets and options
```

Toolchain: gcc ≥ 14 or Apple clang ≥ 16 (`-std=c23`). On macOS,
`brew install libomp` for the multi-threaded `mac-omp` target. Linux links
OpenBLAS (`libopenblas-dev`) by default; `GEMM_PROVIDER=native` builds without
it. The x86-64 default is `BACKENDS="cpu_x86 cpu_scalar"` (AVX-512/VNNI
runtime-dispatched over an x86-64-v3 baseline). Run `make clean` after
changing `BACKENDS` or `EXTRA_CFLAGS`; make does not track them.

## Tests

There is no test framework: each test is a `main()` and the **exit code** is
the contract (see [tests/README.md](tests/README.md)).

```sh
make test            # public headers + unit + integration + Python (fetches the model if missing)
make test-unit       # fast, kernel-level, no model needed
make test-int test-e2e   # real-model product path (needs the GGUF)
make test-py         # hermetic Python suites
make test FILTER=q3k # substring filter
make MODE=asan test  # sanitizer pass
```

| Exit | Meaning |
|------|---------|
| 0    | PASS |
| 77   | SKIPPED (precondition not met — no GGUF, wrong hardware) |
| 99   | ERROR (harness broke) |
| else | FAIL |

New tests follow the `*_unit` / `*_int` / `*_e2e` naming convention and exit
77 when their preconditions aren't met. `*_unit` tests use
`tests/test_helpers.h` (`geist_expect`, `GEIST_REQUIRE_GGUF`).

> **POSIX symbols:** under strict `-std=c23` glibc hides POSIX declarations, so
> a TU using `setenv`/`fork`/`opendir`/… needs `#define _POSIX_C_SOURCE 200809L`
> (or `_GNU_SOURCE` for raw sockets) as its **first line**. The musl CI leg
> catches it if you forget.

CI gates every PR on: build + unit on macOS, Linux glibc and Linux musl;
integration + e2e against real models on Linux; a coarse perf floor; the API
contract (`scripts/check-api-contract.sh`, `make agent-contract-smoke`);
ASan + UBSan; TSan; and clang-format. Details:
[docs/CI_COVERAGE.md](docs/CI_COVERAGE.md). Before pushing, run the checks in
[AGENT.md §7](AGENT.md#7-before-you-push).

## Benchmarks

```sh
make bench           # reproducible run vs any llama.cpp / bitnet.cpp found on the machine
make bench-small     # perf suite; records a new best to benchmark/results/APPLE.md
make bench-synth     # no download: synthetic GGUF with a real model's geometry
```

See [benchmark/METHODOLOGY.md](benchmark/METHODOLOGY.md) for methodology and
the quality/compare-ref procedures. **Never hand-edit recorded benchmark
numbers** — regenerate them on the relevant hardware.

Before/after numbers for a change come from `tools/bench_revision_ab.py`,
which builds both revisions from scratch and measures them against a control
run of the baseline: an incremental build differs from a clean one in code
layout, which alone moves timings by a few percent.

## Formatting

```sh
make format          # rewrite in place (clang-format 22, .clang-format at root)
make format-check    # verify only (hard CI gate)
```

## Pull requests

1. Branch from `main`.
2. One logical change per PR; explain *why*, not just *what*.
3. `make MODE=asan test` should pass (or document the skips).
4. For kernel/perf changes, include before/after numbers and the host.

Open an issue first for anything non-trivial.
