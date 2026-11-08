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
  `EXPERIMENTAL`. Changes to `include/` need a changelog note; what
  `STABLE` promises is in [docs/API_CONTRACT.md](docs/API_CONTRACT.md).
- **Changelog:** a user-visible change adds a fragment
  `changes/<slug>.md` (format: [changes/README.md](changes/README.md)) and
  does not touch `CHANGELOG.md`; the release folds the fragments in. Editing
  `[Unreleased]` directly made every two open pull requests conflict.
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
runtime-dispatched over an x86-64-v3 baseline); Apple Silicon builds
`BACKENDS="cpu_neon cpu_scalar metal"`. Run `make clean` after
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

CI runs in two tiers ([docs/CI_COVERAGE.md](docs/CI_COVERAGE.md#tiers)).
Every PR waits for the required tier: clang-format, the API contract and
version checks, and build + unit tests on Linux x86_64 (musl), Linux arm64
and macOS arm64. After the merge, on every push to `main` and nightly, the
post-merge tier runs the rest: integration + e2e against real models, the
Metal gate, the perf floors on the Pi 5 and the M1 Max, ASan + UBSan, TSan,
coverage, Vulkan on lavapipe and the other build legs; a job failing there
opens a Gitea issue (`CI on main: <job>`, label `ci-failure`) that the next
green run closes. A PR that touches kernels, a GPU backend, memory handling,
threading or other sanitizer-relevant code should run that tier before the
merge: add the label `ci:full` and push. Any other PR leaves the label off
(unless the user asks for it) and relies on the nightly run; each `ci:full`
run adds ~25 jobs that every other PR queues behind
([docs/CI_COVERAGE.md](docs/CI_COVERAGE.md#tiers), "When to set
`ci:full`"). Before pushing, run the checks in
[AGENT.md §7](AGENT.md#7-before-you-push).

`main` merges only when every required job is green; the branch protection
names each one in `.gitea/required-checks.txt`. A job is post-merge exactly
when its `if:` tests `needs.changes.outputs.full`. Adding, renaming or
removing a CI job (edit `.github/workflows/ci.yml`, then `python3
scripts/gitea-ci.py`; never `.gitea/workflows/ci.yml` by hand, which CI's
`gitea-ci.py --check` rejects) means regenerating that list with
`scripts/required_checks.py print` in the same PR (CI fails until it
matches; a new job must also go into the `ci-issues` job's `needs`) and,
after the merge, applying it with `GITEA_TOKEN=<admin token>
scripts/required_checks.py sync`.

## Benchmarks

On the shared M1 Max, wrap every benchmark, reported timing measurement and
heavy real-model run in `~/bin/with-mac-lock <command...>`. Builds and ordinary
unit tests do not require it. On the Pi 5, use `~/bin/with-pi-lock` on the
board and check temperature below 60 °C and load below 0.5 before measuring.
See [shared-host procedure](benchmark/METHODOLOGY.md#shared-host-locks).

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

For a PR's Apple "no regression" gate, run it on the Mac mini M6 through CI
rather than on a shared workstation:

```sh
tea actions workflows dispatch --login git.geisten.net --repo geisten/geistlib --ref main \
  -i new=origin/<branch> -i model=qwen35-0.8b-q8 mac-ab.yml
# Metal: -i backends="cpu_neon cpu_scalar metal" -i env="GEIST_BENCH_BACKEND=metal"
```

`model` is a key of `benchmark/cross_engine_cpu_protocol.json` (host cache,
sha256-checked). On the quiet M6 the control interval stays within ±0.3 %;
the M1 Max, shared by agents, campaigns and CI, gave ±2-28 % in the same week
(#279/#282). `pi5-ab.yml` and `vulkan-ab.yml` do the same on the Pi 5 and the
desktop GPU.

## Formatting

```sh
make format          # rewrite in place (clang-format 22, .clang-format at root)
make format-check    # verify only (hard CI gate)
```

`benchmark/c23_metrics.json` holds a ceiling for each construct the C23
work of #333 removes: `switch` with `default:`, enums without a fixed type,
`void *`, `#if`, `goto` and the rest. CI fails when a total rises above its
ceiling. A change that adds one raises that ceiling with `make c23-metrics`
(it only raises) and commits the file, so the review sees the count move; a
change that removes some passes as is. `tools/c23_metrics.py lower` sets
every ceiling to the tree's totals, from a branch that has main.
`python3 tools/c23_metrics.py show` has the per-file counts.

`make check-layers` proves the layer table `tools/layers.json` (shown in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#dependency-rules)): include
edges and symbol references across layers, internal headers that do not
compile alone and portability `#if`s. A new violation fails; fix it, or, for
a deliberate exception, change the table. `tools/layers_baseline.txt` holds
the old ones and only shrinks: remove a line once its violation is fixed.

## Pull requests

Development happens on `git.geisten.net/geisten/geistlib`; this GitHub
repository is its public mirror. Pull requests here are welcome: a maintainer
reviews them, brings them over to Gitea (`scripts/import-github-pr.sh`) and
merges them there, after which GitHub shows them as merged.


1. Branch from `main`.
2. One logical change per PR; explain *why*, not just *what*.
3. `make MODE=asan test` should pass (or document the skips).
4. For kernel/perf changes, include before/after numbers and the host.

Open an issue first for anything non-trivial.
