# Building & consuming the library

The supported engine artefact is `libgeist.a` plus the public headers. The release
workflow also compiles `examples/simple_generate.c` into thin Linux convenience
CLIs, including a BitNet binary with embedded weights; these are raw-completion
examples, not an additional library API or daemon. This page is about producing
and consuming the SDK.

## Build locally

```sh
make lib                   # auto-detect target, MODE=release -> lib/<target>/<mode>/libgeist.a
make TARGET=pi5 CC=aarch64-linux-gnu-gcc-14 lib # cross-compile for Pi 5
make bin                   # tests, benchmarks and eval tools — not SDK contents
make test                  # unit + integration + Python suites (auto-fetches the model)
make run ARGS='model.gguf "The capital of France is"'   # examples/simple_generate
```

### Self-contained / dependency-free build

`GEMM_PROVIDER=native` drops the BLAS dependency. The release workflow builds
inside Alpine/musl, then links the convenience executables with `-static`:

```sh
# Linux: run inside a matching musl environment (as release.yml does)
make TARGET=linux CC=gcc GEMM_PROVIDER=native \
     lib
```

The release builds run inside `alpine:3.21` so the archive is musl-built.
Linking a glibc consumer against a musl static library is a link-time coin
flip, so match the libc you built against.

## The packaged SDK

Each release attaches `libgeist-<platform>.tar.gz` for `macos-arm64`,
`linux-arm64` and `linux-x86_64`, holding `libgeist.a`, `include/*.h` and
`LICENSE`. Digests are in `SHA256SUMS`.

The archive contains geist's objects, not its dependencies. **An `.a` is not a
link:** the build is an OpenMP build, so a consumer supplies the OpenMP runtime
itself.

```sh
# Linux
cc -std=c23 -I "$d/include" my_app.c "$d/libgeist.a" -fopenmp -lm -o my_app
# macOS
cc -std=c23 -I "$d/include" my_app.c "$d/libgeist.a" \
   -framework Accelerate "$(brew --prefix libomp)/lib/libomp.a" -lm -o my_app
```

The release workflow compiles `examples/embed_smoke.c` and
`examples/agent_contract_smoke.c` against each packaged artefact with exactly
these lines, so a packaging break fails before publication rather than in a
consumer's build.

## Consuming from another repository

Pin a release: exact `vX.Y.Z` tag, immutable asset URL, SHA-256 verified before
extraction — never `latest`.

What the engine promises across that boundary is in
[API_CONTRACT.md](API_CONTRACT.md), and `make agent-contract-smoke` fails here
if a signature moves — in this repository, not in the consumer's build.

## GitHub options

| option | how | best for |
|---|---|---|
| **Release artifacts** ✅ `.github/workflows/release.yml` | push a `v*` tag → builds `libgeist.a` + headers for linux-arm64/x86-64 (musl) and macos-arm64, attaches them with `SHA256SUMS` | consuming the SDK from another project |
| **Linux convenience CLIs** ✅ `.github/workflows/release.yml` | the same tag also attaches `geist-linux-{arm64,x86_64}` and `geist-bitnet-linux-{arm64,x86_64}` | quick raw-completion smoke or the one-file BitNet appliance |
| **GHCR container** *(not wired up)* | there is no image to publish: a library is not a runnable artefact | — |
