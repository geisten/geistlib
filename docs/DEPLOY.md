# Building and consuming the library

The engine artifact is `libgeist.a` plus the public headers. The release also
ships Linux CLIs built from `examples/simple_generate.c` (one with BitNet
weights embedded); they are raw-completion examples, not a second API.

## Build locally

```sh
make lib                   # auto-detected target, MODE=release -> lib/<target>/<mode>/libgeist.a
make TARGET=pi5 CC=aarch64-linux-gnu-gcc-14 lib   # cross-compile for Pi 5
make bin                   # tests, benchmarks and eval tools (not SDK contents)
make test                  # headers, unit, integration and Python suites (fetches the model)
make run ARGS='model.gguf "The capital of France is"'   # examples/simple_generate
```

### Dependency-free build

`GEMM_PROVIDER=native` drops the BLAS dependency. The release builds inside
`alpine:3.21`, so the archive is musl-built and the CLIs link `-static`:

```sh
make TARGET=linux CC=gcc GEMM_PROVIDER=native lib   # inside a musl environment, as release.yml does
```

Link a consumer against the same libc the archive was built with; mixing a
glibc program with a musl static library is not reliable.

## The packaged SDK

Each release attaches `libgeist-<platform>.tar.gz` for `macos-arm64`,
`linux-arm64` and `linux-x86_64`, holding `libgeist.a`, `include/*.h` and
`LICENSE`. Digests are in `SHA256SUMS`; every asset carries a signed
build-provenance attestation (`gh attestation verify <file> --repo geisten/geistlib`).

The archive holds geist's objects, not its dependencies. It is an OpenMP
build, so the consumer supplies the OpenMP runtime:

```sh
# Linux
cc -std=c23 -I "$d/include" my_app.c "$d/libgeist.a" -fopenmp -lm -o my_app
# macOS
cc -std=c23 -I "$d/include" my_app.c "$d/libgeist.a" \
   -framework Accelerate "$(brew --prefix libomp)/lib/libomp.a" -lm -o my_app
```

The release workflow compiles `examples/embed_smoke.c`,
`examples/agent_contract_smoke.c` and `examples/runtime_contract_smoke.c`
against each packaged archive with these lines, so a packaging break fails
before publication.

## Single-file deployment

`geist_model_load_from_memory` loads a GGUF embedded in your executable
(e.g. via `.incbin`); weights are aliased zero-copy from the read-only data,
so RAM cost equals the mmap path. `geist-bitnet` is built this way — see
[release.yml](../.github/workflows/release.yml) and
[QUICKSTART.md](QUICKSTART.md#3-ship-one-file).

## Consuming from another repository

Pin a release: exact `vX.Y.Z` tag, immutable asset URL, SHA-256 verified before
extraction — never `latest`. What the engine promises across that boundary is
in [API_CONTRACT.md](API_CONTRACT.md); `make agent-contract-smoke` and
`make runtime-contract-smoke` fail in this repository if a contracted
signature changes.

## Release assets

| Asset | Built by | Use |
| :-- | :-- | :-- |
| `libgeist-<platform>.tar.gz` + `SHA256SUMS` | `.github/workflows/release.yml` | the SDK |
| `geist-linux-{arm64,x86_64}` | same workflow | raw-completion CLI for any GGUF |
| `geist-bitnet-linux-{arm64,x86_64}` | same workflow | one-file BitNet appliance ([PI5_BITNET.md](PI5_BITNET.md)) |

There is no container image: a library is not a runnable artifact.
