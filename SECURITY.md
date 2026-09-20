# Security Policy

geistlib is a C library that **loads model files and runs inference on them**.
Its security-relevant job is narrow and concrete: a GGUF or safetensors file is
a whole binary format handed to us from outside, and the parsers must reject
malformed input instead of trusting it. This document says what is trusted,
what is not, and how to report a problem.

> Status: **v0.11.x, pre-1.0.** There has been no independent security audit.
> The parser paths are bounds-checked, sanitizer-built and fuzzed in CI; treat
> everything else as ordinary library code, not as a sandbox.

## Reporting a vulnerability

**Do not open a public GitHub issue for security problems.**

Email **g.schlegel@geisten.com** with a description and the impact you believe
it has, a minimal reproduction (the file or the fuzz input, plus the command),
and the commit hash. You will get an acknowledgement; please allow a reasonable
window for a fix before public disclosure. There is no bug-bounty program.

## Trust model

**Untrusted — treated as adversarial input:**

- the **GGUF file**: header, metadata keys and values, tensor names, dimensions,
  offsets, quantization types — all of it, including the tokenizer arrays
  (`tokenizer.ggml.tokens`, `scores`, `token_type`, `merges`) that
  `src/engine/gguf_tokenizer.c` reads out of the same file,
- the **safetensors file** (JSON header, dtypes, shapes, data offsets),
- **caller text** passed to the tokenizer, including invalid UTF-8,
- token IDs handed to `decode` (in or out of vocab).

**Trusted** (the library assumes these are honest):

- the caller and the process it runs in — geistlib is a library, not a
  privilege boundary, and a caller that hands it a path can read that path,
- the host OS, the build toolchain, and the compiled binary,
- the **numerical content** of the weights. We check that a tensor lies inside
  the file; we do not and cannot judge what a model computes. A file that
  parses is a file that will be executed as a model.

## What the library does

- **Bounds by subtraction, never by pointer arithmetic.** The GGUF reader's
  only "is there room?" primitive is `cur_avail()`
  (`src/io/gguf_reader.c`): `end - p`, because forming `p + n` for an
  attacker-chosen `n` is undefined behaviour the compiler may fold away.
- **Checked arithmetic for every untrusted size.** Allocation goes through
  `src/base/heap.h`, which multiplies with `ckd_mul`/`ckd_add` and fails
  instead of wrapping (AGENT.md §3). Counts read from a file are additionally
  bounded against the bytes actually present before they become allocation
  counts.
- **No `system()`, no shell, no network.** The library opens files the caller
  names and computes; it starts no processes and makes no connections.
- **Malformed-input tests** (`tests/test_io_malformed_unit.c`) plus two fuzz
  harnesses (`tests/fuzz_gguf.c`, `tests/fuzz_tokenizer.c`) that run on every
  PR — deterministically under ASan/UBSan, and coverage-guided under libFuzzer
  for 30 s per target. See [docs/CI_COVERAGE.md](docs/CI_COVERAGE.md).
- **`src/io` holds a hard 35 % line-coverage floor** in the coverage gate,
  independent of the ratchet, because it is the parser surface.

## What the library does NOT do

- **No sandbox, no isolation, no privilege drop.** Parse a hostile file in a
  process you are willing to lose.
- **A GGUF is mmap'd, and the mapping stays live for the context's lifetime.**
  Metadata and tensor pointers point *into* that mapping. If the file is
  modified or truncated underneath us, reads through those pointers are the
  kernel's problem (`SIGBUS`), not something the library can detect. Do not
  point it at a file another writer controls.
- **No signature or provenance check on model files.** Checksums are the
  caller's job (geistkit pins fixtures by SHA for exactly this reason).
- **Resource limits are the caller's job.** A legitimate model file may ask for
  gigabytes. The library refuses arithmetic it cannot represent and sizes the
  file cannot support, but it has no policy for "too large".
- **Nothing here concerns model behaviour.** Prompt injection, unsafe
  generations and tool use live in the caller (geistshell has its own
  SECURITY.md); geistlib only does the arithmetic.

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.11.x  | Best-effort; pre-1.0, no security SLA |

Fixes land on `main`; there is no backport guarantee.
