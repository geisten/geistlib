# C23 review (#333): what was proven, changed and rejected

Goal of #333: whatever can be proven at compile time or by a static check is
proven there, and no benchmark gets slower. Runtime validation of untrusted
input (GGUF metadata, caller geometry, file contents) stays; the work only
replaced runtime checks of the library's own wiring.

This report closes #333. Baseline is main at `6642260d`, the last commit
before any #333 change; "now" is main at `46a9aede`. The tickets hold the
per-item numbers; this page holds the outcome and, as much as the rest, what
was rejected and why.

## What now fails the build or a required check

| mistake | where it fails | since |
|---|---|---|
| a new value in an enum that a `switch` does not handle | `-Wswitch-enum` (no `default:` over enums) | #394 (#346) |
| an enum without a fixed underlying type | `c23_metrics` ceiling `enum_named_unfixed = 0` | #394, #415 |
| a family missing from the list, or a list line without a file | make, from `family_list.h` | #428 (#345) |
| a family without `attention_block`, `ffn_block` or another required hook | compile error, `TRANSFORMER_FAMILY(...)` | #428 |
| a backend without a required vtable or prims slot | compile error, `GEIST_BACKEND_VTBL/PRIMS/DESCRIPTOR` | #433 (#345) |
| a family or CPU backend that no longer builds alone | `check-static-wiring` (each one alone) | #428, #433 |
| an include or a symbol across a layer the table does not allow | `check-layers`, from the compiler and `nm` | #424 (#344) |
| an internal header that does not compile on its own | `check-layers` | #424, #425 |
| one more portability `#if` in a file outside `src/base/` | `check-layers` (per-file ceiling) | #424 |
| a wrong Metal handle or architecture handle | compile error (opaque types); `check-handles` keeps 10 cases | #421, #434 (#347) |
| a new clang-tidy finding (null dereference, use-after-free, ...) | `check-static-tidy` ratchet; NOLINT needs a reason | #390, #418 (#342) |
| a printf-like helper called with mismatched arguments | `-Wformat=2` + `[[gnu::format]]` | #417 (#343) |
| fallthrough without `[[fallthrough]]`; a function without a prototype | `-Wimplicit-fallthrough`, `-Wmissing-prototypes` | #396, #398 (#343) |
| a null dereference gcc's optimizer can see | `-Wnull-dereference` | #430 (#343) |
| a value- or sign-changing conversion in `src/base`, `src/io`, `src/engine`, `vision_siglip`, `cpu_scalar`, Metal, Vulkan, `tools/` | `-Wconversion -Wsign-conversion` | #436 (#343) |
| `_Static_assert`, `__attribute__`, `__builtin_memcpy` and kin | `tests/test_spelling_py.py` | #395, #416, #432 (#348) |
| more `void *`, `#define` constants, `goto`, ... than before | `c23_metrics` ceilings (fail on growth only) | #397, #415 (#341) |
| scope-exit cleanup (`GEIST_CLEANUP`, `[[gnu::cleanup]]`) in a kernel or forward pass | `scripts/check-cleanup-scope.sh` | #364 (#349) |
| a toolchain gaining or losing a C23 feature the tree relies on | `toolchain_probe.sh` on every CI toolchain | #423 (#341) |

All of them run without a model and without a GPU: `make check-static`
(clang-tidy, layers, handles, wiring matrix) in the required static-analysis
job, the warnings in every build leg.

## Bugs the new checks found

| bug | found by | fixed in |
|---|---|---|
| KIVI attention computed with an undefined value unless `head_dim % 4 == 0`; now refused at session allocation | clang-tidy | #392 (#391) |
| `make FAMILIES=<subset>` without qwen35 and qwen3moe linked no program (since #245) | one-family build matrix | #428 |
| `BACKENDS=cpu_x86` alone did not link | one-backend build matrix | #433 |
| `buffer_create_aliased` documented optional, called without a null check | required-slot macros | #433 |
| NEON dequant fallback decoded F16 subnormals wrong in the scalar tail | duplicate removal | #435 |
| a malformed UTF-8 byte became 0xFFFFFFFF, not U+FFFD, in the tokenizer | `-Wsign-conversion` | open: #427 |
| 7 printf-like helpers had unchecked callers | `-Wformat=2` | #417 (no caller was wrong) |

## Metrics

Counted by `tools/c23_metrics.py` (a C lexer; comments and literals never
count), the same script on both trees.

| construct | 6642260d | now | change |
|---|--:|--:|--:|
| `void *` | 1992 | 1262 | -730 |
| `(void *)` casts | 64 | 23 | -41 |
| `(void **)` casts | 356 | 319 | -37 |
| object-like `#define` | 159 | 62 | -97 |
| function-like `#define` | 74 | 97 | +23 |
| `enum { }` constants | 44 | 3 | -41 |
| named enums without a fixed type | 44 | 0 | -44 |
| `switch` with `default:` | 47 | 20 | -27 |
| `_Static_assert` | 17 | 0 | -17 |
| `__attribute__` | 52 | 0 | -52 |
| `__builtin_*` | 82 | 76 | -6 |
| `goto` | 46 | 24 | -22 |
| `#if` / `#ifdef` / `#ifndef` | 507 | 501 | -6 |
| `constexpr` | 198 | 383 | +185 |
| `static_assert` | 53 | 122 | +69 |
| `[[gnu::...]]`, `[[clang::...]]` | 74 | 87 | +13 |

The tree grew by 8 733 lines over the same period, most of it unrelated
work; growing counts in the "already done" constructs (`nullptr`,
`static_assert`, `constexpr`, `[static n]`) are the new code following the
rules. `define_function` rose on purpose: `TRANSFORMER_FAMILY`, the
family-list X-macro and the backend table macros are what make a missing hook
a compile error.

Size and build time (macOS arm64, `make TARGET=mac-omp -j8 lib` after
`make clean`, under `with-mac-lock`, three runs):

| | 6642260d | now |
|---|--:|--:|
| `.text` of libgeist.a (`tools/text_size.py`) | 896 972 B | 969 184 B |
| clean build, real | 4.0–4.2 s | 4.6–4.8 s |
| clean build, user + sys | 17.3–17.8 s | 18.2–18.7 s |

These totals include all the unrelated work merged in between (the
transformer layer alone grew by 39 KB of `.text`), so they bound #333 rather
than measure it. The #333 PRs themselves were measured one by one: most left
every object byte-identical, #428 and #433 shrank `.text` (−231 B on arm64,
−284 B on x86_64), and none came near the +1 % limit. Build CPU time is +6 %
against +9 % lines, inside the +10 % gate; the one-family and one-backend
matrix in `check-static` adds about 20 s to the static-analysis job, not to
the build.

## Rejected, with the reason

| item | ticket | why |
|---|---|---|
| gcc `-fanalyzer` | #342 | 9 findings on 128 sources, none real; gave up partway in 64 files |
| cppcheck | #342 | 11 findings, none real; missed the fixture's null dereference |
| clang `-Wthread-safety` | #342 | `guarded_by` on a C struct field does not bind to the object; 9 findings, none bugs |
| `-Wcast-qual` | #343 | 111 sites, all deliberate (owned `const` fields freed, `const` jobs to `geist_par_for`, read-only mmap) |
| `-Wstrict-prototypes` | #343 | nothing to report in C23 |
| `-Wconversion` in `src/archs`, `src/backends/{common,cpu_neon,cpu_x86}`, `src/formats`, `src/quant`, `tests/` | #343 | 399 sites, no bug; casts only |
| typed handles in `quant/quant.h` | #347 | its 68 `void *` are raw quantized bytes behind uniform dispatch tables |
| `typeof` in `heap.h`'s allocation macros | #348 | a wrong element type is already a compile error there |
| `<stdbit.h>` | #348, #341 | missing on macOS and musl 1.2.5; the tree has no builtin it would replace |
| 75 function-like macros kept | #348 | code generators, type or field arguments, kernel-loop accumulators; each has its reason |
| 73 object-like `#define`s kept | #348 | public API macros, `#if` operands, shared with GLSL (`vk_limits.h`), attribute strings, file-format codes |
| splitting `metal/ops.c`, `vulkan/ops.c`, `arch_state.c`, `weight_resolve.c`, `q4_K.c` | #350 | every clean seam adds lines (+15 to +110) and file-local helpers become layer-wide; the only gain is file size |
| further cross-backend merges | #350 | Waves 1-2 (#276-#278) took the real ones; the rest are `-Wswitch-enum` case lists and vtable signatures |
| `#embed` | #341 | needs gcc 15; CI builds with gcc-14 |
| `counted_by` | #341 | gcc-14 rejects it; would need an empty macro there |

## Performance

No benchmark got slower. Most PRs changed spelling only and kept every object
file byte-identical (compared with Apple clang, gcc-14 arm64 and clang for
x86_64), which needs no timing. The PRs that changed code on a hot path ran
`mac-ab.yml` on the Mac mini M6 (CPU, qwen35-0.8b-q8, 12 interleaved cycles):

| PR | change | result |
|---|---|---|
| #422 | dead code; `transformer_run_one_step` now inlined into the decode step | prefill/decode within ±0.1 %, every interval includes zero (run 2948) |
| #435 | NEON dequant fallback on the shared block layout | within ±0.1 %, control ±0.2 % (run 2952) |

A first run of #435 (2939) showed decode +1 %: the branch then lagged main
by seven merges. Measured against a base the branch contains, it is neutral.
Compare like with like before reading a regression into an A/B.

## Process notes

- **Generated files and concurrent PRs.** An exact-match check on a
  committed generated file makes every concurrent PR conflict over it.
  `c23_metrics.json` became a ceiling ratchet (#415) for that reason;
  `include_graph.json` still has the problem (#442).
- **Combinations.** Two PRs green alone can make main red: a new
  `__attribute__` meeting the spelling test, new symbols meeting the layer
  baseline. Merge one at a time and recheck the combination.
- **Agent heads.** CI on `agent/*` branches waits for `/ci-ok` (AGENT.md §9);
  every merge of main into a branch needs a new one. Fewer, complete pushes
  save rounds.

## Left open

- #420: clang-tidy on the x86_64 and Vulkan sources.
- #427: the tokenizer's replacement character, with a parity test.
- #441: the layer violations in `tools/layers_baseline.txt`, among them the
  235 portability `#if`s in hot kernels (one kernel family at a time).
- #442: `include_graph.json` conflicts between concurrent PRs.
- #443: gcc-14 x86_64 `MODE=tsan` and `-Wstringop-overflow` in a test.
- #444: the M1 Max cells of the Phase 0 baseline.
