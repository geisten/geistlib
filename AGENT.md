# AGENT.md — writing code in this repository

Rules for anyone adding C to geist, human or model. Source comments cite this
file by name ("per AGENT.md", "AGENT.md hot-path rule"); this is that file.

Read this before writing a function. The prose rationale for humans lives in
`CONTRIBUTING.md`; this file is the short, checkable form.

---

## 1. Function parameters — the order is not a matter of taste

**Lengths and capacities come before the arrays they describe.**

```c
/* yes */
void dequant_q4_K_row(size_t n_elems, const void *blocks, float out[static n_elems]);
void rope_compute(size_t seq_len, size_t head_dim, size_t n_rotated, float theta,
                  float *cos_out, float *sin_out);

/* no — the length cannot be used in a contract it comes after */
void dequant_q4_K_row(const void *blocks, float *out, size_t n_elems);
```

This is not style for its own sake. `T arr[static len]` is only expressible
when `len` is already in scope, and it is the only way a signature can say
"non-null, and at least this many elements" to the reader, to the compiler,
and to the next model editing this file.

### The order, in full

```
1. out-params that are pure sizes      (rare; e.g. size_t *out_len)
2. lengths / capacities / dimensions   size_t n, n_rows, n_in, n_out, m
3. policy scalars                      float eps, float theta, size_t sliding_window
4. input arrays                        const T in[static n]
5. nullable / optional inputs          const T *bias        <- plain pointer
6. output arrays                       T out[static n]
7. context / handles                   struct geist_backend *be
```

Handles that are not arrays (`struct geist_backend *be`, `void *state`) are
exempt from rule 1 — they are not sized by anything. Put them where they read
best, conventionally first or last, and stay consistent within a family.

### When to write `[static len]` — and when it is a lie

Use it when the parameter is **always** non-null and **always** at least `len`
elements. Otherwise use a plain pointer.

| situation | write |
|---|---|
| always present, exactly/at least `n` elements | `float x[static n]` |
| legitimately `nullptr` (optional bias, optional weight) | `const float *bias` |
| length is a bound, not a guarantee (may write fewer) | `float out[static cap]` + return the count |
| size is not a parameter (opaque blob, `void *`) | `const void *blocks` |

A false `[static]` is worse than none: the compiler is entitled to believe it.
Real examples in this tree — `linear_fp32`'s `bias` and `rmsnorm_fp32`'s
`weight` are passed `nullptr` at 84 and 12 call sites respectively, so they
stay plain pointers.

The bound may be an expression: `const float x[static m * n_in]` is valid and
preferred over understating it as `[static n_in]`.

### What `[static len]` actually buys

Measured on this project's toolchains, so you can calibrate:

- **Codegen: nothing.** clang 21 and gcc 15 emit identical instructions with
  and without it. Do not justify a change by "the optimizer".
- **Diagnostics: gcc only.** gcc warns `reading 16 bytes from a region of
  size 8` when the caller's array is visibly too small (stack arrays and
  tracked `malloc`). clang warns only on a literal `nullptr`.
- **Nothing catches** a short buffer arriving as an opaque `void *` or a
  struct member across a translation unit. That needs an explicit extent —
  see `geist_weight::raw_nbytes` and `src/base/checked.h`.

So the real payoff is that the signature documents itself and the family
stays uniform. That is enough; do not oversell it in a commit message.

---

## 2. Types and null

- `nullptr`, `true`, `false`, `bool` — never `NULL`, never `int` booleans.
  (`NULL` in a *comment*, describing the concept, is fine.)
- `constexpr` for typed constants, not object-like `#define`, where
  compile-time and ABI requirements allow.
- `[[nodiscard]]` on every returned status or size a caller must check.
- `const` on pointer parameters not written through.
- `restrict` **only** where non-aliasing is proven for every call site in the
  tree. Several kernels here document "y may alias x" — those must not get
  it. Aliasing the compiler was told cannot happen is a miscompile.
- `typeof` / `auto` only where they clarify; never where they hide ownership.

## 3. Memory

- **All allocation flows through `src/base/heap.h`** — `heap_alloc_aligned`,
  `heap_calloc_aligned`, `heap_alloc_n_aligned`, and the array macros. Not
  raw `malloc` / `aligned_alloc`.
- **No runtime heap allocation in hot paths** (per-token, per-layer, per
  block). Use caller-provided workspace or a thread-local high-water buffer.
- Free through `safe_free(&p)`; it tolerates null and nulls your pointer.
- **Every size product from untrusted input goes through
  `src/base/checked.h`** (`ckd_mul`, `ckd_add`, `geist_ckd_round_up_pow2`)
  before it reaches an allocation, an index, or an alignment. Model metadata
  and caller geometry are untrusted.

## 4. Bounds

Check untrusted lengths by **subtraction, before the pointer moves**:

```c
if (len > (size_t) (end - p))
    return GEIST_E_FORMAT;
const uint8_t *src = p;
p += len;
```

`p + len` and `&p[len]` are the same expression, and neither is a valid
check: forming a pointer past one-past-the-end is already undefined, so the
comparison you meant to make may be folded away.

## 5. Errors

- Correctness first. **No silent truncation** — if the answer will not fit,
  fail; do not return a shortened one.
- **Outputs are well-defined on every return path**, including failure. A
  function that returns `GEIST_OK` has written its output buffer.
- Express invariants through **explicit validation**, not `assert`. Release
  builds define `NDEBUG`.
- Portability differences live **behind clear boundaries**, not scattered
  `#ifdef`s in logic.

## 6. Changing an existing API

Signature migrations land in **bounded batches**, one family at a time:

1. Move the whole family together, **including any function-pointer typedef
   it is dispatched through**. Make the typedef match exactly rather than
   casting to it — a call through an incompatible function pointer is
   undefined however compatible the representations look, and an exact
   typedef turns a missed indirect call site into a compile error instead of
   silently reordered arguments.
2. A textual rewrite of call sites is **not idempotent**. If a pass is
   partial, revert and redo; do not patch over it.
3. Rerun the family's focused benchmark plus the standard prefill/decode
   sweep (`bench_perf_sweep`).
4. For hot code, diff the optimized disassembly. A pure reorder shows an
   **identical opcode sequence with permuted argument registers**; anything
   else needs explaining before it lands.
5. A pure signature change must be bit-identical in output. Never widen a
   tolerance to absorb one.

Public headers under `include/` also need a `CHANGELOG.md` note. Check the
`@stability` tag first: `EXPERIMENTAL` may change; anything listed in
`docs/API_CONTRACT.md` is a promise and needs a versioned migration.

## 7. Before you push

```sh
make format-check                  # clang-format is a hard CI gate
make test-unit MODE=release
make MODE=asan test-unit           # ASan + UBSan
```

CI builds every Linux target with **gcc**; local macOS builds use clang, and
they disagree on parts of C23. Syntax-check what you changed with the local
gcc before pushing — a `constexpr` with two declarators compiles here and
fails all nine Linux jobs:

```sh
gcc-15 -std=c23 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wundef \
       -fsyntax-only -D_GNU_SOURCE -Wno-vla-parameter \
       -Iinclude -I. -Isrc/base -Isrc/quant -Isrc/backends/common \
       -Isrc/formats/gguf -Isrc/io -Isrc/engine <changed>.c
```

Commits: one logical change each, and say *why*. For kernel or perf work,
include before/after numbers and the host.
