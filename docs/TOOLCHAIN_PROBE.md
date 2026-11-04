# C23 toolchain probe

Which C23 features every CI toolchain really has, measured by
`tools/toolchain_probe.sh` (#341): one small program per feature, compiled,
linked and run under the tree's flags (`-std=c23 -O2 -Wall -Wextra -Wpedantic
-Werror`; C++ as `check-headers` does, `-std=c++17 -pedantic-errors`). "yes"
means the program built and gave the right answer, not only that the
compiler accepted the syntax.

CI runs the probe on each toolchain below and compares it with
`tools/toolchain_probe.expect`; a runner image that gains or loses a feature
fails its job until the expect file and this page are updated. The
comparison holds for the compiler major version recorded there; a new
compiler is reported and not compared, which is also what happens on the
GitHub mirror's runners and their older Apple clang (and on the
"macOS x86_64" leg, which only runs there). Locally: `make toolchain-probe`
(uses `CC` and `CXX`).

## Toolchains

| id | CI job | compiler | C library |
|---|---|---|---|
| `macos-arm64` | build-test "macOS arm64" | Apple clang 21.0.0 | macOS 27 SDK |
| `linux-arm64` | build-test "Linux arm64" | gcc-14 (Ubuntu 24.04 runner) | glibc 2.39 |
| `debian-13` | readme-build `debian:13` | gcc 14.2.0 | glibc 2.41 |
| `alpine-3-21` | readme-build `alpine:3.21` | gcc 14.2.0 | musl 1.2.5 |
| `ubuntu-24-04` | readme-build `ubuntu:24.04` | gcc-14 14.2.0 | glibc 2.39 |

## Results

| feature | macOS clang | Linux gcc-14 (glibc) | Alpine (musl) | verdict |
|---|---|---|---|---|
| `<stdbit.h>` | no header | yes | no header | **needs a wrapper** |
| `counted_by` | yes, and `__builtin_dynamic_object_size` sees it | rejected (`-Werror=attributes`) | rejected | **needs a wrapper** (empty on gcc 14) |
| fixed enum in a header, C23 + C++17 | yes | yes | yes | **usable as is** |
| `[[gnu::cleanup]]` | yes | yes | yes | **usable as is**, spelled `GEIST_CLEANUP` (AGENT.md §3) |
| `typeof`, `typeof_unqual` | yes | yes | yes | **usable as is** |
| `<stdckdint.h>` | yes | yes | yes | **usable as is** |
| `#embed` | yes | rejected | rejected | **not usable** (gcc 15) |

"Linux gcc-14 (glibc)" is `linux-arm64`, `debian-13` and `ubuntu-24-04`;
they agree on every row.

## What each verdict means for #333

- **`<stdbit.h>`: needs a wrapper.** glibc 2.39+ ships it; macOS and musl
  1.2.5 do not, and gcc 14 has no header of its own (gcc 15 does). Any use
  needs a `src/base/` header whose fallback on two of the five toolchains is
  the `__builtin_ctz`/`clz`/`popcount` calls it would replace, so swapping
  `__builtin_*` for `stdc_*` gains no portability (#348).
- **`counted_by`: needs a wrapper.** clang takes it and feeds it to
  `__builtin_dynamic_object_size` (so `-fsanitize=bounds` and fortify see the
  bound); gcc 14 warns that it ignores the attribute, which `-Werror` turns
  into an error. A `GEIST_COUNTED_BY(n)` macro that is empty unless
  `__has_attribute(counted_by)` is the only form that builds everywhere.
- **Fixed underlying enums** (`enum e : int32_t`) build in a header as C23
  and C++17 with the expected widths (4 and 1 byte for `int32_t` and
  `uint8_t`). #346 relies on this.
- **`[[gnu::cleanup]]`** runs on early and normal return everywhere; the
  project spells it `GEIST_CLEANUP` (`src/base/cleanup.h`) so an unsupported
  compiler is refused in one place.
- **`typeof`** and **`typeof_unqual`** work as C23 keywords everywhere, e.g.
  for the array macros of `src/base/heap.h`.
- **`<stdckdint.h>`** is present on every CI toolchain; the
  `__has_include` fallback in `src/base/checked.h` is not exercised by CI.
- **`#embed`** needs gcc 15; the generated `*_spv.h` and `metal_shaders.h`
  headers stay.
