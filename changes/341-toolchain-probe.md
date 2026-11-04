### Added
- **`tools/toolchain_probe.sh` and `docs/TOOLCHAIN_PROBE.md` (#341).** Which
  C23 features each CI toolchain has, compiled, linked and run under the
  tree's warning flags: `<stdbit.h>` (missing on macOS and musl: needs a
  wrapper), `counted_by` (clang only: needs a macro empty on gcc 14), fixed
  enums in a C++17 header, `[[gnu::cleanup]]`, `typeof`, `<stdckdint.h>`
  (usable as is everywhere) and `#embed` (gcc 15: not usable). The build +
  unit test legs and the documented-build images run it against
  `tools/toolchain_probe.expect`, so a runner image that gains or loses a
  feature fails until the page is updated. `make toolchain-probe` runs it
  locally.
