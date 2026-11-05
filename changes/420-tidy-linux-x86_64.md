### Added
- **`make check-static` analyses the Linux x86_64 build with Vulkan (#420).**
  A second required job runs the pinned clang-tidy 22 on the sources the
  macOS leg never compiles: `cpu_x86`, the Vulkan backend and the x86/Linux
  `#if` branches. It has its own baseline,
  `scripts/static-baseline-linux-x86_64.txt` (258 findings), so neither leg
  sees the other's; each of its analyzer findings has a reason in
  `scripts/static-reasons.txt` (none was a bug: intrinsics stores and
  cross-TU facts the analyzer does not follow, glibc's nonnull memcpy, the
  array-typed x86_64 `va_list`).
