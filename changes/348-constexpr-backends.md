### Changed
- **Backend constants are typed `constexpr` objects (#348).** The
  object-like `#define` and `enum { }` constants of the CPU, Metal and Vulkan
  backends (tile sizes, row caps, ring sizes, ISA mask bits) became
  `constexpr`, each with the type its enumerator or literal had, and the x86
  attention kernel's `AF_LD8`/`AF_LD1` load macros became typed
  `always_inline` functions. The object code is byte-identical (Apple clang
  with Metal and with Vulkan, gcc-14 arm64, clang x86-64), so no kernel
  changed. The function-like macros that stay in the kernels say why.
