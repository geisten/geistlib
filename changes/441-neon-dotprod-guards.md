### Removed
- **The `#if defined(__ARM_FEATURE_DOTPROD)` guards inside
  `src/backends/cpu_neon/` and their dead `#else` branches (#441).** Every
  configuration that builds cpu_neon compiles with dotprod: `-dM -E` with the
  exact `make` flags of all 22 sources defines `__ARM_FEATURE_DOTPROD` for
  Apple clang `TARGET=mac` (with and without metal) and `TARGET=mac-omp`
  (default arm64 CPU, apple-m1) and for gcc-14 `TARGET=linux`
  (`-march=armv8.2-a+fp16+dotprod`) and `TARGET=pi5` (`-mcpu=cortex-a76`).
  The 14 guards were always true; the "NEON dotprod required" stub, the
  fall-through `case` labels for the INT8/INT4 KV attention and the empty
  `install_pq2_0_x8_gemv_if_eligible` body went with them. The run-time ISA
  check (`has_dotprod`, the resolver's `CPU_NEON_ISA_DOTPROD` rows) is
  unchanged, and so is the guard in `internal.h`, which tests include.
  cpu_neon has 28 `#if`s left (42 before); all 22 objects are byte-identical
  before and after in the same five configurations.
