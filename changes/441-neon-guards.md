### Removed
- **The `#if defined(__ARM_NEON)` guards inside `src/backends/cpu_neon/` and
  their dead `#else` branches (#441).** cpu_neon builds only for arm64
  (`mk/target-{mac,mac-omp,linux,pi5}.mk` add it only there;
  `scripts/check_wiring.py` likewise), where `__ARM_NEON` is always defined,
  so the 115 guards were always true and their scalar or "NEON required"
  `#else` branches never compiled. `defined(__ARM_NEON) && defined(X)` became
  `defined(X)`; the `__ARM_FEATURE_DOTPROD` / `__ARM_FP16_FORMAT_IEEE`,
  `GEIST_TARGET_PI5`, `_OPENMP`, Accelerate and compiler guards stay, as does
  the one in `internal.h`, which tests include. Two TQ2_0 helpers only that
  dead branch called and eight unused `<stdio.h>` includes went with them.
  cpu_neon shrinks from 13 842 to 13 096 lines and from 157 to 42 `#if`s;
  all 22 objects are byte-identical before and after for Apple clang
  (`TARGET=mac` with and without metal, `TARGET=mac-omp`) and gcc-14 arm64
  (`TARGET=linux`, `TARGET=pi5`).
