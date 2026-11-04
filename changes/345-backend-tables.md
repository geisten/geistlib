### Changed
- **A backend table without a required slot does not compile (#345).**
  `src/backends/common/backend_tables.h` defines a backend's vtbl,
  primitives and descriptor (`GEIST_BACKEND_VTBL`, `GEIST_BACKEND_PRIMS`,
  `GEIST_BACKEND_DESCRIPTOR`) with every required slot as an argument, so a
  missing one, a `nullptr` one, or one cleared again among the optional
  slots is a compile error, and `scripts/check_wiring.py` (run by
  `make check-static`) refuses a table under `src/backends/` defined any
  other way. `<geist_backend.h>` lists the required slots: in the vtbl
  destroy, the buffer ops and `buffer_create_aliased` (documented as
  nullable before, but the arch layer calls it without a check); in the
  primitives everything but `relu_squared` and `scale_f32` (so `silu` too).
  `geist_backend_create`'s run-time check for a vtbl without `destroy` is
  gone. cpu_x86 now spells its vtbl and primitives out instead of copying
  cpu_scalar's in a constructor (only its fused table, all optional, is
  still built that way), and `BACKENDS=cpu_x86` alone compiles cpu_scalar's
  units it borrows from, so it links; `make check-static` builds every CPU
  backend alone (on an arm64 Mac also cpu_x86, with `clang -arch x86_64`).
  Object code: the same instructions, in a different order in
  `cpu_scalar/backend.o`; `create_from` and cpu_x86's constructor shrink.
