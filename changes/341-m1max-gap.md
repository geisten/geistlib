### Added
- **The M1 Max gap of the Phase 0 baselines, documented (#341).** Two more
  attempts at the M1 Max CPU and Metal cells at 6642260d (bench-matrix run
  2990, then by hand under `with-mac-lock` with the idle gate at 70 %)
  failed the quiet-host and the 5 % repeat-spread gates; the spreads and the
  cause (Spotlight indexing) are in `benchmark/results/C23_BASELINE.md`.
