### Changed
- **`make c23-metrics` only raises ceilings (#441).** It used to set every
  ceiling in `benchmark/c23_metrics.json` to the branch's totals, so a pull
  request regenerating it dropped ceilings below constructs merged since,
  and main's check failed (#447 after #450). `write` now raises only the
  ceilings the tree exceeds; `tools/c23_metrics.py lower` sets all of them
  to the tree's totals, on purpose, from a branch that has main.
