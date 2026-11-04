### Changed
- **`benchmark/c23_metrics.json` is a ratchet (#341).** It holds one ceiling
  per C23 gap instead of exact per-file counts. CI fails only when a total
  rises above its ceiling; a change that removes constructs passes without
  touching the file. Concurrent pull requests no longer conflict over it.
  Per-file counts stay in `tools/c23_metrics.py show`.
