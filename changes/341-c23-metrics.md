### Added
- **`tools/c23_metrics.py` and `benchmark/c23_metrics.json` (#341).** The
  C23 gaps of #333 (`switch` with `default:`, enums without a fixed type,
  `void *`, casts, `#if`, `goto`, ...) counted per file by a C lexer, so
  comments and string literals never count. CI fails when the committed file
  no longer matches the tree; `make c23-metrics` regenerates it.
