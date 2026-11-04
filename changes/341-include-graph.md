### Added
- **`tools/include_graph.py`, `benchmark/include_graph.json` and
  `docs/INCLUDE_GRAPH.md` (#341).** The direct include edges of every file
  under `src/` and `include/`, per layer, taken from clang's own
  preprocessor output (`-E -dI`) for the macOS arm64, Linux arm64 and Linux
  x86_64 configurations, so `-Isrc/base` and friends no longer hide which
  layer a header comes from. Host-independent (system headers are empty
  stand-ins, `CPATH` is ignored): Apple clang 21 and LLVM clang 22 give the
  same graph. CI's static-analysis job fails when the committed graph no
  longer matches the tree; `make include-graph` regenerates it. #344's layer
  check reads the same data (`layer_of`, `graph`, `load`).
