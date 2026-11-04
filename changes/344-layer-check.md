### Added
- **`make check-layers`: the module boundaries, proven from the compiler
  (#344).** `tools/layers.json` is the one table of the layers of `src/` and
  what each may use; `docs/ARCHITECTURE.md` shows it, and a check keeps the
  two equal. `tools/check_layers.py` reads every include edge from the
  compiler's include tree (`-H`) of each library source and internal header,
  checks every undefined symbol of the built objects with `nm` (a prototype
  repeated in a `.c` file is caught too), compiles every internal header on
  its own, and counts the portability `#if`s per file outside `src/base/`
  (235 in 52 files) with the lexer of `tools/c23_metrics.py`. Known
  violations are in `tools/layers_baseline.txt`, which may only shrink: 3
  include edges, 23 symbol references (the transformer arch calls the
  engine's sampler, cpu_x86 borrows cpu_scalar's tables), 2 headers that do
  not compile alone (2 more are templates of their includers, listed in the
  table with the reason). Part of `make check-static`; also runs on the x86 build
  in the musl job. Negative tests in `tests/layers/`; no binary runs, about
  5 seconds after `make lib`.
