### Changed
- **The include graph is a report, not a committed file (#442).**
  `benchmark/include_graph.json` and `docs/INCLUDE_GRAPH.md` are gone, and
  so is their CI check: every pair of pull requests that touched includes
  conflicted over them. `make include-graph` writes both under
  `build/include_graph/`; `make check-layers` enforces the edges that matter.
