### Added
- **`docs/C23_REVIEW.md`, the outcome of the C23 review (#333).** What now
  fails the build or a required check, the bugs those checks found, the
  metrics against the Phase 0 baseline, and what was rejected and why.
  AGENT.md gains the rules that came out of it: opaque handles instead of
  `void *`, one spelling per concept, the layer table, family and backend
  wiring through the required-slot macros, `make check-static` before a
  push, and the current warning flags in the local gcc command.
