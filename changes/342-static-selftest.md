### Changed
- **`make check-static` fails on a defect added to a fixture without its mark
  (#342).** The self-test compared only the expected findings in
  `tests/static/`, so a null dereference added there without an
  `// expect:` line passed. Now the findings must be exactly the marks.
  `check-static` is the umbrella target and clang-tidy runs as
  `check-static-tidy`, so another gate adds one prerequisite line. gcc
  `-fanalyzer`, cppcheck and clang `-Wthread-safety` were measured and not
  adopted; the numbers are in `docs/CI_COVERAGE.md`.
