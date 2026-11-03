### Added
- **`make check-static` (#342).** clang-tidy over the library, no models and
  no GPU, in about 15 seconds. A required CI job on the macOS arm64 sources.
  `scripts/static-baseline.txt` lists the known findings: a new finding fails,
  and a fixed one must leave the list, so it only shrinks. A `NOLINT` needs
  its reason on the same line. x86_64 and Vulkan sources are not analysed yet.
