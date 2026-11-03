### Changed
- **An unannotated fall-through between `case` labels fails the build on clang
  too (#343).** `-Wimplicit-fallthrough` is on; gcc already had it through
  `-Wextra`. An intended fall-through is written `[[fallthrough]];`.
