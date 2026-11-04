### Changed
- **`geist_backend_set_error` checks its arguments against the format
  (#343).** The declaration in `geist_backend.h` carries
  `[[gnu::format(printf, 3, 4)]]`, as does the internal
  `geist_error_set_create_time`, and the tree builds with `-Wformat=2`
  under `-Werror`: a format string that is not a literal, or an argument
  that does not match its conversion, fails the build. The 351 existing
  calls already matched. No ABI change.
