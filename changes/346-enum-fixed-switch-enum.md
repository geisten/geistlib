### Changed
- **A new enum value fails the build at every switch over it (#346).** Every
  named enum has a fixed underlying type with its size asserted, and
  `-Wswitch-enum` runs under `-Werror`. Width, `sizeof` of the public structs
  and the ABI are unchanged.
