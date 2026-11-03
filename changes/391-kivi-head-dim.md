### Fixed
- **KIVI refuses a `head_dim` that is not a multiple of 4 (#391).** A model
  with an even `head_dim` such as 6 or 150 loaded, and a KIVI session then
  read past each row and scored uninitialized floats. Such a session now fails
  at creation and names the reason. Only explicit `GEIST_KV_KIVI` reached this.
