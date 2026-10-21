### Fixed
- **Offload no longer attaches twins with no layers (#92).** When the split
  stage holds every layer (`split_layers` = the layer count, e.g. what
  `GEIST_SPLIT_AUTO` picks on unified memory or when the model fits the
  card), `GEIST_OFFLOAD_M_MIN` (or the discrete-GPU default) still loaded a
  third GGUF handle and an empty twin model, and every session's twin IO
  buffer was then refused ("offload off for this session ... invalid buffer
  create request"). Offload is now off in that case; nothing else changes.
