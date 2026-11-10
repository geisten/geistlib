### Changed
- **Backend API: `prims->add_bias`, a row bias broadcast in one op (#492).**
  New nullable primitive `y = x + b` with `b` one row added to every row of
  `x`, bit-identical to one `add` per row. gpt-oss added its q/k/v/o
  projection biases one row per `add` call — on Metal one kernel launch per
  row, about 196k launches for a 2048-token prefill (~12 % of its GPU
  time); they now take one dispatch per projection. Metal, Vulkan and the
  CPU backends implement it; a backend without it (or refusing a view with
  `GEIST_E_UNSUPPORTED`) keeps the per-row path.
