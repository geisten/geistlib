### Changed
- **The sampler moved from `src/engine/` to its own layer, `src/sampling/`
  (#441).** `sampler.{c,h}` and `sampler_penalties.c` sit below the archs
  now, so the transformer's forward head samples without reaching up into
  the engine; `tools/layers.json` lets both the transformer and the engine
  use the new layer, and its 21 entries left `tools/layers_baseline.txt`.
  A pure move: the same code, the same per-file float rules
  (`-fno-fast-math -ffp-contract=off` on the penalties, `-Wconversion`), and
  object code identical apart from paths. Internal headers only; the public
  API is unchanged.
