### Added
- **A mixture of experts learns its expert profile at run time (#165).**
  The router's selections per layer and expert are counted while sessions
  run and kept with the model: when it is unloaded, they are added to the
  counts in `$XDG_CACHE_HOME/geist/moe-profile/<key>.txt` (or
  `~/.cache/geist/...`; `GEIST_MOE_PROFILE_CACHE=<dir>` moves it, `0` turns
  it off). The next load gives the GPU copies to the experts counted most,
  as `GEIST_MOE_PROFILE` does with a hand-made file (which still wins); the
  first load places the same share of every layer, as before. The key is a
  hash of the GGUF's tensor directory and layer 0's router, so another
  quantization or fine-tune gets a file of its own; the file is the USAGE
  format `tools/moe_usage/analyze.py` reads. A file made for another model
  or one that does not parse is ignored with one line on stderr.
