### Added
- **A mixture of experts can learn its expert profile at run time (#165,
  opt-in).** The router's selections per layer and expert are counted while
  sessions run. With `GEIST_MOE_PROFILE_CACHE=1` (or `=<dir>`) they are
  kept with the model: added to `$XDG_CACHE_HOME/geist/moe-profile/<key>.txt`
  (or `~/.cache/geist/...`) when it is unloaded, and the next load gives the
  GPU copies to the experts counted most, as `GEIST_MOE_PROFILE` does with
  a hand-made file (which still wins). Off by default: with the model on a
  discrete GPU a profile trades prefill for decode (Qwen3.6-35B-A3B on an
  RTX 2080 Ti: decode ~16 % faster, prefill ~10 % slower). The key hashes
  the GGUF's tensor directory and layer 0's router, so another quantization
  or fine-tune gets a file of its own; the file is the USAGE format
  `tools/moe_usage/analyze.py` reads. A file made for another model or one
  that does not parse is ignored with one line on stderr.
