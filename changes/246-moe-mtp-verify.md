### Fixed
- **Speculative decoding on an MoE with experts split across GPU and CPU no
  longer runs 1.25-7x slower than it must (#246).** A verify entered the
  thread region of the model's backend only, so the CPU experts ran outside
  their backend's region; it now enters it on both, as prefill does. MTP
  draft steps run in the decode regime, their head takes the device argmax
  instead of reading the logits row over the bus (~30 ms a token on an RTX
  2080 Ti), a partial accept on a model with an MTP block restores the
  per-row DeltaNet states (#192) instead of replaying the accepted rows, and
  a verify's experts run as decode steps'. Qwen3.6-35B-A3B on the RTX 2080
  Ti: MTP drafting (K = 2) 28 → 67 tok/s, n-gram drafting (K = 3) 53 → 66
  tok/s; plain decode unchanged (79-80). MTP still does not beat decoding
  without it on this model (79.8 tok/s; a verify row costs about a decode
  step there), so it stays off by default; see `docs/OFFLOAD.md`.
  `bench_perf_sweep` gains `--spec K`, `--emit-tokens` and `--emit-margins`.
