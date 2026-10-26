### Changed
- **Metal: the decode argmax runs in two passes (#196).** One 256-thread
  threadgroup scanned the whole logits row and was latency-bound: about
  420 us per token for Gemma 4's 262k vocabulary on the M1 Max, the cost
  of reading 150 MB of weights. Now up to 128 threadgroups scan strided
  slices and one more merges their (value, index) pairs. Same index:
  larger value, then lower index; NaN never wins.
