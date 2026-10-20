### Changed
- **Metal DeltaNet prefill 4x faster per layer (#99).** The chunked
  delta rule (qwen35: Bonsai 2 27B, Qwen3.5) runs as seven dispatches per
  call on simdgroup MMAs, with only the state recurrence in chunk order,
  instead of 12 scalar dispatches per 64-token sub-chunk. One Bonsai layer
  at 320 / 512 tokens: 5.26 / 8.39 -> 1.28 / 1.98 ms (llama.cpp's recurrent
  kernel: 2.20 / 3.91 ms for the scan alone); Qwen3.5 4B 4.00 -> 0.83 ms.
  DeltaNet GPU time in a Bonsai prefill falls from ~1.0 to 0.25 ms per
  token. Not bit-identical (summation order): z within 1e-5 of a double
  oracle like the old kernels, greedy output and the 231-item decision
  check against llama.cpp unchanged. Shapes outside the tiling (head sizes
  not multiples of 32, conv kernel != 4) use the serial mixer.
