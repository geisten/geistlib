### Changed
- **The M1 Max Metal benchmark cells cool the GPU before every repeat (#444).**
  `apple_m1max_metal` in `benchmark/cross_engine_gpu_protocol.json` sets
  `cool_each_repeat`, so each sequence length and repeat is its own engine call
  behind the 65 °C gate, as on the M6 CPU profile. One back-to-back call took
  the GPU to 75–82 °C and the clock left 1293 MHz at 78–81 °C: twelve
  512-token prefills in one call measured 1559–1571 tok/s nine times, then
  1092, 1057 and 994; one call per repeat measured 1560–1567 tok/s in 13 of 14
  calls. Numbers of these cells from before this change were taken partly on a
  throttled clock and are not comparable repeat by repeat.
