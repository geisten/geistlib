### Changed
- **Metal: PQ2_0 prefill GEMM 5-12 % faster per call, Bonsai 2 27B prefill
  7-10 % faster (#103), output bit-identical.** The MMAs of the 64x64-tile
  kernel alone reach ~10 TFLOP/s on the M1 Max; the staging between its
  barriers cost the rest. x now arrives as f16 from one conversion dispatch
  per GEMM (`cvt_f32_f16_rows`, into a grow-only private scratch, 11 MB on
  Bonsai), the weights decode through a 256-entry half4 table, and the next
  k step's codes and x are loaded before the current MMAs. ffn_up at 320
  rows: 7.95 -> 8.4 TFLOP/s. `tests/bench_metal_pq2_gemm` covers every
  Bonsai GEMM shape.
