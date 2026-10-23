### Changed
- **Metal: a pass of 2–8 rows through Bonsai 2 27B costs ~40 % less
  (#192).** Speculative verify and short prefill tails run a few rows
  through every layer. Three costs dominated, measured per kernel on the
  M1 Max:
  - `matmul_pq2_m8` read its 8×8 activation tiles as strided transposed
    loads. It now reads f16 activations laid out k-major
    (`cvt_f32_f16_kmaj8`), one contiguous 128-byte load per tile, and
    loads a block's code words before its MMAs: 90 → 58–62 ms of GPU
    time per pass.
  - The DeltaNet beta/alpha F32 linears (48 outputs) took the 8×8 GEMM,
    i.e. 6 threadgroups, at 2–8 rows. They now take the reduction kernel:
    28 → 1.2 ms per pass.
  - DeltaNet with up to 8 rows padded every call to a 64-row chunk. It now
    runs `dn_dec_qk` per row and one `dn_dec_v_rows` dispatch that keeps S
    in registers across the rows: 15.5 → 5 ms at 2 rows.

  Interleaved A/B against main (609-token prompt, wall per pass) is in the
  pull request. Tried and dropped: four output tiles per simdgroup, codes
  decoded through the f16 mantissa, two accumulators per tile, and a
  multi-row PQ2_0 GEMV.
