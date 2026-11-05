### Changed
- **Metal: the DeltaNet state update of a speculative verify (2–8 rows)
  runs split by value column, ~2× faster per layer on Bonsai 2 27B
  (#469).** `dn_dec_v_rows` ran one 512-thread threadgroup per v-head with
  32 rows of S per thread in registers. Only one fits a core, so Bonsai's
  48 heads on the M1 Max's 32 cores ran the serial row chain twice (#266).
  The new `dn_dec_v_cs` takes the layout of Vulkan's `deltanet_scan_f32`
  (#467): 8 lanes per column, 16 rows each, shuffle reductions with no
  threadgroup barrier in the recurrence, 128-thread threadgroups. The
  gated RMS norm runs afterwards in `dn_dec_vnorm`. `dn_dec_v_rows` stays
  for conv kernels other than 4.

  `tests/bench_metal_deltanet` (q/k prep + state update, 48 calls per
  sample, M1 Max): Bonsai shape at 7 rows 0.325 → 0.156 ms, at 2 rows
  0.116 → 0.074; Qwen3.5 4B shape at 7 rows 0.207 → 0.155. With 48
  separate layer states and the rollback slots written: 0.384 → 0.217 ms
  at 7 rows. Bonsai 2 27B verify of 7 rows (6 rejected drafts, wall per
  call, 7 alternating rounds): −5.5 % median, faster in 6 of 7 rounds,
  tokens identical. Qwen3.5 4B: tokens identical, no difference
  measurable at that noise level. The chunk parity test now covers a
  7-row call and checks the rollback slots against shorter runs.
