### Changed
- **On a CPU backend a Gemma 4 prefill runs the KV-shared layers for the
  last eight rows only, bit for bit the full prefill (#591).** Since #594
  the chunks before the last left those layers out (20 of E2B's 35, 18 of
  E4B's 42); the last chunk still ran them for every row, because its last
  row as a pass of one was up to 8 % of the largest logit away from the
  same row in its chunk. The cause is not the decode kernels' 8-bit
  activations as such, the prefill kernels have them too: a pass of one row
  sums in another order (M=1 kernels and fusions), which is rounding, and
  the Q4_K and Q6_K kernels round each activation to 8 bits, so a value the
  two orders put on either side of a step leaves on another grid point
  (Gemma 4 E2B Q4_K_M, `cpu_neon`, 25 prompt lengths: within 5e-7 at 7 of
  them, 2 to 8 % at 18). Now every row of the last chunk stops after the
  last layer that writes a cache, and its last eight rows go on through the
  KV-shared layers as a pass of eight: a prefill pass, on the chunk's
  kernels, which give a row the same bits whatever the rows around it.
  Eight, a whole token tile of the prefill kernels, where two were enough
  on what was measured: a few rows sit under thresholds of their own
  (`cpu_neon`'s F16-cache attention changes kernel below 16 query-head
  pairs, Accelerate's `sgemm` rounds by the row count below about 16000
  weights). A last chunk shorter than 16 rows runs whole. Pending logits,
  K/V rows, token history and every decode step are bit-identical to
  `GEIST_KV_SHARED_FULL=1` and to main before this, with each KV cache
  kind: Gemma 4 E2B and E4B Q4_K_M on the M1 Max (`cpu_neon`, 18 prompt
  lengths from 15 to 2048; `cpu_scalar`), E2B on the Pi 5 (gcc 14,
  OpenBLAS); `cpu_x86` is held to it by `test_kv_shared_prefill_unit`,
  which now also runs a Gemma 4 with Q4_K matrices built in memory. No
  kernel changed. Prefill ms, `tools/bench_revision_ab.py` against main,
  M1 Max `cpu_neon` (8 threads; control within 2.1 %): E2B 64 tokens 188.1
  -> 90.1, 512 tokens 671.2 -> 554.9, 2048 tokens 2653.5 -> 2524.9 (inside
  the noise); E4B 64 tokens 398.2 -> 259.2, 512 tokens 2071.0 -> 1945.4,
  2048 tokens 8952.9 -> 8845.1 (inside the noise). Pi 5, E2B, five
  alternating runs each: 64 tokens 1950.6 -> 934.7, 512 tokens 7042.5 ->
  5911.5. Decode after each length is within the control's noise.
  On Metal the KIVI cache now takes the one-row pass that FP32 and F16
  have had since #598 (pending logits within 1.0e-3 of the largest, the
  same pending token; M1 Max, E2B Q4_K_M; prefill of 512 tokens 3566.6 ->
  1680.4 ms, its attention being a host path). INT8 and INT4 stay on the full
  last chunk there: a row alone measured up to 8.6e-3, above the 5e-3 the
  one-row pass is held to. Vulkan is unmeasured and unchanged.
