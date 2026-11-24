### Changed
- **On Metal a Gemma 4 prefill runs the KV-shared layers for one row, the
  last (#591).** Since #594 the chunks before the last left those layers
  out, and the last chunk ran them for its last 64 to 191 rows, the
  shortest tail that kept every bit. Now every row of the last chunk stops
  after the last layer that writes a cache, and the last row alone goes on
  through the KV-shared layers as a pass of one row. K/V rows, token
  history and every decode step are bit-identical to before. The logits a
  prefill leaves pending are those of a decode step's kernels instead of a
  chunk's and differ by rounding: Gemma 4 E2B Q4_K_M on the M1 Max, within
  1.7e-3 of the largest logit over the test's prompts, the pending token
  the same in every case. Not on a CPU backend, where a row alone differs
  from the same row in a chunk by 3 to 6 % of the largest logit (the
  decode kernels quantize the activations to 8 bits), not on Vulkan
  (unmeasured) and not with an INT8, INT4 or KIVI cache (5e-3 on Metal):
  there the last chunk runs every layer for every row, without the
  128-row split #594 had on Metal. `GEIST_KV_SHARED_FULL=1` is unchanged.
