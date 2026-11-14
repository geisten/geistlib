### Changed
- **The M1 Max Metal 27B benchmark cell has its own gate, and the Mac power gate checks the battery charge (#444).**
  A host profile may refine its gate for one model (`model_host_gate`).
  `apple_m1max_metal` uses it for bonsai2-27b-pq2: start at 55 °C or cooler
  and a repeat spread of up to 15 % instead of 5 %. llama.cpp throttles the
  GPU inside one 512- or 1024-token measurement of this model, so its repeats
  spread 10–20 % on a quiet host while geist's stay within 0.2 %; the cell
  failed the 5 % limit in 8 tries. It is now recorded in
  `benchmark/c23_baseline.json` (prefill 162 tok/s, decode 21.2 tok/s at 512).
  The power gate also waits while a laptop's battery is below 50 %: on a 30 W
  adapter at 7–10 % charge the GPU clock fell from 1293 to 388 MHz mid-run.
