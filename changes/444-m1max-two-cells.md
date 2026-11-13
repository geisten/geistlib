### Added
- **Two M1 Max Phase 0 cells at 6642260d recorded, and the cause of the failed ones (#444).**
  `benchmark/c23_baseline.json` now holds gemma4-e2b-q4km on `apple_m1max`
  (prefill 328 tok/s, decode 56.2 tok/s at 512; raw record lost, values from
  the run report) and on `apple_m1max_metal` (1556 / 118.4, measured with
  `cool_each_repeat`). qwen35-0.8b-q8 on the CPU and bonsai2-27b-pq2 on Metal
  still fail the 5 % repeat-spread gate; `benchmark/results/C23_BASELINE.md`
  and `benchmark/results/raw/c23-m1max-2026-11-13/` record why (other
  processes on the workstation for the CPU cell, GPU throttling inside a
  single 27B measurement on the llama.cpp side for the Metal cell).
