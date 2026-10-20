### Fixed
- **cpu_neon: the same model gives the same logits however its weights
  are mapped (#76).** The decode-time F32 linear (Gemma 4's per-layer
  `inp_gate`/`proj`) ran Accelerate's `cblas_sgemv`, which rounds
  differently with the matrix's address, so mmap-aliased and copied weights
  (`GEIST_WEIGHT_MMAP=0`) diverged (Gemma 4 E2B: last-step logits 0.77 apart
  after 24 greedy tokens). It is now a fixed-order NEON row dot: bit-identical
  in both modes, `test_awq_int` passes on cpu_neon. Decode within noise
  (bench_revision_ab, M1 Max, -7.9 % median, control +-22 %).
