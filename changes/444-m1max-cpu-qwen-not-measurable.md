### Changed
- **The M1 Max CPU cell of qwen35-0.8b-q8 is recorded as not measurable in the Phase 0 baseline (#444).**
  Six tries failed the 5 % repeat-spread gate while other processes of the
  shared desktop filled the efficiency cores. `benchmark/c23_baseline.json`
  and `benchmark/results/C23_BASELINE.md` say so; the model's Apple CPU value
  is the `apple_m6_mini` cell.
