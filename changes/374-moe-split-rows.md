### Changed
- **MoE decode splits repacked experts over the threads too (#374).** A
  decode step with fewer active experts than threads slices each expert's
  rows over the threads (`moe_split`), but only for weights in the file's
  row-major blocks; cpu_x86's repacked Q4_K (Q4_Kx8) and its Q6_K (with a
  W8A8 prefill blob) were refused, so DeepSeek-V2-Lite's 6 experts ran on 6
  of 16 threads. `geist_weight` gains an optional row-range M=1 entry,
  `linear_m1_rows` with its granularity `linear_m1_rows_align`
  (EXPERIMENTAL), which cpu_x86 installs for those layouts; `moe_split`
  uses it where a row view is not possible. Each output element is the
  same dot product as before: logits bit-identical (`test_moe_rows_unit`).
