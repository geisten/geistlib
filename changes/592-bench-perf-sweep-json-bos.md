### Fixed
- **`bench_perf_sweep` no longer prints `inf` for a rate, and `--prompt-file`
  prepends BOS (#592).** A decode step that the clock read as 0 ms (one step
  after a one-token prefill does no forward pass) printed
  `"decode_tps":inf`, which is not JSON; every rate over a zero duration is
  now 0. `--prompt-file` fed the file's tokens without the BOS the model
  asks for (`geist_model_add_bos`), so a 512-token cell ran 512 text tokens
  where the other engines ran BOS and 511; it now puts BOS first unless the
  text starts with it, and says so on stderr.
