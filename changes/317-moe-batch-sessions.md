### Added
- **`geist_session_opts.batch_sessions` (EXPERIMENTAL, read at load,
  #317):** the sessions a caller will keep at once, e.g. the n of
  `geist_session_decode_batch`. A mixture-of-experts model on a GPU leaves
  room for that many sessions' buffers next to its hot experts (KV at
  `max_seq_len`, scratch, recurrent and routing state) instead of one.
  RTX 2080 Ti, 2,048 positions: nine sessions of Qwen3.6-35B-A3B /
  Qwen3-30B-A3B fit (about 2,000 / 1,550 hot experts instead of 3,919 /
  2,834),
  where the third or fourth failed with `GEIST_E_OOM` before. 0 or 1 keeps
  the placement as it was. `tests/bench_session_vram` measures what a
  session costs on the device.
