### Changed
- **Metal: a greedy decode step is committed without waiting for the one
  before it (#590).** The forward of the token `geist_session_decode_step`
  returns is on the GPU before the call returns, and the next one is queued
  behind it with its input token read on the GPU, so the GPU no longer idles
  between two steps: Qwen3.5 0.8B Q8_0 decodes 3.0 % faster on a Mac mini
  M6 (155.8 to 160.6 tok/s, mlx-lm 160.0-160.3) and 8.1 % faster on an M1
  Max, Llama 3.2 3B Q4_K_M 1.2 % and 1.7 %. A generation that stops costs
  one forward in flight. Outputs, snapshots and every other call are those of a
  session that waits for each step (`tests/test_decode_ahead_unit.c`);
  sessions that sample, use repetition penalties, a quantized KV cache,
  Gemma 4's per-layer embeddings or a mixture of experts decode as before,
  and so does every session with `GEIST_DECODE_AHEAD=0`
  ([BACKENDS.md](docs/BACKENDS.md#metal-apple-gpu-experimental)). Backends
  get four optional `geist_backend_fused` slots for this
  (`argmax_f32_ids`, `embedding_lookup_scaled_ids`, `region_end_async`,
  `async_wait`) and `geist_deltanet_mix_args` two optional out-state views.
