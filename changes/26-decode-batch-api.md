### Added
- **`geist_session_decode_batch` and `geist_model_decode_batch_max`
  (EXPERIMENTAL) decode several independent sessions of one model in one pass
  (#26).** The call is `decode_step` for each of n sessions. Every linear layer
  reads its weights once for all n rows; attention and recurrent state stay
  per session.
  - For a given n, a session's tokens and logits do not depend on its
    companions or its slot, bit for bit. n = 1 is `decode_step`.
  - Refusals leave every member untouched.
  - On Bonsai 2 27B (M1 Max, Metal), 8 sessions decode at 3.47x the aggregate
    tokens/s of one; each session's step takes about 2.3x as long.
  - Qwen3.5-family hybrids on Metal and Vulkan, up to 8 sessions;
    `geist_model_decode_batch_max` is 0 elsewhere.

  `tests/bench_decode_batch.c` measures isolation, agreement with
  `decode_step` and throughput on a real GGUF. `docs/API_CONTRACT.md` lists
  the semantics.
