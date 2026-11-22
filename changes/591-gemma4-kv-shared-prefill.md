### Changed
- **A Gemma 4 prefill no longer runs the KV-shared layers for rows nobody
  reads (#591).** Gemma 4's last layers (20 of E2B's 35, 18 of E4B's 42)
  read an earlier layer's KV cache, project only q and keep nothing per
  position, and a prefill reads one row's output, the last one's. Every
  row ran every layer: on the M1 Max a 512-token prefill of E2B spent 67 %
  (Metal) and 68 % (`cpu_neon`) of its layer time in those layers, E4B
  42 %. Now the rows of every prefill chunk before the last stop after the
  last layer that writes a cache, and on a GPU backend the first rows of
  the last chunk do too, where it can be split at a multiple of 128 rows
  with at least 64 left. Logits, decode and the session state are
  bit-identical to before on every backend: whole chunks run the calls
  they ran, and the split keeps every row in the kernel tile and path it
  had. A shorter tail would not (Metal: 16 of 256 rows kept the bits, 24
  did not; the CPU backends' dense F32 linears go through the BLAS, which
  promises no row-count invariance), so a CPU backend does not split the
  last chunk. Not for the MTP drafter, mean pooling, a verify pass (they
  read every row), a layer split or offload; KIVI does not split its last
  chunk. `GEIST_KV_SHARED_FULL=1` runs every layer for every row as
  before. `tools/bench_revision_ab.py` against main, Gemma 4 E2B Q4_K_M,
  prefill ms: Mac mini M6 Metal (run 4048, 10 cycles, control ±0.3 %) 512
  tokens 188.9 -> 99.2 (2710 -> 5161 tok/s), 2048 tokens 786.0 -> 305.3;
  M6 `cpu_neon` (run 4050) 128 tokens 365.2 -> 242.2, 512 tokens 1563.3 ->
  644.7, 2048 tokens 7360.9 -> 2561.7; M1 Max Metal 512 tokens 336.2 ->
  184.8, 2048 tokens 1381.3 -> 534.4; M1 Max `cpu_neon` (6 cycles) 512
  tokens 1638.1 -> 688.7, 2048 tokens 7579.6 -> 2719.2. Prefills of 16
  and 64 tokens and the decode after each length are within the control's
  noise in every run. `GEIST_NLL_PREFILL=1 bench_nll` over 1024
  tokens gives the same NLL before and after (Metal 7.445414, `cpu_neon`
  7.445627). `test_kv_shared_prefill_int` (a Gemma 4 GGUF) and
  `test_kv_shared_prefill_unit` (a Gemma 4 built in memory, every CPU
  backend) compare a session with and one without the switch bit for bit
  after every call.
