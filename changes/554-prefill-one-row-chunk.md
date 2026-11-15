### Fixed
- **A prompt one token longer than a prefill chunk no longer gets other
  logits than the same prompt in one chunk (#554).** A chunk of one row is
  a decode step to every stage of the forward (`seq == 1`): the M=1 kernels
  and fusions, the dense linears included, which sum in another order than
  a chunk's M>1 GEMMs. A prompt of `m_max + 1` tokens split `m_max + 1`,
  so its last token's bits depended on where the chunk boundary fell.
  `transformer_prefill_chunk` now takes one row less when exactly one would
  be left: the last two chunks are `(m_max - 1) + 2` rows, the number of
  chunks stays `ceil(n / m_max)`. No decode kernel and no dispatch changed,
  so a decode step runs what it ran. Dense models needed a second change
  for that split to be invariant: `cpu_neon`'s Q6_K stream prefill kernel
  (a dense `ffn_down`, `n_in >= 4096`) was not among the expert shapes of
  #527 and still spelled its float step three ways, so every chunk whose
  row count was not a multiple of 4 gave its tail rows other bits (Llama
  3.2 3B's Q6_K 8192->3072: 96 of 128 row counts). It now computes
  `fma(ds * sx, dot, acc)` in the 8-token tile, the 4-token tile and the
  tail (`kq_epi_dsx`); the logits of a model with such a tensor move by
  rounding once. On the M1 Max (`cpu_neon`), 65 tokens at `m_max` 64, 16,
  7, 3 and 97 tokens at 96, 32, 5 against one chunk: bit-identical for
  Llama 3.2 3B, Gemma 4 E2B, Qwen3 0.6B, BitNet b1.58 large,
  DeepSeek-V2-Lite, Granite 3.0 1B-A400M, OLMoE 1B-7B and gpt-oss-20b;
  on main the same prompts differed by a relative L2 of up to 6.7 % (all
  but Qwen3 0.6B). Qwen3.5 0.8B still differs (1.3-1.8 %, as on main): the
  chunked DeltaNet rule depends on the chunk. Left as they are: a prefill
  call of one token, which stays a decode step (teacher forcing and the
  tools feed tokens that way), `m_max` below 3 (2 + 1 has no other split),
  and the soft-token (audio, image) prefill. The joined prompt of a
  decision prefix on a chunk boundary and a one-token suffix therefore ends
  in passes of `chunk - 1` and 2 rows, not in the prefix's passes plus one
  (`geist_decision.h`, `docs/DECISION.md`). `test_prefill_chunk_unit`
  checks the plan for `m_max` 1..9, 16, 64, 128 at every prompt length up
  to `3 * m_max + 2`, and on each CPU backend that one call of
  `m_max + 1` and `2 * m_max + 1` tokens has the bits of `prefill(n - 2)`,
  `prefill(2)`; `test_moe_row_invariant_unit` gets the Q6_K 8192->3072
  shape. Both fail on `cpu_neon` without the change and pass on `cpu_x86`
  (emulated amd64, gcc-14, AVX2).
