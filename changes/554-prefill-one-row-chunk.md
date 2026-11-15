### Fixed
- **A prompt one token longer than a prefill chunk no longer gets other
  logits than the same prompt in one chunk (#554).** A chunk of one row is
  a decode step to every stage of the forward (`seq == 1`): the M=1 kernels
  and fusions, the dense linears included, which sum in another order than
  a chunk's M>1 GEMMs. A prompt of `m_max + 1` tokens split `m_max + 1`,
  so its last token's bits depended on where the chunk boundary fell.
  `transformer_prefill_chunk` now takes fewer rows when exactly one would
  be left, so that the last chunk has 8 rows (a whole token tile of the
  prefill kernels; 4 below `m_max` 16, 2 below 8); the number of chunks
  stays `ceil(n / m_max)`. No decode kernel and no dispatch changed, so a
  decode step runs what it ran. Dense models needed a second change for
  that split to be invariant: `cpu_neon`'s Q6_K stream prefill kernel (a
  dense `ffn_down`, `n_in >= 4096`) was not among the expert shapes of
  #527 and still spelled its float step three ways, so every chunk whose
  row count was not a multiple of 4 gave its tail rows other bits (Llama
  3.2 3B's Q6_K 8192->3072: 96 of 128 row counts). It now computes
  `fma(ds * sx, dot, acc)` in the 8-token tile, the 4-token tile and the
  tail (`kq_epi_dsx`); the logits of a model with such a tensor move by
  rounding once. On the M1 Max (`cpu_neon`), 65 tokens at `m_max` 64, 16,
  8, 7, 4, 3 and 97 tokens at 96, 32, 12, 6, 5 against one chunk:
  bit-identical for Llama 3.2 3B, Gemma 4 E2B, Qwen3 0.6B, BitNet b1.58
  large, DeepSeek-V2-Lite, Granite 3.0 1B-A400M, OLMoE 1B-7B and
  gpt-oss-20b; on main, 65 tokens at 64 and 97 at 32 differed for all but
  Qwen3 0.6B, by a relative L2 of up to 6.7 %. Qwen3.5 0.8B still differs
  (1.5-2.1 %; 1.4-1.5 % on main): the chunked DeltaNet rule depends on the
  chunk. The cost is the prompt whose length is 1 modulo the chunk:
  `tools/bench_revision_ab.py` against main on the M1 Max (`cpu_neon`, AC
  power, 10 cycles, control within +-4.6 %), Llama 3.2 3B: a 65-token
  prefill 265.7 -> 284.5 ms, +6.9 % [+6.4, +7.4] (57 + 8 rows where main
  ran 64 rows and a decode step); prefill of 128 and 512 tokens and 64
  decoded tokens after each within the control's noise. Left as they are:
  a prefill call of one token, which stays a decode step (teacher forcing
  and the tools feed tokens that way), `m_max` below 3 (2 + 1 has no other
  split), and the soft-token (audio, image) prefill. The joined prompt of
  a decision prefix on a chunk boundary and a one-token suffix therefore
  moves rows of the prefix's last pass into the final one, where it had
  the prefix's passes plus one (`geist_decision.h`, `docs/DECISION.md`).
  `test_prefill_chunk_unit` checks the plan for `m_max` 1..9, 16, 64, 128
  at every prompt length up to `3 * m_max + 2`, and on each CPU backend
  that one call of `m_max + 1` and `2 * m_max + 1` tokens has the bits of
  `prefill(n - t)`, `prefill(t)` for the last chunk's t rows;
  `test_moe_row_invariant_unit` gets the Q6_K 8192->3072 shape. Both fail
  on `cpu_neon` without the change and pass on `cpu_x86` (emulated amd64,
  gcc-14, AVX2; no x86 host was run).
