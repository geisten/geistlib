### Fixed
- **A prompt one token longer than a prefill chunk no longer gets other
  logits than the same prompt in one chunk (#554).** A chunk of one row is
  a decode step to every stage of the forward (`seq == 1`): the M=1 kernels
  and fusions, the dense linears included, which sum in another order than
  a chunk's M>1 GEMMs. A prompt of `m_max + 1` tokens split `m_max + 1`,
  so its last token's bits depended on where the chunk boundary fell.
  `transformer_prefill_chunk` now takes one row less when exactly one would
  be left: the last two chunks are `(m_max - 1) + 2` rows, the number of
  chunks stays `ceil(n / m_max)`. No kernel and no dispatch changed, so a
  decode step runs what it ran. Left as they are: a prefill call of one
  token, which stays a decode step (teacher forcing and the tools feed
  tokens that way), `m_max` below 3 (2 + 1 has no other split), and the
  soft-token (audio, image) prefill. The joined prompt of a decision
  prefix on a chunk boundary and a one-token suffix therefore ends in
  passes of `chunk - 1` and 2 rows, not in the prefix's passes plus one
  (`geist_decision.h`, `docs/DECISION.md`). `test_prefill_chunk_unit`
  checks the plan for `m_max` 1..9, 16, 64, 128 at every prompt length up
  to `3 * m_max + 2`, and on each CPU backend that one call of
  `m_max + 1` and `2 * m_max + 1` tokens has the bits of `prefill(n - 2)`,
  `prefill(2)`; it fails on `cpu_neon` without the change.
