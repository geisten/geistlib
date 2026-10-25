### Added
- **The decoder arch vtable gains `decode_batch` / `decode_batch_max`, and the
  transformer implements them for Qwen3.5-family hybrids on GPU backends
  (#26).** One call runs the forwards that n independent sessions of a model
  owe as one batched pass:
  - every linear and the lm_head read their weights once for the n rows;
  - RoPE, the KV append, attention and the fused DeltaNet mixer run per
    session, on its own cache, recurrent state and position;
  - each session samples with its own sampler.

  n = 1 is `decode_step`. For a given n, a session's tokens and logits are
  bit-identical whichever sessions share the call and in whichever slot.
  Refusals change nothing; a failure during the pass resets every member.
  The slots are appended to the experimental vtable without moving existing
  fields. The disassembly of every pre-existing function in the touched
  objects is unchanged. Measured on Bonsai 2 27B on the M1 Max (#26 prototype):
  8 sessions decode at 3.47x the aggregate tokens/s of one.
  `tests/test_decode_batch_unit.c` covers isolation, n = 1 identity,
  agreement and the refusals on every backend in the build. The public entry
  points follow in the next change.
