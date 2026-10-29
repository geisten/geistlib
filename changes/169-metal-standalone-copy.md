### Fixed
- **Metal: a buffer copy outside the open command sequence waited for the
  work it depends on.** A copy that cannot go on the sequence's compute
  encoder ran on a command buffer of its own, ahead of the uncommitted
  sequence, and read a source the sequence had not written yet. It now
  flushes the sequence first when the sequence touches either buffer. The
  MoE path's copy of the normalised activations met it in a chunked
  prefill (cosine 0.976 against the CPU on the qwen35moe fixture).
