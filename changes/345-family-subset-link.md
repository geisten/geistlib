### Fixed
- **`make FAMILIES=<subset>` links again (#345).** Since #245 the batched
  decode step compared the session's family against the Qwen3.5 and
  Qwen3-MoE tables by address, so any selection without both `qwen35` and
  `qwen3moe` left `transformer_family_qwen35`, `_qwen35moe` or `_qwen3moe`
  undefined and every program linking `libgeist.a` failed. The family now
  says it with a capability, `decode_batch`; the behaviour of the full build
  is unchanged.
