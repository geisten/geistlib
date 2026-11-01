### Added
- **`geist_session_opts.batch_m_max` (EXPERIMENTAL, read at load, #317):**
  the m_max the batch's further sessions are created with. The MoE expert
  budget sizes their buffers for it, so decode-sized batch members leave
  more hot experts on the GPU. RTX 2080 Ti, 9 sessions at 2,048 positions,
  members at m_max 8: Qwen3.6-35B-A3B 1,840 -> 2,785 hot experts, 8-session
  decode 109 -> 121 tok/s; Qwen3-30B-A3B 1,605 -> 2,141, 71 -> 83 tok/s.
  Such a member still prefills, in chunks of its m_max: 190 tok/s at 8 and
  430 at 64 against 850 at the default 512 (Qwen3.6).
