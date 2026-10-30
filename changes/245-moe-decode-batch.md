### Added
- **`geist_session_decode_batch` takes mixture-of-experts models (#245):**
  Qwen3-MoE and Qwen3.5/3.6-MoE, up to 8 sessions, where every layer's
  experts route on the device (Metal, all experts resident, no shared
  expert): the n rows route and run their experts there. Elsewhere
  `geist_model_decode_batch_max` stays 0 for an MoE: on the host route the
  rows cost more than single steps (M1 Max, 0.4-0.6x);
  `GEIST_DECODE_BATCH_MOE_HOST=1` batches them anyway. Isolation holds bit
  for bit: each selected expert runs one matvec per session's row. M1 Max,
  Qwen3-30B-A3B: 1.09-1.11x aggregate tokens/s at 8 sessions; the Q4_K/Q6_K
  linears have no multi-row kernel, so each row reads the dense weights
  again.
