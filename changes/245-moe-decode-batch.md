### Added
- **`geist_session_decode_batch` takes mixture-of-experts models (#245):**
  Qwen3-MoE (Qwen3-30B-A3B) and Qwen3.5/3.6-MoE (Qwen3.6-35B-A3B), up to 8
  sessions. The isolation guarantee holds bit for bit: each selected expert
  runs one matvec per session's row, never a GEMM shared with another
  session. On Metal the rows route and run their experts on the device
  (`GEIST_DECODE_BATCH_MOE_DEVICE=0`: on the host). M1 Max, Qwen3-30B-A3B:
  1.09-1.11x aggregate tokens/s at 8 sessions; the Q4_K/Q6_K linears have no
  2-8-row kernel yet, so each row reads the dense weights again.
