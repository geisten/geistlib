### Added
- **`geist_session_decode_batch` takes mixture-of-experts models (#245):**
  Qwen3-MoE and Qwen3.5/3.6-MoE, up to 8 sessions, where it pays:
  - on the device route (Metal, every expert resident, no shared expert)
    the n rows route and run their experts there; M1 Max, Qwen3-30B-A3B:
    1.09-1.11x aggregate tokens/s at 8 sessions (the Q4_K/Q6_K linears have
    no multi-row kernel yet, #316);
  - with the experts split between a discrete GPU and the CPU (Vulkan) the
    host routes the rows; RTX 2080 Ti: Qwen3.6-35B-A3B 1.56-1.61x,
    Qwen3-30B-A3B 1.30-1.34x at 8 sessions.
  The host route on a unified-memory GPU costs more than single steps and
  stays off (`GEIST_DECODE_BATCH_MOE_HOST=1` batches anyway). Isolation holds
  bit for bit: each selected expert runs one matvec per session's row.
