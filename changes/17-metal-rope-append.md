### Changed
- **Metal: Llama and Qwen3 attention prep in one dispatch (#17).** A new
  fused op, `attn_rope_append` (`geist_backend_fused`, probed as
  `GEIST_FUSED_ATTN_ROPE_APPEND`), is the Llama/Qwen3 twin of
  `attn_qkv_prep`: optional per-head q/k RMSNorm, half-split RoPE, the q
  scale and the K/V cache append (F16 or F32) in one kernel, where Qwen3
  took six dispatches per layer and Llama four. Qwen3 0.6B goes from 511
  to 371 dispatches per decode token, Llama 3.2 3B from 441 to 357; it
  also runs at prefill. Each step reads the previous one's result back
  from memory as the separate kernels did, so the output is bit-identical
  (parity test, and teacher-forced logits and greedy output of Llama 3.2
  3B, Qwen3 0.6B, Qwen3.5 0.8B and Gemma 4 E2B match main byte for byte).
  `bench_revision_ab` against main on the M1 Max (Metal, 10 cycles, cool
  start, control copy within noise): decode Qwen3 0.6B −5.9 % at 128
  (95 % interval −7.2..−5.4 %) and −6.6 % at 512, prefill −3.5 % / −2.9 %;
  Llama 3.2 3B decode −1.3 % (−2.3..−0.8 %) and −1.9 %; Gemma 4 E2B
  unchanged.
