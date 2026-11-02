### Added
- **A mixture-of-experts prefill on a discrete GPU copies the cold experts
  with many rows to the GPU (#239).** In GPU mode a prefill chunk's cold
  experts (no copy in VRAM) ran on the CPU and were the critical path. The
  profile's hottest cold experts now get a pinned host copy on the GPU
  backend (4 GiB by default, or a quarter of the host memory if less); a
  prefill step copies those with at least a threshold of rows, at most 16,
  into VRAM staging slots on the transfer queue, beside the resident
  experts' GEMMs, and runs them on the GPU. The threshold comes from a cost
  model measured at load (copy rate, CPU time per row): 32 rows for
  Qwen3.6-35B-A3B, 32-64 for Qwen3-30B-A3B on an RTX 2080 Ti + Ryzen 9
  9950X. The 2,726-token request prefills 8 % faster on Qwen3.6-35B-A3B
  (861 against 796 tok/s, 1.42x Strata's 605; +15 % with every cold expert
  copied, 11.7 GB) and 20 % faster on Qwen3-30B-A3B (812 against 679);
  decode and perplexity are unchanged; the slots (40-61 MiB) come out of
  the planner's VRAM headroom, not the hot experts. Backend API
  (EXPERIMENTAL): `fused->stage_reserve`, `fused->stage_weights` (Vulkan).
  `GEIST_MOE_STAGE=0` turns it off; `GEIST_MOE_STAGE_HOST` sets the host
  memory, `GEIST_MOE_STAGE_M_MIN` and `_MAX` the threshold and the experts
  per step. Metal is unchanged. docs/OFFLOAD.md, "Cold experts staged for
  prefill".
