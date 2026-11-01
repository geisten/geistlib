### Added
- **A mixture-of-experts prefill on a discrete GPU copies the cold experts
  with many rows to the GPU (#239).** In GPU mode a prefill chunk's cold
  experts (no copy in VRAM) ran on the CPU and were the critical path. Each
  now has a host copy on the GPU backend (pinned, as many as a quarter of
  the host memory holds); a prefill step copies those with at least a
  threshold of rows, at most 16, into VRAM staging slots on the transfer
  queue, beside the resident experts' GEMMs, and runs them on the GPU. The
  threshold comes from a cost model measured at load (copy rate, CPU time
  per row): 32 rows for Qwen3.6-35B-A3B, 64 for Qwen3-30B-A3B on an RTX
  2080 Ti + Ryzen 9 9950X. The 2,726-token request prefills at 921 instead
  of 799 tok/s (Qwen3.6-35B-A3B, 1.52x Strata's 605) and 830 instead of 695
  (Qwen3-30B-A3B); decode and perplexity are unchanged; the slots take
  40-61 MiB from the hot experts and the host copies 9-12 GB of host
  memory. Backend API (EXPERIMENTAL): `fused->stage_reserve`,
  `fused->stage_weights` (Vulkan). `GEIST_MOE_STAGE=0` turns it off;
  `GEIST_MOE_STAGE_M_MIN`, `_MAX`, `_HOST` set the threshold, the experts
  per step and the host memory. Metal is unchanged. docs/OFFLOAD.md, "Cold
  experts staged for prefill".
