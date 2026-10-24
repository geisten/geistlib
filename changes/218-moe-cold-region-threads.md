### Fixed
- **A mixture of experts on a GPU no longer stalls decode steps for
  20-380 ms (#218).** The cold experts' OpenMP region ran on the ambient
  team, one thread per logical CPU, because only the model backend (the
  GPU) entered its thread regime. Prefill and decode now also enter the
  regime of the CPU backend that runs the cold experts (cpu_x86: one
  thread per physical core). Qwen3-30B-A3B on the RTX 2080 Ti + 9950X,
  quiet host: p99 step 51-112 ms -> 21-23 ms, median 14.7 -> 14.1 ms,
  decode 56-61 -> 68-70 tok/s, prefill 408-423 -> 525-530 tok/s.
