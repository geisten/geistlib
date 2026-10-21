### Changed
- **cpu_neon: the paired Q4_K decode GEMV splits both weights' rows
  evenly (#16).** It gave each thread a range of the taller weight's rows
  and ran both weights on it, so with Llama's q (3072 rows) and k (1024)
  the threads holding the first 1024 indices did twice the work of the
  rest. It now splits the joint row space. Logits byte-identical.
