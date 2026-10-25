### Removed
- **`struct geist_backend` loses the engine-owned `has_opts` field
  (EXPERIMENTAL layout).** `geist_backend_create_auto_cpu` now always hands
  the CPU backend the stored `opts`, zeroed when the GPU backend was
  created with nullptr; every backend reads zeroed opts as it reads
  nullptr (only cpu_x86 reads one field, `max_threads`, where 0 means the
  default).
