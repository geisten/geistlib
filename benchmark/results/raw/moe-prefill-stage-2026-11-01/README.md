# #239 raw data: cold experts staged for prefill (RTX 2080 Ti + Ryzen 9 9950X)

All runs under `~/bin/with-desktop-lock` and `/tmp/geist-gpu.lock`, started at
load < 3 (the load during a run is the run's own). Request: 2,726 tokens of
`docs/OFFLOAD.md`, 128 decoded, `--m-max 512`, GPU mode with the expert
profiles of runs 1830 / 1225 (`ab.sh`; `sum.py` prints the table).

| file | code | what |
|:--|:--|:--|
| `ceil-1031-1747.jsonl` | main 6b3b72e1 + a skip switch (not merged) | prefill with the CPU's cold experts skipped: the ceiling (outputs wrong) |
| `sweep1.jsonl`, `sweep2.jsonl`, `sweep3.jsonl` | branch drafts: 32 experts per step, two slot sets | threshold 16 / 32 / 64; hot count; one vs two slot sets |
| `sweep30.jsonl` | branch draft, 32 per step | Qwen3-30B-A3B thresholds |
| `final36.jsonl`, `final30.jsonl` | 4e9f2ea0 (the PR) vs main 6642260d | acceptance, 3 cycles each |
| `ppl.jsonl` | 4e9f2ea0 | `bench_nll`, context 2,047 tokens prefilled in 512-row chunks, 1,023 scored |
| `gate-revision_ab.*` | 6642260d vs 4e9f2ea0 | `bench_revision_ab.py`, dense synthetic Llama-1B Q4_K on Vulkan |
