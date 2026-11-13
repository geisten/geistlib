# M1 Max Phase 0 cells at 6642260d, fourth attempt (#444)

2026-11-11 to 2026-11-13. geist at 6642260d; one `with-mac-lock` lease per cell;
gates unchanged (80 % idle, 5 % repeat spread, 3 retries, cool start, AC power).

| cell | result |
|---|---|
| apple_m1max / gemma4-e2b-q4km | passed 2026-11-11 22:26 UTC, try 4 (tooling at 6642260d); raw record lost in a reboot of `/private/tmp`, values in `benchmark/c23_baseline.json` are from the run report quoted on #444 |
| apple_m1max / qwen35-0.8b-q8 | failed, 6 tries |
| apple_m1max_metal / gemma4-e2b-q4km | passed 2026-11-13 18:01 UTC, try 1, with `cool_each_repeat` (PR #536, tooling at 0b27a7bb); 6 failed tries before it |
| apple_m1max_metal / bonsai2-27b-pq2 | failed, 6 tries without and 2 with `cool_each_repeat` |

Files:

- `apple_m1max_metal/gemma4-e2b-q4km/`: the passing run (`.jsonl`, report, `stderr.log`).
- `run444n.sh`, `run444.log`: driver and log of the 2026-11-13 Metal runs.
- `trace_cause_search.py`, `inv-qwen.log`, `inv-llama.log`: `macmon` traces of
  2026-11-13 (temperature / clock next to each measurement) for the CPU qwen
  cell and the llama.cpp side of the Metal 27B cell. No idle gate in these runs.

The earlier traces of the Metal gemma cell (2026-11-12, the reason for
`cool_each_repeat`) are quoted on #444 and in PR #536; their files were lost
in the same reboot.
