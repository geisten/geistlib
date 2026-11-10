# M1 Max Phase 0 cells at 6642260d, third attempt (#444)

2026-11-10 01:31–06:00 CET, the four nightly M1 Max cells of `benchmark/c23_baseline.json`
(apple_m1max: qwen35-0.8b-q8, gemma4-e2b-q4km; apple_m1max_metal: gemma4-e2b-q4km,
bonsai2-27b-pq2), geist and tooling at 6642260d, frozen cross-engine protocols with their
unchanged host gates (80 % idle, 5 % repeat spread, 3 retries, cool start), one
`with-mac-lock` lease per cell (LOCK_PRIO=batch). Before each cell `run444.sh` also waited
outside the lock for Spotlight/media daemons below 10 % CPU and >= 85 % idle.

No cell passed: every try ended `repeat spread stayed above 5 %`. Nothing is recorded in
the baseline. Spread per measurement (the first three of each try were discarded and
re-measured, the fourth failed the try), pre-run idle 84–93 %:

| cell | try | spreads % |
|---|--:|---|
| apple_m1max/qwen35-0.8b-q8 | 1 | 23.9, 22.9, 15.4, 22.4 |
| | 2 | 29.2, 17.2, 21.6, 13.7 |
| | 3 | 9.6, 5.8, 6.6, 12.2 |
| apple_m1max/gemma4-e2b-q4km | 1 | 14.8, 16.7, 45.1, 5.4 |
| | 2 | 8.6, 10.4, 9.4, 7.2 |
| | 3 | 11.4, 5.5, 10.8, 9.6 |
| apple_m1max_metal/gemma4-e2b-q4km | 1 | 12.9, 14.4, 10.5, 14.1 |
| | 2 | 13.8, 25.4, 8.4, 6.2 |
| | 3 | 96.3, 5.8, 6.5, 6.4 |
| apple_m1max_metal/bonsai2-27b-pq2 | 1 | 8.2, 7.7, 6.5, 6.0 |
| | 2 | 24.0, 22.8, 24.3, 24.0 |

Disturbers seen in the window, outside the lock: Google Chrome helper processes (~46 % CPU),
other Claude sessions, Spotlight (`spotlightknowledged` 60–90 %) until 01:46; geist-lab's
qwen-math500 runner alternated with these cells through the lock.

Files: `run444.sh` (the driver), `run444.log`, per cell `stderr.log`/`stdout.log` and the
`.jsonl.partial` of each try (no summary record: a failed run writes none).
