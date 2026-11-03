# MTP draft decoding on the dense Qwen3.8-27B, Metal (#401)

M1 Max (64 GB, AC power), Qwen3.8-27B Q4_0 (`qwen3.8-27b-q4_0.gguf`, one
MTP block), Metal backend, 512-token prompt, 128 greedy tokens. Every run
under `~/bin/with-mac-lock`. The Mac was shared (load average 18-100 at
the starts; other agents' builds do not take the lock), so single runs
scatter by up to ±10 %; the A/B is interleaved and carries a control.

Binaries: `base_bench` = `tests/bench_perf_sweep` of main df9c4201,
`ctl_bench` = a copy of it (the control arm), `new_bench` = the branch.

| file | what |
|:--|:--|
| `ab.jsonl` | `cycle.sh`, 3 cycles (one lease each): per prompt (`docs/ARCHITECTURE.md`, `docs/OFFLOAD.md`) base OFF, new OFF, base MTP K=2, new MTP K=2/3/4, control OFF, all with `--emit-tokens`. `sum.py ab.jsonl` prints the table (median, prefill, tokens identical to base OFF of the same cycle, tokens per call, acceptance per draft position) |
| `mb.txt` | `mb.sh`: GPU time per linear call (30 calls in one command sequence, best of 5) for 1..8 rows on the model's shapes: base = main, new = the branch, n4row = the branch with `GEIST_METAL_MV_M8=0` (one n4 matvec per row). `mbench.c` is the probe |
| `trace/before_*.txt`, `trace/after_*.txt` | per speculative call: `draft` (MTP steps), `verify`, `trunc` (rollback), `corr` (the correction token's forward), in ms, from a diagnostic build with a timer in `spec_verify` (not committed); before = main, after = the branch; 32 tokens, `docs/ARCHITECTURE.md` |
| `trace/after_seq_trace.txt` | `vprobe.c` (decode_step, then `verify_drafts` with k = 2, 3) under `GEIST_METAL_SEQ_TRACE=1`: per Metal command sequence the dispatches and wall time, kind 0 = decode, kind 2 = prefill/verify |

Build the probes against a tree's `lib/mac-omp/release/libgeist.a`:
`cc -std=c23 -O2 -Iinclude mbench.c lib/mac-omp/release/libgeist.a
-framework Accelerate -L/opt/homebrew/opt/libomp/lib -lomp -lm -ldl`.
