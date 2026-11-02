# MTP draft decoding on Qwen3.6-35B-A3B in GPU mode (#246)

RTX 2080 Ti + Ryzen 9 9950X (desktop), Qwen3.6-35B-A3B-UD-Q4_K_XL, Vulkan
GPU mode with the run-1830 expert profile (`GEIST_MOE_PROFILE`), `--m-max
512`, 128 greedy tokens. Every run under `~/bin/with-desktop-lock`, started at
load < 3 (the load printed per run is the run's own threads). Binaries:
`base_bench` = main ad1fb409 with this branch's `tests/bench_perf_sweep.c`
(for `--spec`, `--emit-tokens`, `--emit-margins`), `new_bench` = the branch.
Variants (`run.sh`): `OFF` (decode_step, `GEIST_MTP` unset), `MTP<k>`
(`GEIST_MTP=1`, `decode_speculative` with k_max k), `NG<k>` (n-gram drafts,
k_max k). `sum.py <file>` and `corr.py <file>` print the tables.

| file | what |
|:--|:--|
| `ab-prompt.jsonl` | 3 interleaved cycles (`camp.sh`, one lease per cycle): main/branch OFF, main/branch MTP2, branch MTP3, main/branch NG3 on the 2,726-token request (the first tokens of `docs/OFFLOAD.md` at b358d73e) |
| `ab-high-acceptance.jsonl` | 3 cycles, branch only: OFF/MTP2/MTP3/MTP4 on `docs/ARCHITECTURE.md` (1,024 tokens), OFF/MTP2/MTP3 on `README.md` (256) |
| `equivalence.jsonl` | `corr.sh`: per prompt main OFF with `--emit-margins` (reference tokens and the runner-up with its logit gap per step), branch OFF, MTP2, MTP3 |
| `explore-e1.jsonl`, `explore-e2.jsonl` | single-run attribution on intermediate builds (temporary env switches, removed again): `GEIST_X_ARGMAX` = old host-argmax draft head, `GEIST_X_DNROWS` = replay instead of the per-row DeltaNet states, `GEIST_X_VROWS` = verify experts as prefill GEMMs, `GEIST_X_VREG` = old verify region (load backend only), `GEIST_X_VPRE` = verify region in the prefill regime (what the branch does), `GEIST_X_DREG` = no region for the draft steps. Their `base_bench` was main plus a phase trace in `session.c`. e1's verify region used the decode regime; e2 ran beside a CI coverage job on one core |
| `trace/` | wall-clock phases on a diagnostic build: `tr_off`/`tr_seq`/`tr_offhr` = per-step ms (`GEIST_BENCH_STEP_MS`) for OFF, MTP on with decode_step, OFF with host routing; `tr_mtp2`/`tr_mtp3` = per speculative call `start` (draft) / `verify` / `trunc` / `corr` ms, k and accepted drafts; `vt_mtp2`/`vt_mtp2x` = the verify split into layers (with a forced sync) / MTP block / head, with and without the per-row experts |
| `ab-post-merge.jsonl` | 2 cycles after merging main 57f58a18 (GPT-OSS, #243): `main2_bench` = that main with this branch's bench, `merged_bench` = the branch with it merged |
| `metal-27b-*` | M1 Max (shared, loaded: timings not meaningful), Qwen3.8-27B Q4_0 (dense, MTP block), Metal: OFF with margins / MTP2 / MTP3 over 128 tokens of the same request at 512; `test_speculative_loop_int` with and without `GEIST_MTP=1` |

The models were in the page cache for the final files; a run then takes
~16 s, against ~2.5 min while CI jobs evicted it (explore files).
