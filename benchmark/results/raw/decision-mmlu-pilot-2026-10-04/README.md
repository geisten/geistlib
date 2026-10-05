# Frozen raw evidence: Bonsai 2 classic-MMLU development pilot

This bundle records the approved eight-question pilot from 2026-10-04 on an Apple M1 Max (64 GiB), Metal first and CPU NEON second. It contains 256 completed model calls, with exact paired DENSE/selected logits and stable within-backend repeats. It supplies exploratory evidence only. Both reasoning baselines have four invalid, capped answers; neither report is eligible for a quality-matched speedup claim.

- `cases/`: the actual prepared prompt text/IDs, pinned dataset/template metadata, exact exclusion audit, saved chat template and tokenizer log.
- `metal/`, `cpu_neon/`: untouched pre-inference manifests, exact per-profile command inputs, per-call driver JSONL, stderr, component timings and full reports.
- Root logs retain preparation/runner exit output; stdout paths point to the original temporary campaign location.
- [SHA256SUMS](SHA256SUMS) covers every other file in this bundle, including empty stderr logs. It verifies retained content, not origin authenticity. Model weights and executable binaries are not copied; their hashes are in the manifests and were checked after both runs.

[Human-readable result](../../DECISION_MMLU_PILOT_2026-10-04.md) and [compact JSON](../../DECISION_MMLU_PILOT_2026-10-04.json) give the interpretation and canonical predictions. The manifests record clean source revision `13261ac73fb2841ef565bcabda5473eacd1e8ce0`, exact evaluator source hashes, binary hashes, environment and host metadata. The repository added this result after measurement; do not mistake the later result commit for the measured source revision.

## Reproduction

Use the frozen revision and the recorded model. The driver and tokenizer were built with `DECISION=1` and `BACKENDS='cpu_neon cpu_scalar metal'`; a rebuild has its own binary hash and must be reported as a separate run. The metadata pins classic `cais/mmlu` revision `c30699e8356da336a370243923dbaf21066bb9fe`, the three parquet hashes, GGUF template/native tokenization and preparation package versions. This bundle includes the exact prepared inputs so it does not require re-sampling the dataset.

Run each backend serially with a fresh output directory. From the repository root, set the paths to the checkpoint and compiled driver:

```sh
OMP_NUM_THREADS=6 OMP_WAIT_POLICY=passive python3 tools/eval_decision.py \
  --cases-dir benchmark/results/raw/decision-mmlu-pilot-2026-10-04/cases \
  --split development --purpose pilot \
  --model "$decision_model" --binary "$decision_benchmark_binary" \
  --backend metal --threads 6 --decode-cap 512 --margin-pp 2 \
  --warmup 0 --repeats 1 --out-dir "$decision_fresh_metal_dir"
```

After Metal finishes, use the same options with `--backend cpu_neon` and a new CPU output directory. Avoid overlapping inference with builds or other model benchmarks. Reproduction does not imply identical timings on a shared desktop.

To audit the retained files without inference, verify `SHA256SUMS`, then use `tools/bench_decision.py`'s `enrich_and_check` and `tools/eval_decision.py`'s `summarize_profile` on each driver JSONL and its cases. The stored report's quality, latency, RSS and projection fields reproduce from those samples. Parse-cost timings are recorded observations and should not be expected to repeat during re-parsing. The paired comparison uses canonical question identity; quality N stays eight.
