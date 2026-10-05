# Archived encoder reference evidence

This bundle contains the eight-question CPU/MPS pilot, the four complete
778-question MPS development policies, and the failed 8,192-cap ModernBERT
experiment with a separately declared, disjoint 164-question continuation.
The checkpoint weights are excluded. These are Python reference measurements,
not native geist encoder, held-out acceptance, or quality-matched Jev results.

`SHA256SUMS` covers every file except itself. The compact result outside this
directory binds that index by SHA-256. JSONL is deterministically compressed;
the original report hashes refer to **decompressed** bytes. The continuous
native-context run retains `failure.json` and has no successful `report.json`.

From a checkout containing this branch, run the standard-library verifier:

```sh
python3 benchmark/results/raw/decision-encoders-2026-10-05/verify_evidence.py
```

It checks the inventory/checksums, report/dataset/raw binding, complete trials,
disjoint recovery, question-level quality, separate raw/checkpoint-temperature
probability metrics and p50/p95. It imports no ML dependency or checkpoint.

`validation/` retains format, Python, release and ASan/UBSan outputs and their
commands/revision. They ran serially after inference. Release and sanitizer
tests used temporary build directories with `DECISION=1`, CPU NEON/scalar and
Metal; each suite passed 76 tests with 47 platform/fixture skips. Later commits
package evidence/documentation only; the measurement sources stay pinned in
the per-run manifests. Host paths are recorded provenance, not portable setup
defaults. Download the pinned models separately following the adapter docs.
