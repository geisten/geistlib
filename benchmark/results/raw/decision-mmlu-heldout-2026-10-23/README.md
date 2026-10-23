# Raw evidence: Bonsai 2 classic-MMLU held-out sample (#171), 2026-10-22/23

- `runs.tar.gz` holds every run directory of the campaign, without the prompt case files (`*.cases.jsonl`):
  - Reported: `run-calibration250b` (250 calibration questions, decision_dense, 1 trial; `calibration.json` with T = 1.0546) and `run-test2` (500 test questions, core arms, 1 trial, plus the 20-question determinism spot-check; `report.json`).
  - Superseded, kept for completeness: `run-dryrun` (5 development questions, all arms), `run-calibration` (three-arm calibration on 750, stopped by amendment 1), `run-calibration250` (the 250-question re-cut on the earlier binary `e1dac74e`), `run-test-superseded-2trial` (11 test questions under the pre-amendment-2 design; not inspected, not used).
- Each run directory has:
  - `manifest.json`, frozen before inference and checked on every resume.
  - `prepared_metadata.json`.
  - `chunks/NNNN/`: `chunk.json` (rows, lock note, cool gate), the driver JSONL exactly as streamed, `*.recv.tsv` arrival times, `*.run.json`, and `temps.jsonl` (macmon GPU/CPU temperatures).
  - `status.json` and the reports.
- `cases/*/metadata.json` + `split_audit.json`: the prepared bundles. These include the frozen test-sample quotas (`test_sample`), the calibration sample (`calibration_sample`), the dataset parquet hashes, the template hash and the case-file SHA-256. Prompts are rebuilt with `tools/prepare_decision_eval.py` (`--test-sample 500`, template date 2026-10-04) and `tools/sample_decision_cases.py` at the recorded revisions; the case-file hashes verify the rebuild.
- `campaign.sh`, `run.sh`, `prepare.sh`: the exact chunk driver used, with `with-mac-lock`, `LOCK_PRIO=batch` and `LOCK_MAX=30`. `summarize.py` produces the numbers in the writeup from `run-test2/report.json`.

Model weights and binaries are not copied; their hashes are in the manifests. See the [writeup](../../DECISION_MMLU_HELDOUT_2026-10-23.md).
