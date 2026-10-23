# #204 two-stage policy: scripts

- `thresholds_204.py`: fits the declared thresholds from the #171 calibration split (direct arm only); reads no test data.
- `analyze_204.py <plan.json>`: evaluates the declared policies on the 500 #171 test questions against always-reasoning at cap 2048.

Both read `~/workspace/geistlib-campaigns/171` (run-calibration250b, run-test2, run-cap2048; raw bundle in
`../decision-mmlu-heldout-2026-10-23/`) with that directory's pinned tools checkout (`src`, d159d350).
No inference.
