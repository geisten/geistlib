# #210 B: accuracy/time over the reasoning cap -- scripts

- `analyze_210.py <plan.json> <model.gguf>`: derives reasoning outcomes at caps 256..2048 by truncating the
  cap-2048 generations (greedy, deterministic), scores them with the campaign's `generation_outcome`, models the
  call time from measured points, and evaluates always-reasoning and the two-stage policy (calibration-split
  thresholds) at every cap. Asserts that truncation to 512 reproduces every measured cap-512 outcome.
- `ggufvocab.py`: minimal GGUF vocabulary reader / byte-level detokenizer (needs only the model's metadata).

Reads `~/workspace/geistlib-campaigns/171` (run-calibration250b, run-test2, run-cap2048; raw bundle in
`../decision-mmlu-heldout-2026-10-23/`) with that directory's pinned tools checkout (`src`, d159d350).
No inference.
