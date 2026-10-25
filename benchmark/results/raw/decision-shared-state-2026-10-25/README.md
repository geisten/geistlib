# Raw data: #26 shared-state decisions, 2026-10-25

- `step1-cut256/`: the declared run with 256-aligned cuts. Both arms ran per chunk. Not finalized, because the engine resolved a 320-token chunk. `summary_unfinalized.json` is the runner's summary with that check bypassed (the data behind the amendment).
- `step1-cut320/`: the amendment's re-run of the prefix arm with 320-aligned cuts, plus its finalized `report.json`. Its reference is the dense samples of `step1-cut256`.
- `step2/`: fork-check outputs, the tool (`bench_prefix_fork.c`, built against libgeist from the campaign checkout, FP32 KV) and the prompt token IDs (one prompt per line).
- `step3/`: the batched-decode gate on Bonsai (`bonsai.out`) and Qwen3.5 0.8B (`proto08.out`), the bench source and the prototype diff (branch `research/26-multiseq-proto`, fa3cd6ad).

Per chunk: `prefix.driver.jsonl.gz` (every driver line), `temps.jsonl.gz` (macmon), `chunk.json` (cool gate, timing), `prefix.rows.json` (row indices into the #171 test bundle, sha256 108f815e...).
