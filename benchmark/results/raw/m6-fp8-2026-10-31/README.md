# M6 FP8 comparison raw data

Generated on the target M6 by `tools/bench_metal_fp8.py`, commit 15a088d7.
See ../../M6_FP8.md for protocol, results, limitations and reproduction.

`summary.json` is force-tracked because the repository normally ignores that
scratch filename. Binary logits and token IDs are kept locally, excluded
from the PR; logit-manifest.json records their sizes and SHA-256 values.

Provenance note: the original runner's `working_tree_dirty: true` in
metadata.json is a false positive: the shared command_output helper returns
"unavailable" for empty git status, which the first runner converted to bool.
Tracked source was committed; the kernel revision and executable hash pin
this run. The runner now queries tracked git status directly. This metadata
fix does not alter timing, model inputs or quality calculations.
