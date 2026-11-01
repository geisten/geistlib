# M6 PQ2 campaign archive

`matrix` is the exploratory 65-configuration sweep; empty successful stderr files are omitted.
`model` contains the default-batch six-cycle performance comparison and the earlier raw-completion smoke test.
`model-m512` contains the separate six-cycle 512-row session protocol, with its own same-limit baseline.
`model-chat` reuses the complete, hash-matched performance records and adds the primary quality test in the GGUF Qwen chat format with thinking disabled. See its `quality-metadata.json` and `chat-protocol.json`.
`revision-ab` retains both the preliminary and final clean-build comparisons; the report selects the latest final campaign.
`isolation` checks existing Qwen/Llama formats against clean main and flags enabled/disabled.
`validation` holds final local logs and all per-mode independent oracles. `final-validation-status.tsv` retains actual command exits, including unsupported system-Python-3.9 and bare-Python-3.13/missing-NumPy runs, followed by the successful isolated Python-3.13+NumPy retry. `final-effective-status.json` resolves that retry; all effective final checks passed. `validation-status.tsv` retains the earlier setup pass.
`source-proof.json` proves unchanged GPU kernels between the measured and final executables; the final nine-vector replay additionally checks their actual outputs.

Binary logits, token arrays and copied executables remain local and are excluded. Hashes and lengths of measured logit dumps are retained in manifests. The report is generated from JSON, not entered by hand. These are fixed-protocol, warmed measurements; the twelve-case quality smoke test is not a general reasoning-quality guarantee.
