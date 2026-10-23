# Bonsai 2 classic-MMLU held-out sample: direct scoring vs capped reasoning (M1 Max, Metal, 2026-10-23)

Tracking: geistlib#171 (GitHub #587).

**Direct option scoring without generated reasoning is 67.8 %** on 500 held-out questions. The reasoning baseline, capped at 512 generated tokens, reaches 65.4 %. The paired difference is **+2.4 pp, with a conservative 95 % interval of [−5.1, +9.9] pp**. The 2 pp non-inferiority margin is **not established**. At the observed gain/loss rates the interval would clear −2 pp at about **1,450 questions**.

The comparison also does not show what it seems to. A third of the reasoning answers (164/500) hit the 512-token cap and count as wrong. Where reasoning finished, it was right **97.3 %** of the time, against 75.9 % for direct scoring on the same questions. The near-tie is therefore an artifact of the cap, not evidence that token scoring matches reasoning.

Selected-row scoring is 4.06× faster than the capped reasoning call (sum of per-question times). It is bit-identical to dense scoring on all 500 questions.

This is a 500-question estimate with its interval. It is not proof of the margin, and it is not a stock-MMLU score.

## Frozen scope and provenance

- **Checkpoint:** `Ternary-Bonsai-2-27B-PQ2_0.gguf`, SHA-256 `3907dc1658db…ec1`. Metal on an Apple M1 Max (64 GiB), FP32 KV, greedy, concurrency 1, 6 threads, `OMP_WAIT_POLICY=passive`.
- **Binary:** driver `bench_decision` SHA-256 `bf8fa083d636…`, built from source revision `2a387442`. That revision differs from the pilot's engine only by the zero-repeats argument check.
- **Data:** classic `cais/mmlu` revision `c30699e8…`, five fixed subject-specific dev exemplars, exact-overlap exclusion, actual GGUF chat template (date frozen to 2026-10-04). Same prompt policy as the [pilot](DECISION_MMLU_PILOT_2026-10-04.md).
- **Test population:** 500 questions drawn from the cleaned test split (13,904) by a label-independent, subject-stratified rule. Quotas are proportional by largest remainder (3–55 questions per subject, all 57 subjects); within a subject the first questions by `sha256([seed, question_id])`. The rule was frozen in the prepared metadata before inference.
- **Arms** (amendment 2, declared before the reported test inference):
  - chat_direct `decision_dense`: direct score; accuracy, calibration metrics and the paired comparison.
  - chat_direct `decision_selected`: the optimized selected-row path; carries the speedup and is checked bit-exact against dense.
  - chat_reasoning `generate_long`: thinking enabled, cap 512, greedy until EOS or cap. An answer needs exactly one `</think>` and a final letter. Capped or unparseable answers count as wrong; no letter is guessed.

  scorealt_dense, generate_1 and the cloze profile were dropped as reference arms the gate does not need.
- **Trials:** one per question and arm. Determinism spot-check: a second, independent trial in a separate process on 20 declared questions (seed `geist-decision-determinism-spotcheck-v1`).
- **Amendments:** both are recorded in [DECISION_CAMPAIGN_PLAN.json](../../docs/DECISION_CAMPAIGN_PLAN.json) under `held_out_campaign_171`.
  - Calibration was reduced from 750 to a 250-question stratified sample, run with decision_dense only.
  - The test phase uses core arms and one trial.
  - Eleven test questions had run under the earlier two-trial, all-arm design when amendment 2 was decided. They were stopped, not inspected and not used.

## Quality

| Arm | Accuracy | Log-loss | Brier (sum over classes) | ECE (10 bins) |
| --- | --- | --- | --- | --- |
| chat_direct decision_dense, uncalibrated | 67.8 % (339/500) | 0.8175 | 0.4318 | 0.046 |
| chat_direct decision_dense, T = 1.0546 | 67.8 % | 0.8171 | 0.4321 | 0.037 |
| chat_direct decision_selected | identical to dense (bit-exact logits on 500/500) | | | |
| chat_reasoning generate_long, cap 512 | 65.4 % (327/500) | — | — | — |

Reasoning: 164/500 invalid answers, all of them capped at 512 tokens. Generated tokens: min 40, mean 286, max 512.

**Paired comparison, direct (selected) minus reasoning:** n = 500, gains 85, losses 73, difference +2.4 pp. The conservative joint 95 % interval (exact Clopper-Pearson for gains and losses, Bonferroni) is [−5.11, +9.87] pp. Non-inferior at 2 pp: **no**.

The runner's quality-matched speedup gate is ineligible for two reasons: the margin is not established, and the baseline contains capped and invalid answers.

| Questions | n | Direct accuracy | Reasoning accuracy |
| --- | --- | --- | --- |
| Reasoning finished (EOS) | 336 | 75.9 % | **97.3 %** |
| Reasoning capped at 512 | 164 | 51.2 % | 0 % (counted wrong) |

The 512-token cap is far too small for this model in thinking mode. The capped questions are also the harder ones, where direct scoring drops to 51 %. A reasoning baseline that is allowed to finish would very likely beat direct scoring by a wide margin, but this run does not measure that.

**Calibration.** Temperature was fitted only on the 250 calibration questions (calibration split: 68.0 %, log-loss 0.8443 → 0.8435, ECE 0.039), giving T = 1.0546, not at the search bound. The fit is bound to the checkpoint, binary, Metal/FP32 policy and prompt construction.

On the test sample it lowers ECE from 0.046 to 0.037 and log-loss by 0.0004; Brier rises by 0.0002. The uncalibrated four-option probabilities are already close to calibrated. The temperature fit on the earlier binary `e1dac74e`, from the same 250 questions, was identical to the last digit; that run is kept as evidence only.

## Latency

These are single inference calls: session reset, prefill, then scoring or generation, with no setup. Each driver run started on a cool chip (GPU below 65 °C, macmon). Within a run the GPU settles at about 88–90 °C (p50 of start temperatures, max 97 °C), so these are **sustained-load** figures. Only 2 selected and 10 reasoning calls started below 65 °C.

| Arm | p50 | p95 | Mean | Sequential rate estimate | Peak process RSS |
| --- | --- | --- | --- | --- | --- |
| decision_dense | 4.39 s | 12.32 s | 5.44 s | 0.184 q/s | 2.79 GiB |
| decision_selected | 4.37 s | 12.27 s | 5.45 s | 0.184 q/s | 2.79 GiB |
| generate_long (cap 512) | 18.43 s | 43.79 s | 22.11 s | 0.045 q/s | 1.93 GiB |

Selected rows project 16 rows (64 B readback) instead of 248,320 (993,280 B). Whole-query time is still dominated by prefilling 450–3,000-token five-shot prompts, so selected and dense are equal within noise.

The reasoning/selected ratio is 4.06 (sum of per-question times; 4.06 also with additive preprocessing, surface and parse estimates). It compares a capped baseline and is not a quality-matched speedup. The sequential rate is a reciprocal-latency estimate, not measured concurrent throughput. RSS does not capture all Metal resource memory. There is no cold-cache claim, no abstention and no fallback route.

## Determinism

The 20-question spot-check reproduced every arm bit-exactly: logits and argmax for dense and selected, and token IDs and surface bytes for reasoning (20/20 each). It ran in a separate process after the main run. Selected = dense logits held on all 500 questions.

## Campaign and Mac time

The run used 15 test chunks and 3 calibration chunks, each under `with-mac-lock` with `LOCK_PRIO=batch` and `LOCK_MAX=30`, and a 25-min wall-clock driver deadline. Four test driver runs were stopped at the deadline; their unfinished questions were re-run and none was merged twice.

| Phase | Lease time |
| --- | --- |
| Reported calibration (250, decision_dense, 1 trial) | 57 min, including a 25-min chunk lost to host sleep |
| Reported test (500 core arms + 20-question spot-check) | 291 min |
| Dry run (5 development questions, all arms) | 9 min |
| Superseded: three-arm calibration on 750, its 250-question re-cut, 11 two-trial test questions | 197 + 41 + 18 min, including about 115 min of host sleep inside one lease |
| **Total** | **10.2 h of lease time, about 7.9 h awake** |

Two host sleeps stalled chunks. macOS's monotonic clock stands still during sleep, so the first sleep let a chunk overrun its lease; the deadline is now wall-clock time (PR #182). The chunk runner also holds a `caffeinate -i` assertion.

## What this means for #171

- **The 2 pp margin is not established.** At the observed gain/loss rates (17 % / 14.6 %), about 1,450 questions would make the conservative interval clear −2 pp. That would only matter against this capped baseline.
- **The capped reasoning baseline is the wrong reference.** It loses a third of its answers to the cap and is 97 % accurate where it finishes. A fair quality target needs a generation budget at which reasoning completes: a development-only budget study, then a re-frozen cap.
- **Direct token scoring (67.8 %) is therefore not adequate** as a drop-in replacement for completed reasoning on this kind of five-shot knowledge question. This is the negative finding. Whether a trained head (#588) or backbone adaptation is justified should be judged against an uncapped or adequately capped baseline.
- **The direct probabilities are well calibrated** (ECE 0.04–0.05), which is useful for any abstention or fallback design built on direct scoring.

## Raw evidence

- [Compact JSON](DECISION_MMLU_HELDOUT_2026-10-23.json): summary, calibration and one row per question (prediction, probabilities, reasoning stop and length, per-arm latency).
- [Raw bundle](raw/decision-mmlu-heldout-2026-10-23/README.md):
  - `runs.tar.gz`: every run directory (manifests, driver JSONL, chunk metadata, macmon temperatures, reports). Prompt case files are left out; they are rebuilt from the pinned dataset with the recorded tools and checksums.
  - Prepared-bundle metadata.
  - The campaign scripts.
