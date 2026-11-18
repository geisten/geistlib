# Early exit at τ 0.9: held-out validation on a new sample (Bonsai 2 27B, M1 Max, Metal, 2026-11-18)

Tracking: geistlib#467, follow-up of #286. Declared in `docs/DECISION_CAMPAIGN_PLAN.json` → `early_exit_tau09_467` (PR #480, 3c81cf93), merged on 2026-11-08 20:28 CET, before the first inference at 21:00 CET. The 3,000 questions were analysed once, after the last one finished (2026-11-18 03:35 CET).

Raw bundle: `benchmark/results/raw/decision-early-exit-tau09-2026-11-18/`, with every outcome number below from `analysis.py` → `DECISION_EARLY_EXIT_TAU09_2026-11-18.json`. It holds the run's manifest and report (all probe logits), the targets, the 4,000 excluded question ids, the campaign scripts and the power log.

**The declared gate passes.**
- **Non-inferiority:** at τ 0.9 with probes every 128 tokens, early exit is non-inferior to full reasoning at cap 8192 on 3,000 held-out questions that no earlier test sample contains, at the 2 pp margin.
  - Accuracy: 84.23 % vs 81.70 %. Gains 123, losses 47.
  - Conservative paired 95 % interval: **[+1.17, +3.88] pp**.
  - Generated tokens: 0.35×.

This is a confirmatory analysis. It computes the plan's primary and secondary items and nothing else: no subsets, no thresholds other than the declared 0.9 and the descriptive 0.8, no latency check (the plan has none).

## Setup

- **Sample:** 3,000 held-out questions.
  - Drawn label-independent and subject-stratified from the cleaned classic-MMLU test split (13,904) minus the 500 of #171/#236 and the 3,500 of #286: a population of 9,904.
  - Seed `geist-decision-test-sample-467-v1`; `question_ids_sha256` `b40cb1ab…`, as declared.
  - **Why a new sample:** τ 0.9 came from #286's exploratory table on #286's own 3,500 questions. Testing it there would reuse the data that suggested it.
- **Model and generation:** Bonsai 2 27B PQ2_0 (`3907dc16…`), Metal on the M1 Max, FP32 KV, 6 threads, greedy, chat_reasoning prompts (5-shot), cap **8192**.
  - Batch 8, refilled as questions finish, no padding; per-subject prefix snapshots aligned to 320.
  - Source e5e4aaf8; `bench_reasoning_batch` `5100e991…`.
- **One run gives both arms:** full reasoning recorded the option logits at a probe every 128 generated tokens (threshold 0). Probes are transparent (#236).
  - **Early-exit arm:** the declared rule applied to those probes. The first probe at position ≥ 128 whose top softmax probability over the four options is ≥ 0.9 answers with its argmax; without one, the full answer stands.
  - **Full arm:** a capped answer without a final letter is wrong (#171 scoring).
- **Run:** night-only, chunks of at most 28 min under `with-mac-lock` (`LOCK_PRIO=batch`). 104 chunks, 47.8 h of lease (44.3 h of them inside the driver), 2026-11-08 to 11-18.

## Checks before the analysis

All hold. `analysis.py` asserts those marked ✓ from the bundle before it reads an outcome; the others were made on the campaign's files on 2026-11-18.

| check | value | declared | |
|---|---|---|---|
| `cases_sha256` in the manifest | `b8cfe3f3…127413e` | the same | ✓ |
| sha256 of the campaign's `cases-test/test.jsonl` | `b8cfe3f3…127413e` (3,000 records) | the same | on the campaign's file |
| `question_ids_sha256` (ids in selection order, joined by newlines) | `b40cb1ab…3d5354` | the same | ✓ |
| sha256 of `exclude-171-286.jsonl` | `e716bc88…b9a9c6` | the same | on the campaign's file |
| excluded question ids | 4,000, all unique; the last 3,500 are #286's sample in its order | 4,000 unique | ✓ |
| overlap of the 3,000 with the 4,000 excluded | **0** | 0 | ✓ |
| model sha256 in the manifest | `3907dc16…fae62ec1` | `3907dc16…` | ✓ |
| sha256 of the model file on disk | `3907dc16…fae62ec1` | | on the campaign's file |
| source revision in the manifest | `e5e4aaf8a5d91120fda389b65966030ad4045051` | the same | ✓ |
| HEAD of the campaign's build worktree | the same, no local changes | | on the campaign's files |
| sha256 of the built `bench_reasoning_batch` | `5100e991…` = the manifest's `binary_sha256` | | on the campaign's file |
| questions with a recorded generation | 3,000 of 3,000, all unique, in the prepared order | 3,000 | ✓ |
| finished generations per question over all 104 chunks | exactly 1 for each of the 3,000 | finished questions are never re-run | on the campaign's chunk files |
| stop reasons | 2,741 EOS, 259 cap; none other | | ✓ |
| probes | at every 128 tokens before the end, for all 3,000 | every 128 | ✓ |
| settings in the manifest | cap 8192, batch 8, align 320, probes [198, 248069, 271] every 128, `exit_tau` 0, metal, 6 threads, no pad, `OMP_WAIT_POLICY=passive`, all questions | the same | ✓ |
| declaration merged before inference | merge 2026-11-08 20:28:36 CET; first chunk 21:00:05 CET | before any inference | git history, report |

## Primary: non-inferiority at 2 pp

| | accuracy | generated tokens | capped without an answer |
|---|---|---|---|
| full reasoning, cap 8192 | 81.70 % (2,451) | 4,035,367 | 259 |
| early exit, τ 0.9 | **84.23 %** (2,527) | **1,415,264 (0.351×)** | 36 |

- **`paired_quality(early exit τ 0.9, full, 2 pp)`:** gains 123, losses 47, +2.53 pp, interval **[+1.17, +3.88]**. The lower bound is above −2 pp, so the result is **non-inferior: PASS**.
  - The lower bound is also above 0. Superiority was not declared, so that is a description, not a claim.

## Secondary (as declared)

- **Accuracy:** early exit 84.23 % (2,527 of 3,000), full reasoning 81.70 % (2,451).
- **Gains / losses:** 123 / 47.
- **Early exits:** 1,801 of 3,000 questions.
- **Capped answers:** full reasoning hits the cap on 259 questions; with early exit, 36 of them still do.
- **Token cost:** 1,415,264 generated tokens against 4,035,367, **0.351×**.

### τ 0.8 on this sample (descriptive; declared as "not a test")

The same numbers for the current default threshold, from the same recorded run. No interval and no verdict: the plan declares this row as a description.

| | accuracy | gains / losses | early exits | still capped | generated tokens |
|---|---|---|---|---|---|
| early exit, τ 0.9 (primary) | 84.23 % (2,527) | 123 / 47 | 1,801 | 36 | 1,415,264 (0.351×) |
| early exit, τ 0.8 (descriptive) | 83.07 % (2,492) | 139 / 98 | 1,902 | 9 | 963,676 (0.239×) |

Gains and losses are against full reasoning (81.70 %). The two thresholds are not compared by a test here; the plan declares none.

## Deviations from the plan

None changes which questions were run, how they were generated, or how they are scored. Both arms of a question come from one recorded generation, so nothing below can favour one arm.

- **Duration.** The plan estimated about 17 h of lease over 2.5 to 3 nights. The run took 47.8 h of lease, with chunks on nine nights between 2026-11-08 and 11-18.
  - 83 of the 102 driver runs ended at the 28-minute deadline. Their questions in flight (at most 8 each) were re-run from the prompt in a later chunk, as declared; a question that needs most of the cap can take several chunks. The chunks assigned 6,614 question slots for 3,000 questions.
  - Other jobs held the machine lock for part of most nights.
  - **Effect on the outcome:** none. A re-run generates the same greedy trajectory from the same prompt, and each question has exactly one finished generation.
- **Two interruptions on battery (2026-11-12 and 2026-11-15).** The pinned build's evaluator waits for AC power with Low Power Mode off before it starts a chunk. The plan only declared sampling the power state. On both evenings the machine was on battery; the wait timed out (after 2 h 1 min and after 5 min), the chunk failed without starting the driver, and the scheduler stopped as designed. It was restarted by hand on 2026-11-13 21:11 and 2026-11-16 01:04.
  - On 2026-11-16 and 2026-11-17 two more chunks waited on battery at their start (60 s and 30 s recorded waits). The first of them then ran for 5 s before its deadline and finished nothing.
  - **No inference ran on battery:** of the 5,405 power samples, 26 are on battery with Low Power Mode on, and all 26 fall outside every driver run. All 102 driver runs started on AC with Low Power Mode off.
  - **Effect on the outcome:** none; no generation was produced or lost in those chunks. Power would change speed only.
- **Scheduler restarts and idle nights.** The scheduler was started six times (2026-11-08, 11-10, 11-12 twice, 11-13, 11-16). No chunk ran in the night of 11-09, and in the night of 11-11 chunks ran only from 04:25 (on 11-12); the logs do not record why the scheduler was not running in between. **Effect on the outcome:** none; the run resumes from the frozen manifest.
- **Chunks past 07:00.** Two driver runs began after the window closed: one at 07:36 on 2026-11-15 (its chunk was started at 06:44 and waited for the lock) and one at 07:00:22 on 2026-11-16 (started before the scheduler set its stop marker at 07:00:58). **Effect on the outcome:** none.
- **Declaration merge by a watcher.** The declaration PR #480 was merged by a script that polled it from 2026-11-07 and merged it (merge commit) at 20:28 CET on 2026-11-08, once its head 3c81cf93 had the owner's CI approval. The night scheduler waited for the declaration on `main` and logged it at 20:29:58; the first chunk started at 21:00. The planned first night (2026-11-07) was therefore not used. **Effect on the outcome:** none; it is what kept inference behind the declaration.
- **`progress.py`** in the bundle is #286's file, byte-identical: it still names #286's directory, so it was not this campaign's progress report. The scheduler's own log lines carry counts only.

One departure in the analysis itself: `analysis.py` stopped at its first run on a check of the probe positions that was written wrongly (it expected a probe at the final position of a generation that ends on a multiple of 128; there is none, as in #286). The checks run before any outcome is computed. The check was corrected and the analysis then ran once.

## What this does not cover

- **One model, one task family, one cap:** Bonsai 2 27B PQ2_0 on classic-MMLU four-option questions with the #171 prompt policy, cap 8192, greedy. Nothing here speaks for another model, another cap or free-form tasks.
- **No latency measurement.** The token ratio is the cost measure, as declared. #286 measured live early exit at τ 0.8 against the offline policy (300/300 identical); live τ 0.9 was not run.
- **No breakdown.** Per-subject effects, and the effect on questions that full reasoning finishes within the cap, were not analysed. #286 found a cost there for τ 0.8 (about −2.5 pp on finished questions, post hoc); whether τ 0.9 has one is not answered by this result.
- **No test between τ 0.9 and τ 0.8.** The τ 0.8 row is a description.
- **Not re-checked from the bundle:** the hashes of the 71 MB cases file, of the 96 MB exclusion file and of the model, and the one-generation-per-question count over the chunk files, were verified on the campaign's files, which are not committed.

## What follows (declared)

`if_noninferior`: a geist-runtime PR that changes the Bonsai 2 27B early-exit default from τ 0.8 to 0.9, left for the owner's merge decision. This repository holds no such default; nothing here changes.
