# Early exit from reasoning by answer probes: pilot (Bonsai 2 27B, M1 Max, Metal, 2026-10-26)

Tracking: geistlib#236. Declared in `docs/DECISION_CAMPAIGN_PLAN.json` → `early_exit_pilot_236` (PR #262) before any inference. This is a calibration-split pilot: no held-out question was used.

**Probing the answer during reasoning looks promising. Fifty questions cannot show that it is safe.**
- **Probes leave generation unchanged:** 2/2 generations are token-identical with and without probes.
- **Probe cost:** a median of **179 ms, 3.2 decode steps**, just over the 3-step target.
- **Exit points:**
  - At **τ = 0.8**, the policy stops at a probe from token 128 on and answers with the probe's choice. Accuracy equals full reasoning (78 %) at **0.40×** the generated tokens (0.41× time including probes).
  - At τ = 0.95: +2 pp at 0.69× the tokens.
  - At τ = 0.99: 0 pp at 0.87×.
- **The declared gate** (≥ 25 % fewer tokens at ≤ 1 pp loss) **is met on point estimates only.**
  - The paired 95 % intervals at n = 50 are ±8–15 pp wide.
  - The thresholds were evaluated on the same 50 questions.
  - No policy is frozen. The next step is the user's call (see below).

## Setup

- **Sample:** 50 of the 250 frozen #171 calibration questions, `decision_dataset.select_stratified` with seed `geist-decision-early-exit-pilot-v1` (label-independent, subject-stratified).
- **Generation:** chat_reasoning prompts, greedy, cap 2048, unbatched. Bonsai 2 27B PQ2_0, Metal, FP32 KV, 6 threads. `tools/bench_reasoning_probe` at main b8980d3c.
- **Probe:** right after the prompt and then every 128 generated tokens:
  1. `geist_session_snapshot`;
  2. prefill `[198, 248069, 271]` ("\n</think>\n\n", the model's own thinking close);
  3. read the four option logits;
  4. `geist_session_restore`.
- **Runner:** `tools/eval_decision_probe_pilot.py`, 3 chunks with cool starts, 32 min of lease time in total.
- **Scoring:** #171's `generation_outcome` for full reasoning. The offline policy grid is computed from the recorded probes.

## Results

**Full reasoning on the 50:** 78 % (39/50), 7 capped, 27,198 generated tokens (544 per question on average). The probe at the prompt alone, a direct-scoring-like answer under the reasoning template, gets 64 %.

| policy (exit from token 128) | accuracy | vs full, pp [paired 95 %] | generated tokens | time incl. probes | early exits (agree with full) |
|---|---|---|---|---|---|
| full reasoning | 78 % | — | 1.00× | 1.00× | — |
| τ 0.70 | 76 % | −2.0 [−17.9, +14.3] | 0.26× | 0.27× | 33 (23) |
| **τ 0.80** | **78 %** | **0.0 [−14.9, +14.9]** | **0.40×** | **0.41×** | 31 (22) |
| τ 0.90 | 80 % | +2.0 [−11.8, +15.3] | 0.63× | 0.65× | 28 (21) |
| τ 0.95 | 80 % | +2.0 [−8.4, +12.1] | 0.69× | 0.71× | 25 (21) |
| τ 0.99 | 78 % | 0.0 [−8.4, +8.4] | 0.87× | 0.90× | 19 (17) |

- The "stable" variant requires the same argmax at the previous probe. It changes accuracy nowhere, and it costs 2–5 % more tokens.
- Early answers often differ from full reasoning's. At τ 0.8, 9 of 31 do: full reasoning was wrong in some of those and the early answer was wrong in others, and the two kinds offset each other here.
- That balance is the fragile part. Fifty questions cannot pin it down.

## Proposed next step (user's decision)

**Option 1: confirm on calibration.**
- Run the remaining 200 calibration questions with probes. That takes about 2.7 h of Mac time unbatched, or about 1 h combined with batched decode (#26). Batching would need a probe-aware batch driver: a probe on one member snapshots and restores only that session.
- Then freeze τ on calibration (for example the largest τ with ≤ 1 pp loss and ≥ 25 % fewer tokens).
- Then run the held-out 500 once.

**Option 2: freeze τ = 0.95 now and run the held-out 500.**
- τ = 0.95 is the conservative point: 0 losses, a +2 pp point estimate, 0.71× time.
- The run takes about 2–2.5 h unbatched, or about 1 h batched with probes.

Probes are about 3 decode steps each, so a 256-token interval would halve their cost for the long questions. The pilot data show most exits at 128–384 tokens.

Raw data: `raw/decision-early-exit-pilot-2026-10-26/`. It holds the manifest, `report.json` with every probe's logits per question, and the per-chunk driver lines and temperatures. The compact numbers are in `DECISION_EARLY_EXIT_PILOT_2026-10-26.json`.
