# Shared-state multi-question decisions on Bonsai 2 27B (M1 Max, Metal, 2026-10-25)

Tracking: geistlib#26, phase 2, steps 1–3. Phase-1 analysis: [#26 comment](https://git.geisten.net/geisten/geistlib/issues/26#issuecomment-4923). Declaration: `docs/DECISION_CAMPAIGN_PLAN.json` → `shared_prefix_campaign_26`, with one amendment.

**Summary**

1. **Direct scoring from a per-subject prefix snapshot is bit-identical to independent calls and takes 0.40× the time.** Measured on the 500 held-out MMLU questions of #171 (57 subjects).
   - The option logits are bit-identical on **500/500** questions, and accuracy is the same (67.8 %).
   - Per question, p50 drops from 4.40 s to 2.22 s and p95 from 18.8 s to 3.6 s.
   - End to end, including one prefix prepare per subject, the prefix route takes **0.40×** the time of independent calls.
   - The gain comes from **cache reuse** only. Nothing is batched and nothing is trained.
2. **Generation forked from a prefix snapshot is token-identical to an independent session.** On 16 questions (2 subjects), 256 greedy tokens per question match in forward, reverse and interleaved order, and the final logits are bit-identical.
3. **Batched multi-sequence decode passes its gate: 3.47× aggregate tokens/s at 8 sequences.**
   - Each sequence's output is bit-identical whatever the other batch members are (other questions, other lengths, other slot).
   - It is not bit-identical to single-sequence decode: the logits differ by at most 0.009, and the tokens were identical over 32 steps for all 8 sequences.
   - The gain comes from **batching**: the weights are read once for 8 rows.
   - This is a test-only prototype. The API proposal is below; per the user's instruction, work stops there for a decision.

## 1. Direct scoring from a per-subject prefix (500 questions)

**Setup**
- Bonsai 2 27B PQ2_0 (sha256 3907dc16…), Metal, FP32 KV, 6 threads, `OMP_WAIT_POLICY=passive`.
- `tools/bench_decision --prefix` from PRs #237/#254/#256, binary sha256 86c767c2…
- The 500-question #171 test sample (`cases-test/test.jsonl` sha256 108f815e…), chat_direct profile.
- **The prefix:** per subject, the longest common token prefix of its questions minus one token, aligned down to the engine's prefill chunk.
- **The two arms:**
  - `decision_dense`: every prompt scored independently.
  - `decision_prefix`: restore the subject's snapshot and prefill the suffix.
- One trial each. Every lease started with the GPU below 65 °C. The GPU reaches about 90–95 °C within a run, so these are sustained-load latencies.

**One amendment, disclosed.** The declaration aligned the cuts to 256. That is the chunk Qwen3.5 0.8B resolves, and the phase-1 measurements used it. **Bonsai 2 27B resolves 320**: the qwen35 family adds 64 rows for weights of 4 GiB or more.
- **The 256-cut run** (`run-step1`): the runner's engine-chunk check refused to finalize, as designed.
  - 472/500 option-logit vectors were bit-identical, argmax 500/500, accuracy 67.8 % in both arms, time 0.385×.
  - **All 28 mismatches** (max |difference| 0.003) have a final prefill pass of **≤ 8 rows in exactly one of the two splits**, where the #186/#221 small-row PQ2_0 kernels round differently from the large GEMM.
  - The other 407 non-aligned questions were bit-identical, and so were all 65 whose 256-cut happened to be a multiple of 320.
- **The re-run:** the amendment (PR #254, declared before the re-run) repeated only the prefix arm with cuts aligned to 320 (`run-step1b`). Its reference is the first run's `decision_dense` samples. The binding is identical (same binary file, model, cases and environment), and greedy scoring is deterministic across processes (#171).

**Result (cuts aligned to 320):**

| | independent (`decision_dense`) | from prefix (`decision_prefix`) |
|---|---|---|
| option logits bit-identical | — | **500/500** |
| argmax identical | — | 500/500 |
| accuracy | 67.8 % (339) | 67.8 % (339) |
| per question p50 / p95 / mean | 4.40 / 18.76 / 6.30 s | 2.22 / 3.63 / 2.22 s |
| prefix prepares | — | 54 (10 subjects share an empty cut); p50 2.19 s, max 15.5 s; sum 155 s |
| end to end, 500 questions | 3,150 s | 1,108 s + 155 s = 1,263 s (**0.40×**) |
| snapshot per prefix | — | 150–431 MiB (DeltaNet state ~150 MiB + 128 KiB per cached token) |
| process peak RSS | 3.6 GB (weights are mmapped; not in RSS) | |

**By subject** (shared ÷ independent, including the prepare):

| | buckets |
|---|---|
| questions per subject | 3–4: 0.55× · 5–8: 0.47× · 9–14: 0.44× · 15–55: **0.31×** |
| mean prompt tokens | < 500: 0.62× · 500–800: 0.50× · 800–1,200: 0.42× · ≥ 1,200: **0.23×** |
| cached share of the prompt | empty cut: 0.91× · < 50 %: 0.57× · 50–70 %: 0.48× · ≥ 70 %: **0.24×** |

The saving grows with the number of questions per context and with the context's share of the prompt.

**Alignment costs tokens.** The 320-cut caches 230,720 of 374,065 prompt tokens; the 256-cut cached 257,024. Ten short subjects cache nothing at 320. The ≤ 8-row tail kernel is the only source of mismatch seen with 64-aligned cuts (64 is the DeltaNet chunk). An engine change that makes the final-pass kernel choice independent of the split would allow any 64-aligned cut without losing bit-identity. That is a possible follow-up, not measured here.

**Isolation.** Every question restores the full image: KV, DeltaNet conv and S state, positions and pending logits. The order is subject order, and both arms rotate per case. The identity to independent calls is shown directly, 500/500 bit for bit. In phase 1, Qwen3.5 0.8B showed order and addition invariance (forward, reverse, interleaved) on Metal and cpu_neon, and the repo tests `test_decision_prefix_int` and `test_session_snapshot_unit` cover it too.

## 2. Reasoning fork check (generation from a prefix snapshot)

`step2/bench_prefix_fork.c` (scratch tool, source in the raw bundle):
- **Reference:** an independent session prefills the prefix, then the suffix, then decodes 256 greedy tokens. The joined single prefill is also recorded.
- **Worker:** one session prefills the prefix once and takes a snapshot. Then, per question: restore, suffix, 256 tokens.
- **Prompts:** the chat_reasoning prompts of high_school_statistics (8 questions, 840–1,024 tokens, cut 640) and high_school_world_history (8 questions, 1,338–1,732 tokens, cut 960).

| subject | suffix logits = independent / = joined | 256 tokens + final logits identical | prefill, 8 questions: independent → shared | restore | snapshot |
|---|---|---|---|---|---|
| high_school_statistics | 24/24 / 24/24 | 24/24 (forward, reverse, interleaved) | 53.9 → 23.2 s (0.43×) | 8.3 ms | 231 MiB |
| high_school_world_history | 24/24 / 24/24 | 24/24 | 87.8 → 41.5 s (0.47×) | 9.2 ms | 271 MiB |

The geistlib side of forked reasoning is complete. The remaining work is in geist-runtime: chat forks from snapshots, because today a hybrid chat re-prefills.

## 3. Batched multi-sequence decode: prototype gate

The prototype is on branch `research/26-multiseq-proto` (commit fa3cd6ad, not for merge; `step3/prototype.diff`). It adds `transformer_multi_decode_step`, internal only, for Qwen3.5-family models on Metal. It steps n sequences per pass as follows:
- **Shared across the n rows:** every linear runs once over a carrier session's n rows. That covers the projections, the FFN and the lm_head, on the #186/#221 small-row kernels.
- **Per sequence:** RoPE, the KV append, attention and the fused DeltaNet mixer run once per sequence, on that sequence's own KV cache, recurrent state and position.

The measurement is `tests/bench_multiseq_decode.c`:
- 8 + 8 MMLU reasoning prompts are prefilled and snapshotted, so every run starts from the same state.
- Greedy decoding.
- 32 steps for agreement and invariance, 64 timed steps × 3 interleaved rounds for speed. One lease, cool start.

| Bonsai 2 27B, Metal | ms/step | aggregate tok/s | vs single |
|---|---|---|---|
| single-sequence decode | 52.1 | 19.2 | 1.00× |
| B = 1 (prototype) | 55.0 | 18.2 | 0.95× |
| B = 2 | 89.6 | 22.3 | 1.16× |
| B = 4 | 96.6 | 41.4 | 2.16× |
| **B = 8** | **120.1** | **66.6** | **3.47× — gate (≥ 2×) passed** |

Context: `prefill_tokens(m)` on one sequence costs 105–116 ms at m = 2..8 and 491–557 ms at m = 16 (step 2). The prototype's per-sequence attention and DeltaNet dispatches therefore add only about 5–15 ms at B = 8. B = 16 would need a different kernel tier.

**Isolation and agreement:**
- **B = 1:** bit-identical to single-sequence decode at every step (same kernels).
- **Companion invariance:** A0's logits are **bit-identical in every step** whether A0 sits in slot 0 among A1..A7 or in slot 3 among B0..B6 (other questions, 1.3–1.7k-token contexts). The same holds at B = 4 (slot 0 among A1..A3 against slot 2 among B0..B2). Rows of the m-row GEMM do not interact.
- **Against independent decode:**
  - B = 2/4/8: 8/8 sequences are token-identical over 32 steps.
  - The logits are not bit-identical (max |difference| 0.004 / 0.004 / 0.009), because the m-row kernels round differently from the m = 1 matvec.
  - Over long reasoning (up to 2,048 tokens), a greedy path can diverge. Whether accuracy changes has to be measured; see the proposal.
- Qwen3.5 0.8B Q8_0 (`step3/proto08.out`): the same isolation results. Its speed is no guide, because Q8_0 has its own small-row cliff at m = 2..6 that #186/#221 fixed only for PQ2_0.

## What this means for the #204/#210 economics (model, not measured end to end)

These figures use the #210 per-question time model with the measured factors: the prefix route at 0.40× for prompt prefill, and batched decode at 3.47× for decode throughput. With both, multi-question reasoning workloads come to about:

| policy | today | prefix only | prefix + B = 8 batching |
|---|---|---|---|
| always-direct | 0.114× | 0.05× | 0.05× |
| always-reasoning, cap 2048 | 1.00× | 0.92× | **0.30×** |
| two-stage primary (#204) | 0.97× | 0.85× | 0.31× |
| route 70 %, cap 1536 | 0.82× | 0.70× | 0.26× |

Times are relative to today's always-reasoning at cap 2048. Batching is a **throughput** gain for several questions in flight; each question's latency rises about 2.3× (120 vs 52 ms/step). Accuracy under batched reasoning is not yet measured (see below).

## API proposal (AGENT.md §6): batched decode of independent sessions

This is new experimental API only. No existing signature changes.

```c
/* @stability EXPERIMENTAL (#26) — one decode step for n independent sessions
 * of the same model in one pass. Equivalent to geist_session_decode_step on
 * each session in turn, except numerics: n > 1 runs the linears as one
 * n-row GEMM, whose rounding differs from n = 1. Guarantees: a session's
 * result does not depend on the other members or its slot (bit for bit);
 * n = 1 is identical to geist_session_decode_step. Sessions must be
 * distinct, created on this model and backend, each with pending logits
 * (after a prefill or a decode step). Fails before touching any session for
 * invalid arguments, unsupported members (KIVI, MTP, audio stream open,
 * split/offloaded models) or a full KV cache; a backend failure during the
 * pass resets every member. */
[[nodiscard]] enum geist_status geist_session_decode_batch(size_t n,
                                                           struct geist_session *const sessions[],
                                                           geist_token_t out_tokens[]);

/* @stability EXPERIMENTAL — the largest n decode_batch accepts for this
 * model on its backend; 0 = unsupported (no fused DeltaNet mixer, PLE
 * families, CPU backends for now). */
[[nodiscard]] size_t geist_model_decode_batch_max(const struct geist_model *m);
```

- **Carrier:** the model owns an internal scratch session, created on first use with `m_max` rows and serialized with the model's other operations. It is not public.
- **Sampling:** each session keeps its own sampler, penalties and RNG. The head runs once over the n rows; each row's sampling stays in its own session.

**Bounded batches (§6), each with tests and a benchmark gate:**
1. **arch op.** Add an optional `decode_batch` entry to the `geist_arch` vtable (nullptr = unsupported). Implement it for the transformer from the prototype, qwen35 family on Metal. Tests:
   - n = 1 equals decode_step;
   - companion and slot invariance, bit-exact;
   - token agreement with independent decode;
   - atomic refusal;
   - a member at its KV limit.
2. **Public symbols, docs, changelog.** `docs/API_CONTRACT.md` lists the symbols as experimental. A new `make test` real-model case covers them.
3. **Kernels.** One dispatch per layer for the B sequences' attention (B KV caches) and DeltaNet (B states), replacing the prototype's B dispatches. Vulkan follows.
4. **Families without the gated qwen35 block.** llama/gemma attention split into projection/mix/output stages, and the CPU fallback (the host DeltaNet oracle per sequence).
5. **Consumer.** geist-runtime: continuous batching for `decide` escalation and `decide-batch` over a snapshot-forked prefix. Separate issue in that repository.

**Gates before each merge:**
- `bench_perf_sweep` and `tools/bench_revision_ab.py`: no regression of single-sequence decode or prefill (the existing path is untouched).
- Batched throughput at B = 4/8 at least as measured here.
- Quality: the 500-question reasoning arm at cap 2048 run batched (B = 8). The paired comparison against the #171 reasoning outputs has to show no accuracy change beyond noise.
- Cost: about 2 h of Mac time at 3.47× (23,894 s / 3.47, plus prefill).

**Stop here for the user's decision.**

## Mac time

| part | lease time |
|---|---|
| step 1, 256-cut run (4 chunks) | 73 min |
| step 1, 320-cut prefix re-run (1 chunk) | 21 min |
| step 2 (2 leases) | 24 min |
| step 3 Bonsai (1 lease) + 0.8B checks | 5 + 3 min |
| **total** | **about 2.1 h** |

Raw data: `raw/decision-shared-state-2026-10-25/` (manifests, reports, driver lines and temperatures per chunk, step-2/3 outputs, both tool sources, the prototype diff). The compact numbers are in `DECISION_SHARED_STATE_2026-10-25.json`.
