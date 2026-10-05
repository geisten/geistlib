# Pretrained encoder development comparison — 2026-10-05 (#587)

ModernBERT-Large-Instruct and Laya Typed Decisions now have explicitly enabled,
local Python reference adapters. These are executable comparisons, **not native
geist C encoder support**. On the Apple M1 Max, ModernBERT zero-shot answers the
778 cleaned development questions with 42.42% accuracy and a 42.53ms warmed
adapter-call median. Laya zero-shot reaches 34.70% population accuracy. Neither
result establishes a general Bonsai replacement or Jev-equivalent quality.

The [compact result](DECISION_ENCODERS_2026-10-05.json),
[raw evidence](raw/decision-encoders-2026-10-05/SHA256SUMS) and
[adapter instructions](../../docs/DECISION_ENCODERS.md) retain the policy and
limitations. No training, paid compute, or calibration/test inference took place.

## Complete development policies

Apple M1 Max, 64GiB, PyTorch MPS, float32, six CPU threads, concurrency one;
57 subjects, 778 unique canonical questions, fixed subject exemplars, rotation0.
Each accepted case has one warmup and one measured call. An input which would
be shortened is rejected before inference and counts as incorrect in population
accuracy. Latency covers accepted calls only, not a fallback route.

| Model / examples | Token cap | Correct / 778 | Population accuracy | Rejected | Request p50 | Request p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ModernBERT / 0 | 1,024 | 330 | 42.42% | 0 | 42.53ms | 108.00ms |
| ModernBERT / 5 | 1,024 | 278 | 35.73% | 149 | 210.21ms | 386.11ms |
| Laya / 0 | 1,024 | 270 | 34.70% | 5 | 59.88ms | 157.20ms |
| Laya / 5 | 1,024 | 190 | 24.42% | 149 | 286.25ms | 474.56ms |
| ModernBERT / 5, recovered exploration | 8,192 | 334 | 42.93% | 0 | 223.56ms* | 971.40ms* |

The first four policies and the 1,024-token cap were frozen in the retained
`matrix_plan.json`. The 8,192-token geometry was added **after** those results,
on development only, to distinguish the chosen cap from ModernBERT's native
context. Actual prompts in that extension are at most 3,013 tokens. All retained
first/repeat predictions agree within each policy.

ModernBERT's initial five-shot run accepts 629 questions and answers 278/629
(44.20%) correctly on that subset. Laya accepts 773 zero-shot and 629 five-shot
questions: the accepted-only scores are 34.93% and 30.21%. These subsets differ
from the full population; they are not substitutes for the accuracy column.
Laya's trained context is 1,024 tokens and the native builder also limits each
option to 48 tokens and the question head to 256. Raising its cap would not be
a supported solution. The adapter records the exact rejected inputs/reasons.

### MPS failure and explicit recovery

The continuous 8,192-cap run failed after 614 completed questions. PyTorch
reported 3.02GiB in MPS allocations plus 85.11GiB in other driver allocations,
at its 88.13GiB limit. The failed request contains 1,703 tokens; an earlier
3,013-token request had succeeded. In a fresh process that same failed request
succeeds, with driver-allocated memory increasing from 2.01GiB after model load
to 2.88GiB after the request. This points to growth over the run rather than a
simple inability to process that input; it does not identify the allocator or
graph-cache root cause conclusively.

An explicit fresh-process continuation scores only the remaining 164 questions.
The recovered quality result combines disjoint sets of 614 and 164 questions,
with identical checkpoint/source/device/dtype/prompt policy, exactly two
completed trials each, and no missing or duplicated identities. The original
failure remains a failure record; no successful continuous report is fabricated.
No memory limit was disabled and no automatic CPU fallback was used.

*The recovered p50/p95 summarize successful warmed requests across both
processes. They exclude the failed allocation, process restart, reload and
continuation preparation, and are **not operational end-to-end service latency**.
This geometry needs a bounded-memory runtime before deployment. The compact
result distinguishes completed policies, the 164-question continuation, failed
runs and the combined recovered exploration.

## Eight-question CPU/MPS pilot

The same eight canonical development identities as the historical Bonsai pilot
were evaluated first. Both prompt policies were declared before this pilot.

| Model / examples | Correct / 8, CPU and MPS | CPU request p50 | MPS request p50 |
| --- | ---: | ---: | ---: |
| ModernBERT / 0 | 5 | 142.03ms | 39.76ms |
| ModernBERT / 5 | 6 | 512.82ms | 216.64ms |
| Laya / 0 | 3 | 133.95ms | 44.82ms |
| Laya / 5 | 1 | 546.66ms | 289.35ms |

All CPU/MPS choices match for these eight identities. Maximum candidate-logit
differences are 1.34e-5 for ModernBERT and 4.30e-6 for Laya. No pilot input is
rejected. The eight-run matrix contains 128 forward calls; its wall time is
67.07 seconds. Eight questions are a functional pilot, not a reliable quality
estimate, which is why the larger development population is reported above.

## Provenance and independent controls

| Component | Immutable revision |
| --- | --- |
| `answerdotai/ModernBERT-Large-Instruct` | `9943452941e79c8c35ede72e78a38a8175a79bb5` |
| `convaiinnovations/laya`, `typed-decisions/` | `7b928d828b7b0e022f929d9bd2e44165aa270148` |
| Classic `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` |
| Eight-question inference source | `2e7baff` (full revision in each manifest) |
| Development and recovered inference source | `23e43426e7c68d5b037d18b4d501fe001642ddad` |

The checkpoint weights/config/tokenizer files are hashed before and after
successful runs and read locally without remote checkpoint code. Python 3.14.7,
torch 2.12.0, transformers 5.7.0 and Laya SDK 0.3.26 are recorded. This is the
typed checkpoint, not Laya's default English checkpoint. The measured source
revisions remain frozen even though later commits add setup-failure evidence,
local-layout guards and p95 reporting.

The MMLU preparation reuses the exact-overlap exclusion and deterministic
partition from the Bonsai evaluation: 778 development, 750 calibration and
13,904 test questions after exclusions. Five exemplars per subject come from
the source dev split. Only development questions receive model calls here;
calibration and held-out test receive no inference or fitting. This cleaned
classic-MMLU protocol is different from Bonsai's published MMLU-Redux thinking
protocol. Exact exclusion does not establish semantic deduplication or absence
of pretraining contamination.

ModernBERT's mask-position-only vocabulary head was checked against the full
native `AutoModelForMaskedLM` head for 16 prompts. Every candidate argmax agrees;
maximum candidate-logit error is 2.67e-5. The same backbone input is retained;
small matrix-shape rounding differences are measured, not called bit identity.
Laya's native SDK `Agent.predict` was compared on eight zero-shot questions:
all choices agree, with at most 5.00e-5 probability difference from its four-
decimal rounded output and zero SDK CPU fallbacks. SDK 0.3.26's optional
`backend="eager"` path imports a module absent from that wheel; the independent
control uses the working public `Agent(..., fast=False, compile=False)` path.

Raw-logit and shipped-temperature log-loss, Brier, ECE and reliability bins are
recorded separately. Laya's four-choice temperature is 1.7601518630981445; its
small development ECE is not evidence of calibrated correctness on unseen data.
No MMLU calibration is fitted. The act/escalate head is not used for routing.

The archived bundle contains 99 files plus a SHA-256 index (4,402,588 bytes
excluding the index). JSONL is deterministically gzip-compressed; report hashes
bind the decompressed bytes. Native controls, prompts/IDs, logits/probabilities,
per-call timings, rejections, the Unicode-read failure, MPS failure and recovery
are retained. Checkpoint weights and the temporary SDK copy are not archived.
The completed policies plus recovered geometry contain 7,302 forward calls;
independent native controls and the one-request memory diagnostic are separate.

## Decision and remaining gate

The default-off reference modules make pretrained models testable without a
Bonsai fine-tuning job. ModernBERT zero-shot is the practical next development
baseline here: nearly the recovered five-shot accuracy, much shorter requests,
and a completed continuous run. This is a development choice, not held-out
selection or a guaranteed quality margin.

These measurements do not establish a quality-matched speedup against Bonsai,
Jev equivalence, native Prism parity or production acceptance. The historical
Bonsai comparison uses eight questions, different prompts/tokenizers and an
older C binary; its seconds-per-request cannot be combined with the full
778-question encoder quality as an equal-quality speedup. Request timing also
excludes request construction, startup, artifact hashing and dataset/network
costs. Peak process RSS does not measure all Apple GPU/driver memory, as the
failure demonstrates. No cold-cache or concurrent-service result is claimed.

#587 remains open for native Prism parity, negative/ambiguous controls, label
rotations, calibration and held-out evaluation. Native C encoder support needs
its own architecture/mask-scoring implementation. Bonsai classifier training
(#588) stays gated; the current encoder evidence does not justify paying for
training or claiming the same answer quality from a faster model.

## Repository validation

`make format-check` passes for all 477 C files. `make test-py` passes, including
the 12 encoder regression tests for feature opt-in, local checkpoint layout,
input bounds, setup-failure evidence, Unicode JSONL, population separation,
probability metrics and repeat drift. Release and ASan/UBSan unit suites with
`DECISION=1`, CPU NEON/scalar and Metal each pass 76 tests, with 47 documented
platform/fixture skips and zero failures/errors. Checks ran serially after
model measurements. This iteration changes no C files; logs and the validated
source revision are retained under the raw bundle’s `validation/` directory.

Run `python3 benchmark/results/raw/decision-encoders-2026-10-05/verify_evidence.py`
to verify archived checksums, complete populations and regenerated metrics
without optional ML packages.
