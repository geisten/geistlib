# Offline decision evaluation on Apple Silicon (#587)

This is evaluation infrastructure, not a Bonsai accuracy result or a Jev
speedup claim. It builds on `docs/DECISION.md` and leaves the default build
flag off. The first adapter supports **classic `cais/mmlu`, four choices and
one integer gold label**. It does not implement MMLU-Redux's corrected and
multiple-answer annotations.

Before a labelled campaign, settle the dataset variant, allowable accuracy
loss, run scope and generation cap. The CLI deliberately requires the cap,
margin and purpose. `manifest.json` freezes these and the precise commands
before inference. An existing output directory is never overwritten.

## Confirmed campaign choices (2026-10-04)

The user selected classic MMLU with five fixed subject-specific examples,
a maximum two-percentage-point accuracy loss (paired 95% interval), and a
small development pilot first on Apple CPU/Metal. These choices are saved in
[DECISION_CAMPAIGN_PLAN.json](DECISION_CAMPAIGN_PLAN.json). The pilot supplies
exploratory quality and runtime evidence, not held-out acceptance.

The user also approved explicit overlap exclusion and the bounded pilot:
eight development questions from eight deterministic subjects, cap512,
`--warmup 0 --repeats 1` (two total trials), Metal first and CPU afterwards.
The full held-out population is reported as a cleaned subset, not stock MMLU. Tokenizer-only planning of the proposed eight-question
subset finds 399–695 prompttokens and validates all 96 candidate boundaries.
At the saved planning rates, both trials of all eight arms with a 512-token
generation cap would take approximately 18–26 minutes on Metal and 88–124
minutes on CPU if every reasoning output reaches the cap. These spans exclude
setup and host load; they are not measured campaign timings. No quality result
is attached to this plan yet.

## Data and prompt construction

`tools/prepare_decision_eval.py` reads an existing immutable HF snapshot
locally; it neither downloads nor loads an unpinned `datasets` revision. Pin
`cais/mmlu` revision `c30699e8356da336a370243923dbaf21066bb9fe` and retain the
three parquet checksums. The snapshot directory name must match the declared
revision; content hashes are the provenance evidence, not authentication of a
caller-supplied dataset directory.

Fixed subject-specific `dev` examples remain the few-shot pool. Validation is
split deterministically by subject and question hash into development and
calibration. Test remains held out. Exact question/choice fingerprints detect
reordered options too; only outer whitespace and CRLF are normalized, preserving
code indentation. This is not semantic near-duplicate detection.

The overlap policy is **required**. `reject` fails on a duplicate, earlier-split
overlap or conflicting annotation. `exclude` saves every excluded identity and
reason and reports the resulting population as a **cleaned MMLU subset**.
Conflicting annotations that touch fixed few-shot examples always fail. All
earlier split identities are reserved, including exclusions.

The cached pinned corpus has dev 285, validation 1,531 and test 14,042 rows.
A factual audit under the explicit exclusion policy yields development 778,
calibration 750 and test 13,904 questions (108 within-split duplicates and
33 earlier-split overlaps excluded from validation/test). This audit selects
no campaign policy and produces no model accuracy evidence.

Three profiles are saved with their complete prompt text and token IDs:

| Profile | Prompt / label policy | Benchmark arms |
| --- | --- | --- |
| `chat_direct` | Actual GGUF template, thinking disabled, final letter instruction, bare A/B/C/D | DENSE, selected rows, independent dense reference, one generated token |
| `cloze` | Conventional subject-specific MMLU completion ending in `Answer:`, spaced letters | DENSE, selected rows, independent dense reference |
| `chat_reasoning` | Same user question/exemplars, actual template, thinking enabled, xhigh | Ordinary greedy generation until model EOS or the declared token cap |

Chat rendering uses the model's stored template with Transformers' sandboxed
Jinja implementation. Save that template, its checksum, package versions and
the fixed template date. The runtime `TOK`/`BOS` protocol supplies GGUF-native
tokenization; the BOS policy and ID must agree with metadata. Every **complete
prompt + label** must extend the actual prompt prefix by exactly one distinct
token. Standalone tokenization or truncating a multi-token label is insufficient.
Text exceeding the TOK protocol's lossless extent fails before tokenization.

`--per-subject N` deterministically samples a development pilot independently
of labels. `0` retains the whole prepared population. `--subject-limit N`
selects a bounded set of subjects by a separate deterministic hash, also
independent of labels. For example `--per-subject 1 --subject-limit 8` prepares
eight pilot questions when at least eight subjects are available. `--rotations 4` adds
cyclic option permutations with a correctly remapped gold label; primary quality
uses only variant 0. Rotations and timing repeats never increase quality N.

## Commands

Build the existing public driver and tokenizer, with the module enabled:

```sh
make DECISION=1 BACKENDS='cpu_neon cpu_scalar metal' bin
```

Prepare cases after selecting the policy. Set the shell variables to your
actual paths and agreed choices; the required variables have no implicit
quality defaults:

```sh
python3 tools/prepare_decision_eval.py \
  --snapshot "$decision_mmlu_snapshot" \
  --dataset-revision c30699e8356da336a370243923dbaf21066bb9fe \
  --model "$decision_model" --tokenizer-binary "$decision_tokenizer_binary" \
  --out-dir "$decision_cases_dir" \
  --overlap-policy "${decision_overlap_policy:?choose reject or exclude}" \
  --shots 5 --splits development,calibration,test --per-subject 0 --rotations 1 \
  --template-date "$decision_template_date"
```

Preparation additionally needs locally installed `gguf`, `pyarrow`,
`transformers` and `jinja2`. The metrics, runner and all Python unit tests need
only the standard library. Metadata extraction deliberately skips tensor dtype
interpretation: upstream gguf-py may not recognize Prism's PQ2_0 enum 142.
It does not alter or dequantize the model.

Run a development pilot with an explicit cap and margin, first on one backend
at a time. Use a pilot-prepared bundle (`--per-subject N`, e.g. one per subject)
if a bounded duration is needed:

```sh
python3 tools/eval_decision.py \
  --cases-dir "$decision_cases_dir" --split development --purpose pilot \
  --model "$decision_model" --binary "$decision_benchmark_binary" \
  --backend cpu_neon --threads 6 --warmup 0 --repeats 1 \
  --decode-cap "${decision_decode_cap:?declare cap}" \
  --margin-pp "${decision_margin_pp:?declare accuracy margin}" \
  --out-dir "$decision_run_dir"
```

Use `--backend metal` for a separate serial campaign. Do not overlap inference
with your own builds or another benchmark. Record any external host load; this
runner does not isolate the shared desktop. Environment knobs are retained.

For temperature fitting use `--split calibration --purpose calibration` with
the same bound configuration and a separate output directory. It runs the
three numeric arms on `chat_direct` and saves `calibration.json`. A subsequent
`--split test --purpose quality --calibration /path/calibration.json` requires
all declared test questions (`--per-subject 0 --subject-limit 0`), an identical checkpoint,
binary, tokenizer, device/backend/KV/environment, shots, template date and
prompt policy, and disjoint calibration IDs. The runner rejects a pilot bundle
masquerading as a complete test.

## Measurements and quality gates

The driver accepts `--modes name,name` and `--text`; its existing default arms
are unchanged. Selected rows require an accompanying DENSE arm and explicit
`--selected`. Numeric parity includes exact selected/DENSE float logits on
**every paired trial**, not a tolerance relaxed for this evaluation. Independent
SCOREALT-style candidate scores retain the existing reference tolerance.

The generated output retains every emitted ID (including EOS), a stop reason,
raw surface bytes encoded as hex, a completeness flag and separately measured
surface conversion time. Bytes are concatenated before strict UTF-8 decoding.
An incomplete surface or invalid UTF-8 cannot become a guessed answer. Greedy
IDs, numeric argmax and surfaces must stay stable across repeats. The surface
API returns C strings; missing/empty non-EOS pieces mark output incomplete.

For thinking mode, answer parsing requires exactly one `</think>` terminator
and accepts only the final A/B/C/D, optionally prefixed by `Answer:`. It never
extracts a letter from the reasoning or guesses the last letter of explanatory
text. Invalid answers count as incorrect. Capped outputs are separately
reported even if the available suffix parses as a letter. The single-token arm
reports whether that token forms a valid label; its one-token cap is explicit.

Uncalibrated conditional option probabilities have question-weighted accuracy,
log-loss, Brier **sum across classes**, equal-width ECE, reliability bins and
per-subject results. Temperature scaling minimizes calibration log-loss over
`[0.01, 100]`, reports boundary fits and binds the exact policy. It changes
reported confidence, not argmax. No held-out labels fit that temperature.
Generative arms report final-answer accuracy, invalid/capped counts and actual
generated-token distributions; they do not invent option probability metrics.

Direct versus reasoning accuracy is paired by canonical question identity.
The conservative interval uses exact Clopper-Pearson intervals for gain/loss
probabilities with Bonferroni joint 95% coverage. It assumes independent
questions from the target population. A subject-stratified development pilot
is exploratory. Zero observed disagreements in a tiny pilot still have a wide
interval; agreement alone does not demonstrate the margin.

A quality-matched comparison is eligible only for a complete held-out quality
run, when the interval establishes the declared margin and the reasoning
baseline has **no invalid or capped answers**. Otherwise the report explicitly
lists blockers and retains descriptive latency ratios. Eligibility does not
authorize deployment or establish external runtime parity.

Warm sample p50/p95, question-median distributions and first-call medians are
separate. Ratios sum per-question median latencies across the same population;
they do not average convenient speedup examples. The sequential QPS value is
a reciprocal latency estimate, not a measured concurrent service throughput.
Preparation loads and checksums the model: first calls are **not cold cache**.
No cache eviction or independently isolated device cache is claimed.

The additive component estimate includes recorded prompt construction/TOK,
inference, surface conversion and answer parsing. It is not measured service
end-to-end latency. Dataset preparation, boundary-validation tokenizations,
model startup, checksum reads and reference arms are setup/campaign costs;
preparation wall time/TOK call count and run wall time/model-query count retain
those costs separately.
There is no abstention or fallback route. Peak process RSS includes the model
and all retained session/decision handles, not incremental memory per arm.

## Remaining evidence for #587

The framework does not fulfill the whole evaluation ticket. A pinned native
Prism **same-token** reference must still compare identical prompts and forced
suffixes (including longer/high-entropy cases), with both attention and
recurrent state reset. REST top-k alternatives with missing candidate scores
are not sufficient. Keep this separate from free-running accuracy comparison.

Actual calibration/test runs, representative ambiguous/negative controls,
external parity, agreed quality margins and the resulting decision on whether
classifier training is justified remain outstanding. No claim follows from
synthetic CPU/Metal wire-protocol smoke tests alone.

Bonsai's published 89.09 PQ2_0 score is **zero-shot MMLU-Redux 2.0 with
thinking**, up to 32,768 generated tokens and sampled decoding (T=1, top-p=.95,
top-k=20), not this classic five-shot greedy policy. It is contextual evidence,
not the acceptance target. Pin the source rather than conflating variants:

- [Pinned Bonsai GGUF model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/README.md)
- [Pinned Bonsai whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/bfaea577522626b883f755236878e4583f3d6e68/bonsai-2-27b-whitepaper.pdf)
- [Pinned classic MMLU snapshot](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)
- [Original MMLU evaluator](https://github.com/hendrycks/test/blob/master/evaluate.py)
