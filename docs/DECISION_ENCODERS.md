# Optional pretrained encoder baselines (#587)

ModernBERT-Large-Instruct and Laya Typed Decisions are **Python reference
adapters**, not native geist C architectures. They provide a modular comparison
before deciding whether to train a Bonsai classifier. The existing C API,
`DECISION=0` default and GGUF/model capability checks are unchanged.

`tools/decision_encoders.py` imports no ML library until
`create_backend(..., enabled=True)` is called. The runner additionally requires
`--enable-encoder-backends`; a missing opt-in fails before loading artifacts.
There is no automatic model, device or generation fallback.

## Pinned first comparison

| Adapter | Public checkpoint | Revision |
| --- | --- | --- |
| ModernBERT | `answerdotai/ModernBERT-Large-Instruct` | `9943452941e79c8c35ede72e78a38a8175a79bb5` |
| Laya | `convaiinnovations/laya`, `typed-decisions/` | `7b928d828b7b0e022f929d9bd2e44165aa270148` |

Use local safetensors, configuration and tokenizer files. The runner loads
offline, refuses remote checkpoint code and hashes every model/tokenizer
artifact before and after the campaign. Do not substitute the Laya English
root checkpoint or download all of the multilingual variants accidentally.
Laya is pinned to SDK 0.3.26; upstream main may contain different behavior.

The optional dependencies are recorded in
`tools/requirements-decision-encoders.txt`. Install them in a separate virtual
environment; the default build and standard-library Python tests need none of
these packages. The recorded first campaign uses Python 3.14, float32 weights
and arithmetic, six CPU threads, concurrency one, and explicit `mps` or `cpu`.
MPS uses the Apple GPU through PyTorch; it is not geist's Metal backend.

## Data and policies

`tools/prepare_encoder_eval.py` uses the existing immutable classic
`cais/mmlu` snapshot and `decision_dataset.partition`. The exact-overlap
exclusion, development/calibration/test separation, fixed dev exemplars and
deterministic selection therefore agree with the Bonsai pilot. The eight
development identities are the same as the 2026-10-04 pilot. Calibration and
test are not used in the first comparison.

Preparation is token-independent. Zero-shot and five-shot runs share the same
questions and answer options, but have model-specific templates, tokenizers,
and token counts. Both policies must be declared before examining results.
They are not the published zero-shot MMLU-Redux/thinking Bonsai protocol, and
the historical Bonsai binary predates subsequent main-branch integration.

```sh
python3 tools/prepare_encoder_eval.py \
  --snapshot "$decision_mmlu_snapshot" \
  --dataset-revision c30699e8356da336a370243923dbaf21066bb9fe \
  --overlap-policy exclude --split development --shots 5 \
  --per-subject 1 --subject-limit 8 --rotations 1 \
  --out-dir "$decision_encoder_cases"

python3 tools/eval_decision_encoders.py --enable-encoder-backends \
  --cases-dir "$decision_encoder_cases" --split development --purpose pilot \
  --adapter modernbert --model-dir "$decision_modernbert_dir" \
  --model-repo answerdotai/ModernBERT-Large-Instruct \
  --revision 9943452941e79c8c35ede72e78a38a8175a79bb5 \
  --device mps --dtype float32 --max-tokens 1024 --threads 6 \
  --shots 0 --warmup 1 --repeats 1 --margin-pp 2 \
  --out-dir "$decision_encoder_run"
```

Run each model/device/prompt policy serially in its own new output directory.
Use `--shots 5` for the second frozen policy and `--device cpu` for the CPU
comparison. Laya's `--model-dir` points to the local `typed-decisions/`
directory, `--model-repo` is `convaiinnovations/laya`, and its revision is the
second one above. No automatic downloads happen during inference.

## Adapter semantics and validation

ModernBERT uses `AutoModelForMaskedLM`, its native instruction template,
`[unused0] [MASK]`, and distinct single-token answer keys. All prompt tokens
pass through the bidirectional backbone; the normal vocabulary head is
evaluated only at the answer position. A control run compares this with the
full native head. Floating-point matrix shapes can cause small reassociation
differences; record the measured candidate-logit error rather than claiming
bit identity. There is no answer generation or prose parsing.

Laya loads its full decision model with strict state-dict matching and uses
SDK `build_sequence`, `collate_items` and `DecisionModel.forward`. Its adapter
also exposes `decide(state, questions)` for typed `choice`, `score` and `noul`
questions. No SDK CPU/AMP fallback is invoked. Its checkpoint temperature
tables use the SDK's temperature bounds; they are not calibration fitted or
verified on MMLU. The supplied `choice:11+` value is below the SDK minimum
and is clamped; four-option MMLU uses a different, valid table entry.
The act/escalate head is not used to route requests.

The native Laya builder limits each option description to 48 tokens, may
further shorten the instruction/options to its head budget, and may cut the
state to its remaining context. The adapter independently builds the complete
expected token stream and requires exact equality with the native builder.
Any shortening or loss of a marker is an explicit `InputRejected` **before
inference**. Literal model-control strings in input also fail; they are not
silently removed. Token caps cannot exceed the checkpoint's native/trained
context. This validation is stricter than the general-purpose SDK.

## Evidence and limitations

`manifest.json` freezes the dataset policy, checkpoint/file hashes, source
hashes and revision/status, package versions, explicit device/dtype/thread
configuration and command before inference. Raw `samples.jsonl` preserves
prompt/state, options, token IDs, mask positions, logits, probabilities,
timings, repeats and rejected inputs. `report.json` binds the manifest and raw
samples by checksum. A failed inference leaves raw evidence and a failure
record; it does not become a successful report.

One warmup and one measured call **per case** give two trials. Repeats and
rotations never inflate quality N. Population accuracy counts rejected inputs
as incorrect and reports coverage separately; probability metrics and latency
cover accepted cases only. Raw-logit metrics (temperature 1) and the
checkpoint-temperature probability metrics are explicitly separate. Neither
implies calibrated correctness on this dataset. First-trial predictions supply quality; repeat
prediction and logit drift are separately recorded. A rejection-heavy fast
model is not a quality-matched speedup.

The synchronized adapter-call timing includes tokenizer/input preparation,
device transfer, forward pass and result extraction. It excludes construction
of the MMLU request, process/model startup, file hashing, dataset preparation,
and network/service costs. Process peak RSS is not total Apple GPU memory.
Weights are read and checked before timing; no cold-cache claim is made. Keep
inference separate from builds and other benchmarks. Host load is not isolated.

These exploratory results do not establish the two-percentage-point margin,
calibrated correctness confidence, Jev equivalence, or deployment acceptance.
The existing #587 calibration/test, native Prism parity and control tasks
remain outstanding. Encoder support in the public geist C API would require
a distinct architecture/mask-scoring contract; these reference adapters do not
claim that implementation.

Sources:

- [ModernBERT checkpoint and native MMLU example](https://huggingface.co/answerdotai/ModernBERT-Large-Instruct)
- [ModernBERT technical report](https://arxiv.org/abs/2502.03793)
- [Laya model family and limits](https://huggingface.co/convaiinnovations/laya)
- [Laya reference implementation](https://github.com/NandhaKishorM/laya)
