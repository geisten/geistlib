# Optional numeric decisions (experimental)

Build with `make DECISION=1`. Default builds (`DECISION=0`) retain linkable
symbols but return `GEIST_E_UNSUPPORTED` from create/reset/score; capability
queries return false/0. Switching the flag rebuilds the implementation and
relinks consumers, including `1 -> 0`, without `make clean`.
`make test-decision` verifies both modes in the same output directory;
`make MODE=asan test-decision` repeats that gate with sanitizers. The runtime query
`geist_decision_available()` describes the linked library, not the caller's
compiler flags. Existing generation and Bonsai support stay available.

## Integration

`include/geist_decision.h` is an independent, additive API. A handle owns a
private session and candidate workspace. It borrows the model and backend;
destroy handles before the model, then the backend. Architecture dispatch
owns KV/SSM state and model-conformant logits. The appended decoder vtable
capability `logits_vocab_size` lets the engine reject embedding-only models and
validate IDs before inference. Architectures without independent sessions,
reset, prefill or logits are unsupported. A backend must expose host-readable
logits; a runtime failure is explicit rather than a fallback to generation.

```c
#include <geist_decision.h>

/* m and be were loaded normally; both outlive d. */
struct geist_decision_opts opts = {
    .mode = GEIST_DECISION_DENSE,
    .max_prompt_tokens = 512,
    .max_candidates = 4,
};
struct geist_decision *d = nullptr;
enum geist_status status = geist_decision_create(m, be, &opts, &d);
if (status == GEIST_OK) {
    /* Arrays supplied by the application's tokenizer/template. */
    struct geist_decision_result result;
    status = geist_decision_score(d, n_prompt, n_candidates,
                                   prompt_ids, candidate_ids, &result);
    if (status == GEIST_OK) {
        /* candidate_ids[result.best_index] selects the domain label. */
        consume_scores(result.n_candidates, result.logits, result.probabilities);
    }
    geist_decision_destroy(d);
}
```

Each score starts from empty attention/recurrence state. Results preserve
candidate order. Score/reset, including failed calls, invalidate borrowed
results on that handle; operations on other handles do not. The result struct
is cleared on every failure (count 0, null pointers, best index `SIZE_MAX`).
Null/invalid/duplicate IDs and candidate overflow return `GEIST_E_INVALID_ARG`;
prompt overflow returns `GEIST_E_TOO_MANY_TOKENS`. Selected NaN/Inf logits or
missing runtime logits return `GEIST_E_BACKEND`. Non-selected logits do not
enter normalization. A failed prefill propagates its status, and the next
valid query resets again. No partially filled result escapes.

Setup/teardown are serialized with every other operation on the model.
Steady-state scoring may run concurrently on different handles, one thread
per handle, according to the decoder session contract. No tokenizer is called
by scoring; serialize the existing tokenizer separately. Weights, generation
sampler settings and a shared mode are never mutated. Candidate buffers and
the duplicate-detection table are sized with checked arithmetic at creation;
scoring allocates no candidate workspace. Existing architecture/backend
high-water scratch may warm on first use, as with ordinary prefill.

## Score semantics and model suitability

DENSE uses ordinary prefill followed by `geist_session_peek_logits`, including
model-specific final-logit softcaps. It computes the full LM head. This is the
reference path for #586; it does not yet save work in the output projection.
It avoids generating an answer after the prompt, but still runs the model's
entire prompt forward pass.

The selected logits are raw model logits. The returned probability for label
`i` is `exp(logit[i] - max) / sum_candidates exp(logit[j] - max)`, computed in
double precision, with no sampling temperature, top-k or top-p. It is
conditional on the supplied candidate set. Changing that set changes the
probabilities; a singleton always has probability 1. Neither these values nor
full-vocabulary probabilities measure calibrated correctness confidence.

Candidates are distinct **single token IDs**, not token sequences. Applications
must tokenize each label with the correct boundary/template and verify it is
one token. A two-token label cannot be passed as two alternatives, nor may it
be truncated to its first token. Sequence scoring needs a separate API.
Templates, arbitrary question construction, domain label mappings, calibration
and fallback policy belong to the consumer.

Any loaded generative model satisfying the capability contract can use this
path, including Bonsai's existing PQ2_0/Hadamard path. That does not turn it
into a trained classifier or guarantee Jev quality/speed. A reasoning model's
immediate next-token answer can have different quality from its answer after
reasoning. Measure that on held-out task data (#587) before deploying.

## Reproducible reference protocol

Build `make DECISION=1 bin`. Prepare JSONL with one case per line:

```json
{"id":"example-1","prompt_ids":[1,5,9],"candidate_ids":[7,21],"target_index":0}
```

`target_index` is the expected candidate position, optional for perf-only
workloads. The example IDs illustrate the format; they are not a task dataset.
Preserve the actual IDs, tokenizer revision, template, BOS policy and how
labels were tokenized. `eval_geist TOK`/`BOS` can construct inputs using the
GGUF tokenizer. Do not take label token 0 if TOK returns multiple tokens.

```sh
python3 tools/bench_decision.py \
    --model /absolute/path/model.gguf \
    --cases /absolute/path/cases.jsonl \
    --binary bin/mac-omp/release/tools/bench_decision \
    --backend cpu_neon --threads 6 --decode-n 32 --warmup 2 --repeats 10 \
    --tokenization 'GGUF tokenizer; revision ...; template ...; BOS ...; labels ...' \
    --out-dir /tmp/new-decision-run
```

The public-API C driver times reset + prefill + numeric scoring or generation.
It rotates four paths across trials: decision DENSE, independent dense
candidate scoring adapted from SCOREALT, optimized ordinary greedy generation
of one token (the first decode consumes the cached prediction), and generation
up to the configured longer cap, stopping at model EOS. It records actual
emitted IDs/counts. The independent numeric path uses candidate normalization;
`eval_geist SCOREALT` itself returns full-vocabulary log probabilities.

The Python runner saves raw stdout/stderr, enriched samples, exact cases,
model/dataset/runtime-binary SHA256, source revision/status, hardware/OS,
backend, FP32 KV policy, environment knobs, prompt/candidate lengths, threads,
concurrency **1**, warmups, repeats, and interpolated p50/p95 for warm samples.
It checks paired dense/reference parity and rejects incomplete measurements.
Hashes are computed after timing so checksum reads do not prewarm the model.
An existing result directory is never overwritten.

The first mode call for each case is recorded separately. Page/device caches
are not forcibly evicted, modes share them, and loading/session initialization
may touch weights: these samples are **not controlled cold-cache latency**.
For a cold experiment, use a separately documented eviction/restart procedure
and retain its host/cache evidence. This protocol does not measure concurrent
throughput; add a separate concurrency experiment before claiming scaling.

Quality is candidate argmax accuracy for numeric paths and first emitted token
label accuracy for generation; an out-of-candidate token is incorrect. Without
labels, quality is null. Repeated timings are not independent quality examples.
Longer generated outputs are preserved for a downstream answer evaluator;
first-token accuracy does not assess their reasoning or final-answer quality.
Use held-out independent examples, thresholds/calibration fixed before the
test, and the same quality criterion for any product speedup claim. A latency
ratio versus 32 emitted tokens is workload-specific and is not evidence of
Jev parity or a quality-matched reasoning speedup.

## Initial protocol smoke baseline

[Raw baseline, 2026-10-03](../benchmark/results/DECISION_REFERENCE_2026-10-03.json)
contains all samples and hashes for Apple M1 Max, cpu_neon, six threads, FP32
KV, concurrency one. Both models used the raw completion `2 + 2 =` (five
prompt tokens), four independently verified single-token digit alternatives,
one warmup and five measured repeats. No chat wrapper or generated reasoning.

| Model | Decision p50 | Independent dense p50 | One-token p50 | 16-token p50 |
|---|---:|---:|---:|---:|
| Qwen3 0.6B Q8_0 | 117.55 ms | 119.76 ms | 119.30 ms | 297.37 ms |
| Bonsai 2 27B PQ2_0 | 2974.13 ms | 2925.23 ms | 2880.10 ms | 5641.53 ms |

The new interface performs the same dense forward as the reference. On this
single workload, it has essentially one-token latency and avoids subsequent
decode work (2.53x/1.90x versus 16 tokens respectively). The numeric paths
selected the expected digit; generation's first token was outside the digit
candidates, so its first-token label score was zero. The generated sequences
are retained but their final-answer quality has not been evaluated. This is
one smoke example, not independent evidence of task accuracy, reasoning
quality, calibrated confidence, or Jev parity. #586 optimizes the head; #587
must establish quality and latency on representative held-out tasks.
