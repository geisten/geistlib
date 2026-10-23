# Optional numeric decisions (experimental)

Build with `make DECISION=1`. Default builds (`DECISION=0`) retain linkable
symbols but return `GEIST_E_UNSUPPORTED` from create/score; capability
queries return false/0. Switching the flag rebuilds the implementation and
relinks consumers, including `1 -> 0`, without `make clean`. Flag transitions
force compile/archive/link even with one-second build timestamps.
`make test-decision` verifies both modes in the same output directory;
`make MODE=asan test-decision` repeats that gate with sanitizers. The runtime query
`geist_decision_available()` describes the linked library, not the caller's
compiler flags.

## Integration

`include/geist_decision.h` is an independent, additive API. A handle owns a
private session and candidate workspace. It borrows the model and backend;
destroy handles before the model, then the backend. Architecture dispatch
owns KV/SSM state and model-conformant logits. The decoder vtable
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
candidate order. Score, including failed calls, invalidates borrowed
results on that handle; operations on other handles do not. The result struct
is cleared on every failure (count 0, null pointers, best index `SIZE_MAX`).
Null/invalid/duplicate IDs and candidate overflow return `GEIST_E_INVALID_ARG`;
prompt overflow returns `GEIST_E_TOO_MANY_TOKENS`. Selected NaN/Inf logits or
missing runtime logits return `GEIST_E_BACKEND`. Non-selected logits do not
enter normalization. The numerical API and its adversarial/reference checks
compile with `-fno-finite-math-only`, so target-wide GCC fast-math cannot remove
NaN/Inf validation; inference kernel flags are unaffected. A failed prefill
propagates its status, and the next valid query resets again. No partially filled result escapes.

Setup/teardown are serialized with every other operation on the model.
Steady-state scoring may run concurrently on different handles, one thread
per handle, according to the decoder session contract. No tokenizer is called
by scoring; serialize the existing tokenizer separately. Weights, generation
sampler settings and a shared mode are never mutated. Candidate buffers and
the duplicate-detection table are sized with checked arithmetic at creation;
scoring allocates no candidate workspace. Existing architecture/backend
high-water scratch may warm on first use, as with ordinary prefill.

## Several questions over one context: prefix scoring (#5)

`make DECISION=1 DECISION_CONTEXT=1` adds a DENSE-only path that prefills a
shared prefix once and answers each question from that state.
`DECISION_CONTEXT=0` is the default; `DECISION=0 DECISION_CONTEXT=1` is
rejected by the build. Off builds keep the symbols: they return
`GEIST_E_UNSUPPORTED`, clear their outputs, and `geist_decision_prefix_available()`
is false. `make test-decision` walks 1/1, 1/0, 0/0 and back to 1/1 in one
output directory.

```c
/* d: a DENSE handle created with .kv_mode = GEIST_KV_FP32. */
size_t chunk = 0;
geist_decision_prefill_chunk(&chunk, d);  /* cut the prefix at a multiple */
struct geist_decision_prefix *p = nullptr;
status = geist_decision_prefix_prepare(d, max_snapshot_bytes,
                                       n_prefix, prefix_ids, &p);
for (each question) {
    status = geist_decision_score_from_prefix(d, p, n_suffix, n_candidates,
                                              suffix_ids, candidate_ids, &result);
}
geist_decision_prefix_destroy(p);  /* before the model */
```

The prefix is the session's complete state from `geist_session_snapshot`
(#548): every used KV row, each gated-DeltaNet layer's conv and recurrent
state, positions and pending logits. There is no second checkpoint format
and no attention-only pinning, which cannot rewind a hybrid's recurrence.
Every score restores the whole image before prefilling the suffix, so no
question sees another's state, and a failed restore leaves the session reset
for the next call to restore again.

- **Binding.** A prefix records the model, backend, resolved prefill chunk
  and `max_prompt_tokens` it was taken under, and requires a plain FP32 KV
  cache. Any handle on the same model with the same geometry may score from
  it; another model, backend, chunk or capacity is `GEIST_E_INVALID_ARG`.
  INT8/INT4/F16/KIVI caches, an enabled MTP drafter, `SELECTED_ROWS` and
  architectures without snapshots are capability-negative
  (`geist_decision_prefix_supported`) and `GEIST_E_UNSUPPORTED`.
- **Memory.** `prepare` checks the image size against `max_snapshot_bytes`
  before allocating (`GEIST_E_OOM` with nothing allocated above it);
  `geist_decision_prefix_get_info` reports cached tokens, snapshot bytes,
  chunk, capacity and the cache mode. Warmed scoring allocates nothing.
- **Capacity.** `n_prefix` may be 0 (the reset state; no zero-token prefill
  runs) and must be below `max_prompt_tokens`; cached + suffix above it is
  `GEIST_E_TOO_MANY_TOKENS`. The suffix carries the uncached tail of the
  context plus the question and must be at least one token.
- **Bit-identity.** Restore + suffix equals a fresh session that prefills the
  prefix and then the suffix, bit for bit, on CPU NEON, scalar and Metal
  (`tests/test_decision_prefix_unit.c`, `tests/test_decision_prefix_int.c`).
  With `n_prefix` a multiple of the chunk it also equals scoring the joined
  prompt in one call. Off the boundary the joined prompt splits into other
  prefill passes; on the test fixtures the candidate logits then differ by at
  most 8.5e-7 (recorded, not asserted).

Measured on an M1 Max, Qwen3.5 0.8B Q8_0, a two-chunk prefix plus a 16-token
question, per question: cpu_neon 86 ms from the prefix against 350 ms for the
full prompt (4.1x; chunk 64, 128-token prefix, 23 MiB snapshot); Metal 34 ms
against 228 ms (6.8x; chunk 256, 512-token prefix, 32 MiB snapshot).

## Selected-row execution on Apple Silicon (#586)

Set `opts.mode = GEIST_DECISION_SELECTED_ROWS` explicitly and query
`geist_decision_mode_supported(model, opts.mode)` for the loaded model/backend
pair. The zero-initialized default remains `GEIST_DECISION_DENSE`. Unsupported
pairs return `GEIST_E_UNSUPPORTED`; the library never silently changes modes.
Consumers may deliberately create a DENSE handle instead.

The transformer reuses the ordinary chunked prompt backbone. At its final
hidden row it applies the same output normalization and Bonsai Hadamard
rotation, then invokes a backend-resolved row kernel instead of the vocabulary
projection, sampler and `peek_logits`. Head gains and final-logit softcaps
retain their existing semantics. Generation retains its original head path.

Each readout preallocates a bounded result buffer, tile map and hash table.
Candidates sharing a tile reuse one projection. Source CPU rows normally use
one-row views of the same resolved kernel. NEON's Bonsai PQ2_0 repack uses
complete eight-row groups without changing the stored weights or activation
quantization. Metal preserves the original dense pipeline selection: for
example PQ2 n8 remains n8 even for four candidates. Native tiles can include
neighboring output rows; they are discarded before conditional normalization.
The borrowed result still has exactly the requested candidates in input order.

Capability boundaries:

| Backend | Selected-row support |
| --- | --- |
| CPU Scalar | Resolved, row-separable source formats, including F32/F16/Q8/PQ2 |
| Apple NEON | Native source-row kernels, including F16/Q8, and PQ2_0 x8 |
| Metal | Supported dense and quantized head pipelines, including Q8/Q6/PQ2 n4/n8 |
| Other backends | Explicitly unsupported for this mode; DENSE remains available |

NEON F32 is excluded because changing BLAS matrix geometry changes rounding.
NEON repacks other than PQ2_0 x8 are excluded rather than reinterpreted as
source rows. I2_S **output heads** have a shared tensor scale and are excluded;
this does not exclude models with an I2_S backbone and a supported F16 head.
Backend errors and capacity failures remain explicit.

Results report `mode`, actual `projected_rows` (including valid neighboring
rows), `logit_readback_bytes` (the staged tile buffer, including tail padding)
and `head_ns` for elapsed time from final normalization through result access.
On asynchronous backends that interval can include pending backbone device
work; it is not an exclusive GPU head timer. Use the standalone head
microbenchmark to isolate projection cost.
DENSE reports its vocabulary size and logical logit-buffer byte count, with
`head_ns = 0`. Byte counts describe logical staging, not physical bus traffic
on unified memory. Private sessions still reserve the architecture's ordinary
scratch buffers, including their dense-logit storage.
Existing backend thread-local high-water buffers may grow on their first use;
repeated selected kernel calls are checked for zero geist heap allocations.
CPU tiles run the same row kernels with one OpenMP thread to avoid repeatedly
launching teams for a handful of rows; the caller's task setting is restored
on success and failure.

For a paired benchmark use `tools/bench_decision.py --selected` with the usual
arguments below. It adds `decision_selected`, requires exact float-logit parity
with the paired DENSE trial, and records the instrumentation and process peak
RSS. `tests/test_selected_rows_unit` compares dense and selected backend tiles
bit for bit, including PQ2 n8, source/repacked CPU layouts and tail rows. Set
`GEIST_BENCH_SELECTED_ROWS=1` to also measure synthetic, model-sized Qwen/Bonsai
head geometries. These are head microbenchmarks, not model quality evaluations.

Head speedup and end-to-end speedup are different: the backbone still
processes every prompt token, so neither head timing nor short-label latency
establishes Jev parity or a quality-matched reasoning win. Quality evaluation
(#587): [DECISION_EVALUATION.md](DECISION_EVALUATION.md).

## Score semantics and model suitability

DENSE uses ordinary prefill followed by `geist_session_peek_logits`, including
model-specific final-logit softcaps. It computes the full LM head. This remains the
reference for SELECTED_ROWS and does not save work in the output projection.
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

## Direct scoring first, reasoning when unsure (#204, #210)

A common consumer policy answers from the decision probabilities when the
top option is confident enough, and otherwise falls back to generated
reasoning with a token cap. This is application policy, so it is not part of
this API. The policy needs a reasoning prompt, generation, answer parsing, a
calibration and a threshold, and none of these exist here. The API supplies
what the policy reads: the candidate logits, from which a consumer computes
temperature-scaled probabilities, and an ordinary session for the fallback.
geist-runtime offers the policy as a selectable command:
`geistr decide … --threshold P --escalate chat --max-tokens N --temperature T`.
Its diagnostics report which stage answered and what each stage cost.

The trade-off is specific to the task and the model. Measured on Bonsai 2 27B
PQ2_0 (Metal, five-shot classic MMLU, 500 held-out questions), against
always-reasoning at a cap of 2048 tokens (79.0 %), with conservative paired
95 % intervals:

| Routed to reasoning | Cap | Accuracy | Difference, pp | Time |
| --- | --- | --- | --- | --- |
| 0 % (direct only) | — | 67.8 % | −11.2 [−18.2, −4.0] | 0.11× |
| 49 % | 2048 | 76.4 % | −2.6 [−6.7, +1.6] | 0.75× |
| 67 % | 1536 | 77.8 % | −1.2 [−3.9, +1.5] | 0.82× |
| 69 % (declared, precision 0.95) | 2048 | 78.8 % | −0.2 [−2.4, +2.0] | 0.97× |
| 100 % | 512 | 65.4 % | −13.6 [−17.4, −9.5] | 0.46× |

There is no large saving at reasoning-level accuracy: the confident questions
are the ones reasoning answers cheaply. A moderate cap together with routing
beats either one alone, while a short cap is worse than no reasoning. On
JevBench, direct scoring alone reaches 85.3 % (#171), so the curve sits
elsewhere for other tasks. Fit the threshold, temperature and cap on the
consumer's own calibration data.

Sources:
[DECISION_TWO_STAGE_2026-10-23.md](../benchmark/results/DECISION_TWO_STAGE_2026-10-23.md)
(the declared policy) and
[DECISION_CAP_CURVE_2026-10-23.md](../benchmark/results/DECISION_CAP_CURVE_2026-10-23.md)
(the curve over caps; exploratory).

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

## Selected-row measurements (2026-10-04)

[Full protocol, hashes, samples and sweeps](../benchmark/results/DECISION_SELECTED_ROWS_2026-10-04.json)
record M1 Max / 64 GiB, six threads, passive OpenMP waits and concurrency one.
The cases use raw synthetic integer prompts of 5/32 tokens and 4/16 spread
candidates, with one warmup and three warm repeats per case. They carry no
quality labels. This shared desktop has no CPU pinning or thermal control.

Standalone model-shaped projection means (ten warm repeats, four tiles):

| Shape / backend | Dense head | Selected head | Mean ratio | Projected rows |
| --- | ---: | ---: | ---: | ---: |
| Qwen Q8 / cpu_neon | 1671.9 µs | 10.1 µs | 165.5x | 4 |
| Bonsai PQ2 / cpu_neon | 3359.3 µs | 19.4 µs | 173.2x | 32 |
| Qwen Q8 / metal | 762.4 µs | 272.7 µs | 2.8x | 32 |
| Bonsai PQ2 / metal | 1939.4 µs | 412.7 µs | 4.7x | 64 |

Whole-query example: five prompt tokens, four candidates; p50 in milliseconds:

| Model / backend | DENSE | Selected | Up to 16 generated tokens | Dense / selected |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q8_0 / metal | 20.64 | 20.14 | 102.36 | 1.025x |
| Bonsai 2 27B PQ2_0 / metal | 496.47 | 509.10 | 1388.41 | 0.975x |
| Qwen3 0.6B Q8_0 / cpu_neon | 73.67 | 72.17 | 282.77 | 1.021x |
| Bonsai 2 27B PQ2_0 / cpu_neon | 2297.42 | 2285.64 | 4535.12 | 1.005x |

Across the four geometries, Qwen gains about 1–6% versus DENSE. Bonsai is
essentially flat and can be a few percent slower; this matrix demonstrates
no reliable Bonsai end-to-end gain. Its backbone dominates query latency.
The large head-only ratios therefore do not describe an LLM-wide speedup.
Keep mode selection explicit and measure the consumer's workload.

Every paired candidate logit is exact. Selected kernels report zero warm
geist heap allocations. For four spread Bonsai candidates, NEON stages 128
bytes (32 rows), Metal 256 bytes (64 rows), against the logical dense
993,280-byte vocabulary buffer. Private sessions still reserve ordinary
dense scratch; process peak RSS is recorded, not a per-handle/GPU memory saving.

Normal generation is covered by the 32/128-token prefill/decode sweep and a
five-repeat confirmation at length 32 in reverse process order. Raw samples
show substantial run-to-run drift and do not support fine-grained speedup
claims. Full logits and eight generated token IDs remain byte-identical
before/after on Qwen, Bonsai and Gemma, CPU and Metal.

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

The decision API performs the same dense forward as the reference. On this
single workload, it has essentially one-token latency and avoids subsequent
decode work (2.53x/1.90x versus 16 tokens respectively). The numeric paths
selected the expected digit; generation's first token was outside the digit
candidates, so its first-token label score was zero. The generated sequences
are retained but their final-answer quality has not been evaluated. This is
one smoke example, not independent evidence of task accuracy, reasoning
quality, calibrated confidence, or Jev parity. #586 optimizes the head; #587
must establish quality and latency on representative held-out tasks.

Offline MMLU preparation, final-answer parsing, calibration and paired quality
gates: [DECISION_EVALUATION.md](DECISION_EVALUATION.md).
