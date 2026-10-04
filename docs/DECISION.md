# Optional numeric decisions (experimental)

Build with `make DECISION=1`. Default builds (`DECISION=0`) retain linkable
symbols but return `GEIST_E_UNSUPPORTED` from create/reset/score; capability
queries return false/0. Switching the flag rebuilds the implementation and
relinks consumers, including `1 -> 0`, without `make clean`. Flag transitions
force compile/archive/link even with one-second build timestamps.
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
enter normalization. The numerical API and its adversarial/reference checks
compile with `-fno-finite-math-only`, so target-wide GCC fast-math cannot remove
NaN/Inf validation. Inference kernel flags are unaffected. A failed prefill propagates its status, and the next
valid query resets again. No partially filled result escapes.

Setup/teardown are serialized with every other operation on the model.
Steady-state scoring may run concurrently on different handles, one thread
per handle, according to the decoder session contract. No tokenizer is called
by scoring; serialize the existing tokenizer separately. Weights, generation
sampler settings and a shared mode are never mutated. Candidate buffers and
the duplicate-detection table are sized with checked arithmetic at creation;
scoring allocates no candidate workspace. Existing architecture/backend
high-water scratch may warm on first use, as with ordinary prefill.

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

Initial capability boundaries:

| Backend | Selected-row support |
| --- | --- |
| CPU Scalar | Resolved, row-separable source formats, including F32/F16/Q8/PQ2 |
| Apple NEON | Native source-row kernels, including F16/Q8, and PQ2_0 x8 |
| Metal | Supported dense and quantized head pipelines, including Q8/Q6/PQ2 n4/n8 |
| Other backends | Explicitly unsupported for this mode; DENSE remains available |

NEON F32 is excluded because changing BLAS matrix geometry changes rounding.
NEON repacks other than PQ2_0 x8 are currently excluded rather than reinterpreted
as source rows. I2_S **output heads** have a shared tensor scale and are excluded;
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
scratch buffers; this change does not claim to remove their dense-logit storage.
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

The rollout order is Apple Silicon CPU/Metal, followed by other backends.
Ticket #587 starts with MMLU and must define its quality baseline and acceptance
margin before evaluation. Head speedup and end-to-end speedup are different:
the backbone still processes every prompt token, so neither head timing nor
short-label latency establishes Jev parity or a quality-matched reasoning win.

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
before/after on Qwen, Bonsai and Gemma, CPU and Metal. Extraction of the shared
head preparation adds 20 glue instructions on arm64; projection arithmetic
is unchanged. MMLU quality and calibration remain the next stage in #587.

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
