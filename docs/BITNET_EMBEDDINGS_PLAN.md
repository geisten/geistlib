# BitNet embedding models (July 2026) — analysis and implementation plan

Status: **phases 0-3 in the tree; nothing verified against real weights.** This
document records what Microsoft released in July 2026, which of it is
reproducible, and what geistlib has to grow to run it. Sources are pinned
inline so the claims can be re-checked against upstream.

Where things stand: the converter (phase 0), the per-projection input norms
(phase 1), the pooling path plus embedding API (phase 2) and the measurement
apparatus (phase 3) are in the tree and tested. The pipeline is complete end
to end on paper — a converted GGUF loads, prefills, and yields a vector — and
the gate that would prove it is built and self-tested.

What is missing underneath all of it is a numerical oracle: **no part of this
has been run against real BitNet embedding weights**, which are reachable from
neither CI nor the development container. Every test pins structure,
bookkeeping and refusal behaviour — none pins a value. Until the phase 0
reference vectors exist, treat the whole chain as "wired up and
self-consistent", not "correct".

Upstream sources read for this analysis:

- `microsoft/BitNet` `README.md` (News section, Model Releases, Supported
  Models table) — <https://github.com/microsoft/BitNet/blob/main/README.md>
- `microsoft/BitNet` `docs/bitnet-embeddings-i2s-guide.md` — the conversion and
  benchmark guide, which carries every number quoted below.

---

## 1. What actually shipped in July 2026

Three entries in the bitnet.cpp changelog, all July 2026:

| Date | Release | Relevance to geistlib |
| :-- | :-- | :-- |
| 07/16/2026 | *BitNet Embeddings 0.6B/270M: I2_S Conversion and Inference Optimization* guide | The full spec — architecture, tensor names, packing, benchmarks |
| 07/20/2026 | `BitNet-embedding-0.6B` + `BitNet-embedding-270M` on Hugging Face | **The models. In scope for this plan.** |
| 07/23/2026 | `VibeASR.cpp` — real-time multilingual ASR on CPU using BitNet I2_S, RTF < 1 | Adjacent, larger track — see §7 |

So the "new July 2026 models" are **embedding models, not generative LLMs**.
They map text to a dense vector; they emit no tokens. That is a different
product shape from everything geistlib currently runs, and it is the single
biggest thing this plan has to add.

### The models

Both are trained natively at W1.58A8 (ternary weights `{-1, 0, +1}` via absmean
quantization, 8-bit per-token absmax activations) — **not** post-training
quantized. Both use last-token (EOS) pooling followed by L2 normalization, and
both drop the LM head entirely (`output.weight` is skipped at conversion).

| | `BitNet-embedding-0.6B` | `BitNet-embedding-270M` |
| :-- | :-- | :-- |
| Backbone | Qwen3-0.6B | Gemma3 |
| GGUF `general.architecture` | `qwen3` | `gemma3` |
| Parameters | ~0.6B (595.78M) | ~270M |
| Embedding dimension | 1,024 | 640 |
| Hidden layers | 28 | 18 |
| Attention heads (KV) | 16 (8) | 4 (1) |
| `head_dim` | 128 (≠ hidden/heads = 64) | 256 (≠ hidden/heads = 160) |
| Intermediate size | 3,072 | 2,048 |
| Activation | SiLU | GELU |
| Tokenizer | Qwen3 BPE (151,936) | Gemma BPE (262,144) |
| `rope_theta` | 1,000,000 | 10,000 |
| `rms_norm_eps` | 1e-6 | 1e-6 |
| `query_pre_attn_scalar` | — | 256 |
| Embedding scaling | none | `sqrt(hidden_size)` |
| Post-attn / post-FFW norms | no | yes |
| Tied word embeddings | yes | yes |
| Max context | 32,768 | 32,768 |
| License | MIT | MIT |

### The architectural novelty: per-projection input RMSNorm

This is the part that does not exist in any current architecture, upstream or
here. Every linear projection (q, k, v, o, gate, up, down) is preceded by its
own RMSNorm applied to the projection's *input*:

```
x → RMSNorm(x, norm.weight) → activation_quant(int8) → matmul(ternary weight)
```

That is **7 extra norm tensors per layer**, named `blk.{i}.*_norm_in.weight` in
the GGUF mapping (`attn_q_norm_in`, `attn_k_norm_in`, `attn_v_norm_in`,
`attn_output_norm_in`, `ffn_gate_norm_in`, `ffn_up_norm_in`,
`ffn_down_norm_in`). Upstream states plainly that this pattern exists in
neither standard Qwen3/Gemma3 nor standard BitNet.

**Two of the seven are less new than that makes them sound.** BitNet b1.58's
SubLN is a norm on o_proj's input (`[q_out]`) and on down_proj's input
(`[intermediate]`) — which is exactly what `o_proj.norm` and `down_proj.norm`
are here, same shape, same position. geistlib already carries those two as
`attn_sub_norm` / `ffn_sub_norm` and already applies them in the forward path.
The genuinely new ones are the **five** that precede q, k, v, gate and up, all
`[d_model]`. Phase 1 below is sized accordingly.

---

## 2. Are there specific benchmarks? Yes — and they are unusually complete

Upstream publishes both quality and throughput, with the protocol.

### Quality — MTEB v2 (16-bit embeddings)

| Model | Weights | Bitext | Class. | Clust. | PairCls. | Rerank. | Retr. | STS | **Mean** |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| bitnet-embeddings-270m | 1.58-bit | 80.47 | 71.09 | 52.37 | 79.72 | 60.50 | 66.71 | 74.35 | **66.26** |
| bitnet-embeddings-0.6b | 1.58-bit | 81.47 | 72.65 | 53.06 | 80.47 | 62.12 | 68.33 | 74.97 | **67.49** |

Against their own bf16 teachers (`harrier-oss-v1-*`): 66.26 vs **66.5** for the
270M, 67.49 vs **69.0** for the 0.6B. So ternary costs ~0.24 and ~1.5 MTEB
points respectively.

On MMTEB (eng, v2) the 0.6B scores **67.60** against the FP16 teacher's
**67.95** — 0.35 points down — at **870.90 t/s vs 382.15 t/s**, i.e. 2.28×
throughput.

### Conversion fidelity (safetensors → F16 GGUF → I2_S GGUF)

For the 0.6B, averaged over an 8-task subset: safetensors **0.7188**, F16 GGUF
**0.7212**, I2_S GGUF **0.7180**. The I2_S conversion costs 0.0008 against
safetensors and 0.0032 against F16 — negligible. This is the number to
reproduce as a *conversion* gate, separately from the model's own quality.

Note the 270M table in the guide is not a like-for-like comparison: its F16
column comes from a different (non-BitNet-trained) baseline model, so the
column-to-column delta there is an evaluation-setup artifact, not conversion
loss. Only the F16-vs-I2_S delta (0.0019) is meaningful in that table.

### Throughput — Intel Xeon Platinum 8573C, 8 threads, clang, no OpenMP, `GGML_NATIVE=ON`

Prefill (`llama-bench -p ... -r 3 -ngl 0`), tokens/second:

**0.6B**

| Test | F16 | I2_S | Speedup |
| :-- | --: | --: | --: |
| pp128 | 382.15 | **870.90** | 2.28× |
| pp256 | 373.95 | **827.75** | 2.21× |
| pp512 | 371.86 | **716.27** | 1.93× |
| pp1024 | 341.55 | **620.58** | 1.82× |
| pp2048 | 298.21 | **481.14** | 1.61× |
| pp4096 | 236.76 | **336.32** | 1.42× |

**270M**

| Test | F16 | I2_S | Speedup |
| :-- | --: | --: | --: |
| pp128 | 1212.68 | **2019.59** | 1.67× |
| pp256 | 1221.28 | **2119.50** | 1.74× |
| pp512 | 1394.99 | **2181.23** | 1.56× |
| pp1024 | 1265.22 | **2086.46** | 1.65× |
| pp2048 | 1024.47 | **1471.60** | 1.44× |
| pp4096 | 785.54 | **1033.46** | 1.32× |

The speedup falls with sequence length because attention is not quantized and
comes to dominate. That is the same shape geistlib already sees on b1.58.

### Two caveats worth internalizing before promising numbers

1. **These are x86-only figures.** Upstream's own Supported Models table marks
   I2_S ✅ on x86 and ❌ on **ARM** for *both* embedding models — bitnet.cpp has
   no ARM path for them at all. There is no published Pi-5 or Apple-silicon
   baseline to beat, only one to establish.
2. **The token-embedding table stays F16.** For the 270M that table is
   262,144 × 640 ≈ 336 MB in F16 — larger than the entire ternary body, and
   ~62% of the parameter count. That is why the 270M's speedup is the *smaller*
   of the two despite being the smaller model, and it sets a hard floor on
   footprint that ternary weights cannot lower. Upstream reports the 0.6B at
   ~699 MiB I2_S against 1.11 GiB F16.

---

## 3. Can geistlib implement them? Yes — and it starts from an unusually good position

geistlib already has, in tree, most of what these models need:

| Requirement | Status in geistlib | Where |
| :-- | :-- | :-- |
| I2_S dtype + ternary kernels | ✅ **and on ARM, which upstream lacks** | `src/backends/cpu_neon/weight_resolve.c:1349` (`i2_s/q8a`, SDOT), `src/backends/cpu_x86/kernel_i2s_avx512_vnni.c`, `src/backends/cpu_x86/kernel_i2s.c`, scalar fallback |
| W1.58A8 (int8 per-token activation quant) | ✅ the existing b1.58 path | same kernels |
| `qwen3` architecture family | ✅ incl. per-head QK-norm and `head_dim` decoupled from `hidden/heads` via `qwen3.attention.key_length` | `src/archs/transformer/arch_family.c:263` (`populate_qwen3`) |
| RoPE `freq_base` override, `rms_norm_eps` | ✅ | `arch_family.c:297` (`populate_layers_qwen3`) |
| Qwen3 BPE tokenizer (151,936, GPT-2 style w/ merges) | ✅ — already the CI reference model | `src/engine/gguf_tokenizer.c` |
| Gemma BPE tokenizer (262,144) | ✅ via the Gemma 4 path | same |
| Gemma-style extra attention norms | ✅ flag exists (`has_gemma_attn_norms`) | `arch_family.c:55` |
| Embedding scaling by `sqrt(n_embd)` | ✅ used by the Gemma family | head/embed path |
| SubLN plumbing (extra RMSNorm around BitLinear) | ⚠️ exists, but at the **b1.58 positions**, 2 per layer, not 7 | `arch_state.h:88-93`, `weight_load/layer_wiring.c:363,427`, `forward/layer_attn.c:454`, `forward/layer_ffn.c:253` |
| Prefill without decode | ✅ | `geist_session_prefill_tokens`, `geist_session_peek_logits` |
| GGUF loader, mmap, zero-copy weights | ✅ | `src/io/gguf_reader.c` |
| safetensors reader (for a native converter) | ✅ | `src/io/safetensors_reader.c` |

The 0.6B in particular declares `general.architecture = qwen3`, which geistlib
**already registers and accepts**. Its I2_S weights, its decoupled `head_dim`,
its QK-norms and its tokenizer all land on existing, exercised code paths.

### What is genuinely missing

**A. Per-projection input RMSNorm — five new tensors per layer.** The core
engine work, and the subject of Phase 1. The norms before o_proj and down_proj
land in geistlib's existing `attn_sub_norm` / `ffn_sub_norm` slots; the five
before q, k, v, gate and up are new. Touches the layer struct
(`arch_state.h`), the loader (`weight_load/layer_wiring.c`), and both forward
paths (`forward/layer_attn.c`, `forward/layer_ffn.c`).

The real cost is not the norms themselves but what they cost the fusions:
q/k/v share one normalised input today, and so do gate/up, which is precisely
what the fused triple-QKV and gate_up kernels are built on. Per-projection
norms give each projection its own input, so those paths cannot express the
computation at all.

**B. A `gemma3` architecture family** (for the 270M only). `gemma3` is not in
`REGISTRY` or `geist_arch_transformer_gguf_names` (`arch_family.c:637-666`);
only `gemma4` is, and the engine gate fails closed on unknown arch strings. A
new populator needs `post_attention_norm` / `post_ffw_norm` wiring,
`query_pre_attn_scalar = 256`, `sqrt(hidden)` embedding scale, `head_dim = 256`
decoupled from 640/4 = 160, `rope_theta = 10000`, and GELU.

**C. An embedding output path and public API.** Today every forward ends in
`finalize_logits_*` (`forward/head.c`) — project through the LM head, sample.
Embedding models need the opposite: stop at the final `output_norm`, take the
last non-padding token's hidden state, L2-normalize, hand back a `float`
vector, and **never touch the head**. Note that a GGUF without `output.weight`
currently falls back silently to tied embeddings (`layer_wiring.c:617`), which
is harmless but wasteful — the head must be skipped, not tied.

**D. A converter.** Microsoft ships safetensors plus
`utils/convert-bitnet-embedding-to-gguf.py`, and that script only works against
a **pinned llama.cpp branch** (`release-bitnet-embedding-0.6b-270m`). There is
no canonical I2_S GGUF on the Hub to download. Same situation geistlib already
documented for `TQ2_0` (`benchmark/results/TERNARY.md`).

**E. Benchmarks and a quality gate.** No ARM baseline exists upstream, so
geistlib would be establishing the first one — which is exactly the repo's
headline claim shape, and exactly why the numbers have to be produced under the
frozen protocol in `benchmark/METHODOLOGY.md` rather than quoted.

---

## 4. The plan

Sequenced so each phase is independently mergeable under AGENT.md §6 (bounded
batches, disassembly and benchmark gate) and so the 0.6B — the cheaper, more
valuable target — lands before any Gemma3 work starts.

### Phase 0 — Converter and reference vectors (no engine changes) — **converter done**

Shipped: `tools/convert_bitnet_embedding.py` (safetensors → GGUF, `--outtype
i2_s|f16`), covered by `tests/test_bitnet_embedding_convert_py.py` in the
hermetic `make test-py` suite. It needs **numpy and the standard library only**
— no llama.cpp branch, no torch, no transformers, unlike upstream's script.
Tensors are streamed one at a time out of an mmapped checkpoint, so peak memory
tracks the largest tensor rather than the model.

Three things the implementation settled that the upstream guide did not:

- **The I2_S packing is strided, not consecutive.** Upstream's prose ("each
  byte stores 4 values: `(c0<<6)|(c1<<4)|(c2<<2)|c3`") does not say *which*
  four. Element `b*256 + h*128 + g*32 + bb` lives in byte `b*64 + h*32 + bb`
  at shift `6-2g` — the four values sharing a byte are 32 apart, not adjacent.
  Written naively the file loads and produces silent garbage. The converter's
  output is verified **byte-identical** against `pack_i2_s` from
  `tests/test_i2_s_parity.c:30`, frozen as a golden vector in the test.
- **The trailing f32 scale is a multiplier, not its reciprocal.** geistlib
  dequants as `trit * scale` (`cpu_scalar/weight_resolve.c:109`), so the value
  written is `mean(|w|)`. Upstream's "`scale = 1/mean(|w|)`" describes its
  quantisation step, not the stored number.
- **`key_length`/`value_length` must be written explicitly** — the default
  `hidden/heads` derivation is wrong for both models, as upstream notes.

Verified end to end: a synthetic Qwen3-shaped checkpoint converts, and
geistlib's own `gguf_reader` opens the result, reports `arch=qwen3`, resolves
every metadata key the `qwen3` populator reads, passes the extent check on all
38 tensors (I2_S `nbytes` = `n_in*n_out/4 + 4`, so the reader's tail-byte
accounting agrees with the writer's), and finds the `*_norm_in` tensors.

**Still open in this phase**, and it needs a machine with the real weights:
capture reference embeddings from upstream's `llama-embedding` for a fixed
prompt set into `tests/data/` as the Phase 2 parity oracle. Hugging Face is not
reachable from CI, so this is a local step. Until it is done, the converter is
verified for *format*, not for *numerical agreement with Microsoft's own
converter* — the scale convention in particular is inferred from geistlib's
working b1.58 path, not from a byte-diff against a Microsoft-produced
embedding GGUF.

### Phase 1 — Per-projection input RMSNorm (the 0.6B path) — **implemented, not yet numerically verified**

Shipped:

- `config.has_projection_input_norms`, set from
  `bitnet.embedding.projection_input_norms` and confirmed against the tensors
  themselves — metadata alone must not promise norms the file lacks
  (`arch_family.c`, `populate_qwen3`).
- Five new layer tensors (`q_norm_in`, `k_norm_in`, `v_norm_in`,
  `gate_norm_in`, `up_norm_in`); the o_proj and down_proj norms load into the
  existing `attn_sub_norm` / `ffn_sub_norm` slots under their own GGUF names.
- One extra scratch slice, `scratch_proj_in` — sized 0 for every family
  without the norms, so nothing else pays for it.
- Forward: each of q/k/v/gate/up re-normalises the shared attn_norm/ffn_norm
  output into that slice immediately before its matmul. The BitNet activation
  fake-quant moves with it: what the matmul reads is what has to be quantised,
  so it applies to the per-projection vector, not the shared one.
- Fusion gating: the fused triple-QKV path and all three gate_up fusions are
  disabled while the flag is set, because they assume a shared input.

**Verified:** builds under `-Werror` with gcc-14; `make test-unit` 37 passed /
0 failed with no regressions; `make MODE=asan test-unit` clean; `make
format-check` clean under the pinned clang-format 22; a new
`tests/test_projection_input_norms_unit.c` pins the scratch conditionality and
pool accounting, and the converter-side test pins the seven GGUF names the
loader looks up.

**Not verified, and this is the honest gap:** no numerical check against the
real model. Every test here pins structure and bookkeeping, not values — the
forward path has never run on real BitNet embedding weights, because they are
not reachable from CI or this environment. The Phase 0 reference vectors are
the missing oracle, and until they exist Phase 1 should be read as "wired up
and self-consistent", not "correct".

- **Still to do:** per-layer hidden states matching the reference within the
  existing logit tolerance, on both NEON and x86; and the measurement of what
  the disabled fusions actually cost, which the plan has treated as a
  hypothesis from the start.

### Phase 2 — Pooling and the embedding API — **implemented, not yet numerically verified**

Shipped:

- `const float *geist_session_peek_embedding(size_t *n_dims, struct geist_session *s)`
  (`@stability EXPERIMENTAL`, `include/geist_util.h`). The sketch in this plan
  proposed a status-returning `geist_session_embed`; the shipped shape borrows
  a pointer instead, with the same "nullptr plus a zeroed count" idiom as
  `geist_session_peek_logits`.

  **The parameter order follows AGENT.md §1 strictly** — out-size first,
  handle last — and therefore differs from `peek_logits`, which takes them the
  other way round. That sibling is `STABLE since 0.6.0` and named in
  `docs/API_CONTRACT.md`, so it cannot move without breaking a published
  promise for a cosmetic gain. The asymmetry is permanent until a 1.0
  migration could align them, and is called out in the header so it reads as
  a decision rather than an oversight.

  The arch vtable slot (`struct geist_arch_ops_decoder::peek_embedding`)
  keeps the session-first order it shares with `peek_logits`, so that struct
  stays internally consistent; the engine thunk in `session.c` does the swap,
  once.
- `finalize_embedding_last_row` (`forward/head.c`): pools one row out of the
  post-layer hidden states, applies `output_norm`, L2-normalises. It shares
  its first two steps with `finalize_logits_one_row` for the same reason —
  `scratch_h_a` is the staging row every backend can read back — and replaces
  only the vocab projection. **The LM head never runs**: these GGUFs have no
  `output.weight`, and the tied fallback would cost a 151,936-wide matmul
  whose result is discarded.
- Pooling kind read from GGUF metadata (`geist_pooling_select`, pure and
  unit-tested), not hardcoded. `mean` is *recognised* but unimplemented, and a
  model declaring it is refused at load — as is any unknown value. Pooling the
  wrong way produces a vector that looks entirely plausible and means
  something else, which nothing downstream would catch.
- `geist_session_decode_step` returns `GEIST_E_UNSUPPORTED` on an embedding
  model, rather than the misleading `INVALID_STATE` ("prefill first") it would
  otherwise give after a prefill that did run.
- The L2 sum accumulates in `double`. At d_model ≈ 1024 that is the one place
  a float accumulator would visibly move the cosine similarities this vector
  exists to produce.

The engine/application boundary holds: geistlib returns a vector and stops.
Query instruction prefixes — which upstream's FAQ says are **required** or
quality degrades — embedding quantization for storage, and any index belong to
whoever links the library. That is stated in the header.

**Verified:** `-Werror` build under gcc-14; `make test-unit` 38 passed /
0 failed with no regressions; `scripts/check-api-contract.sh` clean;
`tests/test_pooling_select_unit.c` pins that an absent key means *generative*
(every existing model depends on this) and that an unrecognised value is
refused rather than guessed.

**Not verified:** the vector's values. As with phase 1, nothing here has run
on real weights.

- **Still to do:** cosine similarity ≥ 0.999 against the upstream reference
  vectors on the fixed prompt set — the phase 0 oracle, which needs a machine
  that can fetch the model.

### Phase 3 — Quality and performance gates — **machinery built; no numbers produced**

Phase 3's deliverable is numbers, and numbers need the weights. What is in
the tree is the apparatus that produces them; running it needs a machine that
can fetch the model, which is neither CI nor the development container. **No
measurement in this repository has been taken.** Nothing below should be read
as a result.

Shipped:

- `tools/dump_geist_embedding.c` — the first real consumer of
  `geist_session_peek_embedding`. Takes a model and a prompts file (one per
  line; all prompts share one model load, since reloading ~700 MB per prompt
  would dominate the measurement) and writes a self-describing `.gemb`.
- `tools/eval_embedding_fidelity.py` — the cosine-similarity gate, floor
  0.999. Serves both jobs with one mechanism: parity against upstream's
  `llama-embedding`, and F16-vs-I2_S conversion fidelity. Has `--selftest`,
  following `benchmark/perf_gate.py`, so the gate is covered where data is
  not.
- `benchmark/embedding_protocol.json` — the pinned protocol: pp128…pp4096,
  3 repeats, median with MAD, per-host thread counts, and `decode_n: 0`
  (an embedding model emits no tokens, so decode throughput is not a defined
  quantity). Upstream's Xeon figures are recorded inside it as *orientation*
  and explicitly not as a baseline any host profile is compared against.
- `tests/test_embedding_fidelity_py.py` — pins the gate's behaviour and the
  `.gemb` header as a cross-language contract between the C writer and the
  Python reader, which nothing else would catch.

Two things the protocol file makes explicit because they are easy to get
wrong when writing the results up:

- **On ARM there is nothing to divide by.** Upstream marks I2_S unsupported
  on ARM for both models, so a Pi 5 or Apple number is a *first*, not a
  speedup. Reporting a ratio there would require inventing a denominator.
- **Upstream's F16 baseline is the teacher, not the student at F16.** Their
  1.42×–2.28× is BitNet-student vs multilingual-e5-teacher. A geistlib
  F16-vs-I2_S ratio measures something else and must not be compared to it.

**How to actually run it** (needs the 0.6B locally):

```sh
python3 tools/convert_bitnet_embedding.py /path/to/bitnet-embedding-0.6b \
    --outtype i2_s --outfile bitnet-embedding-0.6b-i2_s.gguf
make bin
./bin/<platform>/release/tools/dump_geist_embedding \
    bitnet-embedding-0.6b-i2_s.gguf prompts.txt geist.gemb
# upstream side, same prompts, same order -> up.npy
python3 tools/eval_embedding_fidelity.py --ref up.npy --got geist.gemb
```

- **Done when:** the parity gate passes at ≥ 0.999 — which is also what
  retroactively verifies phases 1 and 2 — and a throughput row exists under
  `benchmark/results/` with pinned revision, model hash and raw samples per
  the [#364](https://github.com/geisten/geistlib/issues/364) protocol.

### Phase 4a — the `gemma3` family — **implemented**

Shipped, and useful on its own: geistlib now accepts `general.architecture =
gemma3`.

- `populate_gemma3` / `populate_layers_gemma3` plus `REGISTRY` and
  `geist_arch_transformer_gguf_names` entries. Structurally Gemma 3 sits
  between llama and gemma4 — Gemma's extra residual norms and QK-norms and the
  `sqrt(d_model)` embedding scale, but no per-layer embeddings, no KV sharing
  and no sliding-window pattern.
- `config.has_embed_scale`, split out from `has_ple`. The `sqrt(d_model)`
  embedding scale used to ride on the per-layer-embedding flag, which worked
  only while the one family with the scale also had PLE. Gemma 3 has the scale
  and no PLE, so the two had to stop being the same question. gemma4 sets both
  and is unchanged.
- `head_dim` comes from `gemma3.attention.key_length`, not `d_model/n_heads`:
  for the 270M those are 256 and 160, the same trap `qwen3` already documents.
- `query_pre_attn_scalar` is **not** plumbed into the attention scale. For this
  model it is 256 and so is `head_dim`, so `1/sqrt(head_dim)` is already the
  right number; a Gemma 3 variant where the two differ would need it, and that
  path is deliberately not written on speculation.

Scope, stated plainly: this populator is written for and exercised by the
BitNet embedding 270M. Whether a stock Google Gemma-3 GGUF loads through it is
untested.

`tests/test_gemma3_family_unit.c` pins that the name is accepted, that no
previously registered family fell out of the list while editing it — the
`static_assert` in `arch_family.c` only compares list *lengths* — and that the
embed scale can now exist without PLE.

### Phase 4b — 128-granular I2_S for the 270M — **not done, and possibly not worth doing**

With 4a the 270M loads and runs **as F16**. What it cannot do is use ternary
weights, and that is the entire reason to want it.

geistlib walks whole 256-element I2_S blocks (`I2S_BLOCK_ELEMS = 256`;
`cpu_scalar/weight_resolve.c:109` loops `b < n_in/256`), but Gemma3-270M's
hidden size is **640** and `640 % 256 = 128`. Five of the seven projections per
layer take a 640-wide input — q, k, v, gate, up — so only `o_proj` (1024) and
`down_proj` (2048) are representable. bitnet.cpp is unaffected: it blocks at
128 elements / 32 bytes, and `640 % 128 = 0`.

The good news is that the on-disk format is already 128-granular. Every
decoder and kernel in the tree has the shape

```c
for (b = 0; b < n_in / 256; b++)          /* block */
    for (h = 0; h < 2; h++)               /* <- this is the 128-element unit */
        for (bb = 0; bb < 32; bb++) ...
```

so the inner `h` loop *is* a 128-element half-block. Nothing on disk changes;
the byte layout is identical. What changes is the loop bound and the block
constants, in: `quant.h`, the `gguf_reader` dtype row, the scalar decoder, the
NEON SDOT kernels and the x86 scalar + AVX-512/VNNI kernels. The size formula
is unaffected either way (128 elems → 32 bytes and 256 → 64 are both `n/4`), so
no existing model's bytes move — the divisibility check merely loosens.

**Why it is still not done here.** It is a change to the hottest code in the
repository across three backends, and AGENT.md §6 requires a disassembly diff
and a parent/patch benchmark on identical hardware before such a change lands.
Neither is possible in this environment: there is no model to run and
cloud-container timings are worthless under `benchmark/METHODOLOGY.md`. Writing
it unmeasured would be exactly the "speculative push" the contributing rules
warn against.

**And the cost/benefit has not improved.** The 270M scores 1.2 MTEB points
below the 0.6B, gets a smaller speedup (1.32–1.74× vs 1.42–2.28×), its F16
token-embedding table alone is ~336 MB — so on a 4 GB Pi it is not meaningfully
cheaper than the 0.6B — and it alone needs this kernel work. The 0.6B is fully
representable today. If 4b is done, it should be because a 128-granular I2_S
path is wanted for its own sake, not to reach this model.

### Phase 5 — Documentation

- `docs/MODELS.md`: new rows, and correct the opening line, which currently
  lists the registered architectures.
- `docs/API_CONTRACT.md`: the embedding API's stability tag.
- `ROADMAP.md`: fold into the "Max quantization" track.

---

## 5. Effort and risk

| Phase | Size | Main risk |
| :-- | :-- | :-- |
| 0 Converter | S | ✅ done for the 0.6B; residual risk is the scale convention, unverified against a Microsoft-produced embedding GGUF |
| 1 Projection norms | **M** | ✅ implemented; smaller than feared (five new tensors, not seven) but numerically unverified, and the disabled fusions are still unmeasured |
| 2 Pooling + API | M | ✅ implemented; `EXPERIMENTAL`, parameter order per AGENT.md §1 and deliberately unlike `peek_logits`, which is STABLE and cannot move |
| 3 Gates | M | ✅ apparatus built and self-tested; **zero measurements taken** — it needs the weights, which this environment cannot fetch |
| 4a gemma3 family | S | ✅ implemented; scoped to the 270M's geometry, stock Gemma-3 GGUFs untested |
| 4b 128-granular I2_S | **L** | not done — three backends' hot kernels, and AGENT.md §6 needs a benchmark this environment cannot produce |

The dominant risk is Phase 1: seven norms per layer sit directly in the hot
path, and the fusion planner currently *disables* optimizations when a sub-norm
is present. A naive implementation could plausibly give back more than ternary
weights win. That argues for measuring the fusion cost early — during Phase 1,
not at Phase 3 — and for treating "the projection norms are cheap enough" as a
hypothesis to test rather than a premise.

## 6. What this is not

Not a RAG stack, not a vector store, not a reranker. geistlib returns a
normalized `float` vector and stops, consistent with the engine/application
boundary in `ROADMAP.md` and `docs/README.md`. Query instruction prefixes,
embedding quantization to 8/4/2/1 bit for storage (upstream supports it, and it
is a *storage* decision), and index structures all belong to whoever links the
library.

## 7. Adjacent: VibeASR.cpp (07/23/2026)

Released three days after the embedding models: a real-time multilingual ASR
engine on CPU using BitNet I2_S quantization, RTF < 1 with very few threads on
x86 AVX2 and ARM NEON. Code at `microsoft/VibeASR.cpp`, models at
`microsoft/VibeVoice-ASR-BitNet`, report at arXiv:2607.21075.

This is interesting for geistlib because the pieces are already here — a
Conformer audio arch (`src/archs/audio_conformer/`), streaming PCM ingestion
(`geist_session_audio_push`), a WER harness (`tools/eval_audio_wer.py`) and
I2_S kernels on both NEON and x86. It is also a substantially larger piece of
work than the embedding models and needs its own analysis. Out of scope here;
worth opening separately.
