# BitNet embedding models (July 2026) — analysis and implementation plan

Status: **proposal**. Nothing here is implemented yet. This document records
what Microsoft released in July 2026, which of it is reproducible, and what
geistlib would have to grow to run it. Sources are pinned inline so the claims
can be re-checked against upstream.

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
neither standard Qwen3/Gemma3 nor standard BitNet — BitNet b1.58's SubLN sits
at *different* positions (after attention, after gate*up). geistlib has SubLN
at those b1.58 positions, not these.

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

**A. Per-projection input RMSNorm (7 tensors/layer).** The core engine work.
geistlib's `attn_sub_norm` / `ffn_sub_norm` are two tensors at the wrong
positions; these models need a norm on the *input* of each of q/k/v/o/gate/up/
down. Touches `struct transformer_layer` (`arch_state.h`), the loader
(`weight_load/layer_wiring.c`), and both forward paths (`forward/layer_attn.c`,
`forward/layer_ffn.c`). It also interacts with the fusion planner: `exec_plan.c`
already disables GEGLU tile fusion and the fused scaled-GELU path when a
sub-norm is present (`exec_plan.c:111,135,142`) — seven more norms will gate
more fusions, and the cost of that has to be measured, not assumed.

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

### Phase 0 — Get the weights and a reference vector (no engine changes)

- Port Microsoft's conversion to `tools/convert_bitnet_embedding.py`, reusing
  the I2_S packing geistlib already reads: 128 values per block into 32 bytes,
  `-1→0, 0→1, +1→2`, two bits per value packed
  `(c0<<6)|(c1<<4)|(c2<<2)|c3`, float32 scale appended after the packed data,
  `scale = 1/mean(|w|)`. Embeddings and all 1-D norms stay F16; `output.weight`
  is skipped. Must write `key_length`/`value_length` — the default
  `hidden/heads` derivation is **wrong for both models**.
- Capture reference embeddings from upstream's `llama-embedding` for a fixed
  prompt set, checked into `tests/data/`, as the parity oracle for Phase 3.
- **Done when:** an I2_S GGUF exists locally and `gguf_reader` opens it and
  enumerates the `*_norm_in` tensors.

### Phase 1 — Per-projection input RMSNorm (the 0.6B path)

- Extend `struct transformer_layer` with the seven `*_norm_in` tensors; load
  them in `weight_load/layer_wiring.c` behind a config flag
  (`has_projection_input_norms`) so no existing family pays for them.
- Apply them in `forward/layer_attn.c` and `forward/layer_ffn.c` on each
  projection's input, before activation quantization.
- Re-check `exec_plan.c` fusion gating and record what each newly-disabled
  fusion costs.
- **Done when:** per-layer hidden states match the reference within the
  existing logit tolerance, on both NEON and x86, and `make MODE=asan test`
  is clean.

### Phase 2 — Pooling and the embedding API

- New `EXPERIMENTAL` surface in `include/geist_util.h` — shape to be settled in
  review, sketch:
  `enum geist_status geist_session_embed(struct geist_session *s, size_t *out_dim, const float **out_vec);`
  following the existing borrow-a-pointer convention of
  `geist_session_peek_logits`, and AGENT.md §1 parameter order.
- Last-token pooling + L2 normalization; skip the LM head entirely rather than
  falling back to the tied table.
- Read the pooling type from the GGUF metadata the converter writes
  (sentence-transformers `modules.json` convention) instead of hardcoding it,
  so a mean-pooling model later doesn't need a second API.
- Document the boundary: geistlib returns a vector. Instruction prefixes on the
  query side — which upstream's FAQ says are **required** or quality degrades —
  are the caller's business, per the engine/application split in `ROADMAP.md`.
- **Done when:** cosine similarity against the upstream reference vectors is
  ≥ 0.999 on the fixed prompt set.

### Phase 3 — Quality and performance gates

- Reproduce the 8-task conversion-fidelity subset (BornholmBitextMining,
  FinancialPhrasebank, KorHateSpeechML, KorSarcasm, PoemSentiment, SICK-R,
  STS17, STSBenchmark) and gate on the F16-vs-I2_S delta, which upstream puts
  at 0.0032 for the 0.6B.
- Prefill throughput at pp128…pp4096 under `benchmark/METHODOLOGY.md`, on x86
  (comparable to upstream's Xeon numbers) and on the Pi 5 (**no upstream
  baseline exists — this is a first**).
- Record to `benchmark/results/` with pinned revisions, model hashes and raw
  samples, per the `#364` protocol.
- **Done when:** results are reproducible and the perf gate has a baseline.

### Phase 4 — `gemma3` family and the 270M (optional, decide after Phase 3)

- New populator + `REGISTRY` and `geist_arch_transformer_gguf_names` entries
  (the `static_assert` mirrors them).
- `query_pre_attn_scalar`, `sqrt(hidden)` embed scale, post-attn/post-FFW norms,
  `head_dim = 256`, GELU.
- **Worth a cost/benefit check first:** the 270M scores 1.2 MTEB points below
  the 0.6B, gets a smaller speedup (1.32–1.74× vs 1.42–2.28×), and its F16
  embedding table alone is ~336 MB — so on a 4 GB Pi it is not dramatically
  cheaper than the 0.6B while being meaningfully worse. Phase 4 may be the
  right thing to *not* do.

### Phase 5 — Documentation

- `docs/MODELS.md`: new rows, and correct the opening line, which currently
  lists the registered architectures.
- `docs/API_CONTRACT.md`: the embedding API's stability tag.
- `ROADMAP.md`: fold into the "Max quantization" track.

---

## 5. Effort and risk

| Phase | Size | Main risk |
| :-- | :-- | :-- |
| 0 Converter | S | Packing/metadata mismatch — caught immediately by the loader |
| 1 Projection norms | **L** | Fusion regressions in `exec_plan.c`; the hot path is the thing this repo guards hardest |
| 2 Pooling + API | M | API shape is a public commitment; get it reviewed before it ships |
| 3 Gates | M | MTEB tooling is Python-side; keep it in `tools/`, out of the engine |
| 4 gemma3 | M | Gemma3 ≠ Gemma4 in ways not yet mapped; may not be worth it |

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
