# BitNet embedding models

Microsoft's `BitNet-embedding-0.6B` and `BitNet-embedding-270M` (July 2026,
MIT) map text to a dense vector; they emit no tokens. geistlib loads them both
as Microsoft ships them — the ready-made GGUFs on the model pages, or the
safetensors through the in-tree converter — and returns the pooled,
L2-normalized vector.

**Status:** correctness is verified (geist reproduces the embedding the model
cards print to RMSE 8.5e-4); **throughput is unmeasured**
(`benchmark/embedding_protocol.json` says `UNMEASURED`). The 270M runs only as
F16 (see [Open work](#open-work)).

Upstream sources: the [`microsoft/BitNet` README](https://github.com/microsoft/BitNet/blob/main/README.md)
and its `docs/bitnet-embeddings-i2s-guide.md` (conversion and benchmark
guide; every upstream number below comes from it).

## The models

Both are trained natively at W1.58A8 (ternary absmean weights, 8-bit
per-token absmax activations), have no LM head, and pool by **mean** followed
by L2 normalization. The model cards say "last-token pooling" in prose, but
the GGUFs carry `{arch}.pooling_type = 1` (MEAN), the cards' own
`llama-embedding` command takes that key, and only mean reproduces the printed
vector (RMSE 8.5e-4, against 0.24 for last-token).

| | `BitNet-embedding-0.6B` | `BitNet-embedding-270M` |
| :-- | :-- | :-- |
| Backbone / `general.architecture` | Qwen3-0.6B / `qwen3` | Gemma3 / `gemma3` |
| Embedding dimension | 1,024 | 640 |
| Layers, heads (KV) | 28, 16 (8) | 18, 4 (1) |
| `head_dim` | 128 (≠ hidden/heads) | 256 (≠ hidden/heads) |
| Intermediate size, activation | 3,072, SiLU | 2,048, GELU |
| Tokenizer | Qwen3 BPE (151,936) | Gemma BPE (262,144) |
| `rope_theta` | 1,000,000 | 10,000 |
| Embedding scale | none | `sqrt(hidden_size)` |
| Post-attn / post-FFW norms | no | yes |
| Max context | 32,768 | 32,768 |

**Per-projection input RMSNorm.** Every linear projection (q, k, v, o, gate,
up, down) has its own RMSNorm on its input, `blk.{i}.*_norm_in.weight` in the
GGUF. The o and down norms sit where BitNet b1.58's SubLN does and load into
the existing `attn_sub_norm` / `ffn_sub_norm` slots; the five before q, k, v,
gate and up are extra.

### Upstream numbers (orientation only)

- **MTEB v2 mean:** 66.26 (270M) and 67.49 (0.6B), against 66.5 and 69.0 for
  their bf16 teachers.
- **Conversion fidelity** (0.6B, 8-task subset): safetensors 0.7188, F16 GGUF
  0.7212, I2_S GGUF 0.7180.
- **Throughput** (Xeon Platinum 8573C, 8 threads, prefill): 0.6B I2_S
  870.90 t/s at pp128 falling to 336.32 at pp4096, 2.28×–1.42× over the F16
  **teacher** (not the student at F16, so not comparable to a geist F16-vs-I2_S
  ratio). The speedup falls with length because attention is not quantized.
- **x86 only:** upstream marks I2_S unsupported on ARM for both models, so a
  Pi 5 or Apple number is a first, not a speedup.
- **F16 token embeddings:** the 270M's table is 262,144 × 640 ≈ 336 MB, larger
  than its ternary body; the 0.6B is ~699 MiB as I2_S.

## How geistlib runs them

- **Loading.** `qwen3` and `gemma3` are registered families.
  `config.has_projection_input_norms` is set from
  `bitnet.embedding.projection_input_norms` and confirmed against the tensors.
  The `gemma3` populator (Gemma norms and QK-norms, `sqrt(d_model)` embedding
  scale, no PLE, KV sharing or sliding window; `head_dim` from
  `gemma3.attention.key_length`) is written for the 270M; stock Google Gemma-3
  GGUFs are untested. `query_pre_attn_scalar` is not plumbed: it equals
  `head_dim` (256) for this model.
- **Forward.** Each of q/k/v/gate/up re-normalizes the shared norm output into
  a `scratch_proj_in` slice (sized 0 for other families) before its matmul,
  and the activation fake-quant applies to that per-projection vector. The
  fused triple-QKV and gate_up paths assume a shared input and are disabled
  for these models.
- **Pooling.** Read from GGUF metadata (`geist_pooling_select`,
  `geist_pooling_from_gguf_type`): last-token or mean, then `output_norm` and
  L2 normalization (accumulated in `double`). An unknown pooling value is
  refused at load. The LM head never runs.
- **API.** `const float *geist_session_peek_embedding(size_t *n_dims, struct geist_session *s)`
  (`EXPERIMENTAL`, `include/geist_util.h`): prefill, then peek. The buffer
  belongs to the session, as with `geist_session_peek_logits`.
  `geist_session_decode_step` returns `GEIST_E_UNSUPPORTED` on an embedding
  model. Query instruction prefixes (required for quality per upstream),
  storage quantization and indexes belong to the caller.

## Converting and checking

`tools/convert_bitnet_embedding.py` converts a Hugging Face checkpoint to GGUF
with numpy and the standard library only (no llama.cpp branch, no torch). It
streams tensors from an mmapped checkpoint, so peak memory tracks the largest
tensor.

```sh
python3 tools/convert_bitnet_embedding.py /path/to/bitnet-embedding-0.6b \
    --outtype i2_s --outfile bitnet-embedding-0.6b-i2_s.gguf
make bin
./bin/<target>/release/tools/dump_geist_embedding \
    bitnet-embedding-0.6b-i2_s.gguf prompts.txt geist.gemb
python3 tools/eval_embedding_fidelity.py --ref up.npy --got geist.gemb   # cosine floor 0.999
```

Format details the upstream guide leaves open, as the converter implements
them:

- **I2_S packing is strided.** Element `b*256 + h*128 + g*32 + bb` lives in
  byte `b*64 + h*32 + bb` at shift `6-2g`; the four values in a byte are 32
  apart. Output is byte-identical to `pack_i2_s` in
  `tests/test_i2_s_parity_unit.c`.
- **The trailing f32 scale is `mean(|w|)`**, a multiplier (geist dequantizes
  `trit * scale`).
- **`key_length` / `value_length` are written explicitly**; `hidden/heads` is
  wrong for both models.

Tests:

| Test | Checks |
| :-- | :-- |
| `tests/test_published_embedding_e2e.c` | the model card's printed vector (RMSE); skips without `GEIST_EMBED_GGUF_PATH` |
| `make test-embedding` | converter → GGUF → loader → norms → pooling → `peek_embedding` on a synthetic model |
| `tests/test_bitnet_embedding_convert_py.py` | converter packing and GGUF names (`make test-py`) |
| `tests/test_embedding_fidelity_py.py` | the cosine gate and the `.gemb` format (`make test-py`) |
| `tests/test_projection_input_norms_unit.c`, `test_pooling_select_unit.c`, `test_gemma3_family_unit.c` | scratch sizing, pooling selection, family registration |

## Open work

- **Throughput.** `benchmark/embedding_protocol.json` pins the protocol
  (pp128…pp4096, 3 repeats, median with MAD, `decode_n: 0`). A run needs a
  quiesced host per `benchmark/METHODOLOGY.md`; on a Pi 5, check temperature
  before every sequence length and discard throttled runs. The cost of the
  disabled fusions is also unmeasured.
- **128-granular I2_S for the 270M.** geist walks whole 256-element I2_S
  blocks, but the 270M's hidden size is 640 (`640 % 256 = 128`), so q, k, v,
  gate and up cannot be I2_S and the model runs as F16. The on-disk layout is
  already 128-granular (the inner `h` loop is a half block), so the change is
  loop bounds and block constants in `quant.h`, the `gguf_reader` dtype row,
  the scalar decoder and the NEON and x86 kernels — hot code in three backends,
  which needs the AGENT.md §6 disassembly and benchmark gate. With a smaller
  speedup, lower MTEB and a 336 MB F16 table, the 270M alone does not justify
  it.
- **VibeASR.cpp** (`microsoft/VibeASR.cpp`, BitNet I2_S real-time ASR, July
  2026) would reuse the Conformer arch, streaming PCM ingestion, the WER
  harness and the I2_S kernels; it needs its own analysis.
