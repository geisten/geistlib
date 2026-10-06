# Models

A GGUF loads when its `general.architecture` is registered (`gemma4`,
`gemma3`, `llama`, `qwen3`, `qwen35`, `bitnet-b1.58`, `bitnet`) and the
selected backend supports its dtypes. The table lists the models exercised by
tests and benchmarks. Text weights are one download; Gemma vision and audio
also need their tower weights.

| Model | Modality | Quant | ~Size | RAM | Best on | Get it |
| :-- | :-- | :-- | --: | --: | :-- | :-- |
| **BitNet b1.58 2B-4T** | text (ternary) | `i2_s` | 1.1 GiB | ≥ 4 GB | **Pi 5 · x86** | `make fetch-bench-model` · [⬇ gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/a1f2f1c765812aa8af3f6eda4a313707064bba15/ggml-model-i2_s.gguf) — or the [self-contained `geist-bitnet` binary](https://github.com/geisten/geistlib/releases/latest) |
| **Gemma 4 E2B-it** | text · vision · audio | `Q4_K_M` | 3.1 GB | ≥ 4 GB | Mac / Pi 5 | `make fetch-model` · [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2/gemma-4-E2B-it-Q4_K_M.gguf) |
| Gemma 4 E4B-it | text · audio | `Q4_K_M` | 4.6 GB | ≥ 8 GB | Mac | `make fetch-e4b-model` · [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/bfc15c382204943c3a8fff0c750b94ae2364d7a3/gemma-4-E4B-it-Q4_K_M.gguf) |
| Llama family (e.g. SmolLM2, Llama 3.2) | text | supported GGUF dtypes | varies | varies | everywhere | `make fetch-llama-model` · [⬇ SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/smollm2-360m-instruct-q8_0.gguf) (the CI reference) |
| Qwen3 (0.6B / 1.7B / 4B) | text | supported GGUF dtypes | 0.6–4 GB | ≥ 1 GB | everywhere | `make fetch-qwen3-model` · [⬇ Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf) (the CI reference) |
| Qwen3.5/3.6/3.8 dense (0.8B – 27B) | text | supported GGUF dtypes incl. UD mixed (IQ4_XS/IQ4_NL/Q3_K/IQ3_S) | 0.8–16 GB | ≥ 2 GB | CPU; Metal on Mac (27B: ≥ 32 GB); Vulkan (27B needs a ≥ 16 GB device) | `make fetch-qwen35-model` · [⬇ Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q8_0.gguf) (the CI reference) — hybrid DeltaNet/attention; MoE variants (A3B etc.) unsupported |
| Ternary-Bonsai-2-27B (Qwen3.8) | text (ternary) | `PQ2_0` + Hadamard | 7.2 GB | ≥ 16 GB | Metal; Vulkan (fits an 11 GiB card) | [⬇ prism-ml GGUF ↗](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) — numbers in [`TERNARY.md`](../benchmark/results/TERNARY.md) |
| BitNet b1.58-large | text (ternary) | `TQ2_0` | 207 MB | ≥ 1 GB | smallest footprint | convert from [1bitLLM ↗](https://huggingface.co/1bitLLM/bitnet_b1_58-large) |

```bash
make fetch-bench-model   # BitNet b1.58 2B-4T — the ~2× decode win on a Pi 5
make fetch-model         # Gemma 4 E2B-it text model; fetch towers separately
make fetch-audio-tower   # matching E2B audio tower
```

## Vision & audio

Gemma 4 models take image and audio input through a built-in SigLIP vision
tower and Conformer audio tower, both on the CPU backends. Attach inputs with
`geist_session_attach_image` / `_audio` / `_video` (`EXPERIMENTAL`, see
[`ARCHITECTURE.md`](ARCHITECTURE.md#multimodal-soft-token-prefixes)); a
microphone loop is in [`VOICE.md`](VOICE.md).

Towers are per variant: their output width matches the text model's residual
stream (1536 on E2B, 2560 on E4B), and the engine refuses a mismatched tower.
E4B audio is tested weekly in CI (`e4b-smoke`); extract its tower with
`python3 tools/fetch_audio_tower.py --url .../gemma-4-E4B-it/resolve/4e22d7e59e078e63a14f351efdc5232ed366b621/model.safetensors`.
E4B vision is untested; `tools/dump_vision_tower.py` extracts the tower.

## Ternary (BitNet b1.58)

BitNet b1.58 (canonical `i2_s` and compact `TQ2_0`) runs with integer-only dot
products: ARM SDOT and x86 AVX-512 VNNI. Decode is ~2× bitnet.cpp on a Pi 5
and faster on an AMD 9950X; numbers in
[`../benchmark/README.md`](../benchmark/README.md).

`TQ2_0` has no canonical upstream GGUF; the conversion recipe is in
[`../benchmark/results/TERNARY.md`](../benchmark/results/TERNARY.md).

## Embedding models

Microsoft's BitNet embedding models (`qwen3` and `gemma3` GGUFs) return a
pooled vector through `geist_session_peek_embedding` (`EXPERIMENTAL`).
Conversion, verification and limits:
[`BITNET_EMBEDDINGS_PLAN.md`](BITNET_EMBEDDINGS_PLAN.md).
