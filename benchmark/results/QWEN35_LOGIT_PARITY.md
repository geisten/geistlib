# Qwen3.5/3.8 logit parity against llama.cpp

Controlled next-token comparison on 2026-08-27. Both engines used the same
GGUF file, the exact same 31 token IDs, greedy-compatible raw logits, and their
Metal backends on an Apple M1 Max. The llama.cpp build was Homebrew 9820
(`3fc4e1052`). No chat template or sampling was involved.

Prompt: [`benchmark/prompts/logit-parity.txt`](../prompts/logit-parity.txt)

| model | top-1 | top-5 overlap | top-10 overlap | centered cosine | centered RMSE | relative L2 | JS divergence (nats) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B Q4_0 | no | 4/5 | 9/10 | 0.950990 | 0.651149 | 0.317637 | 0.124834 |
| Qwen3.8-27B Q4_0 | yes | 5/5 | 9/10 | 0.978152 | 0.443435 | 0.208164 | 0.016160 |

For 4B, Geist ranked token 15666 (`Answer`, p=0.4104) first and token 14162
(`Question`, p=0.0579) fifth. llama.cpp reversed that preference: token 14162
first (p=0.4007), token 15666 second (p=0.1777). A Geist Metal-vs-cpu_neon
control retained the same top-1 and all top-5 tokens, with centered cosine
0.999868 and RMSE 0.034264. The 4B discrepancy is therefore not primarily a
Metal numerical effect.

For 27B, both engines selected token 19 (`4`) and shared all top-5 candidates.
The JS divergence was 0.016160 nats, substantially below the 4B result.

The centered metrics subtract each vector's mean because a constant logit
offset has no effect on probabilities. JS divergence is computed after a
full-vocabulary softmax.

## Reproduction

```sh
make TARGET=mac-omp MODE=release BACKENDS='metal cpu_neon cpu_scalar' \
  bin/mac-omp/release/tools/dump_geist_logits
cc -std=c11 -O2 $(pkg-config --cflags llama ggml) \
  -o /tmp/dump_llamacpp_logits benchmark/dump_llamacpp_logits.c \
  $(pkg-config --libs llama ggml)

GEIST_BACKEND=metal bin/mac-omp/release/tools/dump_geist_logits \
  MODEL.gguf benchmark/prompts/logit-parity.txt ids.bin geist.bin
/tmp/dump_llamacpp_logits MODEL.gguf ids.bin llama.bin
```
