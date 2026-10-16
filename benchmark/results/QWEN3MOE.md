# Qwen3-MoE (Qwen3-30B-A3B) on the CPU backends (#48, phase 1)

**Result.** The `qwen3moe` family runs Qwen3-30B-A3B on `cpu_neon`,
`cpu_x86` and `cpu_scalar`.
- **Accuracy.** Over 768 teacher-forced positions, geist's log-probabilities
  are closer to the exact reference (`cpu_scalar`) than llama.cpp's.
- **Single-position logits.** On cpu_neon they meet the tolerance declared in
  #48 before the comparison, on all three prompts.
- **Speed on the 9950X.** geist decodes faster than llama.cpp (36-37 vs 35
  tok/s) and prefills at 86-91 % of it.
- **Speed on the M1 Max.** The CPU measurement was taken under heavy host load
  and is not yet conclusive.

## Setup

- Model: Qwen/Qwen3-30B-A3B-GGUF @ `e4d4bafd`, `Qwen3-30B-A3B-Q4_K_M.gguf`,
  sha256 `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.
  It has 48 layers, 128 experts of width 768, 8 used per token, and an F32
  router.
- llama.cpp `d81235049`, CPU build (`scripts/ensure-llama-bench.sh cpu`).
- Hosts:
  - M1 Max under `with-mac-lock`, clang;
  - AMD Ryzen 9 9950X (the `geist-avx512` runner), gcc 15.2, 16 threads.
- Prompts: `benchmark/prompts/moe-{short,long,code}.txt` (5, 880 and 173
  tokens). Held-out texts for teacher forcing: `tools/moe_usage/corpus.py`,
  `heldout/en0` (*Moby-Dick*) and `heldout/c0` (llama.cpp source).
- Tools:
  - `tools/dump_geist_logits` with `benchmark/dump_llamacpp_logits.c`;
  - `tools/dump_geist_greedy` with `benchmark/dump_llamacpp_greedy.c`;
  - `tools/eval_geist` (`SCORE`) with `benchmark/dump_llamacpp_nll.c`;
  - `tools/compare_logits.py`.

## Next-token logits vs llama.cpp

Tolerance, fixed in #48 before the first comparison: same top-1, at least
4/5 of the top-5, centered cosine >= 0.99, JS <= 0.02 nats.

| backend | prompt | top-1 | top-5 | centered cosine | JS (nats) |
| :-- | :-- | :-- | --: | --: | --: |
| cpu_neon | short | same | 4/5 | 0.9963 | 0.0025 |
| cpu_neon | long | same | 5/5 | 0.9936 | 0.0054 |
| cpu_neon | code | same | 5/5 | 0.9982 | 0.00005 |
| cpu_x86 | short | same | 5/5 | 0.9924 | 0.0148 |
| cpu_x86 | long | same | 5/5 | 0.9907 | **0.068** |
| cpu_x86 | code | same | 5/5 | 0.9966 | 0.00001 |

**One position is a noisy measure for an MoE.**
- A router picks 8 of 128 experts per layer. When the 8th and the 9th are
  nearly tied, a rounding difference swaps an expert, and everything after
  it moves.
- On cpu_neon, decode and prefill of the same tokens route identically up
  to layer 8. There one expert swaps at a router-logit margin of 0.007.
- llama.cpp itself, ubatch 32 vs 880 on the same Mac, moves the long
  prompt's logits by **JS 0.017**.
- On cpu_x86, three bit-different but equally exact forms of SiLU give JS
  0.025-0.041 at the same position. The previous block, which launched one
  parallel region per expert projection, gave 0.0084 there.

## Teacher-forced log-probabilities (cpu_x86, 2 x 384 positions)

The measure that averages that noise out: the log-probability of each
held-out token given its prefix, through the decode path. The error column
is the mean |log p - log p(cpu_scalar)| per position. cpu_scalar is the
reference because it computes the same quantized model in plain F32.

| text | engine | perplexity | mean \|Δ log p\| vs cpu_scalar |
| :-- | :-- | --: | --: |
| *Moby-Dick* | geist (this block) | 11.13 | **0.106** |
| | geist (previous block) | 11.21 | 0.109 |
| | llama.cpp | 10.93 | 0.157 |
| | cpu_scalar | 11.23 | 0 |
| llama.cpp source | geist (this block) | 2.85 | **0.074** |
| | geist (previous block) | 2.82 | 0.080 |
| | llama.cpp | 2.73 | 0.105 |
| | cpu_scalar | 2.88 | 0 |

geist's two blocks are within 0.003-0.006 of each other and both closer to
the reference than llama.cpp.

## Greedy decode, 256 tokens (cpu_neon, previous block)

- **code:** 256/256 tokens identical.
- **short:** diverges at step 5, a near-tie (gap 0.17 vs 0.16).
- **long:** diverges at step 2. In the logits this is not a near-tie
  (llama.cpp leads by 0.75), but it comes from the router near-tie described
  above.

## KV modes (cpu_neon, previous block, long prompt vs llama.cpp)

| KV | top-1 | cosine | JS |
| :-- | :-- | --: | --: |
| FP32 (the Mac default) | same | 0.9918 | 0.0118 |
| INT8 | same | 0.9898 | 0.0364 |
| INT4 | different | 0.787 | 0.270 |
| KIVI | different | 0.815 | 0.238 |

Control, dense Qwen3-0.6B on the same prompt:

| KV | JS |
| :-- | --: |
| FP32 | 0.00018 |
| INT8 | 0.0030 |
| INT4 | 0.36 |

- The MoE amplifies the KV error about 60× through routing.
- INT4 and KIVI break Qwen3 in general, not only the MoE: #53.

## Speed

9950X, 16 threads, frozen protocol, 2 alternating rounds:

| | geist | llama.cpp |
| :-- | --: | --: |
| prefill 32 | 153 tok/s | 167-168 |
| prefill 512 | 218-219 | 254-255 |
| decode 64 | **36.0-37.1** | 34.5-34.8 |
| peak RSS | 16.8-17.4 GB | |

**What changed the numbers:**
- **Experts run in one parallel region per layer, one expert per worker.**
  Before, each expert projection launched its own region, more than 1,000
  per decode token. With that block, prefill 512 was 159 tok/s and decode
  33-34 tok/s.
- **A repacked Q4_K matrix no longer stays resident twice on cpu_x86.**
  Peak RSS fell from 31.1 to 17.4 GB, with logits byte-identical
  (Llama 3.2 3B Q4_K_M: 4.0 to 2.9 GB).

**M1 Max (cpu_neon):** load 13-25 during the measurement, not conclusive.

| | geist | llama.cpp |
| :-- | --: | --: |
| prefill 32 | 31-65 tok/s | 110-116 |
| prefill 512 | 41-69 | 115-119 |
| decode | 13-26 | 41-44 |

The FFN profile (`GEIST_PROFILE_FORWARD`) puts 96-98 % of the MoE time in
the expert kernels. Remeasuring on a quiet host is open in #48.

## Other families

Logits are byte-identical before and after this change on cpu_neon and
Metal for llama-3.2-3b, qwen3-0.6b, qwen3.5-0.8b, qwen3.5-4b, gemma4-e2b and
bitnet b1.58-large.
