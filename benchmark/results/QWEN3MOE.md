# Qwen3-MoE (Qwen3-30B-A3B) on the CPU backends (#48, phase 1)

**Result.** The `qwen3moe` family runs Qwen3-30B-A3B on `cpu_neon`,
`cpu_x86` and `cpu_scalar`.
- **Next-token logits** meet the tolerance declared in #48 on cpu_neon.
- **Accuracy with an FP32 KV cache.** Against the exact reference, geist's
  logits are as close as llama.cpp's: mean per-position JS 0.0020 vs 0.0020
  on prose, 0.0010 vs 0.0010 on code.
- **cpu_x86's default INT8 KV cache costs that accuracy.** Mean JS rises to
  0.0060, and single positions reach 0.19 on prose and 0.69 on code. The
  MoE amplifies this KV error about 16 times more than a dense model of the
  same quantization does. See "Where the divergence comes from".
- **Speed on the 9950X.** geist decodes faster than llama.cpp (36-37 vs 35
  tok/s) and prefills at 86-91 % of it.

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
- llama.cpp itself, ubatch 32 vs 880 on the same Mac, moves the long
  prompt's logits by **JS 0.017**.
- On cpu_x86, three bit-different but equally exact forms of SiLU give JS
  0.025-0.041 at the same position. That is measured against cpu_scalar
  with its Linux default INT8 KV, so the reference carries the same KV
  error; see below.
- cpu_x86's 0.068 on the long prompt (against llama.cpp) comes from its
  default INT8 KV cache. The next section measures it at 768 positions
  against an exact reference.

## Where the divergence comes from (cpu_x86, 2 x 384 positions)

**Method.**
- Teacher-forced through the decode path (`tools/dump_geist_tf_logits`,
  `benchmark/dump_llamacpp_tf.cpp`): every position's full next-token
  logits, on two held-out texts (*Moby-Dick*, llama.cpp source), 128 prompt
  tokens then 384 positions.
- `tools/js_streams.c` gives the JS of each position against the reference.
- **Reference:** `cpu_scalar` with an **FP32 KV cache**. On Linux,
  cpu_scalar defaults to INT8 KV (`caps.preferred_kv_mode`), so an
  unspecified cpu_scalar run is not exact. A first version of this study
  fell into that trap, and its conclusion (geist closer than llama.cpp) did
  not hold.

### JS per position against the reference

| text | engine | mean | median | p99 | max | > 0.02 | top-1 same |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| *Moby-Dick* | geist cpu_x86, FP32 KV | 0.0020 | 0.0010 | 0.017 | 0.076 | 0.8 % | 97.1 % |
| | llama.cpp (F16 KV) | 0.0020 | 0.0011 | 0.018 | 0.087 | 0.5 % | 95.3 % |
| | geist cpu_x86, INT8 KV (default) | 0.0060 | 0.0020 | 0.124 | 0.188 | 4.4 % | 95.1 % |
| | cpu_scalar, INT8 KV | 0.0061 | 0.0011 | 0.139 | 0.195 | 4.9 % | 95.3 % |
| llama.cpp source | geist cpu_x86, FP32 KV | 0.0010 | 0.0001 | 0.013 | 0.042 | 0.5 % | 98.2 % |
| | llama.cpp (F16 KV) | 0.0010 | 0.0001 | 0.015 | 0.033 | 0.5 % | 98.4 % |
| | geist cpu_x86, INT8 KV (default) | 0.0046 | 0.0002 | 0.037 | **0.692** | 1.8 % | 97.7 % |
| | cpu_scalar, INT8 KV | 0.0034 | 0.0001 | 0.023 | 0.692 | 1.6 % | 97.7 % |

Dense control, Llama 3.2 3B Q4_K_M (*Moby-Dick*, same method):

| engine | mean | max |
| :-- | --: | --: |
| geist cpu_x86, FP32 KV | 0.00044 | 0.0039 |
| geist cpu_x86, INT8 KV | 0.00068 | 0.0115 |
| llama.cpp | 0.0030 | 0.248 |

### Reading

- **geist's arithmetic matches llama.cpp's** on the MoE once the KV cache
  is FP32. On the dense model it is about 7 times closer to the reference.
- **The INT8 KV cache is the source of cpu_x86's larger divergence.**
  cpu_scalar with INT8 KV shows the same error as cpu_x86 with INT8 KV.
- **The MoE amplifies KV error.** INT8 KV adds 0.0002 to the mean JS of the
  dense model and 0.004 to the MoE's, about 16×. The single-position JS of
  0.068 reported for the long prompt earlier is this tail.
- **Routing differs often and mostly harmlessly.** Traced with
  `GEIST_MOE_TRACE` and compared with `tools/moe_trace_compare.py`:
  - With FP32 KV, geist and llama.cpp each choose a different expert set
    than the reference for 16 % of the (layer, position) records; with INT8
    KV, 23-27 %.
  - Nearly always one expert is swapped, at a router margin of about 0.02
    (median).
  - The number of layers that route differently at a position does not
    predict that position's JS (Pearson r -0.11 to +0.04, Spearman ρ
    -0.25 to +0.03).
  - The swaps sit at near-ties and cost little. The JS tail follows the KV
    error, which every layer's expert outputs carry.
- **The fix, measured.** `qwen3moe` asks for an F16 KV cache (#48:
  `transformer_family.kv_auto`). The CPU backends gained F16 KV, and
  cpu_x86 now takes F16 for this family instead of INT8. Same 384
  positions, against the exact reference:

  | KV on cpu_x86 | mean | p99 | max | > 0.02 | top-1 same |
  | :-- | --: | --: | --: | --: | --: |
  | F16 (the family's default now) | 0.0018 | 0.011 | 0.080 | 0.5 % | 98.2 % |
  | FP32 | 0.0020 | 0.017 | 0.076 | 0.8 % | 97.1 % |
  | INT8 (the backend's default) | 0.0060 | 0.124 | 0.188 | 4.4 % | 95.1 % |

  Speed, 9950X, 2 rounds:

  | KV | decode | prefill 512 |
  | :-- | --: | --: |
  | F16 | 35.6-37.5 tok/s | 214 tok/s |
  | INT8 | 35.4-36.5 | 219 |
  | FP32 | 34.2-37.6 | 214 |

  F16 is as accurate as FP32 and costs nothing measurable over INT8.
- **Hypothesis, not tested here.** INT8 KV quantizes K and V per head and
  token. With 4 KV heads of 128, an error in one head reaches all 8 query
  heads of its group, and from there every routing decision downstream.

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
