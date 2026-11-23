## Metal profile, geist against mlx-lm (#590)

### Qwen3.5 0.8B, 8-bit

geist and mlx-lm tokens equal: True

| engine | cell | prefill tok/s | decode tok/s | tokens against the engine's plain cell |
| :-- | :-- | --: | --: | :-- |
| geist | plain | 7170 (7147–7182) | 152.5 (152.4–152.5) | same |
| mlx | plain | 7358 (7165–7682) | 160.8 (160.8–160.9) | same |
| geist | trace | 7180 (7156–7183) | 152.6 (152.4–152.6) | same |
| mlx | trace | 7188 (7180–7365) | 160.1 (160.0–160.2) | same |
| geist | library concurrent encoders | 7420 (7394–7431) | 154.7 (154.6–154.8) | same |
| geist | first buffer after 8 | 7181 (7158–7181) | 154.6 (154.6–154.7) | same |
| geist | first buffer after 16 | 7186 (7153–7188) | 154.3 (154.2–154.4) | same |
| geist | first buffer after 32 | 7191 (7159–7192) | 153.9 (153.9–154.2) | same |
| geist | spin instead of wait | 7187 (7154–7188) | 153.3 (153.2–153.4) | same |
| geist | branch defaults (concurrent, first buffer after 16) | 7431 (7390–7431) | 157.0 (156.8–157.1) | same |
| geist | branch defaults, untraced | 7411 (7380–7412) | 156.9 (156.7–156.9) | same |
| geist | branch defaults, spin | 7433 (7410–7440) | 156.9 (156.9–157.0) | same |
| geist | 128 rows per prefill pass | 6381 (6351–6384) | 152.6 (152.5–152.6) | same |
| geist | 512 rows per prefill pass | 7360 (7300–7376) | 152.6 (152.4–152.6) | same |
| geist | 512 rows per prefill pass, branch defaults | 7478 (7383–7486) | 156.9 (156.6–157.0) | same |
| geist | plain again | 7181 (7155–7184) | 152.6 (152.6–152.6) | same |
| mlx | plain again | 7233 (7171–7471) | 160.1 (160.0–160.1) | same |

GPU state over a 768-token decode (macmon, samples with the GPU over 50 % active): {"geist": {"samples": 103, "busy_samples": 93, "gpu_freq_mhz": 1678, "gpu_active": 0.9676541090011597, "gpu_w": 5.757907390594482, "cpu_w": 0.5981455445289612, "decode_tps": 150.96469986792243}, "mlx": {"samples": 149, "busy_samples": 132, "gpu_freq_mhz": 1302.0, "gpu_active": 0.7504666745662689, "gpu_w": 3.5657613277435303, "cpu_w": 2.800515055656433, "decode_tps": 160.50602460209154}}

#### A/B, a1751c84f0d446e122ffabe5f5dfab7fadd21c13 against HEAD

# Revision A/B, each revision built from scratch

Model: `qwen3.5-0.8b-q8_0.gguf` (`0ad885ffd4bb022f`)
Revisions: base = `a1751c84f0d4` (a1751c84f0d446e122ffabe5f5dfab7fadd21c13), new = `a93ea444ed98` (HEAD); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 10 cycles, order rotating; sequences 512, decode 128, repeats 1 (median), warmup 64; prompt `moe-long.txt`; OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 GEIST_PREFILL_THREADS=4 GEIST_DECODE_THREADS=4 GEIST_TEXT_ONLY=1 GEIST_BENCH_BACKEND=metal
Host: Apple M6, 12 CPUs, Darwin 27.2.0, THP n/a

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 512 | new | 71.6 | 69.5 | -2.9 % | [-3.0 %, -2.8 %] | 10/10 | faster |
| prefill_ms @ 512 | control | 71.6 | 71.5 | -0.1 % | [-0.2 %, +0.1 %] | 5/10 | within noise |
| decode_ms @ 512 | new | 839.6 | 816.8 | -2.7 % | [-2.8 %, -2.7 %] | 10/10 | faster |
| decode_ms @ 512 | control | 839.6 | 839.5 | -0.0 % | [-0.1 %, +0.1 %] | 7/10 | within noise |
| ttft_ms @ 512 | new | 71.6 | 69.5 | -2.9 % | [-3.0 %, -2.8 %] | 10/10 | faster |
| ttft_ms @ 512 | control | 71.6 | 71.6 | -0.1 % | [-0.2 %, +0.1 %] | 5/10 | within noise |

Control: the baseline's own binary again; its intervals reach ±0.2 %, the resolution of this run.

Cool start: every run began at GPU 39.7-47.0 C (limit 65), CPU 38.9-48.7 C (limit 65); 33 s spent cooling.

Power: all 30 runs started on AC power with Low Power Mode off; 0 s spent waiting for it.

3 comparisons: at 95 % about one interval in twenty excludes zero by chance.


Settings before #590 against the branch's defaults: process start to first token, ms: {"serial": {"median": 56.765145999989386, "min": 55.68175000001929, "max": 60.0738750000005, "n": 10}, "branch defaults": {"median": 58.60014550000869, "min": 56.08779199999958, "max": 60.53604100000598, "n": 10}}

bench_nll logits identical across the arms: True {"decode": {"serial": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}, "branch defaults": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}, "origin/main": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}}, "prefill": {"serial": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}, "branch defaults": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}, "origin/main": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}}}

bench_nll prefill-path logits at 128 and 512 rows per pass against the default 256: {"128": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f", "same_as_256": true}, "512": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f", "same_as_256": true}}


### prefill

| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | commit to GPU start, median us | waitUntilCompleted calls | ms in them |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| geist trace | 3 | 71.38 | 71.05 | 71.05 | 0.33 | 4.4 | 61 | 0 | 981 | 0% | 0 | 17136 | 45 | 69.98 |
| mlx trace | 3 | 70.69 | 56.70 | 56.84 | 13.99 | 45.7 | 46 | 2 | 1718 | 100% | 1038 | 2297 | 0 | 0.00 |
| geist library concurrent encoders | 3 | 69.05 | 68.74 | 68.74 | 0.31 | 4.6 | 61 | 0 | 981 | 100% | 822 | 16542 | 45 | 67.31 |
| geist first buffer after 8 | 3 | 71.38 | 71.06 | 71.06 | 0.32 | 4.1 | 61 | 0 | 981 | 0% | 0 | 17138 | 45 | 70.06 |
| geist first buffer after 16 | 3 | 71.35 | 71.05 | 71.05 | 0.31 | 4.1 | 61 | 0 | 981 | 0% | 0 | 17147 | 45 | 70.05 |
| geist first buffer after 32 | 3 | 71.31 | 70.98 | 70.98 | 0.32 | 4.2 | 61 | 0 | 981 | 0% | 0 | 17115 | 45 | 69.96 |
| geist spin instead of wait | 3 | 71.35 | 71.08 | 71.08 | 0.27 | 72.7 | 61 | 0 | 981 | 0% | 0 | 17145 | 45 | 70.69 |
| geist branch defaults (concurrent, first buffer after 16) | 3 | 69.03 | 68.75 | 68.75 | 0.28 | 4.6 | 61 | 0 | 981 | 100% | 822 | 16547 | 45 | 67.29 |
| geist branch defaults, spin | 3 | 68.93 | 68.63 | 68.63 | 0.30 | 70.4 | 61 | 0 | 981 | 100% | 822 | 16531 | 45 | 68.10 |
| geist 128 rows per prefill pass | 3 | 80.36 | 80.05 | 80.05 | 0.30 | 7.1 | 121 | 0 | 1957 | 0% | 0 | 9685 | 105 | 78.19 |
| geist 512 rows per prefill pass | 3 | 69.71 | 69.34 | 69.34 | 0.37 | 2.0 | 31 | 0 | 493 | 0% | 0 | 33617 | 15 | 69.09 |
| geist 512 rows per prefill pass, branch defaults | 3 | 68.74 | 68.39 | 68.39 | 0.35 | 2.1 | 31 | 0 | 493 | 100% | 413 | 33135 | 15 | 67.99 |

geist trace, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 27.10 | 38.1 % |
| DeltaNet mixer | 252 | 19.86 | 27.9 % |
| elementwise / copies | 183 | 12.10 | 17.0 % |
| norm | 121 | 7.98 | 11.2 % |
| attention | 48 | 2.71 | 3.8 % |
| lm head / sampling | 2 | 1.19 | 1.7 % |
| embedding | 2 | 0.12 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 26.50 | 37.3 % | 71.2 |
| `rmsnorm_rows_simd` | 121 | 7.98 | 11.2 % | 66.0 |
| `add_rows` | 96 | 6.42 | 9.0 % | 66.9 |
| `silu_mul_rows` | 48 | 3.46 | 4.9 % | 72.1 |
| `dnf_cst_copy` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_conv` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_norm` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_roll` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_chunk` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_state` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_out` | 36 | 2.84 | 4.0 % | 78.8 |
| `rope_rows` | 24 | 1.40 | 2.0 % | 58.4 |

mlx trace, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| elementwise / copies | 720 | 24.74 | 43.5 % |
| quantized matmul | 499 | 16.64 | 29.3 % |
| norm | 279 | 8.44 | 14.9 % |
| DeltaNet mixer | 135 | 3.96 | 7.0 % |
| attention | 70 | 2.86 | 5.0 % |
| embedding | 12 | 0.12 | 0.2 % |
| lm head / sampling | 3 | 0.06 | 0.1 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsbfloat16` | 279 | 8.44 | 14.9 % | 30.3 |
| `affine_qmm_t_nax_bfloat16_t_gs_64_b_8_bm64_bn64_bk64_wm2_wn2_alN_true_` | 133 | 8.39 | 14.8 % | 63.1 |
| `g2_copybfloat16bfloat16` | 80 | 4.13 | 7.3 % | 51.6 |
| `vvn_Addbfloat16` | 46 | 2.89 | 5.1 % | 62.9 |
| `steel_gemm_splitk_accum_bfloat16_float32` | 48 | 2.79 | 4.9 % | 58.2 |
| `gg1_copybfloat16bfloat16` | 90 | 2.67 | 4.7 % | 29.7 |
| `affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0_pr_0_hgs_0_r_4` | 270 | 2.66 | 4.7 % | 9.9 |
| `vsn_Multiplybfloat16` | 36 | 2.27 | 4.0 % | 62.9 |
| `affine_qmm_t_splitk_bfloat16_t_gs_64_b_8_alN_false` | 36 | 2.21 | 3.9 % | 61.4 |
| `CV2ISigmoidADV2IBroadcastACEV2IBroadcastCAFV2IMultiplyDEGV2IBroadcastF` | 57 | 1.83 | 3.2 % | 32.0 |
| `v_Sigmoidbfloat16bfloat16` | 53 | 1.42 | 2.5 % | 26.7 |
| `Cf4IAsTypeADf4ISigmoidCEf4IBroadcastCDFf4IBroadcastDCGf4IMultiplyEFHf4` | 45 | 1.40 | 2.5 % | 31.0 |

geist library concurrent encoders, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 26.24 | 38.2 % |
| DeltaNet mixer | 252 | 18.99 | 27.6 % |
| elementwise / copies | 183 | 11.78 | 17.1 % |
| norm | 121 | 7.77 | 11.3 % |
| attention | 48 | 2.67 | 3.9 % |
| lm head / sampling | 2 | 1.18 | 1.7 % |
| embedding | 2 | 0.11 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 25.65 | 37.3 % | 69.0 |
| `rmsnorm_rows_simd` | 121 | 7.77 | 11.3 % | 64.2 |
| `add_rows` | 96 | 6.22 | 9.0 % | 64.8 |
| `silu_mul_rows` | 48 | 3.37 | 4.9 % | 70.3 |
| `dnf_cst_copy` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_conv` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_norm` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_roll` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_chunk` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_state` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_out` | 36 | 2.71 | 3.9 % | 75.3 |
| `rope_rows` | 24 | 1.39 | 2.0 % | 57.8 |

geist first buffer after 8, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 27.10 | 38.1 % |
| DeltaNet mixer | 252 | 19.86 | 27.9 % |
| elementwise / copies | 183 | 12.11 | 17.0 % |
| norm | 121 | 7.99 | 11.2 % |
| attention | 48 | 2.71 | 3.8 % |
| lm head / sampling | 2 | 1.19 | 1.7 % |
| embedding | 2 | 0.12 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 26.50 | 37.3 % | 71.2 |
| `rmsnorm_rows_simd` | 121 | 7.99 | 11.2 % | 66.0 |
| `add_rows` | 96 | 6.43 | 9.0 % | 66.9 |
| `silu_mul_rows` | 48 | 3.46 | 4.9 % | 72.1 |
| `dnf_cst_copy` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_conv` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_norm` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_roll` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_chunk` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_state` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_out` | 36 | 2.84 | 4.0 % | 78.8 |
| `rope_rows` | 24 | 1.40 | 2.0 % | 58.5 |

geist first buffer after 16, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 27.09 | 38.1 % |
| DeltaNet mixer | 252 | 19.86 | 28.0 % |
| elementwise / copies | 183 | 12.10 | 17.0 % |
| norm | 121 | 7.98 | 11.2 % |
| attention | 48 | 2.71 | 3.8 % |
| lm head / sampling | 2 | 1.19 | 1.7 % |
| embedding | 2 | 0.12 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 26.50 | 37.3 % | 71.2 |
| `rmsnorm_rows_simd` | 121 | 7.98 | 11.2 % | 66.0 |
| `add_rows` | 96 | 6.42 | 9.0 % | 66.9 |
| `silu_mul_rows` | 48 | 3.46 | 4.9 % | 72.1 |
| `dnf_cst_copy` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_conv` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_norm` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_roll` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_chunk` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_state` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_out` | 36 | 2.84 | 4.0 % | 78.8 |
| `rope_rows` | 24 | 1.40 | 2.0 % | 58.4 |

geist first buffer after 32, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 27.07 | 38.1 % |
| DeltaNet mixer | 252 | 19.82 | 27.9 % |
| elementwise / copies | 183 | 12.10 | 17.0 % |
| norm | 121 | 7.98 | 11.2 % |
| attention | 48 | 2.71 | 3.8 % |
| lm head / sampling | 2 | 1.19 | 1.7 % |
| embedding | 2 | 0.11 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 26.47 | 37.3 % | 71.2 |
| `rmsnorm_rows_simd` | 121 | 7.98 | 11.2 % | 65.9 |
| `add_rows` | 96 | 6.42 | 9.0 % | 66.9 |
| `silu_mul_rows` | 48 | 3.46 | 4.9 % | 72.0 |
| `dnf_cst_copy` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_conv` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_norm` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_roll` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_chunk` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_state` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_out` | 36 | 2.83 | 4.0 % | 78.7 |
| `rope_rows` | 24 | 1.40 | 2.0 % | 58.4 |

geist spin instead of wait, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 27.10 | 38.1 % |
| DeltaNet mixer | 252 | 19.87 | 28.0 % |
| elementwise / copies | 183 | 12.11 | 17.0 % |
| norm | 121 | 7.98 | 11.2 % |
| attention | 48 | 2.71 | 3.8 % |
| lm head / sampling | 2 | 1.19 | 1.7 % |
| embedding | 2 | 0.12 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 26.51 | 37.3 % | 71.3 |
| `rmsnorm_rows_simd` | 121 | 7.98 | 11.2 % | 66.0 |
| `add_rows` | 96 | 6.43 | 9.0 % | 66.9 |
| `silu_mul_rows` | 48 | 3.46 | 4.9 % | 72.1 |
| `dnf_cst_copy` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_conv` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_norm` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_roll` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_chunk` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_state` | 36 | 2.84 | 4.0 % | 78.8 |
| `dnf_out` | 36 | 2.84 | 4.0 % | 78.8 |
| `rope_rows` | 24 | 1.40 | 2.0 % | 58.4 |

geist branch defaults (concurrent, first buffer after 16), prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 26.24 | 38.2 % |
| DeltaNet mixer | 252 | 18.97 | 27.6 % |
| elementwise / copies | 183 | 11.79 | 17.1 % |
| norm | 121 | 7.78 | 11.3 % |
| attention | 48 | 2.67 | 3.9 % |
| lm head / sampling | 2 | 1.18 | 1.7 % |
| embedding | 2 | 0.11 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 25.65 | 37.3 % | 69.0 |
| `rmsnorm_rows_simd` | 121 | 7.78 | 11.3 % | 64.3 |
| `add_rows` | 96 | 6.22 | 9.0 % | 64.8 |
| `silu_mul_rows` | 48 | 3.38 | 4.9 % | 70.3 |
| `dnf_cst_copy` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_conv` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_norm` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_roll` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_chunk` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_state` | 36 | 2.71 | 3.9 % | 75.3 |
| `dnf_out` | 36 | 2.71 | 3.9 % | 75.3 |
| `rope_rows` | 24 | 1.39 | 2.0 % | 57.9 |

geist branch defaults, spin, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 26.20 | 38.2 % |
| DeltaNet mixer | 252 | 18.93 | 27.6 % |
| elementwise / copies | 183 | 11.77 | 17.1 % |
| norm | 121 | 7.77 | 11.3 % |
| attention | 48 | 2.67 | 3.9 % |
| lm head / sampling | 2 | 1.19 | 1.7 % |
| embedding | 2 | 0.11 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 25.60 | 37.3 % | 68.8 |
| `rmsnorm_rows_simd` | 121 | 7.77 | 11.3 % | 64.2 |
| `add_rows` | 96 | 6.21 | 9.0 % | 64.7 |
| `silu_mul_rows` | 48 | 3.37 | 4.9 % | 70.2 |
| `dnf_cst_copy` | 36 | 2.70 | 3.9 % | 75.1 |
| `dnf_conv` | 36 | 2.70 | 3.9 % | 75.1 |
| `dnf_norm` | 36 | 2.70 | 3.9 % | 75.1 |
| `dnf_roll` | 36 | 2.70 | 3.9 % | 75.1 |
| `dnf_chunk` | 36 | 2.70 | 3.9 % | 75.1 |
| `dnf_state` | 36 | 2.70 | 3.9 % | 75.1 |
| `dnf_out` | 36 | 2.70 | 3.9 % | 75.1 |
| `rope_rows` | 24 | 1.39 | 2.0 % | 57.8 |

geist 128 rows per prefill pass, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 745 | 30.64 | 38.3 % |
| DeltaNet mixer | 504 | 21.76 | 27.2 % |
| elementwise / copies | 365 | 13.91 | 17.4 % |
| norm | 241 | 9.16 | 11.4 % |
| attention | 96 | 3.27 | 4.1 % |
| lm head / sampling | 2 | 1.18 | 1.5 % |
| embedding | 4 | 0.13 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 744 | 30.04 | 37.5 % | 40.4 |
| `rmsnorm_rows_simd` | 241 | 9.16 | 11.4 % | 38.0 |
| `add_rows` | 192 | 7.32 | 9.1 % | 38.1 |
| `silu_mul_rows` | 96 | 3.96 | 4.9 % | 41.2 |
| `dnf_cst_copy` | 72 | 3.11 | 3.9 % | 43.2 |
| `dnf_conv` | 72 | 3.11 | 3.9 % | 43.2 |
| `dnf_norm` | 72 | 3.11 | 3.9 % | 43.2 |
| `dnf_roll` | 72 | 3.11 | 3.9 % | 43.2 |
| `dnf_chunk` | 72 | 3.11 | 3.9 % | 43.2 |
| `dnf_state` | 72 | 3.11 | 3.9 % | 43.2 |
| `dnf_out` | 72 | 3.11 | 3.9 % | 43.2 |
| `rope_rows` | 48 | 1.69 | 2.1 % | 35.2 |

geist 512 rows per prefill pass, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 187 | 26.38 | 38.0 % |
| DeltaNet mixer | 126 | 19.24 | 27.8 % |
| elementwise / copies | 92 | 11.90 | 17.2 % |
| norm | 61 | 7.82 | 11.3 % |
| attention | 24 | 2.69 | 3.9 % |
| lm head / sampling | 2 | 1.18 | 1.7 % |
| embedding | 1 | 0.12 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 186 | 25.78 | 37.2 % | 138.6 |
| `rmsnorm_rows_simd` | 61 | 7.82 | 11.3 % | 128.2 |
| `add_rows` | 48 | 6.26 | 9.0 % | 130.4 |
| `silu_mul_rows` | 24 | 3.39 | 4.9 % | 141.3 |
| `dnf_cst_copy` | 18 | 2.75 | 4.0 % | 152.7 |
| `dnf_conv` | 18 | 2.75 | 4.0 % | 152.7 |
| `dnf_norm` | 18 | 2.75 | 4.0 % | 152.7 |
| `dnf_roll` | 18 | 2.75 | 4.0 % | 152.7 |
| `dnf_chunk` | 18 | 2.75 | 4.0 % | 152.7 |
| `dnf_state` | 18 | 2.75 | 4.0 % | 152.7 |
| `dnf_out` | 18 | 2.75 | 4.0 % | 152.7 |
| `rope_rows` | 12 | 1.38 | 2.0 % | 114.6 |

geist 512 rows per prefill pass, branch defaults, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 187 | 26.03 | 38.1 % |
| DeltaNet mixer | 126 | 18.88 | 27.6 % |
| elementwise / copies | 92 | 11.76 | 17.2 % |
| norm | 61 | 7.74 | 11.3 % |
| attention | 24 | 2.67 | 3.9 % |
| lm head / sampling | 2 | 1.19 | 1.7 % |
| embedding | 1 | 0.12 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 186 | 25.43 | 37.2 % | 136.7 |
| `rmsnorm_rows_simd` | 61 | 7.74 | 11.3 % | 126.9 |
| `add_rows` | 48 | 6.18 | 9.0 % | 128.7 |
| `silu_mul_rows` | 24 | 3.35 | 4.9 % | 139.7 |
| `dnf_cst_copy` | 18 | 2.70 | 3.9 % | 149.9 |
| `dnf_conv` | 18 | 2.70 | 3.9 % | 149.9 |
| `dnf_norm` | 18 | 2.70 | 3.9 % | 149.9 |
| `dnf_roll` | 18 | 2.70 | 3.9 % | 149.9 |
| `dnf_chunk` | 18 | 2.70 | 3.9 % | 149.9 |
| `dnf_state` | 18 | 2.70 | 3.9 % | 149.9 |
| `dnf_out` | 18 | 2.70 | 3.9 % | 149.9 |
| `rope_rows` | 12 | 1.37 | 2.0 % | 114.1 |

### decode

| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | commit to GPU start, median us | waitUntilCompleted calls | ms in them |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| geist trace | 3 | 839.19 | 804.24 | 804.24 | 34.95 | 36.1 | 254 | 127 | 45847 | 0% | 0 | 1317 | 127 | 820.21 |
| mlx trace | 3 | 798.16 | 796.97 | 796.99 | 1.19 | 638.2 | 2169 | 778 | 78466 | 100% | 50257 | 3161 | 1 | 4.62 |
| geist library concurrent encoders | 3 | 827.25 | 803.08 | 803.08 | 24.17 | 46.4 | 254 | 127 | 45847 | 100% | 38735 | 1319 | 127 | 800.59 |
| geist first buffer after 8 | 3 | 827.82 | 804.15 | 804.15 | 23.67 | 38.8 | 381 | 127 | 45847 | 0% | 0 | 123 | 127 | 808.05 |
| geist first buffer after 16 | 3 | 829.60 | 804.47 | 804.47 | 25.13 | 39.5 | 381 | 127 | 45847 | 0% | 0 | 243 | 127 | 809.33 |
| geist first buffer after 32 | 3 | 831.24 | 805.17 | 805.17 | 26.07 | 39.4 | 381 | 127 | 45847 | 0% | 0 | 512 | 127 | 811.02 |
| geist spin instead of wait | 3 | 834.99 | 804.33 | 804.33 | 30.66 | 848.8 | 254 | 127 | 45847 | 0% | 0 | 1317 | 127 | 818.66 |
| geist branch defaults (concurrent, first buffer after 16) | 3 | 815.50 | 799.20 | 803.19 | 16.30 | 49.1 | 381 | 127 | 45847 | 100% | 38862 | 199 | 127 | 788.02 |
| geist branch defaults, spin | 3 | 815.60 | 798.72 | 799.56 | 16.87 | 835.3 | 381 | 127 | 45847 | 100% | 38862 | 213 | 127 | 790.36 |
| geist 128 rows per prefill pass | 3 | 839.00 | 804.23 | 804.23 | 34.77 | 35.7 | 254 | 127 | 45847 | 0% | 0 | 1316 | 127 | 820.13 |
| geist 512 rows per prefill pass | 3 | 839.09 | 804.21 | 804.21 | 34.88 | 36.0 | 254 | 127 | 45847 | 0% | 0 | 1316 | 127 | 819.99 |
| geist 512 rows per prefill pass, branch defaults | 3 | 816.20 | 800.15 | 804.33 | 16.05 | 48.0 | 381 | 127 | 45847 | 100% | 38862 | 195 | 127 | 788.59 |

geist trace, decode, one token (median of the per-repeat medians, us): period 6606, gpu 6331, GPU end to wait returns 73, wait returns to first commit 93, first commit to GPU start 113, GPU idle between the token's buffers 0

geist library concurrent encoders, decode, one token (median of the per-repeat medians, us): period 6514, gpu 6230, GPU end to wait returns 74, wait returns to first commit 118, first commit to GPU start 95, GPU idle between the token's buffers 0

geist first buffer after 8, decode, one token (median of the per-repeat medians, us): period 6519, gpu 6331, GPU end to wait returns 71, wait returns to first commit 30, first commit to GPU start 90, GPU idle between the token's buffers 1

geist first buffer after 16, decode, one token (median of the per-repeat medians, us): period 6534, gpu 6331, GPU end to wait returns 72, wait returns to first commit 46, first commit to GPU start 88, GPU idle between the token's buffers 1

geist first buffer after 32, decode, one token (median of the per-repeat medians, us): period 6549, gpu 6336, GPU end to wait returns 72, wait returns to first commit 51, first commit to GPU start 89, GPU idle between the token's buffers 1

geist spin instead of wait, decode, one token (median of the per-repeat medians, us): period 6573, gpu 6332, GPU end to wait returns 57, wait returns to first commit 72, first commit to GPU start 112, GPU idle between the token's buffers 0

geist branch defaults (concurrent, first buffer after 16), decode, one token (median of the per-repeat medians, us): period 6419, gpu 6230, GPU end to wait returns 74, wait returns to first commit 42, first commit to GPU start 77, GPU idle between the token's buffers 1

geist branch defaults, spin, decode, one token (median of the per-repeat medians, us): period 6416, gpu 6242, GPU end to wait returns 57, wait returns to first commit 17, first commit to GPU start 97, GPU idle between the token's buffers 1

geist 128 rows per prefill pass, decode, one token (median of the per-repeat medians, us): period 6606, gpu 6331, GPU end to wait returns 72, wait returns to first commit 93, first commit to GPU start 113, GPU idle between the token's buffers 0

geist 512 rows per prefill pass, decode, one token (median of the per-repeat medians, us): period 6603, gpu 6329, GPU end to wait returns 72, wait returns to first commit 94, first commit to GPU start 113, GPU idle between the token's buffers 0

geist 512 rows per prefill pass, branch defaults, decode, one token (median of the per-repeat medians, us): period 6422, gpu 6234, GPU end to wait returns 74, wait returns to first commit 42, first commit to GPU start 76, GPU idle between the token's buffers 1

geist trace, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 361.48 | 44.9 % |
| elementwise / copies | 8636 | 152.95 | 19.0 % |
| norm | 7747 | 135.19 | 16.8 % |
| DeltaNet mixer | 4572 | 79.13 | 9.8 % |
| attention | 3810 | 68.03 | 8.5 % |
| lm head / sampling | 254 | 5.79 | 0.7 % |
| embedding | 127 | 1.64 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 308.31 | 38.3 % | 17.5 |
| `rmsnorm_rows_simd` | 7747 | 135.19 | 16.8 % | 17.5 |
| `add_rows` | 6096 | 106.34 | 13.2 % | 17.4 |
| `gate_up_silu_q80_n4` | 3048 | 53.17 | 6.6 % | 17.4 |
| `dn_dec_qk` | 2286 | 39.56 | 4.9 % | 17.3 |
| `dn_dec_v` | 2286 | 39.56 | 4.9 % | 17.3 |
| `rope_rows` | 1524 | 27.21 | 3.4 % | 17.9 |
| `qgate_split` | 762 | 13.61 | 1.7 % | 17.9 |
| `scale_rows` | 762 | 13.61 | 1.7 % | 17.9 |
| `kv_append_rows_f16` | 762 | 13.61 | 1.7 % | 17.9 |
| `attention_decq_d256_f16` | 762 | 13.61 | 1.7 % | 17.9 |
| `attention_dec_combine` | 762 | 13.61 | 1.7 % | 17.9 |

mlx trace, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| elementwise / copies | 29870 | 299.42 | 37.6 % |
| quantized matmul | 23853 | 252.11 | 31.6 % |
| norm | 14668 | 151.02 | 18.9 % |
| DeltaNet mixer | 6885 | 48.76 | 6.1 % |
| attention | 2298 | 36.37 | 4.6 % |
| lm head / sampling | 384 | 8.25 | 1.0 % |
| embedding | 508 | 1.03 | 0.1 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0_pr_0_hgs_0_r_4` | 23853 | 252.11 | 31.6 % | 10.6 |
| `rmsbfloat16` | 14668 | 151.02 | 18.9 % | 10.3 |
| `vv_Addbfloat16` | 6123 | 70.97 | 8.9 % | 11.6 |
| `CV2ISigmoidADV2IBroadcastACEV2IBroadcastCAFV2IMultiplyDEGV2IBroadcastF` | 3062 | 35.75 | 4.5 % | 11.7 |
| `gg1_copybfloat16bfloat16` | 4590 | 32.48 | 4.1 % | 7.1 |
| `vs_Multiplybfloat16` | 4590 | 32.19 | 4.0 % | 7.0 |
| `v_Sigmoidbfloat16bfloat16` | 3061 | 28.05 | 3.5 % | 9.2 |
| `gg2_copybfloat16bfloat16` | 1548 | 24.50 | 3.1 % | 15.8 |
| `rope_single_bfloat16_` | 1532 | 24.31 | 3.0 % | 15.9 |
| `g2_copybfloat16bfloat16` | 1532 | 24.15 | 3.0 % | 15.8 |
| `Cf4IAsTypeADf4ISigmoidCEf4IBroadcastCDFf4IBroadcastDCGf4IMultiplyEFHf4` | 2295 | 23.18 | 2.9 % | 10.1 |
| `custom_kernel_gated_delta_step_packed_btree__bfloat16_t_float_128_128_` | 2295 | 16.54 | 2.1 % | 7.2 |

geist library concurrent encoders, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 355.66 | 44.3 % |
| elementwise / copies | 8636 | 150.53 | 18.7 % |
| norm | 7747 | 133.02 | 16.6 % |
| DeltaNet mixer | 4572 | 77.84 | 9.7 % |
| attention | 3810 | 66.97 | 8.3 % |
| lm head / sampling | 254 | 5.73 | 0.7 % |
| embedding | 127 | 1.60 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 303.35 | 37.8 % | 17.2 |
| `rmsnorm_rows_simd` | 7747 | 133.02 | 16.6 % | 17.2 |
| `add_rows` | 6096 | 104.63 | 13.0 % | 17.2 |
| `gate_up_silu_q80_n4` | 3048 | 52.31 | 6.5 % | 17.2 |
| `dn_dec_qk` | 2286 | 38.92 | 4.8 % | 17.0 |
| `dn_dec_v` | 2286 | 38.92 | 4.8 % | 17.0 |
| `rope_rows` | 1524 | 26.79 | 3.3 % | 17.6 |
| `qgate_split` | 762 | 13.39 | 1.7 % | 17.6 |
| `scale_rows` | 762 | 13.39 | 1.7 % | 17.6 |
| `kv_append_rows_f16` | 762 | 13.39 | 1.7 % | 17.6 |
| `attention_decq_d256_f16` | 762 | 13.39 | 1.7 % | 17.6 |
| `attention_dec_combine` | 762 | 13.39 | 1.7 % | 17.6 |

geist first buffer after 8, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 359.57 | 44.7 % |
| elementwise / copies | 8636 | 153.68 | 19.1 % |
| norm | 7747 | 136.92 | 17.0 % |
| DeltaNet mixer | 4572 | 77.50 | 9.6 % |
| attention | 3810 | 68.95 | 8.6 % |
| lm head / sampling | 254 | 5.90 | 0.7 % |
| embedding | 127 | 1.63 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 305.72 | 38.0 % | 17.3 |
| `rmsnorm_rows_simd` | 7747 | 136.92 | 17.0 % | 17.7 |
| `add_rows` | 6096 | 106.41 | 13.2 % | 17.5 |
| `gate_up_silu_q80_n4` | 3048 | 53.86 | 6.7 % | 17.7 |
| `dn_dec_qk` | 2286 | 38.75 | 4.8 % | 16.9 |
| `dn_dec_v` | 2286 | 38.75 | 4.8 % | 16.9 |
| `rope_rows` | 1524 | 27.58 | 3.4 % | 18.1 |
| `qgate_split` | 762 | 13.79 | 1.7 % | 18.1 |
| `scale_rows` | 762 | 13.79 | 1.7 % | 18.1 |
| `kv_append_rows_f16` | 762 | 13.79 | 1.7 % | 18.1 |
| `attention_decq_d256_f16` | 762 | 13.79 | 1.7 % | 18.1 |
| `attention_dec_combine` | 762 | 13.79 | 1.7 % | 18.1 |

geist first buffer after 16, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 357.84 | 44.5 % |
| elementwise / copies | 8636 | 154.37 | 19.2 % |
| norm | 7747 | 136.15 | 16.9 % |
| DeltaNet mixer | 4572 | 78.57 | 9.8 % |
| attention | 3810 | 69.81 | 8.7 % |
| lm head / sampling | 254 | 5.99 | 0.7 % |
| embedding | 127 | 1.72 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 304.59 | 37.9 % | 17.3 |
| `rmsnorm_rows_simd` | 7747 | 136.15 | 16.9 % | 17.6 |
| `add_rows` | 6096 | 106.50 | 13.2 % | 17.5 |
| `gate_up_silu_q80_n4` | 3048 | 53.25 | 6.6 % | 17.5 |
| `dn_dec_qk` | 2286 | 39.29 | 4.9 % | 17.2 |
| `dn_dec_v` | 2286 | 39.29 | 4.9 % | 17.2 |
| `rope_rows` | 1524 | 27.92 | 3.5 % | 18.3 |
| `qgate_split` | 762 | 13.96 | 1.7 % | 18.3 |
| `scale_rows` | 762 | 13.96 | 1.7 % | 18.3 |
| `kv_append_rows_f16` | 762 | 13.96 | 1.7 % | 18.3 |
| `attention_decq_d256_f16` | 762 | 13.96 | 1.7 % | 18.3 |
| `attention_dec_combine` | 762 | 13.96 | 1.7 % | 18.3 |

geist first buffer after 32, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 358.65 | 44.5 % |
| elementwise / copies | 8636 | 154.44 | 19.2 % |
| norm | 7747 | 134.14 | 16.7 % |
| DeltaNet mixer | 4572 | 77.96 | 9.7 % |
| attention | 3810 | 71.70 | 8.9 % |
| lm head / sampling | 254 | 6.33 | 0.8 % |
| embedding | 127 | 1.92 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 305.33 | 37.9 % | 17.3 |
| `rmsnorm_rows_simd` | 7747 | 134.14 | 16.7 % | 17.3 |
| `add_rows` | 6096 | 106.64 | 13.2 % | 17.5 |
| `gate_up_silu_q80_n4` | 3048 | 53.32 | 6.6 % | 17.5 |
| `dn_dec_qk` | 2286 | 38.98 | 4.8 % | 17.1 |
| `dn_dec_v` | 2286 | 38.98 | 4.8 % | 17.1 |
| `rope_rows` | 1524 | 28.68 | 3.6 % | 18.8 |
| `scale_rows` | 762 | 14.34 | 1.8 % | 18.8 |
| `kv_append_rows_f16` | 762 | 14.34 | 1.8 % | 18.8 |
| `attention_decq_d256_f16` | 762 | 14.34 | 1.8 % | 18.8 |
| `attention_dec_combine` | 762 | 14.34 | 1.8 % | 18.8 |
| `sigmoid_mul_rows` | 762 | 14.34 | 1.8 % | 18.8 |

geist spin instead of wait, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 361.53 | 44.9 % |
| elementwise / copies | 8636 | 152.97 | 19.0 % |
| norm | 7747 | 135.21 | 16.8 % |
| DeltaNet mixer | 4572 | 79.14 | 9.8 % |
| attention | 3810 | 68.04 | 8.5 % |
| lm head / sampling | 254 | 5.79 | 0.7 % |
| embedding | 127 | 1.64 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 308.35 | 38.3 % | 17.5 |
| `rmsnorm_rows_simd` | 7747 | 135.21 | 16.8 % | 17.5 |
| `add_rows` | 6096 | 106.35 | 13.2 % | 17.4 |
| `gate_up_silu_q80_n4` | 3048 | 53.18 | 6.6 % | 17.4 |
| `dn_dec_qk` | 2286 | 39.57 | 4.9 % | 17.3 |
| `dn_dec_v` | 2286 | 39.57 | 4.9 % | 17.3 |
| `rope_rows` | 1524 | 27.22 | 3.4 % | 17.9 |
| `qgate_split` | 762 | 13.61 | 1.7 % | 17.9 |
| `scale_rows` | 762 | 13.61 | 1.7 % | 17.9 |
| `kv_append_rows_f16` | 762 | 13.61 | 1.7 % | 17.9 |
| `attention_decq_d256_f16` | 762 | 13.61 | 1.7 % | 17.9 |
| `attention_dec_combine` | 762 | 13.61 | 1.7 % | 17.9 |

geist branch defaults (concurrent, first buffer after 16), decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 351.99 | 43.8 % |
| elementwise / copies | 8636 | 151.98 | 18.9 % |
| norm | 7747 | 133.98 | 16.7 % |
| DeltaNet mixer | 4572 | 77.28 | 9.6 % |
| attention | 3810 | 68.78 | 8.6 % |
| lm head / sampling | 254 | 5.93 | 0.7 % |
| embedding | 127 | 1.68 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 299.60 | 37.3 % | 17.0 |
| `rmsnorm_rows_simd` | 7747 | 133.98 | 16.7 % | 17.3 |
| `add_rows` | 6096 | 104.79 | 13.0 % | 17.2 |
| `gate_up_silu_q80_n4` | 3048 | 52.39 | 6.5 % | 17.2 |
| `dn_dec_qk` | 2286 | 38.64 | 4.8 % | 16.9 |
| `dn_dec_v` | 2286 | 38.64 | 4.8 % | 16.9 |
| `rope_rows` | 1524 | 27.51 | 3.4 % | 18.1 |
| `qgate_split` | 762 | 13.76 | 1.7 % | 18.1 |
| `scale_rows` | 762 | 13.76 | 1.7 % | 18.1 |
| `kv_append_rows_f16` | 762 | 13.76 | 1.7 % | 18.1 |
| `attention_decq_d256_f16` | 762 | 13.76 | 1.7 % | 18.1 |
| `attention_dec_combine` | 762 | 13.76 | 1.7 % | 18.1 |

geist branch defaults, spin, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 352.51 | 44.1 % |
| elementwise / copies | 8636 | 152.22 | 19.0 % |
| norm | 7747 | 134.18 | 16.8 % |
| DeltaNet mixer | 4572 | 77.39 | 9.7 % |
| attention | 3810 | 68.89 | 8.6 % |
| lm head / sampling | 254 | 5.94 | 0.7 % |
| embedding | 127 | 1.68 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 300.04 | 37.5 % | 17.0 |
| `rmsnorm_rows_simd` | 7747 | 134.18 | 16.8 % | 17.3 |
| `add_rows` | 6096 | 104.95 | 13.1 % | 17.2 |
| `gate_up_silu_q80_n4` | 3048 | 52.47 | 6.6 % | 17.2 |
| `dn_dec_qk` | 2286 | 38.69 | 4.8 % | 16.9 |
| `dn_dec_v` | 2286 | 38.69 | 4.8 % | 16.9 |
| `rope_rows` | 1524 | 27.56 | 3.4 % | 18.1 |
| `qgate_split` | 762 | 13.78 | 1.7 % | 18.1 |
| `scale_rows` | 762 | 13.78 | 1.7 % | 18.1 |
| `kv_append_rows_f16` | 762 | 13.78 | 1.7 % | 18.1 |
| `attention_decq_d256_f16` | 762 | 13.78 | 1.7 % | 18.1 |
| `attention_dec_combine` | 762 | 13.78 | 1.7 % | 18.1 |

geist 128 rows per prefill pass, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 361.48 | 44.9 % |
| elementwise / copies | 8636 | 152.95 | 19.0 % |
| norm | 7747 | 135.19 | 16.8 % |
| DeltaNet mixer | 4572 | 79.13 | 9.8 % |
| attention | 3810 | 68.03 | 8.5 % |
| lm head / sampling | 254 | 5.79 | 0.7 % |
| embedding | 127 | 1.64 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 308.31 | 38.3 % | 17.5 |
| `rmsnorm_rows_simd` | 7747 | 135.19 | 16.8 % | 17.5 |
| `add_rows` | 6096 | 106.34 | 13.2 % | 17.4 |
| `gate_up_silu_q80_n4` | 3048 | 53.17 | 6.6 % | 17.4 |
| `dn_dec_qk` | 2286 | 39.56 | 4.9 % | 17.3 |
| `dn_dec_v` | 2286 | 39.56 | 4.9 % | 17.3 |
| `rope_rows` | 1524 | 27.21 | 3.4 % | 17.9 |
| `qgate_split` | 762 | 13.61 | 1.7 % | 17.9 |
| `scale_rows` | 762 | 13.61 | 1.7 % | 17.9 |
| `kv_append_rows_f16` | 762 | 13.61 | 1.7 % | 17.9 |
| `attention_decq_d256_f16` | 762 | 13.61 | 1.7 % | 17.9 |
| `attention_dec_combine` | 762 | 13.61 | 1.7 % | 17.9 |

geist 512 rows per prefill pass, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 361.47 | 44.9 % |
| elementwise / copies | 8636 | 152.95 | 19.0 % |
| norm | 7747 | 135.19 | 16.8 % |
| DeltaNet mixer | 4572 | 79.12 | 9.8 % |
| attention | 3810 | 68.03 | 8.5 % |
| lm head / sampling | 254 | 5.79 | 0.7 % |
| embedding | 127 | 1.64 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 308.30 | 38.3 % | 17.5 |
| `rmsnorm_rows_simd` | 7747 | 135.19 | 16.8 % | 17.5 |
| `add_rows` | 6096 | 106.34 | 13.2 % | 17.4 |
| `gate_up_silu_q80_n4` | 3048 | 53.17 | 6.6 % | 17.4 |
| `dn_dec_qk` | 2286 | 39.56 | 4.9 % | 17.3 |
| `dn_dec_v` | 2286 | 39.56 | 4.9 % | 17.3 |
| `rope_rows` | 1524 | 27.21 | 3.4 % | 17.9 |
| `qgate_split` | 762 | 13.61 | 1.7 % | 17.9 |
| `scale_rows` | 762 | 13.61 | 1.7 % | 17.9 |
| `kv_append_rows_f16` | 762 | 13.61 | 1.7 % | 17.9 |
| `attention_decq_d256_f16` | 762 | 13.61 | 1.7 % | 17.9 |
| `attention_dec_combine` | 762 | 13.61 | 1.7 % | 17.9 |

geist 512 rows per prefill pass, branch defaults, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 352.35 | 43.8 % |
| elementwise / copies | 8636 | 152.14 | 18.9 % |
| norm | 7747 | 134.11 | 16.7 % |
| DeltaNet mixer | 4572 | 77.35 | 9.6 % |
| attention | 3810 | 68.85 | 8.6 % |
| lm head / sampling | 254 | 5.93 | 0.7 % |
| embedding | 127 | 1.68 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 299.90 | 37.3 % | 17.0 |
| `rmsnorm_rows_simd` | 7747 | 134.11 | 16.7 % | 17.3 |
| `add_rows` | 6096 | 104.89 | 13.0 % | 17.2 |
| `gate_up_silu_q80_n4` | 3048 | 52.45 | 6.5 % | 17.2 |
| `dn_dec_qk` | 2286 | 38.68 | 4.8 % | 16.9 |
| `dn_dec_v` | 2286 | 38.68 | 4.8 % | 16.9 |
| `rope_rows` | 1524 | 27.54 | 3.4 % | 18.1 |
| `qgate_split` | 762 | 13.77 | 1.7 % | 18.1 |
| `scale_rows` | 762 | 13.77 | 1.7 % | 18.1 |
| `kv_append_rows_f16` | 762 | 13.77 | 1.7 % | 18.1 |
| `attention_decq_d256_f16` | 762 | 13.77 | 1.7 % | 18.1 |
| `attention_dec_combine` | 762 | 13.77 | 1.7 % | 18.1 |


### Llama 3.2 3B Instruct, 4-bit

geist and mlx-lm tokens equal: False

| engine | cell | prefill tok/s | decode tok/s | tokens against the engine's plain cell |
| :-- | :-- | --: | --: | :-- |
| geist | plain | 2058 (2058–2059) | 67.5 (67.5–67.5) | same |
| mlx | plain | 2635 (2633–2636) | 77.1 (77.1–77.1) | same |
| geist | trace | 2056 (2052–2057) | 67.4 (67.3–67.4) | same |
| mlx | trace | 2639 (2637–2642) | 77.0 (77.0–77.0) | same |
| geist | library concurrent encoders | 2054 (2038–2056) | 67.4 (67.0–67.4) | same |
| geist | first buffer after 8 | 2057 (2057–2058) | 67.9 (67.8–67.9) | same |
| geist | first buffer after 16 | 2059 (2057–2061) | 67.7 (67.7–67.8) | same |
| geist | first buffer after 32 | 1986 (1982–1996) | 64.6 (64.2–64.6) | same |
| geist | spin instead of wait | 1940 (1934–1948) | 61.0 (60.6–61.7) | same |
| geist | branch defaults (concurrent, first buffer after 16) | 1940 (1928–1948) | 62.2 (62.2–62.6) | same |
| geist | branch defaults, untraced | 1955 (1953–1957) | 62.4 (62.3–62.5) | same |
| geist | branch defaults, spin | 1955 (1945–1956) | 62.8 (62.8–62.8) | same |
| geist | 128 rows per prefill pass | 1892 (1850–1905) | 61.4 (58.0–62.4) | same |
| geist | 512 rows per prefill pass | 1952 (1947–1955) | 62.4 (62.4–62.6) | same |
| geist | 512 rows per prefill pass, branch defaults | 1941 (1935–1956) | 62.4 (62.1–62.7) | same |
| geist | plain again | 1960 (1957–2036) | 62.8 (62.5–63.9) | same |
| mlx | plain again | 2623 (2600–2624) | 76.5 (75.9–76.6) | same |

GPU state over a 768-token decode (macmon, samples with the GPU over 50 % active): {"geist": {"samples": 242, "busy_samples": 232, "gpu_freq_mhz": 1548.0, "gpu_active": 0.8923743665218353, "gpu_w": 10.519766807556152, "cpu_w": 0.3768514394760132, "decode_tps": 64.46078764783707}, "mlx": {"samples": 321, "busy_samples": 304, "gpu_freq_mhz": 1296.0, "gpu_active": 0.7471326589584351, "gpu_w": 5.649856328964233, "cpu_w": 1.4169241189956665, "decode_tps": 73.46980430712455}}

#### A/B, a1751c84f0d446e122ffabe5f5dfab7fadd21c13 against HEAD

# Revision A/B, each revision built from scratch

Model: `llama-3.2-3b-Q4_K_M.gguf` (`6c1a2b4116103267`)
Revisions: base = `a1751c84f0d4` (a1751c84f0d446e122ffabe5f5dfab7fadd21c13), new = `a93ea444ed98` (HEAD); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 10 cycles, order rotating; sequences 512, decode 128, repeats 1 (median), warmup 64; prompt `moe-long.txt`; OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 GEIST_PREFILL_THREADS=4 GEIST_DECODE_THREADS=4 GEIST_TEXT_ONLY=1 GEIST_BENCH_BACKEND=metal
Host: Apple M6, 12 CPUs, Darwin 27.2.0, THP n/a

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 512 | new | 262.3 | 262.0 | -0.2 % | [-0.6 %, +0.5 %] | 7/10 | within noise |
| prefill_ms @ 512 | control | 262.3 | 261.9 | -0.0 % | [-0.6 %, +0.9 %] | 5/10 | within noise |
| decode_ms @ 512 | new | 2038.1 | 2027.4 | -0.6 % | [-0.8 %, -0.2 %] | 9/10 | faster |
| decode_ms @ 512 | control | 2038.1 | 2041.9 | -0.0 % | [-0.3 %, +0.5 %] | 5/10 | within noise |
| ttft_ms @ 512 | new | 262.3 | 262.0 | -0.2 % | [-0.6 %, +0.5 %] | 7/10 | within noise |
| ttft_ms @ 512 | control | 262.3 | 261.9 | -0.0 % | [-0.6 %, +0.9 %] | 5/10 | within noise |

Control: the baseline's own binary again; its intervals reach ±0.9 %, the resolution of this run.

Cool start: every run began at GPU 42.3-52.4 C (limit 65), CPU 40.3-48.0 C (limit 65); 33 s spent cooling.

Power: all 30 runs started on AC power with Low Power Mode off; 0 s spent waiting for it.

3 comparisons: at 95 % about one interval in twenty excludes zero by chance.


Settings before #590 against the branch's defaults: process start to first token, ms: {"serial": {"median": 97.71083350000254, "min": 94.56808300001285, "max": 103.87362500000563, "n": 10}, "branch defaults": {"median": 100.01397849998739, "min": 96.94616699994185, "max": 107.32008299999052, "n": 10}}

bench_nll logits identical across the arms: True {"decode": {"serial": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}, "branch defaults": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}, "origin/main": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}}, "prefill": {"serial": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}, "branch defaults": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}, "origin/main": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}}}

bench_nll prefill-path logits at 128 and 512 rows per pass against the default 256: {"128": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76", "same_as_256": true}, "512": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76", "same_as_256": true}}


### prefill

| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | commit to GPU start, median us | waitUntilCompleted calls | ms in them |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| geist trace | 3 | 249.14 | 248.92 | 248.92 | 0.22 | 3.0 | 43 | 1 | 683 | 0% | 0 | 86187 | 27 | 248.01 |
| mlx trace | 3 | 193.99 | 190.46 | 191.19 | 3.53 | 19.6 | 50 | 24 | 1301 | 100% | 797 | 12271 | 0 | 0.00 |
| geist library concurrent encoders | 3 | 249.82 | 249.57 | 249.57 | 0.25 | 3.3 | 43 | 1 | 683 | 100% | 640 | 86328 | 27 | 248.42 |
| geist first buffer after 8 | 3 | 248.89 | 248.66 | 248.66 | 0.23 | 3.0 | 43 | 1 | 683 | 0% | 0 | 86096 | 27 | 247.78 |
| geist first buffer after 16 | 3 | 248.67 | 248.44 | 248.44 | 0.23 | 3.0 | 43 | 1 | 683 | 0% | 0 | 86003 | 27 | 247.57 |
| geist first buffer after 32 | 3 | 257.59 | 257.31 | 257.31 | 0.28 | 3.1 | 43 | 1 | 683 | 0% | 0 | 89183 | 27 | 256.37 |
| geist spin instead of wait | 3 | 263.82 | 263.46 | 263.46 | 0.35 | 264.2 | 43 | 1 | 683 | 0% | 0 | 91185 | 27 | 262.90 |
| geist branch defaults (concurrent, first buffer after 16) | 3 | 264.14 | 263.77 | 263.77 | 0.37 | 3.7 | 43 | 1 | 683 | 100% | 640 | 91415 | 27 | 262.36 |
| geist branch defaults, spin | 3 | 262.28 | 262.02 | 262.04 | 0.25 | 263.3 | 43 | 1 | 683 | 100% | 640 | 90972 | 27 | 261.50 |
| geist 128 rows per prefill pass | 3 | 272.08 | 271.70 | 271.70 | 0.38 | 6.7 | 85 | 1 | 1361 | 0% | 0 | 47247 | 69 | 269.10 |
| geist 512 rows per prefill pass | 3 | 262.40 | 262.03 | 262.03 | 0.37 | 1.4 | 22 | 1 | 344 | 0% | 0 | 126139 | 6 | 261.86 |
| geist 512 rows per prefill pass, branch defaults | 3 | 263.39 | 262.96 | 262.97 | 0.43 | 1.7 | 22 | 1 | 344 | 100% | 322 | 126694 | 6 | 262.63 |

geist trace, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 143.60 | 57.7 % |
| norm | 113 | 40.76 | 16.4 % |
| attention | 112 | 39.34 | 15.8 % |
| elementwise / copies | 61 | 23.53 | 9.5 % |
| lm head / sampling | 2 | 1.00 | 0.4 % |
| embedding | 2 | 0.69 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 122.38 | 49.2 % | 364.2 |
| `rmsnorm_add_norm_rows_simd` | 110 | 39.58 | 15.9 % | 359.8 |
| `silu_mul_rows` | 56 | 21.30 | 8.6 % | 380.4 |
| `matmul_q6k_mm_tc` | 56 | 20.72 | 8.3 % | 369.9 |
| `rope_append_rows_f16` | 56 | 19.67 | 7.9 % | 351.2 |
| `attention_flash_tc_d128` | 56 | 19.67 | 7.9 % | 351.2 |
| `copy_u32` | 3 | 1.36 | 0.5 % | 454.5 |
| `rmsnorm_rows_simd` | 3 | 1.19 | 0.5 % | 395.8 |
| `add_rows` | 2 | 0.87 | 0.3 % | 432.5 |
| `embed_lookup_scaled_rows` | 2 | 0.69 | 0.3 % | 344.4 |
| `matvec_q6k_n4` | 1 | 0.50 | 0.2 % | 498.4 |
| `argmax_f32_part` | 1 | 0.50 | 0.2 % | 498.4 |

mlx trace, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 475 | 70.30 | 36.8 % |
| elementwise / copies | 471 | 70.13 | 36.7 % |
| attention | 203 | 30.05 | 15.7 % |
| norm | 137 | 20.29 | 10.6 % |
| embedding | 12 | 0.37 | 0.2 % |
| lm head / sampling | 3 | 0.04 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmm_t_nax_float16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_b` | 191 | 62.93 | 32.9 % | 329.5 |
| `rmsfloat16` | 137 | 20.29 | 10.6 % | 148.1 |
| `sn_copyfloat16float16` | 82 | 18.85 | 9.9 % | 229.9 |
| `rope_freqs_float16_` | 55 | 18.04 | 9.4 % | 328.0 |
| `vvn_Addfloat16` | 54 | 17.82 | 9.3 % | 330.0 |
| `gg2_copyfloat16float16` | 160 | 12.24 | 6.4 % | 76.5 |
| `Cf2ISigmoidADf2IBroadcastACEf2IBroadcastCAFf2IMultiplyDEGf2IBroadcastF` | 67 | 9.95 | 5.2 % | 148.4 |
| `gg3_copyfloat16float16` | 28 | 9.17 | 4.8 % | 327.4 |
| `steel_attention_float16_bq64_bk32_bd128_bv128_wm4_wn1_maskfloat16_alig` | 27 | 8.91 | 4.7 % | 329.9 |
| `affine_qmv_fast_float16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4` | 284 | 7.37 | 3.9 % | 25.9 |
| `vv_Addfloat16` | 80 | 2.11 | 1.1 % | 26.4 |
| `rope_single_freqs_float16_` | 81 | 2.06 | 1.1 % | 25.4 |

geist library concurrent encoders, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 143.94 | 57.7 % |
| norm | 113 | 40.86 | 16.4 % |
| attention | 112 | 39.45 | 15.8 % |
| elementwise / copies | 61 | 23.60 | 9.5 % |
| lm head / sampling | 2 | 1.01 | 0.4 % |
| embedding | 2 | 0.68 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 122.67 | 49.2 % | 365.1 |
| `rmsnorm_add_norm_rows_simd` | 110 | 39.68 | 15.9 % | 360.7 |
| `silu_mul_rows` | 56 | 21.35 | 8.6 % | 381.3 |
| `matmul_q6k_mm_tc` | 56 | 20.77 | 8.3 % | 370.9 |
| `rope_append_rows_f16` | 56 | 19.72 | 7.9 % | 352.2 |
| `attention_flash_tc_d128` | 56 | 19.72 | 7.9 % | 352.2 |
| `copy_u32` | 3 | 1.37 | 0.6 % | 458.0 |
| `rmsnorm_rows_simd` | 3 | 1.19 | 0.5 % | 396.0 |
| `add_rows` | 2 | 0.87 | 0.3 % | 435.4 |
| `embed_lookup_scaled_rows` | 2 | 0.68 | 0.3 % | 342.4 |
| `matvec_q6k_n4` | 1 | 0.50 | 0.2 % | 503.3 |
| `argmax_f32_part` | 1 | 0.50 | 0.2 % | 503.3 |

geist first buffer after 8, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 143.44 | 57.7 % |
| norm | 113 | 40.72 | 16.4 % |
| attention | 112 | 39.31 | 15.8 % |
| elementwise / copies | 61 | 23.51 | 9.5 % |
| lm head / sampling | 2 | 1.00 | 0.4 % |
| embedding | 2 | 0.68 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 122.24 | 49.2 % | 363.8 |
| `rmsnorm_add_norm_rows_simd` | 110 | 39.54 | 15.9 % | 359.5 |
| `silu_mul_rows` | 56 | 21.28 | 8.6 % | 380.0 |
| `matmul_q6k_mm_tc` | 56 | 20.70 | 8.3 % | 369.7 |
| `rope_append_rows_f16` | 56 | 19.66 | 7.9 % | 351.0 |
| `attention_flash_tc_d128` | 56 | 19.66 | 7.9 % | 351.0 |
| `copy_u32` | 3 | 1.36 | 0.5 % | 454.3 |
| `rmsnorm_rows_simd` | 3 | 1.18 | 0.5 % | 393.9 |
| `add_rows` | 2 | 0.86 | 0.3 % | 432.3 |
| `embed_lookup_scaled_rows` | 2 | 0.68 | 0.3 % | 341.7 |
| `matvec_q6k_n4` | 1 | 0.50 | 0.2 % | 498.3 |
| `argmax_f32_part` | 1 | 0.50 | 0.2 % | 498.3 |

geist first buffer after 16, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 143.31 | 57.7 % |
| norm | 113 | 40.68 | 16.4 % |
| attention | 112 | 39.26 | 15.8 % |
| elementwise / copies | 61 | 23.49 | 9.5 % |
| lm head / sampling | 2 | 1.00 | 0.4 % |
| embedding | 2 | 0.68 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 122.16 | 49.2 % | 363.6 |
| `rmsnorm_add_norm_rows_simd` | 110 | 39.50 | 15.9 % | 359.1 |
| `silu_mul_rows` | 56 | 21.27 | 8.6 % | 379.7 |
| `matmul_q6k_mm_tc` | 56 | 20.66 | 8.3 % | 368.9 |
| `rope_append_rows_f16` | 56 | 19.63 | 7.9 % | 350.6 |
| `attention_flash_tc_d128` | 56 | 19.63 | 7.9 % | 350.6 |
| `copy_u32` | 3 | 1.36 | 0.5 % | 454.8 |
| `rmsnorm_rows_simd` | 3 | 1.18 | 0.5 % | 394.0 |
| `add_rows` | 2 | 0.86 | 0.3 % | 432.4 |
| `embed_lookup_scaled_rows` | 2 | 0.68 | 0.3 % | 341.3 |
| `matvec_q6k_n4` | 1 | 0.50 | 0.2 % | 499.5 |
| `argmax_f32_part` | 1 | 0.50 | 0.2 % | 499.5 |

geist first buffer after 32, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 148.46 | 57.7 % |
| norm | 113 | 42.16 | 16.4 % |
| attention | 112 | 40.53 | 15.8 % |
| elementwise / copies | 61 | 24.39 | 9.5 % |
| lm head / sampling | 2 | 1.05 | 0.4 % |
| embedding | 2 | 0.72 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 126.45 | 49.1 % | 376.4 |
| `rmsnorm_add_norm_rows_simd` | 110 | 40.91 | 15.9 % | 371.9 |
| `silu_mul_rows` | 56 | 22.06 | 8.6 % | 393.9 |
| `matmul_q6k_mm_tc` | 56 | 21.47 | 8.3 % | 383.5 |
| `rope_append_rows_f16` | 56 | 20.27 | 7.9 % | 361.9 |
| `attention_flash_tc_d128` | 56 | 20.27 | 7.9 % | 361.9 |
| `copy_u32` | 3 | 1.43 | 0.6 % | 476.4 |
| `rmsnorm_rows_simd` | 3 | 1.25 | 0.5 % | 417.4 |
| `add_rows` | 2 | 0.90 | 0.4 % | 451.1 |
| `embed_lookup_scaled_rows` | 2 | 0.72 | 0.3 % | 362.5 |
| `matvec_q6k_n4` | 1 | 0.53 | 0.2 % | 527.2 |
| `argmax_f32_part` | 1 | 0.53 | 0.2 % | 527.2 |

geist spin instead of wait, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 151.98 | 57.7 % |
| norm | 113 | 43.10 | 16.4 % |
| attention | 112 | 41.58 | 15.8 % |
| elementwise / copies | 61 | 24.96 | 9.5 % |
| lm head / sampling | 2 | 1.09 | 0.4 % |
| embedding | 2 | 0.74 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 129.40 | 49.1 % | 385.1 |
| `rmsnorm_add_norm_rows_simd` | 110 | 41.82 | 15.9 % | 380.2 |
| `silu_mul_rows` | 56 | 22.53 | 8.6 % | 402.4 |
| `matmul_q6k_mm_tc` | 56 | 22.04 | 8.4 % | 393.6 |
| `rope_append_rows_f16` | 56 | 20.79 | 7.9 % | 371.3 |
| `attention_flash_tc_d128` | 56 | 20.79 | 7.9 % | 371.3 |
| `copy_u32` | 3 | 1.49 | 0.6 % | 496.3 |
| `rmsnorm_rows_simd` | 3 | 1.29 | 0.5 % | 428.4 |
| `add_rows` | 2 | 0.94 | 0.4 % | 471.0 |
| `embed_lookup_scaled_rows` | 2 | 0.74 | 0.3 % | 369.1 |
| `matvec_q6k_n4` | 1 | 0.55 | 0.2 % | 547.0 |
| `argmax_f32_part` | 1 | 0.55 | 0.2 % | 547.0 |

geist branch defaults (concurrent, first buffer after 16), prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 152.11 | 57.7 % |
| norm | 113 | 43.18 | 16.4 % |
| attention | 112 | 41.60 | 15.8 % |
| elementwise / copies | 61 | 25.01 | 9.5 % |
| lm head / sampling | 2 | 1.11 | 0.4 % |
| embedding | 2 | 0.74 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 129.46 | 49.1 % | 385.3 |
| `rmsnorm_add_norm_rows_simd` | 110 | 41.89 | 15.9 % | 380.8 |
| `silu_mul_rows` | 56 | 22.55 | 8.6 % | 402.7 |
| `matmul_q6k_mm_tc` | 56 | 22.09 | 8.4 % | 394.5 |
| `rope_append_rows_f16` | 56 | 20.80 | 7.9 % | 371.4 |
| `attention_flash_tc_d128` | 56 | 20.80 | 7.9 % | 371.4 |
| `copy_u32` | 3 | 1.50 | 0.6 % | 501.3 |
| `rmsnorm_rows_simd` | 3 | 1.29 | 0.5 % | 430.0 |
| `add_rows` | 2 | 0.95 | 0.4 % | 474.7 |
| `embed_lookup_scaled_rows` | 2 | 0.74 | 0.3 % | 367.9 |
| `matvec_q6k_n4` | 1 | 0.55 | 0.2 % | 554.3 |
| `argmax_f32_part` | 1 | 0.55 | 0.2 % | 554.3 |

geist branch defaults, spin, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 151.14 | 57.7 % |
| norm | 113 | 42.88 | 16.4 % |
| attention | 112 | 41.37 | 15.8 % |
| elementwise / copies | 61 | 24.81 | 9.5 % |
| lm head / sampling | 2 | 1.06 | 0.4 % |
| embedding | 2 | 0.72 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 128.69 | 49.1 % | 383.0 |
| `rmsnorm_add_norm_rows_simd` | 110 | 41.62 | 15.9 % | 378.4 |
| `silu_mul_rows` | 56 | 22.44 | 8.6 % | 400.7 |
| `matmul_q6k_mm_tc` | 56 | 21.92 | 8.4 % | 391.4 |
| `rope_append_rows_f16` | 56 | 20.69 | 7.9 % | 369.4 |
| `attention_flash_tc_d128` | 56 | 20.69 | 7.9 % | 369.4 |
| `copy_u32` | 3 | 1.45 | 0.6 % | 484.6 |
| `rmsnorm_rows_simd` | 3 | 1.26 | 0.5 % | 418.5 |
| `add_rows` | 2 | 0.92 | 0.4 % | 461.4 |
| `embed_lookup_scaled_rows` | 2 | 0.72 | 0.3 % | 362.3 |
| `matvec_q6k_n4` | 1 | 0.53 | 0.2 % | 531.0 |
| `argmax_f32_part` | 1 | 0.53 | 0.2 % | 531.0 |

geist 128 rows per prefill pass, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 785 | 156.98 | 57.8 % |
| norm | 225 | 45.07 | 16.6 % |
| attention | 224 | 43.50 | 16.0 % |
| elementwise / copies | 121 | 24.87 | 9.2 % |
| embedding | 4 | 0.72 | 0.3 % |
| lm head / sampling | 2 | 0.56 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 672 | 134.16 | 49.4 % | 199.6 |
| `rmsnorm_add_norm_rows_simd` | 220 | 44.07 | 16.2 % | 200.3 |
| `silu_mul_rows` | 112 | 22.95 | 8.4 % | 204.9 |
| `matmul_q6k_mm_tc` | 112 | 22.54 | 8.3 % | 201.2 |
| `attention_flash_tc_d128` | 112 | 21.82 | 8.0 % | 194.8 |
| `rope_append_rows_f16` | 112 | 21.68 | 8.0 % | 193.5 |
| `copy_u32` | 5 | 1.10 | 0.4 % | 219.6 |
| `rmsnorm_rows_simd` | 5 | 1.00 | 0.4 % | 199.8 |
| `add_rows` | 4 | 0.82 | 0.3 % | 204.5 |
| `embed_lookup_scaled_rows` | 4 | 0.72 | 0.3 % | 179.8 |
| `matvec_q6k_n4` | 1 | 0.28 | 0.1 % | 279.7 |
| `argmax_f32_part` | 1 | 0.28 | 0.1 % | 279.7 |

geist 512 rows per prefill pass, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 197 | 150.56 | 57.5 % |
| norm | 57 | 44.10 | 16.8 % |
| attention | 56 | 41.44 | 15.8 % |
| elementwise / copies | 31 | 23.89 | 9.1 % |
| lm head / sampling | 2 | 1.36 | 0.5 % |
| embedding | 1 | 0.67 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 168 | 128.54 | 49.1 % | 765.1 |
| `rmsnorm_add_norm_rows_simd` | 55 | 42.75 | 16.3 % | 777.3 |
| `silu_mul_rows` | 28 | 21.84 | 8.3 % | 780.0 |
| `matmul_q6k_mm_tc` | 28 | 21.34 | 8.1 % | 762.1 |
| `rope_append_rows_f16` | 28 | 20.72 | 7.9 % | 740.0 |
| `attention_flash_tc_d128` | 28 | 20.72 | 7.9 % | 740.0 |
| `copy_u32` | 2 | 1.36 | 0.5 % | 681.8 |
| `rmsnorm_rows_simd` | 2 | 1.35 | 0.5 % | 677.0 |
| `add_rows` | 1 | 0.68 | 0.3 % | 681.8 |
| `matvec_q6k_n4` | 1 | 0.68 | 0.3 % | 681.8 |
| `argmax_f32_part` | 1 | 0.68 | 0.3 % | 681.8 |
| `argmax_f32_merge` | 1 | 0.68 | 0.3 % | 681.8 |

geist 512 rows per prefill pass, branch defaults, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 197 | 151.05 | 57.4 % |
| norm | 57 | 44.25 | 16.8 % |
| attention | 56 | 41.60 | 15.8 % |
| elementwise / copies | 31 | 23.98 | 9.1 % |
| lm head / sampling | 2 | 1.37 | 0.5 % |
| embedding | 1 | 0.68 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 168 | 128.99 | 49.1 % | 767.8 |
| `rmsnorm_add_norm_rows_simd` | 55 | 42.88 | 16.3 % | 779.6 |
| `silu_mul_rows` | 28 | 21.92 | 8.3 % | 782.8 |
| `matmul_q6k_mm_tc` | 28 | 21.37 | 8.1 % | 763.3 |
| `rope_append_rows_f16` | 28 | 20.80 | 7.9 % | 742.8 |
| `attention_flash_tc_d128` | 28 | 20.80 | 7.9 % | 742.8 |
| `copy_u32` | 2 | 1.37 | 0.5 % | 687.1 |
| `rmsnorm_rows_simd` | 2 | 1.37 | 0.5 % | 684.5 |
| `add_rows` | 1 | 0.69 | 0.3 % | 687.1 |
| `matvec_q6k_n4` | 1 | 0.69 | 0.3 % | 687.1 |
| `argmax_f32_part` | 1 | 0.69 | 0.3 % | 687.1 |
| `argmax_f32_merge` | 1 | 0.69 | 0.3 % | 687.1 |

### decode

| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | commit to GPU start, median us | waitUntilCompleted calls | ms in them |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| geist trace | 3 | 1899.46 | 1863.21 | 1863.21 | 36.25 | 33.8 | 254 | 127 | 33020 | 0% | 0 | 4718 | 127 | 1882.41 |
| mlx trace | 3 | 1660.53 | 1658.97 | 1659.04 | 1.56 | 263.3 | 1788 | 1534 | 61957 | 100% | 42224 | 9903 | 1 | 11.07 |
| geist library concurrent encoders | 3 | 1903.72 | 1867.66 | 1867.66 | 36.06 | 34.4 | 254 | 127 | 33020 | 4% | 1376 | 4734 | 127 | 1886.20 |
| geist first buffer after 8 | 3 | 1886.41 | 1864.34 | 1864.35 | 22.07 | 37.6 | 381 | 127 | 33020 | 0% | 0 | 119 | 127 | 1867.99 |
| geist first buffer after 16 | 3 | 1890.21 | 1865.30 | 1865.32 | 24.90 | 38.1 | 381 | 127 | 33020 | 0% | 0 | 621 | 127 | 1872.06 |
| geist first buffer after 32 | 3 | 1985.35 | 1956.63 | 1956.65 | 28.72 | 39.7 | 381 | 127 | 33020 | 0% | 0 | 1499 | 127 | 1963.75 |
| geist spin instead of wait | 3 | 2095.17 | 2060.40 | 2060.40 | 34.77 | 2097.9 | 254 | 127 | 33020 | 0% | 0 | 4968 | 127 | 2076.98 |
| geist branch defaults (concurrent, first buffer after 16) | 3 | 2053.44 | 2027.23 | 2027.43 | 26.22 | 37.7 | 381 | 127 | 33020 | 4% | 1371 | 657 | 127 | 2034.04 |
| geist branch defaults, spin | 3 | 2037.67 | 2017.46 | 2017.63 | 20.22 | 2049.2 | 381 | 127 | 33020 | 4% | 1371 | 649 | 127 | 2022.24 |
| geist 128 rows per prefill pass | 3 | 2114.71 | 2075.50 | 2075.50 | 39.21 | 35.5 | 254 | 127 | 33020 | 0% | 0 | 5078 | 127 | 2093.58 |
| geist 512 rows per prefill pass | 3 | 2050.44 | 2014.91 | 2014.91 | 35.53 | 31.5 | 254 | 127 | 33020 | 0% | 0 | 4979 | 127 | 2033.12 |
| geist 512 rows per prefill pass, branch defaults | 3 | 2051.46 | 2025.30 | 2025.50 | 26.16 | 38.2 | 381 | 127 | 33020 | 4% | 1371 | 657 | 127 | 2031.35 |

geist trace, decode, one token (median of the per-repeat medians, us): period 14951, gpu 14666, GPU end to wait returns 69, wait returns to first commit 106, first commit to GPU start 111, GPU idle between the token's buffers 0

geist library concurrent encoders, decode, one token (median of the per-repeat medians, us): period 14959, gpu 14667, GPU end to wait returns 69, wait returns to first commit 107, first commit to GPU start 112, GPU idle between the token's buffers 0

geist first buffer after 8, decode, one token (median of the per-repeat medians, us): period 14855, gpu 14679, GPU end to wait returns 69, wait returns to first commit 42, first commit to GPU start 61, GPU idle between the token's buffers 0

geist first buffer after 16, decode, one token (median of the per-repeat medians, us): period 14878, gpu 14681, GPU end to wait returns 69, wait returns to first commit 48, first commit to GPU start 82, GPU idle between the token's buffers 0

geist first buffer after 32, decode, one token (median of the per-repeat medians, us): period 15486, gpu 15278, GPU end to wait returns 70, wait returns to first commit 54, first commit to GPU start 85, GPU idle between the token's buffers 0

geist spin instead of wait, decode, one token (median of the per-repeat medians, us): period 16512, gpu 16246, GPU end to wait returns 62, wait returns to first commit 104, first commit to GPU start 101, GPU idle between the token's buffers 0

geist branch defaults (concurrent, first buffer after 16), decode, one token (median of the per-repeat medians, us): period 16167, gpu 15957, GPU end to wait returns 71, wait returns to first commit 48, first commit to GPU start 86, GPU idle between the token's buffers 0

geist branch defaults, spin, decode, one token (median of the per-repeat medians, us): period 16027, gpu 15858, GPU end to wait returns 59, wait returns to first commit 19, first commit to GPU start 87, GPU idle between the token's buffers 0

geist 128 rows per prefill pass, decode, one token (median of the per-repeat medians, us): period 16339, gpu 16043, GPU end to wait returns 71, wait returns to first commit 108, first commit to GPU start 100, GPU idle between the token's buffers 0

geist 512 rows per prefill pass, decode, one token (median of the per-repeat medians, us): period 16148, gpu 15871, GPU end to wait returns 71, wait returns to first commit 106, first commit to GPU start 98, GPU idle between the token's buffers 0

geist 512 rows per prefill pass, branch defaults, decode, one token (median of the per-repeat medians, us): period 16107, gpu 15898, GPU end to wait returns 71, wait returns to first commit 48, first commit to GPU start 85, GPU idle between the token's buffers 0

geist trace, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 803.74 | 43.1 % |
| attention | 10668 | 598.19 | 32.1 % |
| norm | 7239 | 404.95 | 21.7 % |
| elementwise / copies | 381 | 30.09 | 1.6 % |
| lm head / sampling | 254 | 20.06 | 1.1 % |
| embedding | 127 | 6.15 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 388.77 | 20.9 % | 55.7 |
| `matvec_q4k_n4` | 5334 | 293.28 | 15.7 % | 55.0 |
| `rope_append_rows_f16` | 3556 | 199.40 | 10.7 % | 56.1 |
| `attention_decv_d128_f16` | 3556 | 199.40 | 10.7 % | 56.1 |
| `attention_dec_combine` | 3556 | 199.40 | 10.7 % | 56.1 |
| `gate_up_silu_q4k_n4` | 3556 | 199.40 | 10.7 % | 56.1 |
| `matvec_q6k_n4` | 1905 | 115.54 | 6.2 % | 60.7 |
| `mv3_q4k_q4k_q6k` | 1778 | 105.51 | 5.7 % | 59.3 |
| `mv3_q4k_q4k_q4k` | 1778 | 90.01 | 4.8 % | 50.6 |
| `copy_u32` | 254 | 20.06 | 1.1 % | 79.0 |
| `rmsnorm_rows_simd` | 254 | 16.18 | 0.9 % | 63.7 |
| `add_rows` | 127 | 10.03 | 0.5 % | 79.0 |

mlx trace, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 25129 | 681.40 | 41.1 % |
| elementwise / copies | 17950 | 484.99 | 29.2 % |
| attention | 10715 | 287.72 | 17.3 % |
| norm | 7271 | 197.27 | 11.9 % |
| lm head / sampling | 384 | 4.60 | 0.3 % |
| embedding | 508 | 3.00 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmv_fast_float16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4` | 25129 | 681.40 | 41.1 % | 27.1 |
| `rmsfloat16` | 7271 | 197.27 | 11.9 % | 27.1 |
| `vv_Addfloat16` | 7144 | 193.62 | 11.7 % | 27.1 |
| `gg2_copyfloat16float16` | 7204 | 192.98 | 11.6 % | 26.8 |
| `rope_single_freqs_float16_` | 7143 | 191.61 | 11.5 % | 26.8 |
| `Cf2ISigmoidADf2IBroadcastACEf2IBroadcastCAFf2IMultiplyDEGf2IBroadcastF` | 3572 | 97.72 | 5.9 % | 27.4 |
| `sdpa_vector_float16_t_128_128_nomask_qnt_nc_nosinks` | 3572 | 96.11 | 5.8 % | 26.9 |
| `affine_dequantize_float16_t_gs_64_b_4` | 127 | 2.76 | 0.2 % | 21.7 |
| `looped_logsumexp_float16` | 128 | 1.53 | 0.1 % | 12.0 |
| `vsn_Subtractfloat16` | 128 | 1.53 | 0.1 % | 12.0 |
| `argmax_float16` | 128 | 1.53 | 0.1 % | 12.0 |
| `sn_copyfloat16float16` | 30 | 0.67 | 0.0 % | 22.5 |

geist library concurrent encoders, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 805.47 | 43.1 % |
| attention | 10668 | 599.48 | 32.1 % |
| norm | 7239 | 405.82 | 21.7 % |
| elementwise / copies | 381 | 30.13 | 1.6 % |
| lm head / sampling | 254 | 20.09 | 1.1 % |
| embedding | 127 | 6.17 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 389.61 | 20.9 % | 55.8 |
| `matvec_q4k_n4` | 5334 | 293.92 | 15.7 % | 55.1 |
| `rope_append_rows_f16` | 3556 | 199.83 | 10.7 % | 56.2 |
| `attention_decv_d128_f16` | 3556 | 199.83 | 10.7 % | 56.2 |
| `attention_dec_combine` | 3556 | 199.83 | 10.7 % | 56.2 |
| `gate_up_silu_q4k_n4` | 3556 | 199.83 | 10.7 % | 56.2 |
| `matvec_q6k_n4` | 1905 | 115.77 | 6.2 % | 60.8 |
| `mv3_q4k_q4k_q6k` | 1778 | 105.73 | 5.7 % | 59.5 |
| `mv3_q4k_q4k_q4k` | 1778 | 90.22 | 4.8 % | 50.7 |
| `copy_u32` | 254 | 20.09 | 1.1 % | 79.1 |
| `rmsnorm_rows_simd` | 254 | 16.21 | 0.9 % | 63.8 |
| `add_rows` | 127 | 10.04 | 0.5 % | 79.1 |

geist first buffer after 8, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 811.69 | 43.5 % |
| attention | 10668 | 594.62 | 31.9 % |
| norm | 7239 | 402.71 | 21.6 % |
| elementwise / copies | 381 | 31.87 | 1.7 % |
| lm head / sampling | 254 | 21.25 | 1.1 % |
| embedding | 127 | 2.18 | 0.1 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 389.91 | 20.9 % | 55.8 |
| `matvec_q4k_n4` | 5334 | 290.72 | 15.6 % | 54.5 |
| `gate_up_silu_q4k_n4` | 3556 | 202.33 | 10.9 % | 56.9 |
| `rope_append_rows_f16` | 3556 | 198.21 | 10.6 % | 55.7 |
| `attention_decv_d128_f16` | 3556 | 198.21 | 10.6 % | 55.7 |
| `attention_dec_combine` | 3556 | 198.21 | 10.6 % | 55.7 |
| `matvec_q6k_n4` | 1905 | 120.44 | 6.5 % | 63.2 |
| `mv3_q4k_q4k_q6k` | 1778 | 105.69 | 5.7 % | 59.4 |
| `mv3_q4k_q4k_q4k` | 1778 | 92.51 | 5.0 % | 52.0 |
| `copy_u32` | 254 | 21.25 | 1.1 % | 83.7 |
| `rmsnorm_rows_simd` | 254 | 12.81 | 0.7 % | 50.4 |
| `add_rows` | 127 | 10.62 | 0.6 % | 83.7 |

geist first buffer after 16, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 803.97 | 43.1 % |
| attention | 10668 | 592.45 | 31.8 % |
| norm | 7239 | 407.62 | 21.9 % |
| elementwise / copies | 381 | 33.83 | 1.8 % |
| lm head / sampling | 254 | 22.55 | 1.2 % |
| embedding | 127 | 4.87 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 391.47 | 21.0 % | 56.0 |
| `matvec_q4k_n4` | 5334 | 285.05 | 15.3 % | 53.4 |
| `gate_up_silu_q4k_n4` | 3556 | 198.86 | 10.7 % | 55.9 |
| `rope_append_rows_f16` | 3556 | 197.48 | 10.6 % | 55.5 |
| `attention_decv_d128_f16` | 3556 | 197.48 | 10.6 % | 55.5 |
| `attention_dec_combine` | 3556 | 197.48 | 10.6 % | 55.5 |
| `matvec_q6k_n4` | 1905 | 122.57 | 6.6 % | 64.3 |
| `mv3_q4k_q4k_q6k` | 1778 | 109.91 | 5.9 % | 61.8 |
| `mv3_q4k_q4k_q4k` | 1778 | 87.57 | 4.7 % | 49.3 |
| `copy_u32` | 254 | 22.55 | 1.2 % | 88.8 |
| `rmsnorm_rows_simd` | 254 | 16.15 | 0.9 % | 63.6 |
| `add_rows` | 127 | 11.28 | 0.6 % | 88.8 |

geist first buffer after 32, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 841.39 | 43.0 % |
| attention | 10668 | 612.95 | 31.3 % |
| norm | 7239 | 423.69 | 21.7 % |
| elementwise / copies | 381 | 43.57 | 2.2 % |
| lm head / sampling | 254 | 29.05 | 1.5 % |
| embedding | 127 | 5.98 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 403.18 | 20.6 % | 57.7 |
| `matvec_q4k_n4` | 5334 | 295.52 | 15.1 % | 55.4 |
| `gate_up_silu_q4k_n4` | 3556 | 204.85 | 10.5 % | 57.6 |
| `rope_append_rows_f16` | 3556 | 204.32 | 10.4 % | 57.5 |
| `attention_decv_d128_f16` | 3556 | 204.32 | 10.4 % | 57.5 |
| `attention_dec_combine` | 3556 | 204.32 | 10.4 % | 57.5 |
| `matvec_q6k_n4` | 1905 | 136.71 | 7.0 % | 71.8 |
| `mv3_q4k_q4k_q6k` | 1778 | 113.64 | 5.8 % | 63.9 |
| `mv3_q4k_q4k_q4k` | 1778 | 90.67 | 4.6 % | 51.0 |
| `copy_u32` | 254 | 29.05 | 1.5 % | 114.4 |
| `rmsnorm_rows_simd` | 254 | 20.51 | 1.0 % | 80.7 |
| `add_rows` | 127 | 14.52 | 0.7 % | 114.4 |

geist spin instead of wait, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 888.98 | 43.1 % |
| attention | 10668 | 661.61 | 32.1 % |
| norm | 7239 | 447.90 | 21.7 % |
| elementwise / copies | 381 | 33.04 | 1.6 % |
| lm head / sampling | 254 | 22.02 | 1.1 % |
| embedding | 127 | 6.83 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 430.06 | 20.9 % | 61.6 |
| `matvec_q4k_n4` | 5334 | 324.53 | 15.8 % | 60.8 |
| `rope_append_rows_f16` | 3556 | 220.54 | 10.7 % | 62.0 |
| `attention_decv_d128_f16` | 3556 | 220.54 | 10.7 % | 62.0 |
| `attention_dec_combine` | 3556 | 220.54 | 10.7 % | 62.0 |
| `gate_up_silu_q4k_n4` | 3556 | 220.54 | 10.7 % | 62.0 |
| `matvec_q6k_n4` | 1905 | 127.55 | 6.2 % | 67.0 |
| `mv3_q4k_q4k_q6k` | 1778 | 116.54 | 5.7 % | 65.5 |
| `mv3_q4k_q4k_q4k` | 1778 | 99.82 | 4.8 % | 56.1 |
| `copy_u32` | 254 | 22.02 | 1.1 % | 86.7 |
| `rmsnorm_rows_simd` | 254 | 17.84 | 0.9 % | 70.2 |
| `add_rows` | 127 | 11.01 | 0.5 % | 86.7 |

geist branch defaults (concurrent, first buffer after 16), decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 873.44 | 43.1 % |
| attention | 10668 | 643.73 | 31.8 % |
| norm | 7239 | 442.86 | 21.8 % |
| elementwise / copies | 381 | 36.85 | 1.8 % |
| lm head / sampling | 254 | 24.57 | 1.2 % |
| embedding | 127 | 5.36 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 425.22 | 21.0 % | 60.9 |
| `matvec_q4k_n4` | 5334 | 309.53 | 15.3 % | 58.0 |
| `gate_up_silu_q4k_n4` | 3556 | 216.00 | 10.7 % | 60.7 |
| `rope_append_rows_f16` | 3556 | 214.58 | 10.6 % | 60.3 |
| `attention_decv_d128_f16` | 3556 | 214.58 | 10.6 % | 60.3 |
| `attention_dec_combine` | 3556 | 214.58 | 10.6 % | 60.3 |
| `matvec_q6k_n4` | 1905 | 133.33 | 6.6 % | 70.0 |
| `mv3_q4k_q4k_q6k` | 1778 | 119.62 | 5.9 % | 67.3 |
| `mv3_q4k_q4k_q4k` | 1778 | 94.96 | 4.7 % | 53.4 |
| `copy_u32` | 254 | 24.57 | 1.2 % | 96.7 |
| `rmsnorm_rows_simd` | 254 | 17.64 | 0.9 % | 69.5 |
| `add_rows` | 127 | 12.28 | 0.6 % | 96.7 |

geist branch defaults, spin, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 869.27 | 43.1 % |
| attention | 10668 | 640.69 | 31.8 % |
| norm | 7239 | 440.75 | 21.8 % |
| elementwise / copies | 381 | 36.65 | 1.8 % |
| lm head / sampling | 254 | 24.43 | 1.2 % |
| embedding | 127 | 5.35 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 423.18 | 21.0 % | 60.6 |
| `matvec_q4k_n4` | 5334 | 308.09 | 15.3 % | 57.8 |
| `gate_up_silu_q4k_n4` | 3556 | 214.97 | 10.7 % | 60.5 |
| `rope_append_rows_f16` | 3556 | 213.56 | 10.6 % | 60.1 |
| `attention_decv_d128_f16` | 3556 | 213.56 | 10.6 % | 60.1 |
| `attention_dec_combine` | 3556 | 213.56 | 10.6 % | 60.1 |
| `matvec_q6k_n4` | 1905 | 132.66 | 6.6 % | 69.6 |
| `mv3_q4k_q4k_q6k` | 1778 | 119.04 | 5.9 % | 66.9 |
| `mv3_q4k_q4k_q4k` | 1778 | 94.53 | 4.7 % | 53.2 |
| `copy_u32` | 254 | 24.43 | 1.2 % | 96.2 |
| `rmsnorm_rows_simd` | 254 | 17.56 | 0.9 % | 69.1 |
| `add_rows` | 127 | 12.22 | 0.6 % | 96.2 |

geist 128 rows per prefill pass, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 895.33 | 43.1 % |
| attention | 10668 | 666.35 | 32.1 % |
| norm | 7239 | 451.09 | 21.7 % |
| elementwise / copies | 381 | 33.51 | 1.6 % |
| lm head / sampling | 254 | 22.34 | 1.1 % |
| embedding | 127 | 6.85 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 433.07 | 20.9 % | 62.0 |
| `matvec_q4k_n4` | 5334 | 326.70 | 15.7 % | 61.2 |
| `rope_append_rows_f16` | 3556 | 222.12 | 10.7 % | 62.5 |
| `attention_decv_d128_f16` | 3556 | 222.12 | 10.7 % | 62.5 |
| `attention_dec_combine` | 3556 | 222.12 | 10.7 % | 62.5 |
| `gate_up_silu_q4k_n4` | 3556 | 222.12 | 10.7 % | 62.5 |
| `matvec_q6k_n4` | 1905 | 128.70 | 6.2 % | 67.6 |
| `mv3_q4k_q4k_q6k` | 1778 | 117.53 | 5.7 % | 66.1 |
| `mv3_q4k_q4k_q4k` | 1778 | 100.27 | 4.8 % | 56.4 |
| `copy_u32` | 254 | 22.34 | 1.1 % | 88.0 |
| `rmsnorm_rows_simd` | 254 | 18.02 | 0.9 % | 71.0 |
| `add_rows` | 127 | 11.17 | 0.5 % | 88.0 |

geist 512 rows per prefill pass, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 869.21 | 43.1 % |
| attention | 10668 | 646.92 | 32.1 % |
| norm | 7239 | 437.94 | 21.7 % |
| elementwise / copies | 381 | 32.49 | 1.6 % |
| lm head / sampling | 254 | 21.66 | 1.1 % |
| embedding | 127 | 6.66 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 420.45 | 20.9 % | 60.2 |
| `matvec_q4k_n4` | 5334 | 317.20 | 15.7 % | 59.5 |
| `rope_append_rows_f16` | 3556 | 215.64 | 10.7 % | 60.6 |
| `attention_decv_d128_f16` | 3556 | 215.64 | 10.7 % | 60.6 |
| `attention_dec_combine` | 3556 | 215.64 | 10.7 % | 60.6 |
| `gate_up_silu_q4k_n4` | 3556 | 215.64 | 10.7 % | 60.6 |
| `matvec_q6k_n4` | 1905 | 124.91 | 6.2 % | 65.6 |
| `mv3_q4k_q4k_q6k` | 1778 | 114.08 | 5.7 % | 64.2 |
| `mv3_q4k_q4k_q4k` | 1778 | 97.39 | 4.8 % | 54.8 |
| `copy_u32` | 254 | 21.66 | 1.1 % | 85.3 |
| `rmsnorm_rows_simd` | 254 | 17.49 | 0.9 % | 68.9 |
| `add_rows` | 127 | 10.83 | 0.5 % | 85.3 |

geist 512 rows per prefill pass, branch defaults, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 14351 | 872.62 | 43.1 % |
| attention | 10668 | 643.17 | 31.8 % |
| norm | 7239 | 442.43 | 21.8 % |
| elementwise / copies | 381 | 36.74 | 1.8 % |
| lm head / sampling | 254 | 24.49 | 1.2 % |
| embedding | 127 | 5.37 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 424.81 | 21.0 % | 60.8 |
| `matvec_q4k_n4` | 5334 | 309.33 | 15.3 % | 58.0 |
| `gate_up_silu_q4k_n4` | 3556 | 215.80 | 10.7 % | 60.7 |
| `rope_append_rows_f16` | 3556 | 214.39 | 10.6 % | 60.3 |
| `attention_decv_d128_f16` | 3556 | 214.39 | 10.6 % | 60.3 |
| `attention_dec_combine` | 3556 | 214.39 | 10.6 % | 60.3 |
| `matvec_q6k_n4` | 1905 | 133.11 | 6.6 % | 69.9 |
| `mv3_q4k_q4k_q6k` | 1778 | 119.45 | 5.9 % | 67.2 |
| `mv3_q4k_q4k_q4k` | 1778 | 94.94 | 4.7 % | 53.4 |
| `copy_u32` | 254 | 24.49 | 1.2 % | 96.4 |
| `rmsnorm_rows_simd` | 254 | 17.62 | 0.9 % | 69.4 |
| `add_rows` | 127 | 12.25 | 0.6 % | 96.4 |


