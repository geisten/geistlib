## Metal profile, geist against mlx-lm (#590)

### Qwen3.5 0.8B, 8-bit

geist and mlx-lm tokens equal: True

| engine | cell | prefill tok/s | decode tok/s | tokens against the engine's plain cell |
| :-- | :-- | --: | --: | :-- |
| geist | plain | 7201 (7166–7202) | 152.9 (152.6–153.0) | same |
| mlx | plain | 7278 (7207–7442) | 160.1 (160.0–160.1) | same |
| geist | trace | 7189 (7160–7189) | 152.9 (152.7–152.9) | same |
| mlx | trace | 7265 (7251–7311) | 159.9 (158.0–160.0) | same |
| geist | 1op | 7301 (7296–7302) | 147.5 (145.2–148.3) | same |
| mlx | 1op | 8035 (8033–8060) | 139.7 (139.5–139.7) | same |
| geist | concurrent, barrier before every dispatch | 7203 (7175–7206) | 151.9 (151.7–152.1) | same |
| geist | concurrent, no barrier (wrong output) | 8376 (8363–8431) | 171.4 (171.1–171.4) | DIFFERENT |
| geist | concurrent, barrier where dependent by bind offset | 7442 (7413–7450) | 155.7 (155.4–155.8) | same |
| geist | concurrent, barrier where dependent by buffer | 7290 (7273–7293) | 155.1 (155.0–155.2) | same |
| geist | library concurrent encoders | 7433 (7401–7439) | 155.2 (154.9–155.2) | same |
| geist | library concurrent encoders, untraced | 7418 (7389–7421) | 155.0 (154.9–155.2) | same |
| geist | rotation 48/16 | 7198 (7170–7203) | 154.0 (153.9–154.0) | same |
| geist | rotation 192/64 | 7157 (7148–7188) | 152.8 (152.6–152.9) | same |
| geist | rotation off | 7245 (7202–7247) | 151.0 (150.8–151.1) | same |
| geist | tensor API off | 3462 (3458–3462) | 152.8 (152.8–152.9) | same |
| geist | plain again | 7202 (7176–7204) | 152.9 (152.7–153.0) | same |
| mlx | plain again | 7355 (7349–7489) | 160.0 (160.0–160.1) | same |

GPU state over a 768-token decode (macmon, samples with the GPU over 50 % active): {"geist": {"samples": 103, "busy_samples": 93, "gpu_freq_mhz": 1679, "gpu_active": 0.9681875109672546, "gpu_w": 5.7940473556518555, "cpu_w": 0.6037136912345886, "decode_tps": 151.24353503524026}, "mlx": {"samples": 149, "busy_samples": 131, "gpu_freq_mhz": 1295, "gpu_active": 0.7464534640312195, "gpu_w": 3.5586979389190674, "cpu_w": 2.8216960430145264, "decode_tps": 160.6560728636997}}

Serial against concurrent encoders (GEIST_METAL_CONCURRENT=0/1): process start to first token, ms: {"serial": {"median": 55.83799999999428, "min": 54.91583299999547, "max": 59.15124999998511, "n": 10}, "concurrent": {"median": 58.18906249999145, "min": 55.32799999997451, "max": 59.46391599999856, "n": 10}}

bench_nll logits identical across the arms: True {"decode": {"serial": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}, "concurrent": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}, "origin/main": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}}, "prefill": {"serial": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}, "concurrent": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}, "origin/main": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}}}


### prefill

| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | commit to GPU start, median us | waitUntilCompleted calls | ms in them |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| geist trace | 3 | 71.32 | 70.98 | 70.98 | 0.34 | 4.2 | 61 | 0 | 981 | 0% | 0 | 17121 | 45 | 69.98 |
| mlx trace | 3 | 70.37 | 56.65 | 56.79 | 13.72 | 46.0 | 46 | 2 | 1718 | 100% | 1038 | 2357 | 0 | 0.00 |
| geist 1op | 3 | 70.14 | 69.84 | 69.90 | 0.31 | 19.1 | 764 | 0 | 981 | 0% | 0 | 1460 | 601 | 66.79 |
| mlx 1op | 3 | 63.66 | 58.97 | 68.67 | 4.69 | 32.2 | 1016 | 205 | 2076 | 100% | 611 | 59 | 0 | 0.00 |
| geist library concurrent encoders | 3 | 68.97 | 68.64 | 68.64 | 0.33 | 4.6 | 61 | 0 | 981 | 100% | 822 | 16515 | 45 | 67.26 |

geist trace, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 27.06 | 38.1 % |
| DeltaNet mixer | 252 | 19.84 | 28.0 % |
| elementwise / copies | 183 | 12.09 | 17.0 % |
| norm | 121 | 7.97 | 11.2 % |
| attention | 48 | 2.71 | 3.8 % |
| lm head / sampling | 2 | 1.18 | 1.7 % |
| embedding | 2 | 0.12 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 26.48 | 37.3 % | 71.2 |
| `rmsnorm_rows_simd` | 121 | 7.97 | 11.2 % | 65.9 |
| `add_rows` | 96 | 6.42 | 9.0 % | 66.9 |
| `silu_mul_rows` | 48 | 3.46 | 4.9 % | 72.0 |
| `dnf_cst_copy` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_conv` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_norm` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_roll` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_chunk` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_state` | 36 | 2.83 | 4.0 % | 78.7 |
| `dnf_out` | 36 | 2.83 | 4.0 % | 78.7 |
| `rope_rows` | 24 | 1.40 | 2.0 % | 58.3 |

mlx trace, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| elementwise / copies | 720 | 24.72 | 43.5 % |
| quantized matmul | 499 | 16.64 | 29.3 % |
| norm | 279 | 8.43 | 14.9 % |
| DeltaNet mixer | 135 | 3.96 | 7.0 % |
| attention | 70 | 2.87 | 5.0 % |
| embedding | 12 | 0.11 | 0.2 % |
| lm head / sampling | 3 | 0.06 | 0.1 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsbfloat16` | 279 | 8.43 | 14.9 % | 30.2 |
| `affine_qmm_t_nax_bfloat16_t_gs_64_b_8_bm64_bn64_bk64_wm2_wn2_alN_true_` | 133 | 8.39 | 14.8 % | 63.1 |
| `g2_copybfloat16bfloat16` | 80 | 4.13 | 7.3 % | 51.6 |
| `vvn_Addbfloat16` | 46 | 2.89 | 5.1 % | 62.8 |
| `steel_gemm_splitk_accum_bfloat16_float32` | 48 | 2.79 | 4.9 % | 58.2 |
| `gg1_copybfloat16bfloat16` | 90 | 2.67 | 4.7 % | 29.6 |
| `affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0_pr_0_hgs_0_r_4` | 270 | 2.66 | 4.7 % | 9.8 |
| `vsn_Multiplybfloat16` | 36 | 2.27 | 4.0 % | 62.9 |
| `affine_qmm_t_splitk_bfloat16_t_gs_64_b_8_alN_false` | 36 | 2.21 | 3.9 % | 61.4 |
| `CV2ISigmoidADV2IBroadcastACEV2IBroadcastCAFV2IMultiplyDEGV2IBroadcastF` | 57 | 1.83 | 3.2 % | 32.0 |
| `v_Sigmoidbfloat16bfloat16` | 53 | 1.41 | 2.5 % | 26.7 |
| `Cf4IAsTypeADf4ISigmoidCEf4IBroadcastCDFf4IBroadcastDCGf4IMultiplyEFHf4` | 45 | 1.39 | 2.5 % | 31.0 |

geist 1op, prefill, per repeat (one op per buffer; a buffer of several kernels split by dispatch count):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 45.20 | 64.7 % |
| DeltaNet mixer | 252 | 15.28 | 21.9 % |
| elementwise / copies | 183 | 5.84 | 8.4 % |
| attention | 48 | 1.84 | 2.6 % |
| norm | 121 | 1.70 | 2.4 % |
| embedding | 2 | 0.02 | 0.0 % |
| lm head / sampling | 2 | 0.01 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 43.44 | 62.2 % | 116.8 |
| `silu_mul_rows` | 48 | 2.62 | 3.8 % | 54.6 |
| `dnf_cst_copy` | 36 | 2.18 | 3.1 % | 60.6 |
| `dnf_conv` | 36 | 2.18 | 3.1 % | 60.6 |
| `dnf_norm` | 36 | 2.18 | 3.1 % | 60.6 |
| `dnf_roll` | 36 | 2.18 | 3.1 % | 60.6 |
| `dnf_chunk` | 36 | 2.18 | 3.1 % | 60.6 |
| `dnf_state` | 36 | 2.18 | 3.1 % | 60.6 |
| `dnf_out` | 36 | 2.18 | 3.1 % | 60.6 |
| `add_rows` | 96 | 2.04 | 2.9 % | 21.2 |
| `matvec_q80_n4` | 1 | 1.76 | 2.5 % | 1759.0 |
| `rmsnorm_rows_simd` | 121 | 1.70 | 2.4 % | 14.0 |

mlx 1op, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 603 | 34.11 | 49.7 % |
| elementwise / copies | 872 | 20.88 | 30.4 % |
| norm | 342 | 7.41 | 10.8 % |
| DeltaNet mixer | 162 | 4.23 | 6.2 % |
| attention | 82 | 1.94 | 2.8 % |
| lm head / sampling | 3 | 0.06 | 0.1 % |
| embedding | 12 | 0.03 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmm_t_nax_bfloat16_t_gs_64_b_8_bm64_bn64_bk64_wm2_wn2_alN_true_` | 133 | 20.21 | 29.4 % | 152.0 |
| `rmsbfloat16` | 342 | 7.41 | 10.8 % | 21.7 |
| `affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0_pr_0_hgs_0_r_4` | 374 | 6.66 | 9.7 % | 17.8 |
| `g2_copybfloat16bfloat16` | 88 | 3.95 | 5.8 % | 44.9 |
| `CV2ISigmoidADV2IBroadcastACEV2IBroadcastCAFV2IMultiplyDEGV2IBroadcastF` | 71 | 3.87 | 5.6 % | 54.4 |
| `steel_gemm_splitk_accum_bfloat16_float32` | 48 | 3.62 | 5.3 % | 75.4 |
| `custom_kernel_gated_delta_step_packed_btree__bfloat16_t_float_128_128_` | 54 | 3.01 | 4.4 % | 55.8 |
| `s_copybfloat16bfloat16` | 18 | 2.70 | 3.9 % | 150.2 |
| `vsn_Multiplybfloat16` | 36 | 2.65 | 3.9 % | 73.6 |
| `affine_qmm_t_splitk_bfloat16_t_gs_64_b_8_alN_false` | 36 | 2.61 | 3.8 % | 72.4 |
| `Cf4IAsTypeADf4ISigmoidCEf4IBroadcastCDFf4IBroadcastDCGf4IMultiplyEFHf4` | 54 | 1.52 | 2.2 % | 28.2 |
| `vvn_Addbfloat16` | 46 | 1.49 | 2.2 % | 32.3 |

geist library concurrent encoders, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 373 | 26.20 | 38.2 % |
| DeltaNet mixer | 252 | 18.94 | 27.6 % |
| elementwise / copies | 183 | 11.77 | 17.1 % |
| norm | 121 | 7.76 | 11.3 % |
| attention | 48 | 2.67 | 3.9 % |
| lm head / sampling | 2 | 1.18 | 1.7 % |
| embedding | 2 | 0.11 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q80_mm_tc` | 372 | 25.61 | 37.3 % | 68.8 |
| `rmsnorm_rows_simd` | 121 | 7.76 | 11.3 % | 64.2 |
| `add_rows` | 96 | 6.21 | 9.0 % | 64.7 |
| `silu_mul_rows` | 48 | 3.37 | 4.9 % | 70.2 |
| `dnf_cst_copy` | 36 | 2.71 | 3.9 % | 75.2 |
| `dnf_conv` | 36 | 2.71 | 3.9 % | 75.2 |
| `dnf_norm` | 36 | 2.71 | 3.9 % | 75.2 |
| `dnf_roll` | 36 | 2.71 | 3.9 % | 75.2 |
| `dnf_chunk` | 36 | 2.71 | 3.9 % | 75.2 |
| `dnf_state` | 36 | 2.71 | 3.9 % | 75.2 |
| `dnf_out` | 36 | 2.71 | 3.9 % | 75.2 |
| `rope_rows` | 24 | 1.39 | 2.0 % | 57.8 |

### decode

| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | commit to GPU start, median us | waitUntilCompleted calls | ms in them |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| geist trace | 3 | 837.64 | 802.69 | 802.69 | 34.95 | 36.2 | 254 | 127 | 45847 | 0% | 0 | 1317 | 127 | 818.69 |
| mlx trace | 3 | 802.07 | 798.31 | 798.33 | 3.77 | 670.5 | 2169 | 730 | 78466 | 100% | 50257 | 3059 | 1 | 4.60 |
| geist 1op | 3 | 870.73 | 819.56 | 949.91 | 51.17 | 896.4 | 42672 | 127 | 45847 | 0% | 0 | 169 | 17977 | 663.68 |
| mlx 1op | 3 | 911.31 | 789.49 | 849.34 | 121.82 | 1096.5 | 38864 | 621 | 78108 | 100% | 23115 | 31 | 1 | 1.83 |
| geist library concurrent encoders | 3 | 825.28 | 801.10 | 801.10 | 24.18 | 46.6 | 254 | 127 | 45847 | 100% | 38735 | 1352 | 127 | 798.66 |

geist trace, decode, one token (median of the per-repeat medians, us): period 6594, gpu 6319, GPU end to wait returns 72, wait returns to first commit 93, first commit to GPU start 113, GPU idle between the token's buffers 0

geist library concurrent encoders, decode, one token (median of the per-repeat medians, us): period 6492, gpu 6213, GPU end to wait returns 74, wait returns to first commit 118, first commit to GPU start 96, GPU idle between the token's buffers 0

geist trace, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 360.80 | 44.9 % |
| elementwise / copies | 8636 | 152.64 | 19.0 % |
| norm | 7747 | 134.94 | 16.8 % |
| DeltaNet mixer | 4572 | 78.98 | 9.8 % |
| attention | 3810 | 67.89 | 8.5 % |
| lm head / sampling | 254 | 5.77 | 0.7 % |
| embedding | 127 | 1.64 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 307.73 | 38.3 % | 17.4 |
| `rmsnorm_rows_simd` | 7747 | 134.94 | 16.8 % | 17.4 |
| `add_rows` | 6096 | 106.14 | 13.2 % | 17.4 |
| `gate_up_silu_q80_n4` | 3048 | 53.07 | 6.6 % | 17.4 |
| `dn_dec_qk` | 2286 | 39.49 | 4.9 % | 17.3 |
| `dn_dec_v` | 2286 | 39.49 | 4.9 % | 17.3 |
| `rope_rows` | 1524 | 27.16 | 3.4 % | 17.8 |
| `qgate_split` | 762 | 13.58 | 1.7 % | 17.8 |
| `scale_rows` | 762 | 13.58 | 1.7 % | 17.8 |
| `kv_append_rows_f16` | 762 | 13.58 | 1.7 % | 17.8 |
| `attention_decq_d256_f16` | 762 | 13.58 | 1.7 % | 17.8 |
| `attention_dec_combine` | 762 | 13.58 | 1.7 % | 17.8 |

mlx trace, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| elementwise / copies | 29870 | 299.93 | 37.6 % |
| quantized matmul | 23853 | 252.57 | 31.6 % |
| norm | 14668 | 151.28 | 18.9 % |
| DeltaNet mixer | 6885 | 48.79 | 6.1 % |
| attention | 2298 | 36.46 | 4.6 % |
| lm head / sampling | 384 | 8.23 | 1.0 % |
| embedding | 508 | 1.04 | 0.1 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0_pr_0_hgs_0_r_4` | 23853 | 252.57 | 31.6 % | 10.6 |
| `rmsbfloat16` | 14668 | 151.28 | 18.9 % | 10.3 |
| `vv_Addbfloat16` | 6123 | 71.11 | 8.9 % | 11.6 |
| `CV2ISigmoidADV2IBroadcastACEV2IBroadcastCAFV2IMultiplyDEGV2IBroadcastF` | 3062 | 35.82 | 4.5 % | 11.7 |
| `gg1_copybfloat16bfloat16` | 4590 | 32.50 | 4.1 % | 7.1 |
| `vs_Multiplybfloat16` | 4590 | 32.21 | 4.0 % | 7.0 |
| `v_Sigmoidbfloat16bfloat16` | 3061 | 28.08 | 3.5 % | 9.2 |
| `gg2_copybfloat16bfloat16` | 1548 | 24.56 | 3.1 % | 15.9 |
| `rope_single_bfloat16_` | 1532 | 24.37 | 3.1 % | 15.9 |
| `g2_copybfloat16bfloat16` | 1532 | 24.22 | 3.0 % | 15.8 |
| `Cf4IAsTypeADf4ISigmoidCEf4IBroadcastCDFf4IBroadcastDCGf4IMultiplyEFHf4` | 2295 | 23.23 | 2.9 % | 10.1 |
| `custom_kernel_gated_delta_step_packed_btree__bfloat16_t_float_128_128_` | 2295 | 16.55 | 2.1 % | 7.2 |

geist 1op, decode, per repeat (one op per buffer; a buffer of several kernels split by dispatch count):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 858.27 | 90.4 % |
| DeltaNet mixer | 4572 | 48.49 | 5.1 % |
| norm | 7747 | 15.58 | 1.6 % |
| attention | 3810 | 13.25 | 1.4 % |
| elementwise / copies | 8636 | 12.46 | 1.3 % |
| lm head / sampling | 254 | 1.51 | 0.2 % |
| embedding | 127 | 0.34 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 691.69 | 72.8 % | 39.2 |
| `gate_up_silu_q80_n4` | 3048 | 166.58 | 17.5 % | 54.7 |
| `dn_dec_qk` | 2286 | 24.25 | 2.6 % | 10.6 |
| `dn_dec_v` | 2286 | 24.25 | 2.6 % | 10.6 |
| `rmsnorm_rows_simd` | 7747 | 15.58 | 1.6 % | 2.0 |
| `add_rows` | 6096 | 8.40 | 0.9 % | 1.4 |
| `attention_decq_d256_f16` | 762 | 4.74 | 0.5 % | 6.2 |
| `attention_dec_combine` | 762 | 4.74 | 0.5 % | 6.2 |
| `rope_rows` | 1524 | 2.64 | 0.3 % | 1.7 |
| `qgate_split` | 762 | 1.56 | 0.2 % | 2.1 |
| `kv_append_rows_f16` | 762 | 1.13 | 0.1 % | 1.5 |
| `sigmoid_mul_rows` | 762 | 1.12 | 0.1 % | 1.5 |

mlx 1op, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 23749 | 521.02 | 61.3 % |
| elementwise / copies | 29718 | 155.84 | 18.3 % |
| norm | 14605 | 93.59 | 11.0 % |
| DeltaNet mixer | 6858 | 45.30 | 5.3 % |
| attention | 2286 | 23.95 | 2.8 % |
| lm head / sampling | 384 | 8.90 | 1.0 % |
| embedding | 508 | 0.74 | 0.1 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0_pr_0_hgs_0_r_4` | 23749 | 521.02 | 61.3 % | 21.9 |
| `rmsbfloat16` | 14605 | 93.59 | 11.0 % | 6.4 |
| `vv_Addbfloat16` | 6096 | 56.56 | 6.7 % | 9.3 |
| `CV2ISigmoidADV2IBroadcastACEV2IBroadcastCAFV2IMultiplyDEGV2IBroadcastF` | 3048 | 42.33 | 5.0 % | 13.9 |
| `custom_kernel_gated_delta_step_packed_btree__bfloat16_t_float_128_128_` | 2286 | 33.37 | 3.9 % | 14.6 |
| `gg1_copybfloat16bfloat16` | 4572 | 25.42 | 3.0 % | 5.6 |
| `rope_single_bfloat16_` | 1524 | 17.31 | 2.0 % | 11.4 |
| `v_Sigmoidbfloat16bfloat16` | 3048 | 9.53 | 1.1 % | 3.1 |
| `Ef4IExpAFf4INegativeEGV2IBroadcastBCHV2IBroadcastCBIV2IAddGHJV2IBroadc` | 2286 | 7.78 | 0.9 % | 3.4 |
| `sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks` | 762 | 6.63 | 0.8 % | 8.7 |
| `vv_Multiplybfloat16` | 762 | 6.53 | 0.8 % | 8.6 |
| `argmax_bfloat16` | 128 | 5.87 | 0.7 % | 45.9 |

geist library concurrent encoders, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 20701 | 354.72 | 44.3 % |
| elementwise / copies | 8636 | 150.11 | 18.7 % |
| norm | 7747 | 132.66 | 16.6 % |
| DeltaNet mixer | 4572 | 77.64 | 9.7 % |
| attention | 3810 | 66.78 | 8.3 % |
| lm head / sampling | 254 | 5.70 | 0.7 % |
| embedding | 127 | 1.60 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matvec_q80_n4` | 17653 | 302.54 | 37.8 % | 17.1 |
| `rmsnorm_rows_simd` | 7747 | 132.66 | 16.6 % | 17.1 |
| `add_rows` | 6096 | 104.35 | 13.0 % | 17.1 |
| `gate_up_silu_q80_n4` | 3048 | 52.17 | 6.5 % | 17.1 |
| `dn_dec_qk` | 2286 | 38.82 | 4.8 % | 17.0 |
| `dn_dec_v` | 2286 | 38.82 | 4.8 % | 17.0 |
| `rope_rows` | 1524 | 26.71 | 3.3 % | 17.5 |
| `qgate_split` | 762 | 13.36 | 1.7 % | 17.5 |
| `scale_rows` | 762 | 13.36 | 1.7 % | 17.5 |
| `kv_append_rows_f16` | 762 | 13.36 | 1.7 % | 17.5 |
| `attention_decq_d256_f16` | 762 | 13.36 | 1.7 % | 17.5 |
| `attention_dec_combine` | 762 | 13.36 | 1.7 % | 17.5 |


### Llama 3.2 3B Instruct, 4-bit

geist and mlx-lm tokens equal: False

| engine | cell | prefill tok/s | decode tok/s | tokens against the engine's plain cell |
| :-- | :-- | --: | --: | :-- |
| geist | plain | 2056 (2056–2058) | 67.3 (67.3–67.3) | same |
| mlx | plain | 2634 (2633–2635) | 77.2 (77.1–77.2) | same |
| geist | trace | 2051 (2048–2051) | 67.3 (67.1–67.3) | same |
| mlx | trace | 2640 (2639–2640) | 77.1 (77.1–77.2) | same |
| geist | 1op | 2045 (2044–2046) | 66.7 (66.6–66.7) | same |
| mlx | 1op | 2571 (2569–2571) | 75.4 (75.4–75.5) | same |
| geist | concurrent, barrier before every dispatch | 2057 (2054–2057) | 67.0 (66.9–67.0) | same |
| geist | concurrent, no barrier (wrong output) | 2150 (2144–2151) | 71.2 (71.0–71.2) | DIFFERENT |
| geist | concurrent, barrier where dependent by bind offset | 2083 (2083–2084) | 67.0 (67.0–67.0) | same |
| geist | concurrent, barrier where dependent by buffer | 2056 (2055–2058) | 67.0 (67.0–67.0) | same |
| geist | library concurrent encoders | 2056 (2055–2056) | 66.9 (66.8–67.0) | same |
| geist | library concurrent encoders, untraced | 2052 (2049–2055) | 67.0 (67.0–67.0) | same |
| geist | rotation 48/16 | 2056 (2056–2059) | 67.5 (67.5–67.5) | same |
| geist | rotation 192/64 | 2055 (2054–2055) | 67.3 (67.3–67.3) | same |
| geist | rotation off | 2058 (2057–2060) | 67.1 (67.1–67.1) | same |
| geist | tensor API off | 717 (716–717) | 67.3 (67.1–67.3) | same |
| geist | plain again | 2057 (2056–2058) | 67.3 (67.3–67.3) | same |
| mlx | plain again | 2642 (2641–2642) | 77.1 (76.8–77.1) | same |

GPU state over a 768-token decode (macmon, samples with the GPU over 50 % active): {"geist": {"samples": 228, "busy_samples": 218, "gpu_freq_mhz": 1551.0, "gpu_active": 0.8940917551517487, "gpu_w": 10.511359691619873, "cpu_w": 0.37323926389217377, "decode_tps": 65.72861355448555}, "mlx": {"samples": 299, "busy_samples": 283, "gpu_freq_mhz": 1285, "gpu_active": 0.7407764196395874, "gpu_w": 5.553389549255371, "cpu_w": 1.3943690061569214, "decode_tps": 75.50733679618529}}

Serial against concurrent encoders (GEIST_METAL_CONCURRENT=0/1): process start to first token, ms: {"serial": {"median": 91.87489599997889, "min": 90.18537499997592, "max": 100.5762089999962, "n": 10}, "concurrent": {"median": 93.01606249997008, "min": 90.24570799999765, "max": 101.11745799997607, "n": 10}}

bench_nll logits identical across the arms: True {"decode": {"serial": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}, "concurrent": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}, "origin/main": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}}, "prefill": {"serial": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}, "concurrent": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}, "origin/main": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}}}


### prefill

| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | commit to GPU start, median us | waitUntilCompleted calls | ms in them |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| geist trace | 3 | 249.77 | 249.55 | 249.55 | 0.23 | 3.0 | 43 | 1 | 683 | 0% | 0 | 86337 | 27 | 248.65 |
| mlx trace | 3 | 193.96 | 190.44 | 191.15 | 3.52 | 19.6 | 50 | 24 | 1301 | 100% | 797 | 12293 | 0 | 0.00 |
| geist 1op | 3 | 250.36 | 250.13 | 250.18 | 0.23 | 25.3 | 682 | 1 | 683 | 0% | 0 | 5243 | 666 | 246.23 |
| mlx 1op | 3 | 199.21 | 197.43 | 216.33 | 1.78 | 36.5 | 815 | 419 | 1653 | 100% | 445 | 414 | 0 | 0.00 |
| geist library concurrent encoders | 3 | 249.09 | 248.90 | 248.92 | 0.19 | 3.3 | 43 | 1 | 683 | 100% | 640 | 86249 | 27 | 247.70 |

geist trace, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 143.97 | 57.7 % |
| norm | 113 | 40.86 | 16.4 % |
| attention | 112 | 39.44 | 15.8 % |
| elementwise / copies | 61 | 23.60 | 9.5 % |
| lm head / sampling | 2 | 1.00 | 0.4 % |
| embedding | 2 | 0.69 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 122.69 | 49.2 % | 365.1 |
| `rmsnorm_add_norm_rows_simd` | 110 | 39.67 | 15.9 % | 360.7 |
| `silu_mul_rows` | 56 | 21.36 | 8.6 % | 381.4 |
| `matmul_q6k_mm_tc` | 56 | 20.78 | 8.3 % | 371.1 |
| `rope_append_rows_f16` | 56 | 19.72 | 7.9 % | 352.1 |
| `attention_flash_tc_d128` | 56 | 19.72 | 7.9 % | 352.1 |
| `copy_u32` | 3 | 1.37 | 0.5 % | 455.9 |
| `rmsnorm_rows_simd` | 3 | 1.19 | 0.5 % | 395.1 |
| `add_rows` | 2 | 0.87 | 0.3 % | 433.7 |
| `embed_lookup_scaled_rows` | 2 | 0.69 | 0.3 % | 342.6 |
| `matvec_q6k_n4` | 1 | 0.50 | 0.2 % | 500.2 |
| `argmax_f32_part` | 1 | 0.50 | 0.2 % | 500.2 |

mlx trace, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 475 | 70.29 | 36.8 % |
| elementwise / copies | 471 | 70.12 | 36.7 % |
| attention | 203 | 30.05 | 15.7 % |
| norm | 137 | 20.29 | 10.6 % |
| embedding | 12 | 0.37 | 0.2 % |
| lm head / sampling | 3 | 0.04 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmm_t_nax_float16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_b` | 191 | 62.92 | 32.9 % | 329.4 |
| `rmsfloat16` | 137 | 20.29 | 10.6 % | 148.1 |
| `sn_copyfloat16float16` | 82 | 18.84 | 9.9 % | 229.8 |
| `rope_freqs_float16_` | 55 | 18.04 | 9.4 % | 327.9 |
| `vvn_Addfloat16` | 54 | 17.82 | 9.3 % | 329.9 |
| `gg2_copyfloat16float16` | 160 | 12.24 | 6.4 % | 76.5 |
| `Cf2ISigmoidADf2IBroadcastACEf2IBroadcastCAFf2IMultiplyDEGf2IBroadcastF` | 67 | 9.94 | 5.2 % | 148.4 |
| `gg3_copyfloat16float16` | 28 | 9.16 | 4.8 % | 327.3 |
| `steel_attention_float16_bq64_bk32_bd128_bv128_wm4_wn1_maskfloat16_alig` | 27 | 8.91 | 4.7 % | 329.9 |
| `affine_qmv_fast_float16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4` | 284 | 7.37 | 3.9 % | 26.0 |
| `vv_Addfloat16` | 80 | 2.11 | 1.1 % | 26.4 |
| `rope_single_freqs_float16_` | 81 | 2.06 | 1.1 % | 25.4 |

geist 1op, prefill, per repeat (one op per buffer; a buffer of several kernels split by dispatch count):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 218.08 | 87.2 % |
| attention | 112 | 14.77 | 5.9 % |
| norm | 113 | 9.09 | 3.6 % |
| elementwise / copies | 61 | 8.18 | 3.3 % |
| embedding | 2 | 0.06 | 0.0 % |
| lm head / sampling | 2 | 0.01 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 178.91 | 71.5 % | 532.5 |
| `matmul_q6k_mm_tc` | 56 | 36.92 | 14.8 % | 659.3 |
| `attention_flash_tc_d128` | 56 | 10.87 | 4.3 % | 194.1 |
| `rmsnorm_add_norm_rows_simd` | 110 | 9.02 | 3.6 % | 82.0 |
| `silu_mul_rows` | 56 | 7.98 | 3.2 % | 142.6 |
| `rope_append_rows_f16` | 56 | 3.90 | 1.6 % | 69.6 |
| `matvec_q6k_n4` | 1 | 2.25 | 0.9 % | 2246.9 |
| `add_rows` | 2 | 0.12 | 0.0 % | 59.9 |
| `copy_u32` | 3 | 0.07 | 0.0 % | 25.0 |
| `rmsnorm_rows_simd` | 3 | 0.07 | 0.0 % | 24.5 |
| `embed_lookup_scaled_rows` | 2 | 0.06 | 0.0 % | 31.3 |
| `argmax_f32_part` | 1 | 0.00 | 0.0 % | 3.7 |

mlx 1op, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 582 | 133.65 | 61.8 % |
| elementwise / copies | 639 | 55.58 | 25.7 % |
| norm | 168 | 15.19 | 7.0 % |
| attention | 250 | 11.81 | 5.5 % |
| embedding | 12 | 0.05 | 0.0 % |
| lm head / sampling | 3 | 0.04 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmm_t_nax_float16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_b` | 191 | 118.49 | 54.8 % | 620.4 |
| `Cf2ISigmoidADf2IBroadcastACEf2IBroadcastCAFf2IMultiplyDEGf2IBroadcastF` | 82 | 24.19 | 11.2 % | 295.1 |
| `vvn_Addfloat16` | 54 | 16.00 | 7.4 % | 296.3 |
| `rmsfloat16` | 168 | 15.19 | 7.0 % | 90.6 |
| `affine_qmv_fast_float16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4` | 391 | 15.16 | 7.0 % | 38.8 |
| `sn_copyfloat16float16` | 112 | 10.01 | 4.6 % | 89.4 |
| `steel_attention_float16_bq64_bk32_bd128_bv128_wm4_wn1_maskfloat16_alig` | 27 | 5.97 | 2.8 % | 221.0 |
| `rope_freqs_float16_` | 55 | 4.01 | 1.9 % | 72.9 |
| `vv_Addfloat16` | 111 | 2.96 | 1.4 % | 26.7 |
| `gg2_copyfloat16float16` | 252 | 2.21 | 1.0 % | 8.8 |
| `rope_single_freqs_float16_` | 112 | 0.96 | 0.4 % | 8.6 |
| `sdpa_vector_float16_t_128_128_nomask_qnt_nc_nosinks` | 56 | 0.86 | 0.4 % | 15.4 |

geist library concurrent encoders, prefill, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 393 | 143.55 | 57.7 % |
| norm | 113 | 40.76 | 16.4 % |
| attention | 112 | 39.33 | 15.8 % |
| elementwise / copies | 61 | 23.52 | 9.4 % |
| lm head / sampling | 2 | 1.00 | 0.4 % |
| embedding | 2 | 0.68 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `matmul_q4k_mm_tc` | 336 | 122.32 | 49.1 % | 364.1 |
| `rmsnorm_add_norm_rows_simd` | 110 | 39.57 | 15.9 % | 359.8 |
| `silu_mul_rows` | 56 | 21.29 | 8.6 % | 380.2 |
| `matmul_q6k_mm_tc` | 56 | 20.73 | 8.3 % | 370.1 |
| `rope_append_rows_f16` | 56 | 19.67 | 7.9 % | 351.2 |
| `attention_flash_tc_d128` | 56 | 19.67 | 7.9 % | 351.2 |
| `copy_u32` | 3 | 1.37 | 0.5 % | 455.2 |
| `rmsnorm_rows_simd` | 3 | 1.18 | 0.5 % | 394.3 |
| `add_rows` | 2 | 0.87 | 0.3 % | 432.9 |
| `embed_lookup_scaled_rows` | 2 | 0.68 | 0.3 % | 341.5 |
| `matvec_q6k_n4` | 1 | 0.50 | 0.2 % | 499.7 |
| `argmax_f32_part` | 1 | 0.50 | 0.2 % | 499.7 |

### decode

| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | commit to GPU start, median us | waitUntilCompleted calls | ms in them |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| geist trace | 3 | 1903.22 | 1867.07 | 1867.07 | 36.15 | 33.7 | 254 | 127 | 33020 | 0% | 0 | 4731 | 127 | 1886.32 |
| mlx trace | 3 | 1657.56 | 1656.05 | 1656.11 | 1.51 | 263.7 | 1788 | 1534 | 61957 | 100% | 42224 | 9882 | 1 | 11.09 |
| geist 1op | 3 | 1920.21 | 1895.78 | 1898.67 | 24.43 | 587.2 | 29337 | 127 | 33020 | 0% | 0 | 797 | 13908 | 1818.62 |
| mlx 1op | 3 | 1685.48 | 1677.11 | 1677.66 | 8.37 | 849.4 | 30993 | 15729 | 61605 | 100% | 18038 | 422 | 1 | 1.87 |
| geist library concurrent encoders | 3 | 1913.77 | 1882.99 | 1882.99 | 30.79 | 42.8 | 254 | 127 | 33020 | 100% | 32766 | 4737 | 127 | 1890.27 |

geist trace, decode, one token (median of the per-repeat medians, us): period 14975, gpu 14692, GPU end to wait returns 69, wait returns to first commit 105, first commit to GPU start 111, GPU idle between the token's buffers 0

geist library concurrent encoders, decode, one token (median of the per-repeat medians, us): period 15056, gpu 14731, GPU end to wait returns 70, wait returns to first commit 140, first commit to GPU start 115, GPU idle between the token's buffers 0

geist trace, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 10795 | 609.49 | 32.6 % |
| attention | 10668 | 599.45 | 32.1 % |
| norm | 7239 | 405.81 | 21.7 % |
| elementwise / copies | 3937 | 226.06 | 12.1 % |
| lm head / sampling | 254 | 20.07 | 1.1 % |
| embedding | 127 | 6.17 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 389.60 | 20.9 % | 55.8 |
| `matvec_q4k_n4` | 5334 | 293.93 | 15.7 % | 55.1 |
| `rope_append_rows_f16` | 3556 | 199.82 | 10.7 % | 56.2 |
| `attention_decv_d128_f16` | 3556 | 199.82 | 10.7 % | 56.2 |
| `attention_dec_combine` | 3556 | 199.82 | 10.7 % | 56.2 |
| `gate_up_silu_q4k_n4` | 3556 | 199.82 | 10.7 % | 56.2 |
| `matvec_q6k_n4` | 1905 | 115.74 | 6.2 % | 60.8 |
| `mv3_q4k_q4k_q6k` | 1778 | 105.71 | 5.7 % | 59.5 |
| `mv3_q4k_q4k_q4k` | 1778 | 90.25 | 4.8 % | 50.8 |
| `copy_u32` | 254 | 20.07 | 1.1 % | 79.0 |
| `rmsnorm_rows_simd` | 254 | 16.21 | 0.9 % | 63.8 |
| `add_rows` | 127 | 10.04 | 0.5 % | 79.0 |

mlx trace, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 25129 | 680.24 | 41.1 % |
| elementwise / copies | 17950 | 484.13 | 29.2 % |
| attention | 10715 | 287.17 | 17.3 % |
| norm | 7271 | 196.93 | 11.9 % |
| lm head / sampling | 384 | 4.60 | 0.3 % |
| embedding | 508 | 2.99 | 0.2 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmv_fast_float16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4` | 25129 | 680.24 | 41.1 % | 27.1 |
| `rmsfloat16` | 7271 | 196.93 | 11.9 % | 27.1 |
| `vv_Addfloat16` | 7144 | 193.27 | 11.7 % | 27.1 |
| `gg2_copyfloat16float16` | 7204 | 192.61 | 11.6 % | 26.7 |
| `rope_single_freqs_float16_` | 7143 | 191.24 | 11.5 % | 26.8 |
| `Cf2ISigmoidADf2IBroadcastACEf2IBroadcastCAFf2IMultiplyDEGf2IBroadcastF` | 3572 | 97.58 | 5.9 % | 27.3 |
| `sdpa_vector_float16_t_128_128_nomask_qnt_nc_nosinks` | 3572 | 95.93 | 5.8 % | 26.9 |
| `affine_dequantize_float16_t_gs_64_b_4` | 127 | 2.74 | 0.2 % | 21.6 |
| `looped_logsumexp_float16` | 128 | 1.53 | 0.1 % | 12.0 |
| `vsn_Subtractfloat16` | 128 | 1.53 | 0.1 % | 12.0 |
| `argmax_float16` | 128 | 1.53 | 0.1 % | 12.0 |
| `sn_copyfloat16float16` | 30 | 0.67 | 0.0 % | 22.5 |

geist 1op, decode, per repeat (one op per buffer; a buffer of several kernels split by dispatch count):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 10795 | 1554.69 | 81.9 % |
| elementwise / copies | 3937 | 235.21 | 12.4 % |
| attention | 10668 | 79.35 | 4.2 % |
| norm | 7239 | 28.31 | 1.5 % |
| lm head / sampling | 254 | 0.76 | 0.0 % |
| embedding | 127 | 0.34 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `gate_up_silu_q4k_n4` | 3556 | 692.53 | 36.5 % | 194.7 |
| `matvec_q6k_n4` | 1905 | 552.19 | 29.1 % | 289.9 |
| `matvec_q4k_n4` | 5334 | 309.98 | 16.3 % | 58.1 |
| `mv3_q4k_q4k_q6k` | 1778 | 124.05 | 6.5 % | 69.8 |
| `mv3_q4k_q4k_q4k` | 1778 | 110.60 | 5.8 % | 62.2 |
| `attention_decv_d128_f16` | 3556 | 35.45 | 1.9 % | 10.0 |
| `attention_dec_combine` | 3556 | 35.45 | 1.9 % | 10.0 |
| `rmsnorm_add_norm_rows_simd` | 6985 | 27.50 | 1.4 % | 3.9 |
| `rope_append_rows_f16` | 3556 | 8.46 | 0.4 % | 2.4 |
| `rmsnorm_rows_simd` | 254 | 0.81 | 0.0 % | 3.2 |
| `argmax_f32_part` | 127 | 0.38 | 0.0 % | 3.0 |
| `argmax_f32_merge` | 127 | 0.38 | 0.0 % | 3.0 |

mlx 1op, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 25022 | 1123.86 | 67.0 % |
| elementwise / copies | 17782 | 303.55 | 18.1 % |
| attention | 10668 | 135.57 | 8.1 % |
| norm | 7240 | 108.62 | 6.5 % |
| lm head / sampling | 384 | 4.96 | 0.3 % |
| embedding | 508 | 0.79 | 0.0 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `affine_qmv_fast_float16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4` | 25022 | 1123.86 | 67.0 % | 44.9 |
| `Cf2ISigmoidADf2IBroadcastACEf2IBroadcastCAFf2IMultiplyDEGf2IBroadcastF` | 3557 | 170.74 | 10.2 % | 48.0 |
| `vv_Addfloat16` | 7113 | 126.42 | 7.5 % | 17.8 |
| `rmsfloat16` | 7240 | 108.62 | 6.5 % | 15.0 |
| `sdpa_vector_float16_t_128_128_nomask_qnt_nc_nosinks` | 3556 | 76.23 | 4.5 % | 21.4 |
| `rope_single_freqs_float16_` | 7112 | 59.34 | 3.5 % | 8.3 |
| `gg2_copyfloat16float16` | 7112 | 6.38 | 0.4 % | 0.9 |
| `argmax_float16` | 128 | 3.35 | 0.2 % | 26.2 |
| `looped_logsumexp_float16` | 128 | 0.80 | 0.0 % | 6.3 |
| `vsn_Subtractfloat16` | 128 | 0.80 | 0.0 % | 6.3 |
| `affine_dequantize_float16_t_gs_64_b_4` | 127 | 0.28 | 0.0 % | 2.2 |
| `gather_frontuint32_uint32_int_1` | 127 | 0.26 | 0.0 % | 2.1 |

geist library concurrent encoders, decode, per repeat (buffer time split by dispatch count: counts only):

| class | dispatches | GPU ms | share |
| :-- | --: | --: | --: |
| quantized matmul | 10795 | 611.30 | 32.5 % |
| attention | 10668 | 601.24 | 31.9 % |
| norm | 7239 | 407.02 | 21.6 % |
| elementwise / copies | 3937 | 226.73 | 12.0 % |
| lm head / sampling | 254 | 20.12 | 1.1 % |
| embedding | 127 | 6.19 | 0.3 % |

| kernel | dispatches | GPU ms | share | us per dispatch |
| :-- | --: | --: | --: | --: |
| `rmsnorm_add_norm_rows_simd` | 6985 | 390.77 | 20.8 % | 55.9 |
| `matvec_q4k_n4` | 5334 | 294.81 | 15.7 % | 55.3 |
| `rope_append_rows_f16` | 3556 | 200.41 | 10.6 % | 56.4 |
| `attention_decv_d128_f16` | 3556 | 200.41 | 10.6 % | 56.4 |
| `attention_dec_combine` | 3556 | 200.41 | 10.6 % | 56.4 |
| `gate_up_silu_q4k_n4` | 3556 | 200.41 | 10.6 % | 56.4 |
| `matvec_q6k_n4` | 1905 | 116.08 | 6.2 % | 60.9 |
| `mv3_q4k_q4k_q6k` | 1778 | 106.02 | 5.6 % | 59.6 |
| `mv3_q4k_q4k_q4k` | 1778 | 90.53 | 4.8 % | 50.9 |
| `copy_u32` | 254 | 20.12 | 1.1 % | 79.2 |
| `rmsnorm_rows_simd` | 254 | 16.25 | 0.9 % | 64.0 |
| `add_rows` | 127 | 10.06 | 0.5 % | 79.2 |


