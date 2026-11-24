## Tensor-API GEMM of fragments: origin/main, this checkout, GEIST_METAL_TC_COOP=1 and mlx-lm (#590)

main `16c7df30`, new `b9ec2ea4`

### Qwen3.5 0.8B, 8-bit

| arm | prefill tok/s | against main | decode tok/s | against main | tokens as main's |
| :-- | --: | --: | --: | --: | :-- |
| main | 7431 (7391–7456) | +0.0 % | 162.0 (161.1–162.3) | +0.0 % | True |
| new | 7929 (7884–7947) | +6.7 % | 162.1 (162.0–162.4) | +0.1 % | True |
| coop | 7442 (7401–7455) | +0.1 % | 162.2 (162.0–162.2) | +0.1 % | True |
| mlx | 7256 (7167–7377) | -2.4 % | 160.1 (158.7–160.1) | -1.2 % | True |

bench_nll: {"decode": {"main": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}, "new": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}, "coop": {"tokens": 95, "nll": 3.509791, "ppl": 33.4413, "logits_fnv": "c2748027690c9bf6"}}, "prefill": {"main": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}, "new": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}, "coop": {"tokens": 127, "nll": 3.871956, "ppl": 48.0363, "logits_fnv": "541c4ddcdb237e1f"}}}

process start to first token, one-token prompt, ms: {"main": {"median": 66.41, "min": 62.96, "max": 72.83, "n": 10}, "new": {"median": 66.49, "min": 63.71, "max": 71.95, "n": 10}}

#### A/B, 16c7df30 against HEAD

# Revision A/B, each revision built from scratch

Model: `qwen3.5-0.8b-q8_0.gguf` (`0ad885ffd4bb022f`)
Revisions: base = `16c7df30d986` (16c7df30d9868fe8dc6505aec6c3461c3f86f282), new = `b9ec2ea41c10` (HEAD); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 10 cycles, order rotating; sequences 512, decode 128, repeats 1 (median), warmup 64; prompt `moe-long.txt`; OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 GEIST_PREFILL_THREADS=4 GEIST_DECODE_THREADS=4 GEIST_TEXT_ONLY=1 GEIST_BENCH_BACKEND=metal
Host: Apple M6, 12 CPUs, Darwin 27.2.0, THP n/a

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 512 | new | 69.2 | 65.0 | -6.1 % | [-6.4 %, -5.8 %] | 10/10 | faster |
| prefill_ms @ 512 | control | 69.2 | 69.2 | -0.0 % | [-0.2 %, +0.2 %] | 5/10 | within noise |
| decode_ms @ 512 | new | 789.5 | 789.9 | +0.1 % | [-0.0 %, +0.4 %] | 2/10 | within noise |
| decode_ms @ 512 | control | 789.5 | 789.5 | +0.0 % | [-0.1 %, +0.1 %] | 5/10 | within noise |
| ttft_ms @ 512 | new | 69.5 | 65.3 | -6.1 % | [-6.3 %, -5.8 %] | 10/10 | faster |
| ttft_ms @ 512 | control | 69.5 | 69.6 | -0.0 % | [-0.2 %, +0.3 %] | 5/10 | within noise |

Control: the baseline's own binary again; its intervals reach ±0.3 %, the resolution of this run.

Cool start: every run began at GPU 40.8-43.9 C (limit 65), CPU 39.9-44.0 C (limit 65); 33 s spent cooling.

Power: all 30 runs started on AC power with Low Power Mode off; 0 s spent waiting for it.

3 comparisons: at 95 % about one interval in twenty excludes zero by chance.


### Llama 3.2 3B Instruct, 4-bit

| arm | prefill tok/s | against main | decode tok/s | against main | tokens as main's |
| :-- | --: | --: | --: | --: | :-- |
| main | 2057 (2054–2058) | +0.0 % | 68.4 (68.4–68.5) | +0.0 % | True |
| new | 2278 (2276–2283) | +10.8 % | 68.4 (68.3–68.5) | +0.0 % | True |
| coop | 2055 (2052–2062) | -0.1 % | 68.4 (68.4–68.4) | -0.0 % | True |
| mlx | 2636 (2631–2639) | +28.1 % | 77.1 (76.8–77.2) | +12.6 % | False |

bench_nll: {"decode": {"main": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}, "new": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}, "coop": {"tokens": 95, "nll": 0.840224, "ppl": 2.3169, "logits_fnv": "b2135ee7cbb7a921"}}, "prefill": {"main": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}, "new": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}, "coop": {"tokens": 127, "nll": 2.819346, "ppl": 16.7659, "logits_fnv": "3720d83b04261e76"}}}

process start to first token, one-token prompt, ms: {"main": {"median": 104.26, "min": 103.52, "max": 105.33, "n": 10}, "new": {"median": 104.82, "min": 104.12, "max": 105.28, "n": 10}}

#### A/B, 16c7df30 against HEAD

# Revision A/B, each revision built from scratch

Model: `llama-3.2-3b-Q4_K_M.gguf` (`6c1a2b4116103267`)
Revisions: base = `16c7df30d986` (16c7df30d9868fe8dc6505aec6c3461c3f86f282), new = `b9ec2ea41c10` (HEAD); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 10 cycles, order rotating; sequences 512, decode 128, repeats 1 (median), warmup 64; prompt `moe-long.txt`; OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 GEIST_PREFILL_THREADS=4 GEIST_DECODE_THREADS=4 GEIST_TEXT_ONLY=1 GEIST_BENCH_BACKEND=metal
Host: Apple M6, 12 CPUs, Darwin 27.2.0, THP n/a

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 512 | new | 248.6 | 224.6 | -9.7 % | [-9.9 %, -9.5 %] | 10/10 | faster |
| prefill_ms @ 512 | control | 248.6 | 248.9 | +0.1 % | [-0.0 %, +0.3 %] | 2/10 | within noise |
| decode_ms @ 512 | new | 1871.3 | 1870.5 | -0.1 % | [-0.1 %, +0.0 %] | 7/10 | within noise |
| decode_ms @ 512 | control | 1871.3 | 1871.6 | +0.0 % | [-0.0 %, +0.2 %] | 4/10 | within noise |
| ttft_ms @ 512 | new | 248.8 | 224.8 | -9.7 % | [-9.8 %, -9.5 %] | 10/10 | faster |
| ttft_ms @ 512 | control | 248.8 | 249.1 | +0.1 % | [-0.0 %, +0.3 %] | 2/10 | within noise |

Control: the baseline's own binary again; its intervals reach ±0.3 %, the resolution of this run.

Cool start: every run began at GPU 41.8-50.9 C (limit 65), CPU 40.0-50.7 C (limit 65); 33 s spent cooling.

Power: all 30 runs started on AC power with Low Power Mode off; 0 s spent waiting for it.

3 comparisons: at 95 % about one interval in twenty excludes zero by chance.


### Qwen3.5 4B, 4-bit

| arm | prefill tok/s | against main | decode tok/s | against main | tokens as main's |
| :-- | --: | --: | --: | --: | :-- |
| main | 1568 (1541–1575) | +0.0 % | 49.6 (49.6–49.7) | +0.0 % | True |
| new | 1705 (1669–1711) | +8.8 % | 49.6 (49.6–49.7) | -0.0 % | True |
| coop | 1568 (1541–1572) | -0.0 % | 49.7 (49.6–49.7) | +0.1 % | True |
| mlx | 1811 (1799–1812) | +15.5 % | 57.1 (56.8–57.2) | +15.0 % | False |

bench_nll: {"decode": {"main": {"tokens": 95, "nll": 0.709146, "ppl": 2.0323, "logits_fnv": "0a71f6a2ab84fe6d"}, "new": {"tokens": 95, "nll": 0.709146, "ppl": 2.0323, "logits_fnv": "0a71f6a2ab84fe6d"}, "coop": {"tokens": 95, "nll": 0.709146, "ppl": 2.0323, "logits_fnv": "0a71f6a2ab84fe6d"}}, "prefill": {"main": {"tokens": 127, "nll": 2.935423, "ppl": 18.8295, "logits_fnv": "2c3a08ae15b88eac"}, "new": {"tokens": 127, "nll": 2.935423, "ppl": 18.8295, "logits_fnv": "2c3a08ae15b88eac"}, "coop": {"tokens": 127, "nll": 2.935423, "ppl": 18.8295, "logits_fnv": "2c3a08ae15b88eac"}}}

process start to first token, one-token prompt, ms: {"main": {"median": 105.42, "min": 103.78, "max": 105.8, "n": 10}, "new": {"median": 105.61, "min": 105.24, "max": 106.11, "n": 10}}

#### A/B, 16c7df30 against HEAD

# Revision A/B, each revision built from scratch

Model: `qwen3.5-4b-q4_0.gguf` (`298fcb5fe7a77ccc`)
Revisions: base = `16c7df30d986` (16c7df30d9868fe8dc6505aec6c3461c3f86f282), new = `b9ec2ea41c10` (HEAD); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 10 cycles, order rotating; sequences 512, decode 128, repeats 1 (median), warmup 64; prompt `moe-long.txt`; OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 GEIST_PREFILL_THREADS=4 GEIST_DECODE_THREADS=4 GEIST_TEXT_ONLY=1 GEIST_BENCH_BACKEND=metal
Host: Apple M6, 12 CPUs, Darwin 27.2.0, THP n/a

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 512 | new | 332.2 | 306.4 | -7.8 % | [-7.9 %, -7.7 %] | 10/10 | faster |
| prefill_ms @ 512 | control | 332.2 | 332.4 | +0.0 % | [-0.1 %, +0.1 %] | 3/10 | within noise |
| decode_ms @ 512 | new | 2577.9 | 2578.9 | -0.0 % | [-0.1 %, +0.1 %] | 5/10 | within noise |
| decode_ms @ 512 | control | 2577.9 | 2578.1 | +0.0 % | [-0.2 %, +0.1 %] | 4/10 | within noise |
| ttft_ms @ 512 | new | 332.6 | 306.7 | -7.8 % | [-7.9 %, -7.7 %] | 10/10 | faster |
| ttft_ms @ 512 | control | 332.6 | 332.7 | +0.0 % | [-0.1 %, +0.1 %] | 3/10 | within noise |

Control: the baseline's own binary again; its intervals reach ±0.2 %, the resolution of this run.

Cool start: every run began at GPU 42.2-51.9 C (limit 65), CPU 40.0-46.8 C (limit 65); 33 s spent cooling.

Power: all 30 runs started on AC power with Low Power Mode off; 0 s spent waiting for it.

3 comparisons: at 95 % about one interval in twenty excludes zero by chance.


### Gemma 4 E2B-it, 4-bit

| arm | prefill tok/s | against main | decode tok/s | against main | tokens as main's |
| :-- | --: | --: | --: | --: | :-- |
| main | 7147 (7106–7173) | +0.0 % | 87.8 (87.8–87.9) | +0.0 % | True |
| new | 7649 (7566–7690) | +7.0 % | 87.8 (87.6–87.9) | +0.0 % | True |
| coop | 7150 (7115–7178) | +0.0 % | 87.8 (87.5–87.9) | -0.0 % | True |
| mlx | 7667 (7626–7720) | +7.3 % | 96.7 (96.6–96.8) | +10.1 % | False |

bench_nll: {"decode": {"main": {"tokens": 95, "nll": 10.773558, "ppl": 47741.5889, "logits_fnv": "2d10648daf24f88f"}, "new": {"tokens": 95, "nll": 10.773558, "ppl": 47741.5889, "logits_fnv": "2d10648daf24f88f"}, "coop": {"tokens": 95, "nll": 10.773558, "ppl": 47741.5889, "logits_fnv": "2d10648daf24f88f"}}, "prefill": {"main": {"tokens": 127, "nll": 10.858324, "ppl": 51964.9361, "logits_fnv": "66ba6995bd2e8297"}, "new": {"tokens": 127, "nll": 10.858324, "ppl": 51964.9361, "logits_fnv": "66ba6995bd2e8297"}, "coop": {"tokens": 127, "nll": 10.858324, "ppl": 51964.9361, "logits_fnv": "66ba6995bd2e8297"}}}

process start to first token, one-token prompt, ms: {"main": {"median": 83.8, "min": 82.45, "max": 90.51, "n": 10}, "new": {"median": 84.24, "min": 83.55, "max": 88.76, "n": 10}}

#### A/B, 16c7df30 against HEAD

# Revision A/B, each revision built from scratch

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb8`)
Revisions: base = `16c7df30d986` (16c7df30d9868fe8dc6505aec6c3461c3f86f282), new = `b9ec2ea41c10` (HEAD); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 10 cycles, order rotating; sequences 512, decode 128, repeats 1 (median), warmup 64; prompt `moe-long.txt`; OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 GEIST_PREFILL_THREADS=4 GEIST_DECODE_THREADS=4 GEIST_TEXT_ONLY=1 GEIST_BENCH_BACKEND=metal
Host: Apple M6, 12 CPUs, Darwin 27.2.0, THP n/a

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 512 | new | 72.0 | 67.1 | -6.7 % | [-6.9 %, -6.6 %] | 10/10 | faster |
| prefill_ms @ 512 | control | 72.0 | 71.9 | -0.0 % | [-0.2 %, +0.1 %] | 5/10 | within noise |
| decode_ms @ 512 | new | 1457.7 | 1457.8 | +0.0 % | [-0.3 %, +0.1 %] | 5/10 | within noise |
| decode_ms @ 512 | control | 1457.7 | 1458.6 | +0.1 % | [-0.3 %, +0.1 %] | 4/10 | within noise |
| ttft_ms @ 512 | new | 72.0 | 67.1 | -6.7 % | [-6.9 %, -6.6 %] | 10/10 | faster |
| ttft_ms @ 512 | control | 72.0 | 71.9 | -0.0 % | [-0.2 %, +0.1 %] | 5/10 | within noise |

Control: the baseline's own binary again; its intervals reach ±0.3 %, the resolution of this run.

Cool start: every run began at GPU 40.7-47.6 C (limit 65), CPU 39.3-46.2 C (limit 65); 33 s spent cooling.

Power: all 30 runs started on AC power with Low Power Mode off; 0 s spent waiting for it.

3 comparisons: at 95 % about one interval in twenty excludes zero by chance.


## Metal unit tests

| test | GEIST_METAL_TC_COOP | exit | FAIL lines |
| :-- | --: | --: | --: |
| test_backend_metal_deltanet_chunk_unit | 0 | 0 | 0 |
| test_backend_metal_deltanet_chunk_unit | 1 | 0 | 0 |
| test_backend_metal_deltanet_unit | 0 | 0 | 0 |
| test_backend_metal_deltanet_unit | 1 | 0 | 0 |
| test_backend_metal_parity_unit | 0 | 0 | 0 |
| test_backend_metal_parity_unit | 1 | 0 | 0 |
| test_metal_autorelease_unit | 0 | 77 | 0 |
| test_metal_autorelease_unit | 1 | 77 | 0 |
| test_metal_budget_unit | 0 | 0 | 0 |
| test_metal_budget_unit | 1 | 0 | 0 |
| test_metal_concurrent_unit | 0 | 0 | 0 |
| test_metal_concurrent_unit | 1 | 0 | 0 |
| test_metal_pq2_lowp_unit | 0 | 0 | 0 |
| test_metal_pq2_lowp_unit | 1 | 0 | 0 |
| test_metal_resources_unit | 0 | 77 | 0 |
| test_metal_resources_unit | 1 | 77 | 0 |
| test_decode_ahead_unit | 0 | 0 | 0 |
| test_decode_ahead_unit | 1 | 0 | 0 |

  tensor GEMM kernels: fragments (vector loads) against cooperative

## One prefill GEMM on Metal: cooperative kernel, fragments, mlx (#590)

ms per GEMM, median of three cycles; TFLOP/s = 2 x rows x n_in x n_out / time. mlx's weights are affine groups of 64 at the format's bits, its activations 16-bit floats.

| format n_in x n_out, rows | cooperative ms | fragments ms | change | same output | cooperative TFLOP/s | fragments TFLOP/s | mlx ms | mlx TFLOP/s | difference to cpu_scalar (coop, fragments) |
|:--|--:|--:|--:|:--|--:|--:|--:|--:|:--|
| q4_0 2560 x 8192, 256 | 0.724 | 0.645 | -11 % | yes | 14.8 | 16.6 | 0.598 | 18.0 | 4.25e-04, 4.25e-04 |
| q4_0 2560 x 8192, 512 | 1.414 | 1.257 | -11 % | yes | 15.2 | 17.1 | 1.178 | 18.2 | 4.05e-04, 4.05e-04 |
| q4_0 2560 x 9216, 256 | 0.793 | 0.718 | -9 % | yes | 15.2 | 16.8 | 0.672 | 18.0 | 3.91e-04, 3.91e-04 |
| q4_0 2560 x 9216, 512 | 1.558 | 1.400 | -10 % | yes | 15.5 | 17.3 | 1.325 | 18.2 | 3.91e-04, 3.91e-04 |
| q4_0 9216 x 2560, 256 | 0.965 | 0.793 | -18 % | yes | 12.5 | 15.2 | 0.668 | 18.1 | 4.39e-04, 4.39e-04 |
| q4_0 9216 x 2560, 512 | 2.146 | 1.815 | -15 % | yes | 11.3 | 13.3 | 1.336 | 18.1 | 4.45e-04, 4.45e-04 |
| q4_k 3072 x 1024, 256 | 0.135 | 0.129 | -4 % | yes | 11.9 | 12.5 | 0.389 | 4.1 | 4.36e-04, 4.36e-04 |
| q4_k 3072 x 1024, 512 | 0.381 | 0.281 | -26 % | yes | 8.5 | 11.5 | 0.180 | 17.9 | 4.83e-04, 4.83e-04 |
| q4_k 3072 x 3072, 256 | 0.355 | 0.329 | -7 % | yes | 13.6 | 14.7 | 0.266 | 18.2 | 4.44e-04, 4.44e-04 |
| q4_k 3072 x 3072, 512 | 0.756 | 0.653 | -14 % | yes | 12.8 | 14.8 | 0.516 | 18.7 | 4.44e-04, 4.44e-04 |
| q4_k 3072 x 8192, 256 | 0.914 | 0.808 | -12 % | yes | 14.1 | 15.9 | 0.693 | 18.6 | 4.30e-04, 4.30e-04 |
| q4_k 3072 x 8192, 512 | 1.818 | 1.554 | -15 % | yes | 14.2 | 16.6 | 1.369 | 18.8 | 4.30e-04, 4.30e-04 |
| q4_k 8192 x 3072, 256 | 0.997 | 0.841 | -16 % | yes | 12.9 | 15.3 | 0.699 | 18.4 | 3.66e-04, 3.66e-04 |
| q4_k 8192 x 3072, 512 | 2.325 | 1.804 | -22 % | yes | 11.1 | 14.3 | 1.396 | 18.5 | 3.66e-04, 3.66e-04 |
| q6_k 3072 x 1024, 256 | 0.142 | 0.131 | -8 % | yes | 11.3 | 12.3 | 0.393 | 4.1 | 5.09e-04, 5.09e-04 |
| q6_k 3072 x 1024, 512 | 0.402 | 0.286 | -29 % | yes | 8.0 | 11.3 | 0.192 | 16.8 | 4.77e-04, 4.77e-04 |
| q6_k 8192 x 3072, 256 | 1.232 | 1.016 | -18 % | yes | 10.5 | 12.7 | 0.770 | 16.7 | 4.27e-04, 4.27e-04 |
| q6_k 8192 x 3072, 512 | 2.328 | 2.779 | +19 % | yes | 11.1 | 9.3 | 1.566 | 16.5 | 4.27e-04, 4.27e-04 |
| q8_0 1024 x 3584, 256 | 0.165 | 0.144 | -13 % | yes | 11.4 | 13.0 | 0.112 | 16.8 | 4.26e-04, 4.26e-04 |
| q8_0 1024 x 3584, 512 | 0.305 | 0.264 | -13 % | yes | 12.3 | 14.2 | 0.214 | 17.6 | 4.26e-04, 4.26e-04 |
| q8_0 1024 x 6144, 256 | 0.235 | 0.209 | -11 % | yes | 13.7 | 15.4 | 0.185 | 17.4 | 3.84e-04, 3.84e-04 |
| q8_0 1024 x 6144, 512 | 0.452 | 0.400 | -12 % | yes | 14.3 | 16.1 | 0.360 | 17.9 | 3.84e-04, 3.84e-04 |
| q8_0 2048 x 1024, 256 | 0.097 | 0.080 | -18 % | yes | 11.1 | 13.4 | 0.268 | 4.0 | 4.01e-04, 4.01e-04 |
| q8_0 2048 x 1024, 512 | 0.235 | 0.187 | -20 % | yes | 9.1 | 11.5 | 0.128 | 16.8 | 3.81e-04, 3.81e-04 |
| q8_0 3584 x 1024, 256 | 0.165 | 0.135 | -18 % | yes | 11.4 | 13.9 | 0.446 | 4.2 | 4.29e-04, 4.29e-04 |
| q8_0 3584 x 1024, 512 | 0.461 | 0.376 | -18 % | yes | 8.2 | 10.0 | 0.221 | 17.0 | 4.29e-04, 4.29e-04 |
## DeltaNet prefill: geist's mixer, mlx-lm's recurrence kernel (#590)

ms per call, median of three cycles. geist's call is the whole mixer, mlx-lm's the recurrence alone.

| value heads | rows | geist mixer | mlx-lm gated_delta_kernel |
|--:|--:|--:|--:|
| 16 | 256 | 0.403 | 0.204 |
| 16 | 512 | 0.944 | 0.393 |
| 32 | 256 | 0.811 | 0.373 |
| 32 | 512 | 1.733 | 0.720 |
