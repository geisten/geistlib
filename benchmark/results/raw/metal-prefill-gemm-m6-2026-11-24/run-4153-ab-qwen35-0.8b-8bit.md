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
