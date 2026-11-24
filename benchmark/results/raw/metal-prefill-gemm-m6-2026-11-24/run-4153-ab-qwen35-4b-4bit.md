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
