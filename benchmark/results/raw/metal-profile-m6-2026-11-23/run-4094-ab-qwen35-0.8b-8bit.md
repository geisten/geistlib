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
