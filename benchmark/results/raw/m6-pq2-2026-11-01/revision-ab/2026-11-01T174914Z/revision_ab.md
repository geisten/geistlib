# Revision A/B, each revision built from scratch

Model: `qwen3.5-0.8b-q8_0.gguf` (`0ad885ffd4bb022f`)
Revisions: base = `6642260d55a1` (origin/main), new = `82c71d6df64f` (HEAD); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 6 cycles, order rotating; sequences 128,512, decode 32, repeats 1 (median), warmup 512; OMP_WAIT_POLICY=active GEIST_BENCH_BACKEND=metal GEIST_METAL_TENSOR=1 GEIST_METAL_PQ2_MODE=off
Host: Apple M6, 12 CPUs, Darwin 27.0.0, THP n/a

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 128 | new | 21.4 | 21.4 | -0.0 % | [-2.4 %, +0.5 %] | 4/6 | within noise |
| prefill_ms @ 128 | control | 21.4 | 21.5 | +0.0 % | [-2.4 %, +0.4 %] | 3/6 | within noise |
| prefill_ms @ 512 | new | 71.4 | 71.3 | -0.1 % | [-0.7 %, +0.3 %] | 5/6 | within noise |
| prefill_ms @ 512 | control | 71.4 | 71.5 | +0.2 % | [-0.5 %, +0.2 %] | 1/6 | within noise |
| decode_ms @ 128 | new | 205.7 | 205.7 | -0.0 % | [-0.4 %, +1.4 %] | 3/6 | within noise |
| decode_ms @ 128 | control | 205.7 | 205.7 | +0.0 % | [-0.4 %, +0.3 %] | 3/6 | within noise |
| decode_ms @ 512 | new | 206.3 | 206.3 | +0.0 % | [-1.2 %, +0.5 %] | 2/6 | within noise |
| decode_ms @ 512 | control | 206.3 | 206.3 | +0.0 % | [-1.2 %, +0.2 %] | 3/6 | within noise |
| ttft_ms @ 128 | new | 21.4 | 21.4 | -0.0 % | [-2.4 %, +0.5 %] | 4/6 | within noise |
| ttft_ms @ 128 | control | 21.4 | 21.5 | +0.0 % | [-2.4 %, +0.4 %] | 3/6 | within noise |
| ttft_ms @ 512 | new | 71.4 | 71.3 | -0.1 % | [-0.7 %, +0.3 %] | 5/6 | within noise |
| ttft_ms @ 512 | control | 71.4 | 71.5 | +0.2 % | [-0.5 %, +0.2 %] | 1/6 | within noise |

Control: the baseline's own binary again; its intervals reach ±2.4 %, the resolution of this run.

Cool start: every run began at GPU 39.9-41.8 C (limit 65), CPU 39.1-43.5 C (limit 65); 20 s spent cooling.

Power: all 18 runs started on AC power with Low Power Mode off; 0 s spent waiting for it.

6 comparisons: at 95 % about one interval in twenty excludes zero by chance.
