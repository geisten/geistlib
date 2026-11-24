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
