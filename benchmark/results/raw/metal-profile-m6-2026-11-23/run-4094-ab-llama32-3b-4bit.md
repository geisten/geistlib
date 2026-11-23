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
