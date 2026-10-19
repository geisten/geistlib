# Revision A/B, each revision built from scratch

Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (`3907dc1658db1f78`)
Revisions: base = `d56811839c11` (origin/main), new = `f8da3d1f49b5` (perf/pq2_0-int8-prefill); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 6 cycles, order rotating; sequences 128,512, decode 16, repeats 1 (median), warmup 1; OMP_WAIT_POLICY=active GEIST_BENCH_BACKEND=cpu_neon
Host: Apple M1 Max, 10 CPUs, Darwin 27.0.0, THP n/a

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 128 | new | 8403.2 | 3965.0 | -52.8 % | [-54.5 %, -49.8 %] | 6/6 | faster |
| prefill_ms @ 128 | control | 8403.2 | 8382.0 | -0.3 % | [-1.6 %, +4.0 %] | 4/6 | within noise |
| prefill_ms @ 512 | new | 34078.8 | 16335.0 | -51.8 % | [-53.2 %, -47.5 %] | 6/6 | faster |
| prefill_ms @ 512 | control | 34078.8 | 34155.5 | +0.7 % | [-2.3 %, +0.8 %] | 2/6 | within noise |
| decode_ms @ 128 | new | 1287.1 | 1313.8 | +1.5 % | [+0.7 %, +3.4 %] | 0/6 | slower |
| decode_ms @ 128 | control | 1287.1 | 1281.5 | -0.4 % | [-5.0 %, +0.2 %] | 5/6 | within noise |
| decode_ms @ 512 | new | 1358.9 | 1375.9 | +1.1 % | [-11.0 %, +2.3 %] | 1/6 | within noise |
| decode_ms @ 512 | control | 1358.9 | 1348.6 | -0.8 % | [-11.6 %, +2.5 %] | 4/6 | within noise |
| ttft_ms @ 128 | new | 8403.2 | 3965.0 | -52.8 % | [-54.5 %, -49.8 %] | 6/6 | faster |
| ttft_ms @ 128 | control | 8403.2 | 8382.0 | -0.3 % | [-1.6 %, +4.0 %] | 4/6 | within noise |
| ttft_ms @ 512 | new | 34078.8 | 16335.0 | -51.8 % | [-53.2 %, -47.5 %] | 6/6 | faster |
| ttft_ms @ 512 | control | 34078.8 | 34155.5 | +0.7 % | [-2.3 %, +0.8 %] | 2/6 | within noise |

Control: the baseline's own binary again; its intervals reach ±11.6 %, the resolution of this run.

6 comparisons: at 95 % about one interval in twenty excludes zero by chance.
