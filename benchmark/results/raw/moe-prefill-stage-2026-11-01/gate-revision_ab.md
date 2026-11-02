# Revision A/B, each revision built from scratch

Model: `synth-1b.gguf` (`9b61cff0268cd3f6`)
Revisions: base = `6642260d55a1` (6642260d), new = `4e9f2ea0db84` (b239); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 6 cycles, order rotating; sequences 128,512, decode 32, repeats 1 (median), warmup 16; OMP_WAIT_POLICY=active GEIST_BENCH_BACKEND=vulkan
Host: AMD Ryzen 9 9950X 16-Core Processor, 32 CPUs, Linux 7.0.0-38-generic, THP always [madvise] never

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 128 | new | 22.6 | 22.6 | -0.1 % | [-19.2 %, +0.5 %] | 4/6 | within noise |
| prefill_ms @ 128 | control | 22.6 | 22.6 | -0.0 % | [-18.9 %, +0.5 %] | 4/6 | within noise |
| prefill_ms @ 512 | new | 62.4 | 62.4 | -0.0 % | [-4.8 %, +0.2 %] | 3/6 | within noise |
| prefill_ms @ 512 | control | 62.4 | 62.4 | +0.0 % | [-4.0 %, +0.1 %] | 3/6 | within noise |
| decode_ms @ 128 | new | 87.1 | 87.0 | -0.1 % | [-13.2 %, +0.4 %] | 4/6 | within noise |
| decode_ms @ 128 | control | 87.1 | 87.1 | -0.1 % | [-12.9 %, +0.3 %] | 3/6 | within noise |
| decode_ms @ 512 | new | 90.6 | 90.5 | -0.1 % | [-0.4 %, +0.3 %] | 4/6 | within noise |
| decode_ms @ 512 | control | 90.6 | 90.5 | -0.1 % | [-1.1 %, +2.4 %] | 4/6 | within noise |
| ttft_ms @ 128 | new | 22.6 | 22.6 | -0.1 % | [-19.2 %, +0.5 %] | 4/6 | within noise |
| ttft_ms @ 128 | control | 22.6 | 22.6 | -0.0 % | [-18.9 %, +0.5 %] | 4/6 | within noise |
| ttft_ms @ 512 | new | 62.4 | 62.4 | -0.0 % | [-4.8 %, +0.2 %] | 3/6 | within noise |
| ttft_ms @ 512 | control | 62.4 | 62.4 | +0.0 % | [-4.0 %, +0.1 %] | 3/6 | within noise |

Control: the baseline's own binary again; its intervals reach ±18.9 %, the resolution of this run.

6 comparisons: at 95 % about one interval in twenty excludes zero by chance.
