# Revision A/B, each revision built from scratch

Model: `synth.gguf` (`9b61cff0268cd3f6`)
Revisions: base = `c993379b0746` (c993379b07462403ab14c356a3d631f84278b2d8), new = `cceea150ddcd` (cceea150ddcdbec19665ca8ceb8f00442e668260); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 8 cycles, order rotating; sequences 128,512, decode 16, repeats 1 (median), warmup 1; OMP_WAIT_POLICY=active GEIST_BENCH_BACKEND=vulkan GEIST_VK_DEVICE=0
Host: AMD Ryzen 9 9950X 16-Core Processor, 32 CPUs, Linux 7.0.0-38-generic, THP always [madvise] never

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 128 | new | 22.2 | 22.7 | +0.5 % | [-28.1 %, +3.6 %] | 2/8 | within noise |
| prefill_ms @ 128 | control | 22.2 | 22.2 | +0.0 % | [-30.7 %, +3.6 %] | 3/8 | within noise |
| prefill_ms @ 512 | new | 61.4 | 61.5 | +0.1 % | [-10.2 %, +1.5 %] | 2/8 | within noise |
| prefill_ms @ 512 | control | 61.4 | 61.4 | +0.0 % | [-10.9 %, +0.9 %] | 4/8 | within noise |
| decode_ms @ 128 | new | 41.9 | 43.2 | -0.1 % | [-25.0 %, +7.9 %] | 5/8 | within noise |
| decode_ms @ 128 | control | 41.9 | 42.5 | +2.9 % | [-31.2 %, +18.6 %] | 3/8 | within noise |
| decode_ms @ 512 | new | 43.3 | 44.2 | -0.0 % | [-5.4 %, +10.0 %] | 4/8 | within noise |
| decode_ms @ 512 | control | 43.3 | 44.3 | +1.8 % | [-9.2 %, +4.8 %] | 4/8 | within noise |
| ttft_ms @ 128 | new | 22.2 | 22.7 | +0.5 % | [-28.1 %, +3.6 %] | 2/8 | within noise |
| ttft_ms @ 128 | control | 22.2 | 22.2 | +0.0 % | [-30.7 %, +3.6 %] | 3/8 | within noise |
| ttft_ms @ 512 | new | 61.4 | 61.5 | +0.1 % | [-10.1 %, +1.5 %] | 2/8 | within noise |
| ttft_ms @ 512 | control | 61.4 | 61.4 | +0.0 % | [-10.9 %, +0.9 %] | 4/8 | within noise |

Control: the baseline's own binary again; its intervals reach ±31.2 %, the resolution of this run.

6 comparisons: at 95 % about one interval in twenty excludes zero by chance.

raw: /home/germar/gitea-runner-geisten/work/444be258d8e4464e/hostexecutor/ab-out/2026-10-18T132456Z/revision_ab.jsonl
report: /home/germar/gitea-runner-geisten/work/444be258d8e4464e/hostexecutor/ab-out/2026-10-18T132456Z/revision_ab.md
