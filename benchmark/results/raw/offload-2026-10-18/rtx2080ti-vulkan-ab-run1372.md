# Revision A/B, each revision built from scratch

Model: `synth.gguf` (`968df819bff52e1a`)
Revisions: base = `c993379b0746` (c993379b07462403ab14c356a3d631f84278b2d8), new = `cceea150ddcd` (cceea150ddcdbec19665ca8ceb8f00442e668260); baseline `base`; `control` is a copy of the baseline's binary
Protocol: 8 cycles, order rotating; sequences 128,512, decode 16, repeats 1 (median), warmup 1; OMP_WAIT_POLICY=active GEIST_BENCH_BACKEND=vulkan GEIST_VK_DEVICE=0
Host: AMD Ryzen 9 9950X 16-Core Processor, 32 CPUs, Linux 7.0.0-38-generic, THP always [madvise] never

| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 128 | new | 38.2 | 33.7 | -1.8 % | [-32.0 %, +46.4 %] | 5/8 | within noise |
| prefill_ms @ 128 | control | 38.2 | 17.8 | -10.4 % | [-69.4 %, +42.6 %] | 7/8 | within noise |
| prefill_ms @ 512 | new | 64.5 | 64.8 | +2.0 % | [-30.2 %, +44.6 %] | 2/8 | within noise |
| prefill_ms @ 512 | control | 64.5 | 65.5 | +0.6 % | [-28.6 %, +47.4 %] | 2/8 | within noise |
| decode_ms @ 128 | new | 78.6 | 82.8 | +0.6 % | [-21.3 %, +27.8 %] | 4/8 | within noise |
| decode_ms @ 128 | control | 78.6 | 52.2 | -8.9 % | [-71.9 %, +41.9 %] | 5/8 | within noise |
| decode_ms @ 512 | new | 49.0 | 48.5 | +8.3 % | [-26.5 %, +51.1 %] | 2/8 | within noise |
| decode_ms @ 512 | control | 49.0 | 51.8 | +4.8 % | [-13.6 %, +164.8 %] | 2/8 | within noise |
| ttft_ms @ 128 | new | 38.2 | 33.7 | -1.8 % | [-32.0 %, +46.5 %] | 5/8 | within noise |
| ttft_ms @ 128 | control | 38.2 | 17.8 | -10.4 % | [-69.4 %, +42.6 %] | 7/8 | within noise |
| ttft_ms @ 512 | new | 64.5 | 64.8 | +2.0 % | [-30.2 %, +44.6 %] | 2/8 | within noise |
| ttft_ms @ 512 | control | 64.5 | 65.5 | +0.6 % | [-28.6 %, +47.4 %] | 2/8 | within noise |

Control: the baseline's own binary again; its intervals reach ±164.8 %, the resolution of this run.

6 comparisons: at 95 % about one interval in twenty excludes zero by chance.

raw: /home/germar/gitea-runner-geisten/work/d932ab827cc087b3/hostexecutor/ab-out/2026-10-18T132224Z/revision_ab.jsonl
report: /home/germar/gitea-runner-geisten/work/d932ab827cc087b3/hostexecutor/ab-out/2026-10-18T132224Z/revision_ab.md
