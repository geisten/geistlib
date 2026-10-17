Protocol: 8 cycles, order rotating; sequences 128,512, decode 32, repeats 1 (median), warmup 16; OMP_WAIT_POLICY=active
| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| prefill_ms @ 128 | new | 1861.3 | 1859.6 | -0.0 % | [-0.3 %, +0.5 %] | 4/8 | within noise |
| prefill_ms @ 128 | control | 1861.3 | 1862.0 | +0.1 % | [-0.7 %, +0.3 %] | 3/8 | within noise |
| prefill_ms @ 512 | new | 7781.1 | 7772.9 | +0.0 % | [-0.7 %, +0.6 %] | 4/8 | within noise |
| prefill_ms @ 512 | control | 7781.1 | 7776.4 | +0.1 % | [-0.6 %, +0.6 %] | 4/8 | within noise |
| decode_ms @ 128 | new | 2278.3 | 2258.7 | -1.2 % | [-3.1 %, -0.0 %] | 8/8 | faster |
| decode_ms @ 128 | control | 2278.3 | 2269.6 | -0.5 % | [-2.0 %, +1.0 %] | 5/8 | within noise |
| decode_ms @ 512 | new | 2351.3 | 2326.7 | -1.2 % | [-2.7 %, +0.4 %] | 7/8 | within noise |
| decode_ms @ 512 | control | 2351.3 | 2346.3 | -0.4 % | [-1.9 %, +0.7 %] | 5/8 | within noise |
| ttft_ms @ 128 | new | 1861.3 | 1859.6 | -0.0 % | [-0.3 %, +0.5 %] | 4/8 | within noise |
| ttft_ms @ 128 | control | 1861.3 | 1862.0 | +0.1 % | [-0.7 %, +0.3 %] | 3/8 | within noise |
| ttft_ms @ 512 | new | 7781.1 | 7772.9 | +0.0 % | [-0.7 %, +0.6 %] | 4/8 | within noise |
| ttft_ms @ 512 | control | 7781.1 | 7776.4 | +0.1 % | [-0.6 %, +0.6 %] | 4/8 | within noise |
Control: the baseline's own binary again; its intervals reach ±2.0 %, the resolution of this run.
6 comparisons: at 95 % about one interval in twenty excludes zero by chance.
