# geist Benchmarks — Raspberry Pi 5, audio tower (#234)

Per-stage baseline for the Gemma 4 audio path on the reference board. This is
the number every audio optimization (#235 streaming, #237 x86 W8A8, #238
attention quantization) is judged against.

> The protocol lives in this file and is fully reproducible from a clean
> checkout: fixtures come from `make fetch-audio-tower` (SHA-pinned) and
> `tools/gen_test_wav.py` (deterministic synthesis) — no voice recordings.
> `benchmark/reference_runs.json` is written only by `make bench --record`
> (its own rule), so the audio rows live here instead.

## Setup

- **Board:** Raspberry Pi 5 Model B Rev 1.1, 4× Cortex-A76, 4 GB RAM,
  64-bit Raspberry Pi OS (kernel 6.18.33, Debian Trixie), quiesced
  (load < 0.05), passively started below 55 °C, `vcgencmd get_throttled`
  low nibble 0 throughout (never above 67 °C during the runs).
- **Build:** `make TARGET=pi5` (gcc 14.2), commit `c6b10ea` (main after
  #245 — load-time kernel binding; log line confirms
  `audio_encoder: linear kernels: neon`).
- **Fixtures:** `audio_bench/audio_tower.safetensors`
  (SHA-256 `d6c45a6c…`, via `make fetch-audio-tower`),
  `audio_test_data/mel_constants.bin` (checked in).
- **Clips:** `python3 tools/gen_test_wav.py audio_test_data/clip<N>s.wav <N>`
  for N = 2, 10, 28 (28 stays under the encoder's 30 s buffer limit).
  SHA-256: 2 s `5e9fcb23…`, 10 s `a49db952…`, 28 s `1a9f8105…`.

## Encoder throughput (`bench_audio_encode`, 1 warmup + 10 repeats, median)

| clip | soft tokens | encode (median) | spread | RTF (encode/audio) |
|-----:|------------:|----------------:|-------:|-------------------:|
|  2 s |          51 |          664 ms |  1.6 % |             0.33× |
| 10 s |         251 |        3 596 ms |  1.5 % |             0.36× |
| 28 s |         256¹ |       12 136 ms |  6.6 % |             0.43× |

¹ `geist_session_attach_audio` (and this bench) cap soft tokens at a
hardcoded `max_soft = 256` ≈ 10.2 s of audio — a 28 s clip pays the full
encode cost and silently loses everything past ~10 s. Tracked in #247.

## Where the time goes (`-DGEIST_AUDIO_PROFILE`, shares of pipeline total)

| stage | 2 s clip | 10 s clip | precision today |
|---|---:|---:|---|
| mel + misc | < 1 % | < 1 % | FP32 |
| subsample convs | 21.5 % | 19.3 % | FP32 |
| attention (12×) | **30.0 %** | 25.1 % | W8A32 |
| lconv (12×) | 23.5 % | 18.8 % | FP32 |
| FFN 1+2 (12×) | 22.3 % | 34.7 % | W8A8 |
| output/embed proj | 2.2 % | 1.7 % | W8A32 |

Readings for the open issues:

- **#238 gate: PASSES.** Attention is 25–30 % of encode — well above the
  ≥ 15 % bar the issue set, so W8A8 attention is worth attempting. The
  FP32 lconv (19–24 %) is a second lever of the same size.
- **Mel is free.** < 1 % of the pipeline — incremental mel (#235) buys
  latency overlap, not throughput.
- FFN share grows with clip length (22 → 35 %) — it is the first target
  on long clips, and it is exactly the stage #237 accelerates on x86.

## W8A8 attention / LConv (#238) and live streaming (#235) — measured

Same board/protocol, quiesced re-run (2 s clip, 10 repeats, medians):

| config | encode | vs baseline | RTF |
|---|---:|---:|---:|
| shipping (attn W8A32, lconv FP32) | 665 ms | — | 0.33× |
| `GEIST_AUDIO_ATTN_W8A8=1` | 533 ms | **−20 %** | 0.27× |
| + `GEIST_AUDIO_LCONV_W8A8=1` | 411 ms | **−38 %** | 0.21× |

Quality gates for the quantized configs, all green on this board:
`test_audio_attn_w8a8_parity_int` (soft-token cosine drift mean
1−cos ≈ 9e-4, worst token 0.93) and `test_audio_chat_e2e` **6/6 clips**
with attn+lconv W8A8 — the model still answers the audio correctly.
Defaults stay opt-in per the #238 rollout plan.

Live streaming worker (`GEIST_AUDIO_STREAM=1`, 10 s clip, tail after
end-of-speech):

| config | residual tail | tail / audio |
|---|---:|---:|
| monolithic (encode after end_input) | 3 600 ms | 0.36× |
| streaming worker | **1 091 ms** | **0.11×** |

The worker encodes 48-mel-frame batches while PCM still arrives; output
is bit-identical to the monolithic path
(`test_audio_stream_live_parity_int`, 251 = 251 tokens, max|Δ| = 0).
Tail / full-encode = 0.30 — the #235 target of 0.25 is near but not met;
the remaining tail is the final flush, and the 48-frame kick interval is
the knob to shrink it further.

## Push-to-talk stages (`test_audio_latency_e2e`, 2 s synthetic clip, LM loaded)

Three fresh runs (each loads the 3.1 GB Q4_K_M LM; run 1 pays cold page-in):

| run | attach | template prefill | first token | decode (21 tok) |
|----:|-------:|-----------------:|------------:|----------------:|
| 1 (cold) | 2 313 ms | 846 ms | 168 ms | 3 231 ms |
| 2 | 1 669 ms | 629 ms | 151 ms | 2 984 ms |
| 3 | 1 586 ms | 612 ms | 149 ms | 3 029 ms |

Two observations the encoder-only table cannot show:

- **Memory pressure costs ~2.4×.** Attach (mel+encode+inject) takes
  ~1.6 s warm with the LM resident vs 664 ms encoder-only — on a 4 GB
  board the 3.1 GB LM and the 586 MB tower fight for page cache. Any
  future "keep the tower hot" work should be measured in THIS
  configuration, not standalone.
- **Decode dominates the reply:** ~145 ms/token (≈ 7 t/s), consistent
  with the text benchmarks in `PI5.md`. For short replies the audio
  pipeline is roughly a third of total user-perceived latency.

## Reproduce

```sh
make TARGET=pi5 bin/pi5/release/tests/bench_audio_encode \
                bin/pi5/release/tests/test_audio_latency_e2e
make fetch-audio-tower
python3 tools/gen_test_wav.py audio_test_data/clip2s.wav 2
# warmup, then:
for i in $(seq 10); do
  ./bin/pi5/release/tests/bench_audio_encode audio_test_data/clip2s.wav
done
```

Check `uptime` and the thermal notes in `PI5.md` first; the same two
confounds (background load, heat) apply here.
