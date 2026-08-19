# Voice quickstart — push-to-talk on a Raspberry Pi 5 and macOS

Speak into a microphone, get a spoken-to answer from Gemma 4 running
locally — no cloud, no Python, no audio library. The pattern is
`examples/push_to_talk.c` (~200 lines of public API): a capture tool
pipes raw 16 kHz mono s16le PCM into the program, an energy VAD segments
utterances, each utterance becomes one audio chat turn.

## What you need

```sh
make                     # library + examples' prerequisites
make fetch-model         # Gemma 4 E2B-it Q4_K_M (~3.1 GB)
make fetch-audio-tower   # audio tower (~590 MB, SHA-pinned Range download)
make -C examples         # builds examples/push_to_talk
```

The audio tower + `audio_test_data/mel_constants.bin` (checked in) are
found relative to the working directory — **run from the repo root**, or
point `GEIST_AUDIO_MODEL_PATH` / `GEIST_MEL_CONSTANTS_PATH` anywhere.

Sanity check without any microphone (pipe a WAV + a second of silence):

```sh
python3 tools/gen_test_wav.py audio_test_data/smoke.wav 2
(tail -c +45 audio_test_data/smoke.wav; dd if=/dev/zero bs=32000 count=1) | \
    examples/push_to_talk gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

## Raspberry Pi 5

The Pi 5 has **no analog audio input** — use USB. Any USB microphone (or
a USB conference speakerphone; see below) shows up as an ALSA device with
no driver work:

```sh
arecord -l                          # find the card, e.g. card 1, device 0
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -t raw | \
    examples/push_to_talk gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

`plughw` lets ALSA resample if the mic doesn't do 16 kHz natively.
Expectations on a 4 GB Pi 5 (measured, `benchmark/results/PI5-audio.md`):
~2 s attach for a ~3 s utterance, replies at ~7 tok/s. Transcription
quality is ASR-grade since the #270 injection fix: **4.2 % aggregate WER**
on the LibriSpeech harness set (median 0 %, 16/30 clips verbatim). W8A8
attention/LConv is the default since its quality gates went green
(−38 % encode; 4.3 % WER vs 4.2 % FP32 — inside noise);
`GEIST_AUDIO_ATTN_W8A8=0 GEIST_AUDIO_LCONV_W8A8=0` opts back to the
high-precision path.

Hardware advice: for an assistant that also *speaks*, prefer a USB
speakerphone (Jabra Speak, Anker PowerConf class) — the built-in hardware
echo cancellation is what lets the device listen while it plays audio.
For an embedded form factor, an I2S mic HAT (ReSpeaker 2-Mic) works but
needs a device-tree overlay. And under sustained load the Pi 5 wants
active cooling — the passively cooled board throttles at ~80 °C.

## macOS

No `arecord` on macOS; ffmpeg's avfoundation input is the equivalent:

```sh
ffmpeg -f avfoundation -list_devices true -i ""   # the trailing error is normal
ffmpeg -hide_banner -loglevel error -f avfoundation -i ":1" \
       -ar 16000 -ac 1 -f s16le - | \
    examples/push_to_talk gguf_artifacts/gemma4-e2b-Q4_K_M.gguf 1200
```

- `":1"` is the built-in mic on a typical MacBook — pick your index from
  the device list.
- macOS asks for microphone permission for your terminal on first use.
- `brew install sox` + `rec -q -t raw -r 16000 -e signed -b 16 -c 1 -`
  is the lighter alternative.

On Apple Silicon expect ~0.8 s attach and ~27 tok/s replies (M-class).

## Calibrating the VAD

The second CLI argument is the RMS threshold (default 300). Every
room/mic/gain combination needs the knob:

- **Never triggers while you speak** → lower it.
- **Triggers by itself** (fan, keyboard) → raise it. A MacBook mic idles
  at ~800 frame RMS, so 1200 is a good start there; a quiet USB mic on a
  Pi is usually fine near the default.
- An utterance ends after **0.8 s of silence**; utterances are capped at
  28 s (the encoder buffers at most 30 s).

## Prompting patterns

`GEIST_PTT_PROMPT` sets the per-utterance instruction — it changes what
the loop *is*:

| Goal | Prompt |
| :-- | :-- |
| assistant | `Answer the speaker briefly.` (default) |
| dictation | `Transcribe this audio.` — best measured (4.2 % WER on the LibriSpeech harness set); wording matters mildly since #270 (instruction-heavy default: 6.1 %), see `PI5-audio.md` |
| command recognition | few-shot anchor: `The user gave a smart home command. Common commands: 'Lampe an', 'Licht aus', 'Musik an'. Which command did you hear?` |

Short commands (< 1 s) need the few-shot vocabulary anchor — without it
Gemma 4 E2B falls back to "I don't know" (analysis in
`docs/audio-chunk-streaming/short-command-analysis.md`).

## System-wide dictation

`examples/dictate` is the dictation core: same VAD and streaming turn as
push_to_talk, but each utterance becomes ONE LINE of clean transcript on
stdout (status stays on stderr), with the anti-loop guard built in. That
makes it a pipeline stage — the OS typing integration stays out of tree:

```sh
# Wayland (wtype from your distro's repos)
arecord -f S16_LE -r 16000 -c 1 -t raw | \
    examples/dictate gguf_artifacts/gemma4-e2b-Q4_K_M.gguf | wtype -

# Wayland without compositor virtual-keyboard support: ydotool
... | while IFS= read -r line; do ydotool type -- "$line "; done

# X11
... | while IFS= read -r line; do xdotool type --clearmodifiers -- "$line "; done
```

- `GEIST_DICTATE_PROMPT` overrides the instruction; the default
  `Transcribe this audio.` measures 4.2 % WER on the English LibriSpeech
  harness set. Punctuation comes out naturally (commas, periods, casing).
- The model transcribes the language it hears — German measures **7.1 %
  WER** (FLEURS de_de, 30 clips) with the same default prompt; a German
  prompt buys nothing (7.9 %). Details in `PI5-audio.md`.
- Utterances are typed after end-of-speech (0.8 s silence), not
  word-by-word — the streaming session keeps that tail short.

## What runs where

Everything here runs on the **CPU backends** (`cpu_neon` on ARM with
runtime-probed DOTPROD kernels, AVX-512 VNNI on capable x86). On Apple
Silicon the dense fp32 matmuls go through Accelerate (the AMX matrix
unit); the experimental Metal/Vulkan GPU backends are not used by the
audio path. Latency after end-of-speech is currently utterance-level
(`attach_audio` re-encodes the whole clip); #256 tracks exposing the
internal streaming encoder (measured 202 ms tail on the Pi) through the
public API.

## Troubleshooting

- `this model instance cannot hear` — the audio tower wasn't found:
  run from the repo root, or set `GEIST_AUDIO_MODEL_PATH` and
  `GEIST_MEL_CONSTANTS_PATH`. Only Gemma 4 GGUFs load the tower
  (`geist_model_modalities()` is the programmatic check).
- Garbled or truncated replies on very long utterances — the 28 s cap;
  split your input at pauses.
- Slow on the Pi with other workloads running — timing collapses under
  memory pressure (3.1 GB model on a 4 GB board); keep the board to one
  job at a time.
