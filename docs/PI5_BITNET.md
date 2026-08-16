# BitNet on a Raspberry Pi 5 — user guide

Everything about running the self-contained BitNet binary (release asset
`geist-bitnet-linux-arm64` — renamed to plain `geist-bitnet` on download below)
from the [release page](https://github.com/geisten/geistlib/releases/latest)
on a Raspberry Pi 5. Every number on this page was measured on the reference
board described below; nothing is extrapolated.

## Tested configuration

| | |
| :-- | :-- |
| Board | Raspberry Pi 5 Model B Rev 1.1, **4 GB RAM** |
| OS | Raspberry Pi OS 64-bit (Debian 12 "bookworm") |
| Kernels seen in testing | `6.12.20+rpt-rpi-2712`, `6.18.33+rpt-rpi-2712` |
| Cooling | official active cooler (pwm-fan) |
| Storage | SD card (the cold-start numbers below are SD-bound) |

Any 64-bit OS on a Pi 5 should behave the same (the binary is fully static —
musl, no shared libraries, no glibc version to match). A Pi 4 will run it too,
slower; 32-bit OS images will not (see [Common errors](#common-errors)).
8 GB boards simply have more headroom.

## Disk and RAM

- **Disk:** 1.2 GB for the binary itself — that's model plus engine, there are
  no other files. Keep ~1.3 GB free for the download.
- **RAM:** peak resident memory measured **~1.7 GB** while generating. The
  weights are demand-paged read-only out of the binary (zero-copy), so those
  pages are file-backed and the kernel can evict them under pressure — a 4 GB
  board with a desktop session running still fits comfortably. No swap needed.

## Cooling and thermal throttling

The reference numbers were measured with the official active cooler, board
temperature 52–58 °C, and `vcgencmd get_throttled` reporting `0x0`
(never throttled). A passively cooled or bare Pi 5 will throttle under
sustained generation and lose throughput. Check yours:

```bash
vcgencmd measure_temp && vcgencmd get_throttled   # 0x0 = never throttled
```

## Install, update, uninstall

```bash
# install (the asset carries a platform suffix; your copy doesn't have to)
curl -LO https://github.com/geisten/geistlib/releases/latest/download/geist-bitnet-linux-arm64

# verify (optional, recommended — BEFORE renaming, so the checksum file matches)
curl -LO https://github.com/geisten/geistlib/releases/latest/download/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS

mv geist-bitnet-linux-arm64 geist-bitnet && chmod +x geist-bitnet

# update: download the new release the same way — each release is one file
# uninstall: it wrote nothing anywhere else — no config, no cache, no state
rm geist-bitnet
```

The download is ~1.2 GB (measured ~46 s on the reference board's connection —
yours will vary).

## Usage

```bash
./geist-bitnet "The capital of France is"        # one shot
./geist-bitnet "Explain mmap briefly:" 128       # cap new tokens
./geist-bitnet                                   # interactive REPL
printf 'prompt one\nprompt two\n' | ./geist-bitnet - 64   # batch
./geist-bitnet "Write a haiku:" -t 0.8           # sampled, varies per run
```

| Argument | Meaning |
| :-- | :-- |
| 1st | the prompt; `-` reads prompts line-by-line from stdin; absent on a terminal → REPL |
| 2nd | max new tokens (default 256; `-1` = until the context fills; generation always stops at the model's EOS token) |
| `-t`/`--temperature <float>` | any position; `0` (default) = greedy, bit-identical runs; `>0` = softmax sampling with a fresh seed per run |

**The REPL is deliberately memory-less.** Each line is an independent
completion — the model stays loaded (that is the expensive part), but there is
no conversation state between lines. geistlib applies no chat template, so the
model sees your raw text as a document to continue, not as a question to
answer. Phrase prompts as text to be continued ("The three largest moons of
Jupiter are") rather than questions, and you get much better output.

Performance knobs (environment variables): `OMP_WAIT_POLICY=active` keeps the
OpenMP threads hot between tokens and is worth setting for every run; the rest
(thread counts, prefetch) are documented in
[`QUICKSTART.md`](QUICKSTART.md#4-performance-knobs).

## What to expect: startup and speed

Measured on the tested configuration:

| | |
| :-- | :-- |
| First run after boot (cold page cache) | **~14 s** to the first token — the SD card pages 1.1 GB of weights into memory once |
| Every run after that (warm) | **~0.6 s** to the first token |
| Generation | **15–18 tokens/s** depending on context length (17.9 t/s short prompts, 15.0 t/s at 512 tokens — [methodology](../benchmark/README.md)) |

The cold start is storage-bound: an NVMe HAT makes it near-instant, a slow SD
card makes it slower. Warm starts stay fast until the kernel evicts the pages
(reboot, heavy memory pressure).

## Offline behavior

After the download, the binary makes **no network access of any kind** — no
telemetry, no update checks, no model fetches. Prompts and output never leave
the device. Verify it yourself: run it with the network cable pulled, or under
`strace -f -e trace=network`.

## Common errors

| Symptom | Cause | Fix |
| :-- | :-- | :-- |
| `cannot execute binary file: Exec format error` | 32-bit OS or non-arm64 machine | `uname -m` must say `aarch64`. Install the 64-bit Raspberry Pi OS; on x86 boxes build from source instead |
| `Permission denied` despite `chmod +x` | filesystem mounted `noexec` (some `/tmp` setups, USB sticks, network mounts) | move the binary to your home directory: `mv geist-bitnet ~/` |
| Process killed mid-run, `Killed` in dmesg | out of memory (other big processes competing) | close memory-heavy apps; check `free -h`; the binary itself needs no swap on an otherwise idle 4 GB board |
| `embedded model load failed` at startup | truncated or corrupted download | re-download and verify: `sha256sum -c --ignore-missing SHA256SUMS` |
| Output stops after very few tokens | the model emitted its EOS token — that is normal | phrase the prompt as text to continue, or raise the token cap |

## Model limits and responsible use

The embedded model is **BitNet b1.58 2B-4T** — a 2-billion-parameter ternary
model. Small models are useful and fast, and they confabulate freely. A real
example from this page's own test run: prompted with "The moon is made of",
the model fluently generated *"60% oxygen, 30% silicon, and 10% iron"* —
confident, specific, and wrong.

- **Do not rely on it for facts**, and especially not for medical, legal or
  financial decisions. Treat output as plausible-sounding text, not knowledge.
- **No conversation memory, no instruction tuning applied**: this CLI runs raw
  completions without a chat template, so "instructions" work only as far as
  text-continuation carries them.
- **Context window:** 4096 tokens. Long prompts crowd out generation space.
- The model reflects the biases and gaps of its training data; outputs may be
  offensive or subtly skewed. You are responsible for how you use them.

Microsoft's [model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
documents the intended use, evaluation and limitations in full.

## Attribution and licenses

| Component | Origin | License |
| :-- | :-- | :-- |
| BitNet b1.58 2B-4T weights (embedded) | [microsoft/bitnet-b1.58-2B-4T-gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf), © Microsoft | MIT |
| geistlib engine + CLI | [geisten/geistlib](https://github.com/geisten/geistlib) | Apache-2.0 ([LICENSE](../LICENSE), [NOTICE](../NOTICE)) |

The binary redistributes the unmodified GGUF weights under the MIT license;
the MIT notice travels with this repository's NOTICE file. BitNet is described
in [Ma et al., "The Era of 1-bit LLMs"](https://arxiv.org/abs/2402.17764) and
the [BitNet b1.58 2B-4T technical report](https://arxiv.org/abs/2504.12285).
