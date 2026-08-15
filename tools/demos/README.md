# Demo recording pipeline (issue #218)

The take scripts that produce the recordings in
[`docs/DEMOS.md`](../../docs/DEMOS.md). The originals lived in `/tmp` on the
reference board and were lost; these are faithful reconstructions of the
method documented in DEMOS.md and the PR #214 commit message. Run them on a
Raspberry Pi 5 with the release binary (`geist-bitnet`) in this directory or
pointed to via `BIN=`.

## Prerequisites

```bash
sudo apt install asciinema ffmpeg
cargo install --git https://github.com/asciinema/agg   # cast -> gif renderer
```

`take_bitnet.sh` additionally needs a bitnet.cpp build at the pinned baseline
(clang, not gcc — gcc-12 ICEs on their `quantize_i2_s`):

```bash
git clone --recursive https://github.com/microsoft/BitNet && cd BitNet
git checkout 404980e   # June 2025: current main mis-decodes i2_s on ARM (????)
# build per their README with clang, then point BITNET_CLI at build/bin/llama-cli
```

## The pipeline

Every take follows the same shape; the comparison takes enforce all of it,
the use-case takes skip the thermal gate (no speed claim in frame):

1. **Cool-down gate** — wait until the board is ≤ 57 °C
   (`vcgencmd measure_temp`), so no take starts from a hot chip.
2. **Pre-warm the engine under test** — critical on a 4 GB board: the other
   engine's 1.2 GB mmap evicts your pages and halves tok/s on the first run.
   One short throwaway generation re-pages the weights.
3. **`asciinema rec`** — the recorded shell prints temperature and engine
   version *in frame*, runs the generation, prints the temperature again.
4. **Patch the cast height** — the first line of a `.cast` file is JSON;
   set the same `"height"` on both comparison casts so the composite lines up:

   ```bash
   sed -i '1s/"height": [0-9]*/"height": 24/' take_geist.cast take_bitnet.cast
   ```

5. **Render** each cast to a GIF:

   ```bash
   agg take_geist.cast take_geist.gif
   agg take_bitnet.cast take_bitnet.gif
   ```

6. **Composite** the comparison side by side:

   ```bash
   ffmpeg -i take_geist.gif -i take_bitnet.gif \
     -filter_complex "hstack=inputs=2" ../../assets/versus-bitnetcpp.gif
   ```

   The use-case takes (`take_offline.sh`, `take_writer.sh`) are single-pane:
   `agg` output goes straight to `assets/demo-offline-box.gif` /
   `assets/demo-writing.gif`.

## Fairness controls (comparison takes)

Fixed in the scripts, all visible in the recorded frame: byte-identical GGUF
(`ggml-model-i2_s.gguf`), both engines greedy, 4 threads, the same prompt
(`PROMPT=` — use the one shown in the shipped recording), 110 new tokens,
recorded **sequentially** (running both at once would split the 4 cores and
slow both).
