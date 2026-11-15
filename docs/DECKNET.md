# decknet: a hand-written convolutional net for images (experimental)

decknet is the geistbox deck-check network (#532): two camera images in (the
deck now and the empty deck), one small state vector per deck position out.
It is a fixed convolutional net whose weights were written by hand from the
deck geometry; nothing is trained. It is the library's first model that is
not a language model: no tokenizer, no session, no backend handle.

Build with `make DECKNET=1`. The default build (`DECKNET=0`) compiles none of
it: `libgeist.a`, `libgeist.so`/`.dylib` and every binary are byte for byte
what they are without the arch, and the `geist_decknet_*` symbols do not
exist. A program that includes `<geist_decknet.h>` links against a
`DECKNET=1` library. `make test-decknet` builds it into the same output
directories, runs its tests and its fuzz driver, and goes back;
`make MODE=asan test-decknet` does the same with sanitizers.

## API

`include/geist_decknet.h`, all EXPERIMENTAL:

```c
#include <geist_decknet.h>

struct geist_decknet *d = nullptr;
if (geist_decknet_load("deck.gguf", &d) != GEIST_OK) {
    fprintf(stderr, "%s\n", geist_last_create_error());
    return 1;
}
struct geist_decknet_info info;
(void) geist_decknet_info(d, &info); /* slots, image size, counts, ... */

/* cur, ref: RGB8, info.image_width x info.image_height, rows top to bottom */
struct geist_decknet_result res;
enum geist_status st = geist_decknet_run(d, n_bytes, cur, ref, &res);
if (st == GEIST_OK && res.checkable) {
    for (size_t s = 0; s < info.n_slots; s++) {
        struct geist_decknet_slot slot;
        (void) geist_decknet_slot(d, s, &slot);
        /* slot.occ_rear, slot.rack, slot.tips[0 .. info.n_tips), ... */
    }
}
geist_decknet_destroy(d);
```

| function | what it does | statuses besides `GEIST_OK` |
| :-- | :-- | :-- |
| `geist_decknet_load(path, &d)` | validates and copies the file, allocates the whole workspace | `GEIST_E_IO` (cannot open or parse as GGUF), `GEIST_E_UNSUPPORTED` (another `general.architecture`), `GEIST_E_FORMAT` (breaks the contract below), `GEIST_E_OOM`, `GEIST_E_INVALID_ARG` |
| `geist_decknet_load_from_memory(size, data, &d)` | the same from bytes, which may be released on return | as above; bytes that are not a GGUF are `GEIST_E_FORMAT` |
| `geist_decknet_info(d, &info)` | slots, image size, counts, plane heights, the head thresholds `occ` and `smooth` | `GEIST_E_INVALID_ARG` |
| `geist_decknet_set_homographies(d, n, image, h)` | replaces all homographies of the current or the empty-deck image | `GEIST_E_INVALID_ARG` (`n` is not `n_homographies`, a value is not finite); nothing is replaced then |
| `geist_decknet_run(d, n_bytes, cur, ref, &res)` | one forward pass; `res.checkable` says whether the pair could be compared | `GEIST_E_INVALID_ARG` (null, or `n_bytes` is not width x height x 3) |
| `geist_decknet_slot(d, i, &slot)` | the units of slot `i` from the last run | `GEIST_E_INVALID_STATE` (no run yet, or the last pair was not checkable), `GEIST_E_INVALID_ARG` |
| `geist_decknet_destroy(d)` | frees the handle | |

A load failure leaves the handle `nullptr` and its reason in
`geist_last_create_error()`, naming the key, the tensor and the row.
One handle is used by one thread at a time; different handles are
independent. No call after the load allocates
(`tests/test_decknet_unit.c` counts).

A pair that is **not checkable** (the brightness of the two images on the
bare worktable differs by more than the file allows, or the bare worktable
itself looks changed) has no slot state: `geist_decknet_slot` returns
`GEIST_E_INVALID_STATE` rather than zeros, which would read as an empty deck.

The head (thresholds `occ` and `smooth`, argmax over the type units) stays
with the caller; `geist_decknet_info` hands out the two thresholds of the
file. `geist_model_load` refuses a decknet file with `GEIST_E_UNSUPPORTED`,
as before.

### Calibration without a new file

The 31 homographies follow from the camera pose, which a device measures
per run from its fiducials. `geist_decknet_set_homographies` replaces them
for one image: `n * 9` floats, row-major 3 x 3 each, in the file's order
(`[0]` the bare worktable strip, then the five planes of each slot). Each
maps a crop pixel `(column, row, 1)` to a camera pixel `(u, v, w)`. The
current and the empty-deck image have a set each, because the two were taken
from two poses; both start as the file's. `info.planes_mm` are the plane
heights the caller's projection needs.

Finding the fiducials, the camera pose and lens undistortion are the
caller's.

## Stream

A camera delivers frame after frame from one pose, against one empty deck,
and usually as YUV. `geist_decknet_run` pays for all of that again with each
call: the homography arithmetic of every crop pixel, the resampling of the
empty deck, and, because it takes RGB8, a colour conversion of the whole
frame first. A stream (#533) keeps what does not change between frames:

```c
struct geist_decknet_stream *s = nullptr;
if (geist_decknet_stream_create(d, &s) != GEIST_OK) { /* GEIST_E_OOM */ }

struct geist_decknet_frame f = {
        .format = GEIST_DECKNET_FORMAT_NV12, /* or _RGB8, _YUYV */
        .width = info.image_width, .height = info.image_height,
        .stride = stride, .n_bytes = n_bytes, .data = empty_deck};
(void) geist_decknet_stream_set_reference(s, &f); /* once, and after new homographies */

for (;;) {                                        /* per camera frame */
    f.data      = frame;                          /* the capture buffer itself */
    f.timestamp = t;
    if (geist_decknet_stream_push_frame(s, &f) != GEIST_OK) { break; }
    struct geist_decknet_stream_result res;
    (void) geist_decknet_stream_result(s, &res);  /* res.image, res.timestamp, res.n_frames */
    if (res.image.checkable) {
        struct geist_decknet_slot slot;
        (void) geist_decknet_stream_slot(s, 0, &slot);
    }
}
geist_decknet_stream_destroy(s);
```

| function | what it does | statuses besides `GEIST_OK` |
| :-- | :-- | :-- |
| `geist_decknet_stream_create(d, &s)` | a stream on the handle `d`, which it borrows; allocates everything the stream will ever need | `GEIST_E_OOM`, `GEIST_E_INVALID_ARG` |
| `geist_decknet_stream_set_reference(s, &frame)` | the empty deck, in any format; takes over the handle's homographies as they are now | `GEIST_E_INVALID_ARG` (null, or a frame that is refused, see below); the reference it had stays |
| `geist_decknet_stream_push_frame(s, &frame)` | one forward pass on the frame against the reference; returns when it is done | `GEIST_E_INVALID_ARG` (null, or a frame that is refused): nothing is read, the last result stays. `GEIST_E_INVALID_STATE`: no reference yet, or the homographies were replaced since |
| `geist_decknet_stream_result(s, &res)` | the image check of the last frame, its timestamp, the number of frames so far | `GEIST_E_INVALID_STATE` before the first frame, `GEIST_E_INVALID_ARG` |
| `geist_decknet_stream_slot(s, i, &slot)` | the units of slot `i` from the last frame | as `geist_decknet_slot` |
| `geist_decknet_stream_destroy(s)` | frees the stream; the handle stays | |

The stream computes in the handle's workspace: `d` outlives the stream, and
while the stream exists `geist_decknet_run` is not called on `d`. One stream
is used by one thread at a time. After the create nothing allocates: not the
reference, not the first frame, not a later one.

### Frames

A frame is the caller's memory and is read in place, during the call, and
never kept: a capture buffer can be handed over as it is. Rows run top to
bottom, `stride` bytes apart.

| format | layout | size |
| :-- | :-- | :-- |
| `GEIST_DECKNET_FORMAT_RGB8` | 3 bytes per pixel, R G B | any |
| `GEIST_DECKNET_FORMAT_NV12` | a plane of Y, then at byte `stride * height` a plane of U V pairs, one pair per 2 x 2 pixels, its rows `stride` apart as well | width and height even |
| `GEIST_DECKNET_FORMAT_YUYV` | 4 bytes per 2 pixels: Y0 U Y1 V | width even |

Width and height are the model's. `n_bytes` is the extent of `data`, and
nothing at or behind `data + n_bytes` is read: a frame is refused with
`GEIST_E_INVALID_ARG` when its format is not one of the three, its size is
not the model's (or odd where the format needs it even), its stride is
shorter than a row, or `n_bytes` is smaller than where its last row ends
(the last row needs no padding). The check is done once per frame, with
checked arithmetic; the samplers then index the buffer through tables whose
coordinates are clamped to the image. MJPEG is decoded by the caller.

NV12 and YUYV are read as BT.601 limited range, the chroma of a pixel taken
from its block without interpolation. In integers, for the bytes `Y`, `U`,
`V` of a pixel:

```
c = 298 * (Y - 16) + 128
R = clamp((c + 409 * (V - 128)) >> 8)
G = clamp((c - 100 * (U - 128) - 208 * (V - 128)) >> 8)
B = clamp((c + 516 * (U - 128)) >> 8)          clamp to 0..255
```

A camera that delivers full-range or BT.709 YUV needs another matrix; there
is one because there is one user. Converting in the caller and pushing RGB8
is always possible.

### What it gives, and what it saves

**The result is that of `geist_decknet_run` on the same two images, bit for
bit**: for RGB8 as they are, for NV12 and YUYV as converted to RGB8 by the
formula above. Nothing is approximated; every float the pass reads comes
from the expression `run.c` uses. `tests/test_decknet_stream_unit.c` compares
every output of every slot with `memcmp`, and `tools/bench_stream` hashes
every output of every frame it times and fails on a difference.

What a stream does not repeat (`src/archs/decknet/stream.c`):

- **Sample tables.** Per homography, where each crop pixel reads the frame
  and with which fractions, computed once (when the reference is set)
  instead of two F64 divisions per sample and frame.
- **Region of interest.** Tables hold only the crop pixels the pass reads:
  the pixels of the pooling rows on the floor plane, the kernel windows
  around the pixels a convolution is evaluated at on the others. Nothing
  else of a frame is touched.
- **In place.** The 2 x 2 camera pixels of a sample are read in the frame's
  format; an NV12 or YUYV frame is never converted as a whole.
- **The empty deck** is sampled once, when it is set.

The reference also fixes the homographies. After
`geist_decknet_set_homographies` the tables are those of the old pose, so
`push_frame` returns `GEIST_E_INVALID_STATE` until the reference is set
again (which rebuilds them, without allocating).

### Measured

`tools/bench_stream` feeds frames from memory (a static scene with a block
moving over it) at 1280 x 720 and 1920 x 1080, through `geist_decknet_run`
once per frame ("still"; an NV12 or YUYV frame converted whole first, the
empty deck converted once outside the timing) and through a stream. One
thread. Its network is invented in the bench, in the shape of a deck check:
six slots, 31 homographies, a 256 x 192 crop, 96 raster units with disc and
ring pooling, 9 x 9 and 13 x 13 kernels, two region pools over most of the
floor crop. **How much of a crop a network reads decides what the region of
interest saves, so these are that network's numbers, not a device's**: it
samples 1.78 M crop pixels per frame on the still path and 0.88 M in a
stream.

Raspberry Pi 5 (gcc 14.2, Linux 6.18, board below 52 °C, under the host
lock), 300 frames per row, times per frame in ms:

| size | format | still frames/s | p50 | p99 | stream frames/s | p50 | p99 | stream / still |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| 1280 x 720 | RGB8 | 9.5 | 105.2 | 106.2 | 23.5 | 42.4 | 42.7 | 2.48 |
| 1280 x 720 | NV12 | 9.1 | 109.3 | 109.9 | 19.1 | 52.3 | 52.6 | 2.09 |
| 1280 x 720 | YUYV | 9.1 | 109.5 | 109.9 | 20.0 | 49.9 | 50.1 | 2.19 |
| 1920 x 1080 | RGB8 | 9.4 | 106.5 | 107.2 | 23.1 | 43.3 | 44.1 | 2.46 |
| 1920 x 1080 | NV12 | 8.7 | 115.4 | 115.8 | 19.0 | 52.7 | 53.1 | 2.19 |
| 1920 x 1080 | YUYV | 8.6 | 115.8 | 116.4 | 19.8 | 50.5 | 50.7 | 2.29 |

The crops are a fixed number of pixels, so the frame size matters only
through the memory the samples are spread over, and through the whole-frame
conversion the still path needs for YUV.

What each part contributes, same board, 1280 x 720, p50 in ms (100 frames;
the intermediate rows were measured with a switch that is not in the tree):

| | RGB8 | NV12 | YUYV |
| :-- | --: | --: | --: |
| still path per frame | 105.2 | 109.3 | 109.5 |
| stream: frame read in place, empty deck sampled once | 86.3 | 102.3 | 98.3 |
| + sample tables | 50.0 | 66.7 | 62.3 |
| + region of interest (the tree) | 42.4 | 52.3 | 49.9 |
| region of interest without tables, for comparison | 64.2 | 73.8 | 71.4 |

Reading YUV in place alone gains little: a bilinear sample converts four
pixels, so the 1.48 M samples of the current image convert 5.9 M pixels
where a whole 720p frame has 0.9 M. It pays once the tables and the region
are there.

Mac M1 Max (Apple clang 21, under the host lock), the same run. The machine
was on battery in low-power mode with other jobs running (load average 4 to
5), so take these as an order of magnitude; the outputs are the same bits
as on the Pi (the bench prints the same hashes):

| size | format | still frames/s | p50 | p99 | stream frames/s | p50 | p99 | stream / still |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| 1280 x 720 | RGB8 | 21.9 | 44.5 | 87.3 | 56.0 | 17.8 | 20.4 | 2.55 |
| 1280 x 720 | NV12 | 21.2 | 46.6 | 53.5 | 42.8 | 23.3 | 24.2 | 2.01 |
| 1280 x 720 | YUYV | 21.2 | 46.8 | 52.0 | 43.6 | 22.8 | 24.8 | 2.06 |
| 1920 x 1080 | RGB8 | 22.1 | 44.6 | 49.9 | 55.0 | 18.0 | 22.3 | 2.48 |
| 1920 x 1080 | NV12 | 20.1 | 49.7 | 54.1 | 41.4 | 23.7 | 29.2 | 2.06 |
| 1920 x 1080 | YUYV | 19.8 | 50.0 | 55.9 | 42.6 | 23.3 | 31.3 | 2.15 |

The target of #533 (a Pi 5, 1920 x 1080 at 60 frames per second, p99 below
one frame period of 16.7 ms) is **not met** by this step: 43 ms for RGB8,
53 ms for NV12.

```sh
make DECKNET=1 bin/<target>/release/tools/bench_stream
~/bin/with-mac-lock bin/<target>/release/tools/bench_stream [--frames N] [--only still|stream]
```

Not in this step, and why the frame rate is where it is: with the sampling
cut down, most of a stream's frame is the pass itself, the F64 kernels and
the pooling, which a stream runs exactly as the still path does (on the Mac,
with the pass switched off, sampling was 5.5 of the 18 ms of an RGB8 frame
and 11 of the 23 ms of an NV12 frame). Temporal
gating (unchanged tiles reuse their results), integer kernels, a queue that
drops the older frame instead of blocking the capture thread, and a GPU path
are the next steps of #533; none of them is here.

## File contract

GGUF v3, `general.architecture = "decknet"`. The loader refuses everything
that is not exactly this, including a file with a 14th tensor: a newer
network read by an older loader would otherwise answer with a layer missing.

Metadata (`decknet.*`):

| key | type | constraint |
| :-- | :-- | :-- |
| `slots` | string | 1 to 64 printable letters, one per deck position |
| `image.width`, `image.height` | u32 | 2 to 32768 |
| `crop.width`, `crop.height` | u32 | at least 1, at most 2^22 pixels together |
| `bg.width`, `bg.height` | u32 | at least 1, at most 2^22 pixels together |
| `t_change` `g_change` `t_wall` `g_wall` `t_tip` `t_dark` `t_tube` `g_unit` `t_white` `g_white` `gain_min` `gain_max` `bg_rest_max` `occ` `smooth` | f32 | finite |

Tensors, all F32 and all finite, exactly these 13 (numpy shapes; GGUF lists
the dimensions reversed):

| name | shape | constraint |
| :-- | :-- | :-- |
| `decknet.homography` | (1 + 5 x slots) x 3 x 3 | |
| `decknet.planes_mm` | 5 | floor, plate, deep-well, hole, tip |
| `decknet.bg_nominal` | 3 | |
| `decknet.conv.blob`, `decknet.conv.tube` | k x k | k odd, at most 255 |
| `decknet.change.w`, `decknet.wall.w` | rows x 6 | 1 to 64 rows |
| `decknet.pool.{on,off,tube,wall,occ_front,occ_rear}` | rows x 3 | see below |

A pooling row is `(unit, pixel, weight)`: the indices travel as F32 and must
be integers, `pixel` inside the crop. The rows are ascending by
`(unit, pixel)` starting at unit 0 with no unit left out, as the exporter
writes them, which excludes a pixel listed twice and a unit that pools
nothing. `pool.on` and `pool.off` have the same number of units (the raster,
96), `pool.tube` an even number (two orientations, 8 each), the three region
pools one unit each.

## Forward pass

`sat(x) = relu(x) - relu(x - 1)`. The numpy reference is geistbox's
`software/geistbox/decknet.py` (`warp`, `normalise`, `forward`);
`src/archs/decknet/run.c` follows it operation by operation.

1. **Resample.** Bilinear sampling of the camera image through a homography
   (`u = p0 / p2 - 0.5`, clamped to the image) into a crop, float 0..1. The
   coordinates are computed in F64, as the reference does. A homography that
   divides by zero samples the image corner; it does not fail.
2. **Gain.** On the bare-worktable crop, `g = bg_nominal / max(mean, 1e-3)`
   per colour for each image; each image is multiplied by its gain. The pair
   is checkable when the ratio of the two gains stays inside
   `(gain_min, gain_max)` and the mean changed fraction of the strip is below
   `bg_rest_max`. The mean is an F32 running sum in pixel order, because the
   reference's is (see "Parity").
3. **Change, wall** on the floor plane:
   `sat(g_change * (sum(relu(change.w . [cur, ref])) - t_change))`, pooled by
   `occ_front` and `occ_rear`; the same form with `wall.w`, pooled by `wall`.
4. **Blob energy** on the plate, deep-well, hole and tip planes: each colour
   convolved with `conv.blob` ("same" size, edge pixels repeated);
   `bright = sum relu(d)`, `dark = sum relu(-d)`.
5. **Units** per raster position:
   `sat(g_unit * (pool.on(e) - pool.off(e) - t))`: hole (hole plane, dark,
   `t_dark`), plate and deep likewise; `white = sat(g_white *
   (pool.on(min(R, G, B)) - t_white))` on the tip plane;
   `tip = relu(unit(tip plane, bright, t_tip) + white - 1 - hole)`.
6. **Tubes:** `sat(g_unit * (pool.tube(sum relu(conv(hole plane, conv.tube)))
   - t_tube))`.

`rack` is the mean of `max(tip, hole)`, `plate` and `deep` the means of their
units, `tube_n` and `tube_r` the means of the two tube orientations.

The kernels are applied as a convolution (kernel flipped), which is what the
reference's FFT computes; the shipped kernels are point-symmetric, so a
correlation gives the same. They are evaluated only at the crop pixels a
pooling row reads (about 1 in 16 for the device's file); the pooled values
are those of a full map.

## Where it lives

- `src/archs/decknet/load.c`: the file contract, `run.c`: the pass,
  `stream.c`: the same pass fed from a camera stream (sample tables, region
  of interest, frame formats). One
  implementation in portable C, built without `-ffast-math` and without
  contraction. No backend dispatches it, so there is no `cpu_scalar` /
  `cpu_neon` / `cpu_x86` variant to keep in step: the same code runs on
  every platform, and `test_decknet_unit` is its parity check there.
- `src/engine/decknet.c`: the public entry points, and the composition point
  for the arch (`tools/layers.json`).
- decknet is not in `src/engine/arch_registry.c`. That registry answers
  "which decoder, with which towers, does `geist_model_load` build for this
  `general.architecture`", and every reader of a descriptor goes on to a
  model, a tokenizer and a session. decknet has none of them and its own
  entry point, so its one name is checked where it is loaded, fail-closed.
  A second image network is the moment for a table of them.

## Tests

| test | what it proves |
| :-- | :-- |
| `tests/test_decknet_unit.c` | every output of every slot against the numpy reference, within 1e-4, on an invented network everywhere and on the device's network where its fixture exists; replaced homographies; a pair that is not checkable; no allocation per call, the first included; `geist_model_load` refuses the file |
| `tests/test_decknet_malformed_unit.c` | 54 broken files, each refused with the expected status and message; every truncation; every allocation of a load failing once; units computed by hand (kernel orientation, pooling); argument checks |
| `tests/fuzz_decknet.c` | the loader, and the forward pass on whatever the loader accepts, under libFuzzer and a deterministic driver |
| `tests/test_decknet_stream_unit.c` | a stream against the still path, every output bit for bit: RGB8, NV12 and YUYV, padded rows, replaced homographies, across a frame that is not checkable; no allocation after the create, the first frame included; 227 frame descriptions refused, exact-fit frames accepted; state and argument checks; every allocation of a create failing once |
| `tests/fuzz_decknet_stream.c` | frame geometry (format, size, stride, extent) and homographies from the input, on exact-size buffers; the harness decides for itself whether a frame fits and fails on a wrong verdict, not only on a crash |

`test_decknet_unit` compares against the numpy reference on two fixtures of
the same four files (a network, `cur.ppm`, `ref.ppm`, `expected.txt`):

- **`tests/data/decknet_synth/`, in the repository, on every platform.** A
  network whose every number is invented by
  `tests/scripts/decknet_synth_fixture.py`: three slots, a 160 x 120 image, a
  58 x 42 crop, seeded random homographies, kernels (not symmetric), 1x1
  weights and pooling rows, hand-picked thresholds, on an image pair of
  noise and shapes. It describes no device and makes no sense as a deck
  check; it is the loader's and the pass's arithmetic against the
  reference. The reference keeps its network in module globals, so the
  script replaces each one the forward pass reads with the invented value
  and calls the reference's functions unchanged; a global left at its
  original value makes the two sides compute different networks and the
  test fail.
- **`$GEIST_DECKNET_FIXTURE`, where it exists.** The exported network of the
  device with an image pair and the reference's outputs, in the same
  four-file layout; its generator is kept with the device's sources, not
  here. It holds the deck
  geometry, which is confidential, and every branch of this repository is
  mirrored to GitHub, so it is **not in the repository**: it lives on the
  hosts that test it (CI: the repository variable of that name, on the
  macOS jobs), and the test leaves this part out anywhere else.

The stream test uses the invented network and `tests/decknet_tiny.h`; the
other tests build their files in code.

## Parity

Measured maximum `|C - numpy|` over all outputs is about 1e-6 (F32 rounding).

One point needed more than following the formulas. The reference computes
the worktable mean as `a.mean((0, 1))` on an F32 array, which numpy
evaluates as an F32 running sum over the strip's pixels; for the device's
file its rounding moves the mean by about 1e-5 of its value. The gain scales
the whole image, and behind the gain of the white units an exact mean
differs from the reference by up to 4.4e-4 in a unit, above the 1e-4 the
parity test allows. `run.c` therefore sums in F32
in the same order. If the reference changes to an F64 mean, `gain_check`
changes with it (one line) and both fixtures are regenerated.
