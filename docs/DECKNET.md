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

- `src/archs/decknet/load.c`: the file contract, `run.c`: the pass. One
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

The other two tests build their files in code.

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
