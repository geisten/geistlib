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
frame first. A stream (#533) keeps what does not change between frames,
with gating also what did not change in the frame, and with its queue of
one it takes the newest frame of a camera that is faster than the pass:

```c
struct geist_decknet_stream *s = nullptr;
if (geist_decknet_stream_create(d, &s) != GEIST_OK) { /* GEIST_E_OOM */ }

struct geist_decknet_frame f = {
        .format = GEIST_DECKNET_FORMAT_NV12, /* or _RGB8, _YUYV */
        .width = info.image_width, .height = info.image_height,
        .stride = stride, .n_bytes = n_bytes, .data = empty_deck};
(void) geist_decknet_stream_set_reference(s, &f); /* once, and after new homographies */
(void) geist_decknet_stream_set_gate(s, true, 0); /* optional: unchanged tiles are not recomputed */
(void) geist_decknet_stream_set_threads(s, 2);    /* optional: the slots on two threads */

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
| `geist_decknet_stream_set_gate(s, on, threshold)` | temporal gating on or off (off after the create), see "Gating"; the first call that turns it on allocates what it keeps | `GEIST_E_OOM` (the stream stays as it was), `GEIST_E_INVALID_ARG` |
| `geist_decknet_stream_set_threads(s, n)` | the slots of a frame on one thread (after the create) or on two | `GEIST_E_INVALID_ARG` (null, or `n` is not 1 or 2) |
| `geist_decknet_stream_set_reference(s, &frame)` | the empty deck, in any format; takes over the handle's homographies as they are now | `GEIST_E_INVALID_ARG` (null, or a frame that is refused, see below); the reference it had stays |
| `geist_decknet_stream_push_frame(s, &frame)` | one forward pass on the frame against the reference; returns when it is done | `GEIST_E_INVALID_ARG` (null, or a frame that is refused): nothing is read, the last result stays. `GEIST_E_INVALID_STATE`: no reference yet, or the homographies were replaced since |
| `geist_decknet_stream_offer(s, &frame, &dropped)` | from the thread that captures: the frame waits for the next pass; never waits itself. The frame that was waiting, if any, comes back in `*dropped` and is counted. See "A queue of one" | `GEIST_E_INVALID_ARG` (null, or a frame that is refused): nothing is queued |
| `geist_decknet_stream_push_offered(s, &done)` | `push_frame` on the waiting frame, which comes back in `*done`; `GEIST_OK` and `*done == nullptr` when none waits | as `push_frame`; the frame comes back with any status |
| `geist_decknet_stream_result(s, &res)` | the image check of the last frame, its timestamp, the number of frames so far, what the frame cost (the tiles the network reads, how many of them were taken from this frame, the samples computed), and how many offered frames were dropped | `GEIST_E_INVALID_STATE` before the first frame, `GEIST_E_INVALID_ARG` |
| `geist_decknet_stream_slot(s, i, &slot)` | the units of slot `i` from the last frame | as `geist_decknet_slot` |
| `geist_decknet_stream_destroy(s)` | frees the stream; the handle stays | |

The stream computes in the handle's workspace: `d` outlives the stream, and
while the stream exists `geist_decknet_run` is not called on `d`. One stream
is used by one thread at a time; `geist_decknet_stream_offer` is the one
function a second thread may call meanwhile. After the create, and the first
`geist_decknet_stream_set_gate` that turns gating on, nothing allocates: not
the reference, not the first frame, not a later one.

### Frames

A frame is the caller's memory and is read during the call; no pointer into
it is kept (a frame handed to `geist_decknet_stream_offer` is lent until it
comes back, see "A queue of one"), so a capture buffer can be handed over
as it is. Rows run top to bottom, `stride` bytes apart.

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
checked arithmetic; the code behind it then indexes the buffer through
tables whose coordinates are clamped to the image, and by tiles clipped to
it. A sample of an RGB8 frame reads 16 bytes of each of its two rows, of
which the first six are its pixels, where the buffer has them, and the six
alone at the end of a buffer that stops with its last pixel
(`test_decknet_stream_unit` runs both, for every length of what is behind
the frame). MJPEG is decoded by the caller.

A stream divides a frame into **tiles** of 16 x 16 pixels
(`GEIST_DECKNET_TILE`) from its top left corner and touches only the tiles a
sample reads. An RGB8 frame is sampled in place. Of an NV12 or YUYV frame
those tiles are converted to RGB8 first, each pixel once, into memory the
stream owns, and sampled there: a bilinear sample reads four pixels, and a
frame's pixels are read by several samples, so converting per sample (the
first step of #533 did) converts each pixel several times over.

NV12 and YUYV are read as BT.601 limited range, the chroma of a pixel taken
from its block without interpolation. In integers, for the bytes `Y`, `U`,
`V` of a pixel:

```
c = 298 * (Y - 16) + 128
R = clamp((c + 409 * (V - 128)) >> 8)
G = clamp((c - 100 * (U - 128) - 208 * (V - 128)) >> 8)
B = clamp((c + 516 * (U - 128)) >> 8)          clamp to 0..255
```

`stream.c` evaluates this in 16 bits, so that a compiler makes vectors of
it: the multiples of 256 are taken out of the shift (298 = 256 + 42,
409 = 256 + 153, 208 = 256 - 48, 516 = 512 + 4), which leaves sums between
-20128 and 29597. All 2^24 values of (Y, U, V) give the bytes of the
formula above (checked once, exhaustively, outside the tree;
`test_decknet_stream_unit` runs frames of arbitrary bytes against its own
copy of the formula).

A camera that delivers full-range or BT.709 YUV needs another matrix; there
is one because there is one user. Converting in the caller and pushing RGB8
is always possible.

### Gating

With a fixed camera most of a frame is the frame before. A gated stream
(`geist_decknet_stream_set_gate(s, true, threshold)`) keeps the bytes of
the tiles the network reads and compares each tile of a new frame with the
kept one:

- a tile whose bytes differ from the kept ones by **at most `threshold`**,
  as the sum of absolute differences, is not taken. The kept bytes stand for
  it, and what the pass computed from them is reused: the samples, the
  kernel sums, the 1x1 units;
- any other tile is taken and is the kept one from then on.

The bytes of a tile are those of its pixels in the frame's format: R G B for
RGB8, Y0 U Y1 V for YUYV, for NV12 its Y bytes and the U V pairs of its
2 x 2 blocks. The sum is over all of them, not over the luma alone as #533
first proposed: with luma alone a tile whose colour changed would count as
unchanged at threshold 0, and threshold 0 has to mean "the same bytes".

**The result is exact with respect to the threshold.** It is, bit for bit,
that of `geist_decknet_run` on the image whose tiles are the kept ones. With
threshold 0 that image is the frame, so a gated stream at threshold 0 gives
the bits of a stream without gating. Above 0 the image differs from the
frame only in tiles that are within the threshold of what is kept, and
because a tile is compared with the kept bytes and not with the frame
before, a slow drift does not pass unnoticed: once a tile is more than the
threshold away from what the result was computed from, it is taken. The
first frame after `set_gate`, after a new reference and after a frame of
another format is taken whole.

What is kept, and what a frame still costs (`stream.c`, `run.c`):

- the tiles are compared (`memcmp` first: most tiles of a still camera are
  the same bytes; the sum of absolute differences only where they are not
  and the threshold is above 0), and the taken tiles are copied, neighbours
  of a row in one piece;
- only the samples that read a taken tile are computed. A table of the
  taken tiles summed over rectangles lets whole stretches of the sample
  tables be skipped;
- the kernel sums are kept per slot and plane, and recomputed where the
  kernel's window holds a recomputed pixel. They are sums of camera bytes,
  before the gain (see "Forward pass"), so the brightness of the frame may
  move without making them stale;
- the 1x1 units of the floor plane are kept as long as the gains are the
  same bits, which they are while the tiles of the worktable strip are kept;
- the pooling and everything behind it runs for every frame: it is cheap,
  and it is where the gains enter.

**A frame that changed nearly all over is computed whole.** When more than
7 of 8 of the tiles the network reads were taken, keeping what the pass
computes costs more than computing it: every crop of every slot and plane
would be written into the memory the stream keeps (24 MB for the bench's
network), where a pass that keeps nothing has one crop that stays in the
cache. Such a frame is computed as a frame of a stream without gating, from
the kept tiles, which are the frame's now; the result is the same bits.
Nothing that was kept from earlier frames is valid after it, so the next
frame that mostly stands still is sampled whole once more, into the kept
memory this time, and the frame after that is computed from it. The first
frame of a gated stream is such a frame, and so is the one after a new
reference. What it costs is under "Measured".

A frame that is not checkable has its tiles kept but no slot computed; the
next frame is computed whole.

`threshold` is in units of byte differences per tile (a full 16 x 16 tile
has 768 bytes in RGB8, 512 in YUYV, 384 in NV12). What to set it to on a
camera with noise is the caller's measurement; nothing here chooses it.

### Two threads

The slots of a frame do not depend on each other. With
`geist_decknet_stream_set_threads(s, 2)` one half of them is computed on a
second thread of the library's pool (`src/base/par.h`) in a second
workspace, while `push_frame` computes the other half and waits. The result
is the same bits. The conversion of an NV12 or YUYV frame is split the
same way, by rows of tiles; the tile comparison of a gated stream and the
worktable check stay on the calling thread.

The second thread sleeps between two frames of a camera, and waking it is
part of the next pass. With OpenMP, `OMP_WAIT_POLICY=active` in the
environment keeps it spinning instead, at the price of a core that is busy
all the time. Measured at 60 frames per second it made no difference that
held from one run to the next (see "Measured"); the library does not set
it.

### A queue of one

`push_frame` computes and returns, so a capture loop that calls it is as
slow as the pass. Where the camera must not wait, two threads share the
stream through a queue that holds one frame:

```c
/* the thread that captures: never waits, never allocates */
const struct geist_decknet_frame *dropped;
(void) geist_decknet_stream_offer(s, frame, &dropped); /* frame: lent from here on */
if (dropped != nullptr) { give_back(dropped); }        /* the pass was too slow for it */
wake_the_other_thread();

/* the thread that computes */
const struct geist_decknet_frame *done;
if (geist_decknet_stream_push_offered(s, &done) == GEIST_OK && done != nullptr) {
    struct geist_decknet_stream_result res;            /* res.timestamp is done->timestamp */
    (void) geist_decknet_stream_result(s, &res);       /* res.n_dropped: frames lost so far */
}
if (done != nullptr) { give_back(done); }
```

- **Offer** is one atomic exchange of a pointer. The frame becomes the
  waiting one. If a frame was waiting already, the pass has not come round
  since: that older frame is dropped, handed back in `*dropped` and
  counted. So the pass always gets the newest frame, and the camera thread
  does the same few instructions whatever the pass is doing. The frame's
  geometry is checked here, by the rule of `push_frame`; a frame that is
  refused is not queued and replaces nothing.
- **Push the offered frame** takes the waiting frame and runs `push_frame`
  on it. It does not wait for one: with none waiting it returns `GEIST_OK`
  and `*done == nullptr` at once. When to call it is the caller's (a
  condition variable, a pipe, the capture loop's own event); the stream has
  no thread of its own and no lock.
- **Who owns a waiting frame.** Nothing is copied. From the offer on, the
  `struct geist_decknet_frame` and the bytes it describes are lent to the
  stream: the caller does not write, free or reuse them until the stream
  hands that pointer back. It does so exactly once, as `*dropped` of a later
  offer or as `*done` of the `push_offered` that computed on it (with
  whatever status: a frame offered before a reference is set comes back
  with `GEIST_E_INVALID_STATE`). For a capture queue that is the moment to
  give the buffer back to the driver. A frame that is still out when the
  stream is destroyed is the caller's again.
- **Threads.** `geist_decknet_stream_offer` may be called from any thread,
  also while another is inside any other function of the stream but
  `geist_decknet_stream_destroy`. Everything else stays with one thread at a
  time.
- The **result** carries the timestamp of the frame it was computed from,
  and `n_dropped`, the frames replaced while they waited since the stream
  was created. A dropped frame is not a frame of `n_frames`.

`tests/test_decknet_stream_unit.c` has a consumer that is slow by
construction (it takes one pass for every three frames a second thread
offers: the third is computed and two come back dropped, in each round) and
a second thread that offers without waiting; both under ThreadSanitizer.

### What it gives, and what it saves

**The result is that of `geist_decknet_run` on the same two images, bit for
bit**: for RGB8 as they are, for NV12 and YUYV as converted to RGB8 by the
formula above; for a gated stream on the image of its kept tiles. Nothing is
approximated: a stream and the still path compute a crop pixel, a kernel
sum and a unit with the same functions of `run.c`.
`tests/test_decknet_stream_unit.c` compares every output of every slot with
`memcmp`, and `tools/bench_stream` hashes every output of every frame it
times and fails on a difference.

What a stream does not repeat (`src/archs/decknet/stream.c`):

- **Sample tables.** Per homography, where each crop pixel reads the frame
  and with which weights, computed once (when the reference is set)
  instead of two F64 divisions per sample and frame: 16 bytes a sample,
  three weights and a position. Which crop pixel a sample is the table does
  not say; what a plane reads of its crop is a few hundred runs of
  neighbouring pixels, and the samples of a run lie side by side. A sample
  computes its three colours at once, in four 32-bit lanes of the
  compiler's vector type: two 16-byte loads, a byte shuffle per pixel,
  three multiplications (by the differences from the first pixel, which
  gives the integer `decknet_blend` gives with four).
- **Region of interest.** Tables hold only the crop pixels the pass reads:
  the pixels of the pooling rows on the floor plane, the kernel windows
  around the pixels a convolution is evaluated at on the others. Nothing
  else of a frame is touched.
- **Tiles.** RGB8 is sampled in place; of an NV12 or YUYV frame only the
  tiles a sample reads are converted, each pixel once.
- **The empty deck** is sampled once, when it is set.

The reference also fixes the homographies. After
`geist_decknet_set_homographies` the tables are those of the old pose, so
`push_frame` returns `GEIST_E_INVALID_STATE` until the reference is set
again (which rebuilds them, without allocating).

### Measured

`tools/bench_stream` feeds frames from memory (a static scene with a block
moving over it) at 1280 x 720 and 1920 x 1080, through `geist_decknet_run`
once per frame ("still"; an NV12 or YUYV frame converted whole first, the
empty deck converted once outside the timing), through a stream, and
through a gated stream at threshold 0. Its network is invented in the
bench, in the shape of a deck check: six slots, 31 homographies, a
256 x 192 crop, 96 raster units with disc and ring pooling, 9 x 9 and
13 x 13 kernels of two weights each, two region pools over most of the
floor crop. **How much of a crop a network reads, and how many weights its
kernels have, decide what a stream saves, so these are that network's
numbers, not a device's**: it samples 1.78 M crop pixels per frame on the
still path and 0.88 M in a stream, from 6235 of the 8160 tiles of a
1920 x 1080 frame. The moving block changes 270 of them per frame (4.3 %),
134 of 2890 at 1280 x 720 (4.6 %).

Raspberry Pi 5 (gcc 14.2, board below 60 °C and load below 0.5 before each
run, under the host lock), 300 frames per row after 20 warm-up, ms per
frame. One thread:

| size | format | still frames/s | p50 | p99 | stream frames/s | p50 | p99 | gated frames/s | p50 | p99 |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1280 x 720 | RGB8 | 16.1 | 61.9 | 62.7 | 54.0 | 18.5 | 18.7 | 212.3 | 4.7 | 5.0 |
| 1280 x 720 | NV12 | 15.3 | 65.5 | 66.2 | 51.8 | 19.3 | 19.4 | 222.0 | 4.5 | 4.8 |
| 1280 x 720 | YUYV | 15.2 | 65.6 | 65.9 | 51.3 | 19.5 | 19.6 | 220.3 | 4.6 | 5.0 |
| 1920 x 1080 | RGB8 | 15.9 | 62.9 | 63.3 | 50.4 | 19.8 | 20.7 | 170.1 | 5.9 | 8.0 |
| 1920 x 1080 | NV12 | 14.0 | 71.3 | 71.5 | 44.2 | 22.6 | 23.4 | 185.2 | 5.4 | 5.7 |
| 1920 x 1080 | YUYV | 13.9 | 72.1 | 72.4 | 43.9 | 22.7 | 25.1 | 186.8 | 5.3 | 5.8 |

The streams on two threads (`--threads 2`; the still path has no second
thread):

| size | format | stream frames/s | p50 | p99 | gated frames/s | p50 | p99 |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| 1280 x 720 | RGB8 | 96.9 | 10.3 | 12.0 | 273.4 | 3.8 | 4.2 |
| 1280 x 720 | NV12 | 84.5 | 11.8 | 12.1 | 276.9 | 3.7 | 4.2 |
| 1280 x 720 | YUYV | 83.8 | 11.9 | 15.2 | 283.4 | 3.6 | 4.0 |
| 1920 x 1080 | RGB8 | 81.6 | 12.2 | 12.8 | 207.6 | 5.0 | 5.4 |
| 1920 x 1080 | NV12 | 71.0 | 14.1 | 14.5 | 230.4 | 4.6 | 4.7 |
| 1920 x 1080 | YUYV | 69.0 | 14.5 | 14.9 | 224.5 | 4.5 | 5.1 |

What gating costs when it saves little or nothing, same board,
1920 x 1080, p50 / p99 in ms. `--scene all`: every pixel of a frame differs
from the frame before, so every tile is taken. `--scene burst`: 30 such
frames and 30 with only the block moving, in turn, so that one frame in 60
is the one that fills the caches again. `--scene half`: the upper half of
the frame changes, which is three of the six slots, all of them computed
by the first thread:

| | RGB8 | NV12 | YUYV |
| :-- | --: | --: | --: |
| all: stream, one thread | 19.7 / 19.9 | 22.5 / 22.9 | 22.7 / 23.0 |
| all: gated, one thread | 21.3 / 25.7 | 23.2 / 23.4 | 23.7 / 23.9 |
| all: stream, two threads | 11.8 / 14.0 | 13.8 / 18.6 | 14.2 / 14.5 |
| all: gated, two threads | 13.9 / 14.3 | 15.8 / 16.3 | 16.6 / 18.7 |
| burst: gated, one thread | 21.4 / 24.0 | 23.4 / 24.3 | 23.8 / 24.2 |
| burst: gated, two threads | 13.0 / 18.2 | 15.6 / 18.6 | 16.3 / 16.9 |
| half: stream, one thread | 19.5 / 19.7 | 22.3 / 22.6 | 22.6 / 22.8 |
| half: gated, one thread | 17.0 / 18.0 | 17.8 / 19.8 | 17.8 / 18.8 |
| half: stream, two threads | 12.1 / 12.5 | 14.0 / 14.4 | 14.4 / 16.8 |
| half: gated, two threads | 13.9 / 14.5 | 14.4 / 15.0 | 15.0 / 15.5 |

A gated frame that changed all over is computed whole (see "Gating"), so
what it costs beyond a frame that is not gated is its tiles compared and
copied: 0.7 to 1.6 ms on one thread (3 to 8 %), where it was a third
before that rule (25.7, 28.3 and 29.0 ms against 19.2, 21.6 and 22.6). On
two threads it is 2.0 to 2.4 ms (14 to 18 %), because the comparison and
the copy stay on the calling thread while the rest is halved. The frame
that fills the caches again after such a stretch is the slowest of a burst:
its p99 above. With half of the frame changing, a gated stream on one
thread is ahead of one that is not gated; on two threads it is behind
here, because the three slots that change are the first thread's and the
second has nothing to do: which slots change decides it, and the rule
above looks at the frame, not at a slot.

Every row hashes every output of every frame, and the three ways agree in
every bit.

Where the frame went, same board, 1920 x 1080, one thread, ms per frame
(300 frames, measured with timers that are not in the tree; the first
column is the stream as the second step of #533 left it, the last a gated
stream when every tile changes):

| | before, RGB8 | RGB8 | NV12 | YUYV | gated, RGB8 | gated, all tiles, RGB8 |
| :-- | --: | --: | --: | --: | --: | --: |
| tiles: conversion, or comparison and copy | 0 | 0 | 2.6 | 3.1 | 2.0 | 1.6 |
| sampling | 9.7 | 7.7 | 7.7 | 7.7 | 1.3 | 7.8 |
| floor plane: 1x1 units and region pools | 4.2 | 4.2 | 4.4 | 4.2 | 0.7 | 4.2 |
| 9 x 9 kernel, four planes | 5.9 | 5.9 | 6.0 | 5.9 | 0.8 | 5.9 |
| 13 x 13 kernel | 0.3 | 0.3 | 0.3 | 0.3 | 0.04 | 0.3 |
| unit pooling and the rest of the pass | 1.2 | 1.2 | 1.3 | 1.3 | 1.3 | 1.3 |
| the frame | 21.3 | 19.4 | 22.3 | 22.4 | 6.1 | 21.1 |

On two threads the parts do not halve: summed over both threads, sampling
takes 10.6 ms where it takes 7.7 on one (the two share the last cache level
and the memory behind it), the floor plane 4.7 and the 9 x 9 kernel 6.2, and
the conversion of an NV12 frame 1.8 ms of wall time where it takes 2.6 on
one thread.

With a camera (`--camera 60 --frames 600`: a second thread offers a frame
every 16.7 ms, the pass takes what waits; see "A queue of one"), same
board, 1920 x 1080, the pass on two threads, not gated: time from the offer
of a frame to its result, p50 / p99 in ms, and how many of the 600 frames
had their result later than a frame period. Three runs, and two with
`OMP_WAIT_POLICY=active`:

| | RGB8 | NV12 | YUYV |
| :-- | --: | --: | --: |
| run 1 | 12.4 / 13.3, 0 late | 14.3 / 20.4, 12 late | 14.1 / 18.3, 9 late |
| run 2 | 12.4 / 14.0, 4 late | 14.3 / 18.0, 8 late | 14.2 / 17.8, 6 late |
| run 3 | 12.3 / 13.7, 2 late | 14.3 / 15.1, 4 late | 14.2 / 25.7, 28 late |
| active, run 1 | 11.9 / 14.4, 2 late | 14.2 / 15.1, 3 late | 14.3 / 14.8, 0 late |
| active, run 2 | 12.3 / 12.9, 2 late | 14.3 / 20.5, 11 late | 14.1 / 14.7, 1 late |

Of the 9300 frames offered in those rows two were dropped; a pass that is
late makes the next one start late, which is why late frames come in
groups. Gated at threshold 0 with the block moving, the same camera, two
runs: 4.7 to 5.3 ms, p99 at most 6.7 ms, on two threads, two frames of 3600
later than a period; on one thread 6.1 to 6.3 ms and at most 7.7 ms, one
of 1800 late. At 30 frames per second on one thread, not gated:
19.5 / 19.7, 21.7 / 26.6 and 22.1 / 26.1 ms, none late and none dropped.
The median is the pass; the tail is single passes of 17 to 36 ms, a few in
600 frames, in every format, with and without the active wait. Where they
come from was not found: the board also runs CI runner services, and
nothing was pinned to a core.

What each part gained, same board, 1920 x 1080, p50 in ms, one thread
unless the row says two (200 or 300 frames; each row is the tree at that
point of the work, measured then):

| | RGB8 | NV12 | YUYV |
| :-- | --: | --: | --: |
| the first step of #533 (tables, region of interest) | 43.3 | 52.7 | 50.5 |
| integer crops, kernels as sums per weight | 26.0 | 38.0 | 36.1 |
| + tiles (NV12 and YUYV converted once per pixel), 1x1 layer four rows at a time | 24.4 | 30.7 | 30.7 |
| + the 1x1 layer's `relu` without a branch | 21.4 | 27.8 | 27.8 |
| + the conversion in 16 bits, as loops a compiler makes vectors of | 21.3 | 23.3 | 23.8 |
| gated, tiles compared by sum of absolute differences, every sample looked at | 13.0 | 8.0 | 9.7 |
| gated, tiles compared by `memcmp` first, stretches of the tables skipped | 5.9 | 5.5 | 5.6 |
| two threads, not gated (the conversion on the calling thread) | 12.8 | 15.8 | 16.0 |
| two threads, not gated, the conversion split as well | 13.1 | 15.0 | 15.4 |
| the third step of #533 starts from: one thread | 21.2 | 23.5 | 24.0 |
| + a sample as one vector, tables of 16 bytes: one thread | 19.1 | 21.6 | 22.6 |
| the same on two threads, before and after | 13.2, 12.0 | 15.2, 14.1 | 14.9, 14.0 |
| gated, every tile changes, one thread, before | 25.7 | 28.3 | 29.0 |
| + such a frame computed whole, its tiles copied in runs | 20.9 | 23.1 | 23.4 |

The still path gained from the integer crops as well (1920 x 1080 RGB8:
106.5 to 62.9 ms). The rows of one step were measured in one session; the
tables at the top are the tree as it is, in another, and a build differs
from the next by a few tenths of a millisecond (the same source, one thread,
RGB8: 19.1 to 19.8 ms).

One sample alone, in a loop outside the tree with tables and pixels in the
first-level cache, same board, ns per sample: as it was (twelve
multiplications, 70 instructions) 8.4; as it is 5.1; the same arithmetic
with NEON intrinsics 5.0; red and blue in one 64-bit product 8.8; 16-bit
differences widened to 32 bits 6.0. So the vector types cost nothing
against intrinsics, and what is left is the arithmetic: in the tree a
sample takes 8.5 ns, of which the memory (14 MB of tables, the frame, the
crop) is the smaller part.

**Against the target of #533** (a Pi 5, 1920 x 1080 at 60 frames per
second, p99 below one frame period of 16.7 ms, on at most two cores, six
slots), on the bench's network: a stream on two threads has its median
inside it for all three formats, with 4.5 ms to spare for RGB8 and 2.2 to
2.6 ms for NV12 and YUYV (12.2, 14.1 and 14.5 ms; they were 13.1, 15.0 and
15.4 before the third step). Fed one frame after the other, its p99 is
inside as well in the table above (12.8, 14.5, 14.9 ms), and was up to
18.6 ms for a format in other runs of the same. Fed by a camera thread at
60 frames per second, which is the target as it is written, RGB8 is inside
in every run (p99 12.9 to 14.4 ms, at most 4 frames of 600 late) and NV12
and YUYV are not in every run: their p99 was between 14.7 and 25.7 ms, with
0 to 28 frames of 600 late. So RGB8 meets the target; NV12 and YUYV meet it at the
median and miss it in the tail, by single slow passes and not by the cost
of the pass. On one thread a stream is not inside (20 to 23 ms) unless it
is gated and the scene is mostly still: 5.3 to 5.9 ms, and 6.1 to 6.3 ms
behind a camera at 60 frames per second with p99 at most 7.7 ms. A gated
stream on two threads whose every tile changes is at 13.9, 15.8 and
16.6 ms: inside for RGB8, at the limit for the other two. None of this is
the device's network or a camera; see "Not covered".

```sh
make DECKNET=1 bin/<target>/release/tools/bench_stream
~/bin/with-mac-lock bin/<target>/release/tools/bench_stream \
    [--frames N] [--threads 1|2] [--scene block|all|half|burst] \
    [--only still|stream|gated] [--size 720|1080] [--camera FPS]
```

### Not covered

- **A capture loop.** The queue of one is two calls; the thread that
  captures, what wakes the thread that computes and the buffers of the
  camera are the caller's. No camera was connected: the bench's "camera" is
  a thread with a clock.
- **The slow passes.** Behind a camera at 60 frames per second a few passes
  in 600 take 17 to 36 ms where the median is 14, and they are what keeps
  NV12 and YUYV from the target. Not looked into: which core a thread runs
  on, what else the board does, the frequency governor.
- **A GPU path.** Not looked at.
- **Integral images and separable kernels.** A kernel is summed per distinct
  weight, tap by tap. For a disc or ring kernel that is integer additions
  only, but as many as the kernel has taps: 5.9 ms of a 19 ms frame for the
  bench's 9 x 9 kernel, now as much as the sampling was. The sum over the
  weight with the most taps is the sum over the whole window less the
  others, and the window is a few differences of row sums; that would be
  the same integers. Not measured.
- **8-bit crops and dot-product instructions.** #533 proposed u8 inputs with
  16- and 32-bit accumulators and SDOT. A crop pixel rounded to 8 bits is up
  to 2e-3 of the range away from the reference's F32 pixel, and behind the
  gains of the units that is more than the 1e-4 the parity test allows (an
  estimate from the gains, not a measurement); the crops are 32-bit fixed
  point for that reason. Their kernels then have no multiplications to give
  to a dot product.
- **Sampling beyond this.** A sample is three multiplications of four lanes
  and five byte shuffles, 5 ns on the Pi 5 with everything in the cache and
  8.5 ns in a frame; NEON intrinsics for the same arithmetic gave the same
  5 ns. A byte offset in the table instead of x and y would save 0.7 ns and
  need the tables rebuilt for each stride: not taken. A sample needs every
  bit of its three weights to give the bits it gives, so a table entry
  cannot be much smaller than its 16 bytes.
- **The conversion of NV12 and YUYV** is 2.6 and 3.1 ms of a frame, 1.8 and
  2.0 ms of wall time on two threads, in a region of its own before the
  slots'. Converting the tiles of a slot on the thread that samples them was
  not tried.
- **The whole-frame rule per slot.** A gated frame is computed whole when
  7 of 8 of all its tiles changed. A frame in which one slot changes all
  over and the others stand still is computed from the caches, that slot
  included, which costs about a third more for that slot than computing it
  whole would.
- **A threshold for a real camera.** Gating is exact for any threshold, but
  which threshold keeps the tiles of a noisy sensor is a measurement on that
  sensor. The bench's frames have no noise.
- **More than two threads**, and the tile comparison of a gated stream on
  the second one: with every tile changing it is the 2 ms a gated frame
  costs more than one that is not gated, on two threads.

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
   (`u = p0 / p2 - 0.5`, clamped to the image) into a crop. The coordinates
   are computed in F64, as the reference does. A homography that divides by
   zero samples the image corner; it does not fail. A crop pixel is kept as
   an integer, see "Integer crops" below.
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

### Integer crops

The reference multiplies the image by its gain, samples F32 and convolves
in F64. `run.c` computes the same quantities in another order (#533), for
the still path and a stream alike:

- **A crop pixel is an integer**: the bilinear sample of the four camera
  bytes with weights in 1/2^24 that sum to 2^24 exactly, rounded once to
  1/65536 of a byte value (`decknet_weights`, `decknet_blend` in
  `decknet.h`), in 32-bit arithmetic that cannot overflow. No gain is in
  it. Four 32-bit values per pixel, R G B and a zero.
- **The gain is applied behind the kernels**, which are linear:
  `relu(gain * conv(x))` for `relu(conv(gain * x))`. Where a float pixel is
  needed (the 1x1 units, the white unit, the worktable check) it is
  `(float) (value * gain / (255 * 65536))`.
- **A kernel is a sum over its distinct weights** of weight times the sum of
  the pixels that have it (`decknet_kernel.part`, built by the loader from
  the file's F32 weights). The pixel sums are integer sums: exact, the same
  in any order and at any vector width, in 32 bits within a part and 64
  across. A disc or ring kernel has two or three weights; a kernel whose
  weights are all different is one multiplication per tap, as before.
- The **1x1 layer** computes four of its rows at once, each lane with the
  operations and in the order one row took before; its `relu` selects
  instead of branching. Its values are the bits they were.

What that buys: an integer sum needs no order, so the compiler's vector
types can carry it (`run.c` uses GCC/Clang vector types, no intrinsics and
no `#if`: the same source on every target, and a target without vectors
computes the same integers), and a camera stream can keep a kernel sum from
one frame to the next while the brightness of the image moves.

What it costs: a pixel is no longer the reference's F32, but within F32
rounding of it (the weights are exact to 2^-25, the value to 2^-17 of a
byte), so the outputs differ from the reference's in the place they already
did. On the invented network the largest `|C - numpy|` went from 2.8e-7 to
5.4e-7, against a tolerance of 1e-4 that is unchanged; see "Parity".

## Where it lives

- `src/archs/decknet/load.c`: the file contract, `run.c`: the pass,
  `stream.c`: the same pass fed from a camera stream (sample tables, region
  of interest, tiles, frame formats, gating, the second thread, the queue
  of one). One
  implementation in portable C, built without `-ffast-math` and without
  contraction. Its vectors are the compiler's (vector types for a crop pixel,
  for the sample that computes one and for four rows of the 1x1 layer,
  plain loops for the colour conversion): no intrinsics, no platform `#if`. No backend dispatches it,
  so there is no `cpu_scalar` / `cpu_neon` / `cpu_x86` variant to keep in
  step: the same code runs on every platform, and `test_decknet_unit` is its
  parity check there.
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
| `tests/test_decknet_stream_unit.c` | a stream against the still path, every output bit for bit: RGB8, NV12 and YUYV, padded rows, replaced homographies, across a frame that is not checkable, on one thread and on two; NV12 and YUYV frames of arbitrary bytes (the conversion over its whole range); a gated stream against the still path on the image of its kept tiles, which the test builds with its own tiles and its own sum of absolute differences, at threshold 0 and above it; gating at its edges (a kernel window across a cell of the crop, a tile exactly at the threshold and one above it, each tile changed on its own, one sample at a time across a seam of the sample tables, a frame that is not checkable between two that are); the fixed-point sample itself (weights that never exceed the whole, a flat image sampled to itself, any four bytes to within 2/65536 of the exact value) and a kernel whose pixel sum passes 32 bits; no allocation after the create and the first `set_gate`, the first frame included, the queue's calls among them; 227 frame descriptions refused, exact-fit frames accepted; samples at the last pixels of a frame with 0 to 20 bytes behind it; the queue of one, step by step on one thread, with a consumer that takes one pass per three offers of a second thread, and with that thread offering without waiting (every frame comes back once, dropped or computed, the computed ones in order and with the bits of `push_frame`); state and argument checks; every allocation of a create and of a `set_gate` failing once. The gating tests assert which frames were computed from what was kept (they sample less than a whole frame) |
| `tests/fuzz_decknet_stream.c` | frame geometry (format, size, stride, extent) and homographies from the input, on exact-size buffers; the harness decides for itself whether a frame fits and fails on a wrong verdict, not only on a crash. And gating: four frames (one, the same again, two with bytes changed) through a gated stream and one without, which have to agree in every bit at threshold 0, and the same frame twice has to give one result at any threshold. And the queue of one: every description goes through an offer as well, which has to give the verdict of a push, queue nothing it refuses and hand back the frame it replaces |

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

Measured maximum `|C - numpy|` over all outputs is about 1e-6 (F32 rounding)
on the device's network with the pass as #532 landed it. With the integer
crops of #533 the invented network is at 5.4e-7 (2.8e-7 before); **the
device's network has not been run with them by the author of that change**,
who has no access to its fixture: CI runs it where the fixture is. The
tolerance is 1e-4, as before.

One point needed more than following the formulas. The reference computes
the worktable mean as `a.mean((0, 1))` on an F32 array, which numpy
evaluates as an F32 running sum over the strip's pixels; for the device's
file its rounding moves the mean by about 1e-5 of its value. The gain scales
the whole image, and behind the gain of the white units an exact mean
differs from the reference by up to 4.4e-4 in a unit, above the 1e-4 the
parity test allows. `run.c` therefore sums in F32
in the same order. If the reference changes to an F64 mean, `gain_check`
changes with it (one line) and both fixtures are regenerated.
