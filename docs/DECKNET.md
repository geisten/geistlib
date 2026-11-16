# decknet: a hand-written convolutional net for images (experimental)

decknet is the geistbox deck-check network (#532): two camera images in (the
deck now and the empty deck), one small state vector per deck position out.
It is a fixed convolutional net whose weights were written by hand from the
deck geometry; nothing is trained. It is the library's first model that is
not a language model: no tokenizer, no session, and no backend to create.
Its pass is portable C; a camera stream may hand the integer part of it to
a GPU backend (see "On a GPU").

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
(void) geist_decknet_stream_set_backend(s, metal); /* optional: the integer part on a GPU */

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
| `geist_decknet_stream_set_backend(s, be)` | the integer part of a frame that is not gated on the GPU of `be` (today the `metal` backend), or with `nullptr` on the CPU again, see "On a GPU"; allocates what the GPU needs | `GEIST_E_UNSUPPORTED` (a backend without a decknet path, which every CPU backend is; a model too large for the GPU's 32-bit indices), `GEIST_E_OOM`, `GEIST_E_BACKEND`, `GEIST_E_INVALID_ARG`; the stream stays as it was |
| `geist_decknet_stream_set_reference(s, &frame)` | the empty deck, in any format; takes over the handle's homographies as they are now | `GEIST_E_INVALID_ARG` (null, or a frame that is refused, see below); the reference it had stays |
| `geist_decknet_stream_push_frame(s, &frame)` | one forward pass on the frame against the reference; returns when it is done | `GEIST_E_INVALID_ARG` (null, or a frame that is refused): nothing is read, the last result stays. `GEIST_E_INVALID_STATE`: no reference yet, or the homographies were replaced since. On a GPU also `GEIST_E_BACKEND`: the device failed the frame, and the stream has no result until one succeeds |
| `geist_decknet_stream_offer(s, &frame, &dropped)` | from the thread that captures: the frame waits for the next pass; never waits itself. The frame that was waiting, if any, comes back in `*dropped` and is counted. See "A queue of one" | `GEIST_E_INVALID_ARG` (null, or a frame that is refused): nothing is queued |
| `geist_decknet_stream_push_offered(s, &done)` | `push_frame` on the waiting frame, which comes back in `*done`; `GEIST_OK` and `*done == nullptr` when none waits | as `push_frame`; the frame comes back with any status |
| `geist_decknet_stream_result(s, &res)` | the image check of the last frame, its timestamp, the number of frames so far, what the frame cost (the tiles the network reads, how many of them were taken from this frame, the samples computed), and how many offered frames were dropped | `GEIST_E_INVALID_STATE` before the first frame, `GEIST_E_INVALID_ARG` |
| `geist_decknet_stream_slot(s, i, &slot)` | the units of slot `i` from the last frame | as `geist_decknet_slot` |
| `geist_decknet_stream_destroy(s)` | frees the stream; the handle stays | |

The stream computes in the handle's workspace: `d` outlives the stream, and
while the stream exists `geist_decknet_run` is not called on `d`. One stream
is used by one thread at a time; `geist_decknet_stream_offer` is the one
function a second thread may call meanwhile. After the create, the first
`geist_decknet_stream_set_gate` that turns gating on and a
`geist_decknet_stream_set_backend`, nothing allocates: not the reference,
not the first frame, not a later one.

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
those tiles are converted to RGB8 first, into memory the stream owns, and
sampled there: a bilinear sample reads four pixels, and a frame's pixels
are read by several samples, so converting per sample (the first step of
#533 did) converts each pixel several times over.

A stream that is not gated converts **slot by slot, and packed**: the
thread that computes a slot converts the tiles that slot reads right before
it samples them, into an image that holds the rectangle of those tiles and
nothing else, and its next slot goes into the same memory. What a slot
reads of a 1920 x 1080 frame is then 0.8 MB that the core has just written,
instead of a part of a 6 MB image of the frame that another region wrote
before (see "Where the conversion's 2 ms went"). A tile that two slots read
is converted once for each; the worktable strip's tiles are converted before
the slots. A gated stream converts the tiles it takes into an image of the
whole frame, which is what it keeps.

NV12 and YUYV are read as BT.601 limited range, the chroma of a pixel taken
from its block without interpolation. In integers, for the bytes `Y`, `U`,
`V` of a pixel:

```
c = 298 * (Y - 16) + 128
R = clamp((c + 409 * (V - 128)) >> 8)
G = clamp((c - 100 * (U - 128) - 208 * (V - 128)) >> 8)
B = clamp((c + 516 * (U - 128)) >> 8)          clamp to 0..255
```

`stream.c` evaluates this in 16 bits, 16 pixels at a time in the compiler's
vector types (no intrinsics; the last pixels of a row whose length is not a
multiple of 16 in pairs): the multiples of 256 are taken out of the shift
(298 = 256 + 42, 409 = 256 + 153, 208 = 256 - 48, 516 = 512 + 4), which
leaves sums between -20128 and 29597; the offsets of a term are gathered
into one constant; what depends on the chroma alone is computed once for
the two pixels that share it; and a colour is clamped as bytes, by the sign
of its high byte. All 2^24 values of (Y, U, V) give the bytes of the
formula above, in every lane and in the pairs (checked exhaustively outside
the tree, with clang on arm64 and gcc 14 on arm64 and x86-64;
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
is the same bits. The conversion of an NV12 or YUYV frame goes with the
slots: each thread converts what its slots read (see "Frames"); the tile
comparison of a gated stream, with the conversion of the tiles it takes,
and the worktable check stay on the calling thread.

The second thread is the pool's, and between two frames of a camera it has
nothing to do. Waking it is not what a frame waits for: behind a camera at
60 frames per second on a Raspberry Pi 5 (libgomp) its half of a frame
starts 2 to 20 µs after the first thread's (measured at the fourth step of
#533, when the conversion was a region of its own before the slots', in
both). With OpenMP, `OMP_WAIT_POLICY=active` in the environment
keeps it spinning instead, at the price of a core that is busy all the
time; in five runs per format it changed nothing (see "The tail behind a
camera"), and the library does not set it.

**Binding the threads to cores needs `OMP_NUM_THREADS` as well.** The pool
takes no more threads than the calling thread may run on CPUs
(`src/base/par.c`), unless `OMP_NUM_THREADS` sizes the team. `OMP_PROC_BIND`
and `GOMP_CPU_AFFINITY` bind the calling thread to one core, so with either
of them alone the stream's second thread is not used, without a word:
28 ms a frame on the Pi 5 where two threads take 12. With
`OMP_NUM_THREADS=2` next to the binding both run.

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
approximated: a stream and the still path compute a crop pixel and a unit
with the same functions of `run.c`, and a kernel sum as the same integers.
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
  tiles a sample reads are converted, 16 pixels at a time, per slot and
  packed, by the thread that samples them.
- **The empty deck** is sampled once, when it is set.
- **A kernel's taps one by one.** Where both kernels have few weights (see
  "Integer crops"), a plane is sampled as prefix sums along its runs of crop
  pixels instead of as the pixels, and a kernel takes the sum of a row of
  equal weights from two of them. Not in a gated stream's kept crops, which
  change pixel by pixel.

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
| 1280 x 720 | RGB8 | 16.1 | 62.1 | 62.5 | 61.4 | 16.3 | 16.5 | 202.5 | 5.0 | 5.5 |
| 1280 x 720 | NV12 | 15.2 | 65.8 | 66.6 | 60.6 | 16.5 | 16.6 | 213.8 | 4.7 | 5.0 |
| 1280 x 720 | YUYV | 15.2 | 65.9 | 66.0 | 59.7 | 16.7 | 16.9 | 212.8 | 4.7 | 5.1 |
| 1920 x 1080 | RGB8 | 15.8 | 63.5 | 63.8 | 56.8 | 17.6 | 17.7 | 168.6 | 5.9 | 6.3 |
| 1920 x 1080 | NV12 | 13.8 | 72.2 | 72.5 | 56.3 | 17.7 | 17.9 | 186.4 | 5.4 | 5.7 |
| 1920 x 1080 | YUYV | 13.8 | 72.4 | 73.1 | 53.8 | 18.6 | 18.7 | 184.7 | 5.4 | 5.8 |

The streams on two threads (`--threads 2`, bound to two cores as under "Two
threads"; the still path has no second thread). On this board NV12 at
1920 x 1080 moves more from one session to the next than the others: 12.1
to 12.4 ms in the three runs of the session these tables are from (this
one and the two scenes below), 11.5 to 11.9 ms in eight runs of two other
sessions of the same afternoon, in which the tree before, run in turn with
it, took 13.0 to 13.2:

| size | format | stream frames/s | p50 | p99 | gated frames/s | p50 | p99 |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| 1280 x 720 | RGB8 | 103.1 | 9.7 | 10.5 | 267.3 | 3.9 | 4.3 |
| 1280 x 720 | NV12 | 105.2 | 9.5 | 10.0 | 281.0 | 3.7 | 4.0 |
| 1280 x 720 | YUYV | 103.8 | 9.6 | 9.9 | 280.4 | 3.7 | 4.0 |
| 1920 x 1080 | RGB8 | 88.9 | 11.2 | 11.6 | 202.7 | 5.0 | 5.6 |
| 1920 x 1080 | NV12 | 82.5 | 12.1 | 14.1 | 227.1 | 4.5 | 4.9 |
| 1920 x 1080 | YUYV | 85.0 | 11.7 | 12.0 | 229.0 | 4.4 | 4.9 |

What gating costs when it saves little or nothing, same board,
1920 x 1080, p50 / p99 in ms. `--scene all`: every pixel of a frame differs
from the frame before, so every tile is taken. `--scene burst`: 30 such
frames and 30 with only the block moving, in turn, so that one frame in 60
is the one that fills the caches again. `--scene half`: the upper half of
the frame changes, which is three of the six slots, all of them computed
by the first thread:

| | RGB8 | NV12 | YUYV |
| :-- | --: | --: | --: |
| all: stream, one thread | 17.3 / 17.4 | 17.9 / 18.2 | 18.3 / 19.0 |
| all: gated, one thread | 19.0 / 19.1 | 19.8 / 20.2 | 20.2 / 20.4 |
| all: stream, two threads | 11.3 / 16.1 | 12.4 / 12.8 | 11.8 / 12.3 |
| all: gated, two threads | 13.2 / 13.7 | 15.1 / 15.8 | 14.9 / 15.1 |
| burst: gated, one thread | 18.9 / 23.6 | 19.8 / 23.0 | 20.2 / 22.8 |
| burst: gated, two threads | 13.0 / 17.4 | 14.9 / 16.7 | 14.6 / 16.5 |
| half: stream, one thread | 17.3 / 19.1 | 17.9 / 18.7 | 18.3 / 18.5 |
| half: gated, one thread | 17.2 / 18.6 | 17.4 / 18.6 | 17.5 / 18.6 |
| half: stream, two threads | 11.6 / 11.9 | 12.3 / 12.6 | 11.6 / 12.2 |
| half: gated, two threads | 14.0 / 14.4 | 14.4 / 14.9 | 14.3 / 15.0 |

A gated frame that changed all over is computed whole (see "Gating"), so
what it costs beyond a frame that is not gated is its tiles compared and
copied, and for NV12 and YUYV that it is converted and sampled as an image
of the whole frame and not slot by slot and packed: 1.7 to 1.9 ms on one
thread (10 %), where it was a third before that rule (25.7, 28.3 and 29.0 ms
against 19.2, 21.6 and 22.6, at the third step). On two threads it is 1.9 to
3.1 ms (17 to 26 %), because the comparison, the copy and the conversion
stay on the calling thread while the rest is halved. The frame that fills
the caches again after such a stretch is the slowest of a burst: its p99
above. With half of the frame changing, a gated stream on one thread is
ahead of one that is not gated for NV12 and YUYV and level with it for RGB8
(what it computes from kept crops sums its kernels tap by tap, and the other
no longer does); on two threads it is behind here, because the three slots
that change are the first thread's and the second has nothing to do: which
slots change decides it, and the rule above looks at the frame, not at a
slot.

Every row hashes every output of every frame, and the three ways agree in
every bit.

Where the frame went, same board, 1920 x 1080, one thread, ms per frame
(300 frames, measured with timers that are not in the tree, at the third
step of #533; the first column is the stream as the second step left it,
the last a gated stream when every tile changes). The fourth step's
kernels by spans are not in it: they took 2 ms off a frame that is not
gated, which can only be the 9 x 9 row's; the parts were not timed
again:

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
one thread. The conversion is no longer a part of its own; "Where the
conversion's 2 ms went" has the parts of a frame as they are now.

With a camera (`--camera 60 --frames 600`: a second thread offers a frame
every 16.7 ms, the pass takes what waits; see "A queue of one") the time
from the offer of a frame to its result is the pass at the median, and it
has a tail that is not the pass's. The runs at 60 frames per second, not
gated, and where their tail comes from are under "The tail behind a
camera". Gated at threshold 0 with the block moving, the same camera, two
runs per format (the tree as it is): 4.9 to 5.3 ms, p99 at most 6.5 ms, on
two threads, one result of 3600 later than a period; on one thread 6.2 to
6.6 ms and at most 7.5 ms, one of 3600 late. At 30 frames per second on one
thread, not gated: 17.5 / 17.6, 18.0 / 18.3 and 18.6 / 18.8 ms, none
dropped, and one result of 1800 later than a period of 33.3 ms (a pass of
34.5 ms).

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
| the fourth step of #533 starts from: one thread | 19.4 | 22.1 | 22.4 |
| + a row of equal kernel weights from two prefix sums: one thread | 17.5 | 19.9 | 20.4 |
| the same on two threads, before and after | 12.0, 11.2 | 13.9, 13.1 | 14.2, 13.7 |
| the prefix sums next to the crop instead of in its place, two threads (not taken) | 16.4 | 18.7 | 19.5 |
| the fifth step of #533 starts from: one thread, two threads | 17.6, 11.2 | 19.9, 13.1 | 20.3, 13.7 |
| + NV12 and YUYV converted 16 pixels at a time: one thread, two | 17.6, 11.1 | 19.3, 12.8 | 19.3, 13.2 |
| + converted per slot, packed, by the thread that samples: one thread, two | 17.3, 11.2 | 17.9, 11.6 | 18.2, 11.8 |
| per slot, but into the image of the frame, two threads (not taken) | | 13.8 | 14.4 |
| RGB8 copied into a packed image as well, two threads (not taken) | 11.9 | | |

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
inside it for all three formats, with 5.5 ms to spare for RGB8 and 4.3 to
5.2 ms for NV12 and YUYV (11.2 ms, and 11.5 to 12.4 ms over the sessions of
the fifth step; they were 11.2, 13.1 and 13.7 before it, and 12.2, 14.1 and
14.5 before the fourth). Fed by a camera thread at 60 frames per second,
which is the target as it is written, and bound to two cores, **p99 was
below a frame period in all 10 runs of 600 frames for each of the three
formats** (the last table of "The tail behind a camera": p99 11.4 to
15.2 ms for RGB8, 11.9 to 14.1 for NV12, 12.2 to 12.7 for YUYV; 10, 1 and 9
results of 6000 late), where the tree before, in the same session and run
in turn with it, was inside in 10, 9 and 4 of 10 runs. The slow passes are
still there, and still the board's (memory compaction and the other
services; a pass of 30 to 45 ms hit 9 of those 30 runs and made one to
three results late in each): what changed is that a format with 5 ms to
spare is late once or twice after such a pass and has caught up before the
run's hundredth part is used up, where one with 3 ms was late five times
or more.
So on this board as it is, all three formats met the target in every run
of that session; that is ten runs of ten seconds each, not a guarantee,
and a board without the compaction and the other services was not there to
measure. On one thread a stream is not inside (17 to 19 ms) unless it is
gated and the scene is mostly still: 5.4 to 5.9 ms, and 6.2 to 6.6 ms
behind a camera at 60 frames per second with p99 at most 7.5 ms. A gated
stream on two threads whose every tile changes is at 13.2, 15.1 and
14.9 ms: inside. None of this is the device's network or a camera; see
"Not covered".

```sh
make DECKNET=1 bin/<target>/release/tools/bench_stream
~/bin/with-mac-lock bin/<target>/release/tools/bench_stream \
    [--frames N] [--threads 1|2] [--scene block|all|half|burst] \
    [--only still|stream|gated] [--size 720|1080] [--camera FPS] [--trace] \
    [--backend cpu|metal]
```

Each row also gives the processor time the process used for its timed
frames, all threads together (`getrusage`); behind a camera that includes
the bench's own looking for a frame, which the row names.

### Where the conversion's 2 ms went

After the fourth step an NV12 or YUYV frame took 2 ms longer on two threads
than an RGB8 one, and had 3 ms to spare below a frame period where RGB8 had
5.5. The fifth step of #533 measured where the 2 ms go and took most of
them away. Timers in the stream that are not in the tree, same board,
1920 x 1080, two threads bound to two cores, 300 frames, ms per frame;
where a cell has two numbers they are the two threads:

| | RGB8 | NV12 before | NV12 | YUYV before | YUYV |
| :-- | --: | --: | --: | --: | --: |
| conversion, a region of its own before the slots | 0 | 1.80 | 0.04 | 1.97 | 0.05 |
| worktable strip and gains | 0.15 | 0.16 | 0.13 | 0.16 | 0.13 |
| in the slots: conversion of what the slot reads | 0 | 0 | 1.71, 1.80 | 0 | 1.62, 1.73 |
| in the slots: sampling | 4.75, 5.40 | 5.20, 5.95 | 4.70, 4.75 | 4.97, 5.72 | 4.72, 4.71 |
| in the slots: the planes | 5.43, 5.59 | 5.37, 5.29 | 5.10, 5.05 | 5.79, 5.81 | 5.44, 5.39 |
| the slots of each thread | 10.2, 11.0 | 10.6, 11.2 | 11.5, 11.6 | 10.8, 11.5 | 11.8, 11.8 |
| the frame | 11.2 | 13.2 | 11.8 | 13.7 | 12.0 |

- **All of it was the conversion, and on two threads the conversion was
  memory.** It took 1.8 and 2.0 ms of wall time with both threads busy for
  all of it (1.73 to 1.97 ms each): 3.5 and 3.9 ms of processor time for
  what one thread alone does in 2.6 and 3.1. It wrote an RGB image of the
  frame's size, 4.8 MB of it, which the slots then read back; the board
  does not give two threads twice the memory.
- **The threads did not wait for each other there, and they do in the
  slots.** The second thread's sampling is 0.6 to 0.8 ms slower than the
  first's in every format (its slots are the lower half of the frame), and
  the first waits that long at the end. With packed images it is not
  (4.7 ms on both); for RGB8, which is sampled in the caller's frame, it
  still is.
- **The arithmetic was the smaller part.** The conversion alone, with
  everything in the first-level cache, ns per pixel: as it was (loops over
  small arrays for gcc to make vectors of) 1.37 for NV12 and 1.67 for YUYV;
  16 pixels at a time in vector types 1.06 and 1.09. On one thread in the
  stream that is 2.6 to 1.8 ms and 3.1 to 2.0 ms of a frame.

What changed (`stream.c`):

- **16 pixels at a time** (see "Frames" for the arithmetic). By itself,
  on two threads, 0.2 to 0.4 ms of an NV12 frame and 0.4 to 0.6 ms of a
  YUYV frame; on one thread 0.7 and 1.0 ms.
- **Per slot, packed, on the thread that samples.** The thread that
  computes a slot converts what that slot reads right before it samples
  it, into an image of the rectangle of those tiles (0.8 MB for a slot of
  the bench's network) that its next slot overwrites. Another 1.1 to
  1.5 ms on two threads and 0.9 to 1.6 ms on one: the sampling reads what
  the core has just written (one thread: 6.5 ms where sampling the caller's
  RGB8 frame takes 7.5), nothing waits, and no image of the frame is
  written and read back. A stream allocates a second image of the frame's
  size for it, 6 MB at 1920 x 1080, of which the second thread uses the
  first 0.8.

What was tried and dropped, each measured on the same board (two threads,
p50 of the frame in ms):

- **Per slot on the sampling thread, but into the image of the frame's
  size** (a tile given to the slot that reads it, tiles that two slots read
  or the strip reads converted before the slots): NV12 13.8 and YUYV 14.4,
  against 13.2 and 13.6 before. The conversion was no faster for being
  next to its reader (2.0 ms per thread) and the sampling slower. It is the
  packing that gains, not the place in the pass.
- **Zeroing a row of the packed image before converting into it**, so that
  the cache need not fetch what is about to be overwritten: 12.0 and 12.1
  against 11.8 and 12.0.
- **RGB8 copied into a packed image as well.** Sampling gets faster by 0.2
  and 1.0 ms on the two threads, and the copy costs 1.6 and 2.0 ms: 11.9
  against 11.2 (one thread: 17.6 against 17.3). So **a plain copy of what a
  slot reads costs on two threads what the NV12 conversion now costs**, and
  what is left between the YUV formats and RGB8, 0.4 to 0.7 ms, is the
  price of bringing 0.8 MB per slot through the cache less what the
  sampling gains from it. More arithmetic taken out of the conversion would
  not show there.
- **Two ways to write the 16 pixels** that gcc 14 makes slow code of: a
  half vector widened with `__builtin_convertvector` goes through the
  general registers (2.5 ns per pixel), and the three colours put side by
  side by a loop over arrays instead of by six shuffles (1.8 ns).

### The tail behind a camera

Fed by a camera at 60 frames per second, a stream on two threads has a few
passes in 600 that take 17 to 36 ms where the median is 12 to 14, and
they are what kept NV12 and YUYV from the target of #533 after its third
step. The fourth step looked for where they come from, on the same
Raspberry Pi 5 (Debian 13, kernel 6.18 with 16 KB pages; the board also
runs other services, among them a metrics exporter and CI runner
services). **They come from the board and not from the pass**, and of what
a library can do about them only a faster pass and a word about binding
are left.

How: timers in the stream and the bench (not in the tree) recorded, for
every frame, when it was offered, taken and done, when each thread began
and ended its half of the conversion and of the slots and on which core,
the process's page faults and context switches (`getrusage`), how long each
thread ran and how long it waited for a core
(`/proc/self/task/*/schedstat`), the frequency of the four cores, and the
system's counters in `/proc/vmstat`. 44 runs of 600 frames in the three
formats, threads not bound: 26 399 passes, of which 221 results (0.84 %)
came later than a frame period. `tools/bench_stream --camera 60 --trace`
prints the part of this that needs nothing in the library: each slow pass
with the minor faults, the involuntary switches and the pages migrated
since the pass before it.

- **Memory compaction: 131 of the 221 late results (59 %).** While the bench
  runs, the kernel's compaction thread migrates 160 to 1750 pages every
  2.05 s (`pgmigrate_success` and `compact_isolated` in `/proc/vmstat`; 25
  of the 32 gaps between two such moments in a run are multiples of 2.05 s;
  on the idle board it was 9 pages in 10 s). `compact_stall` stays 0, so it
  is not this process allocating, and `compact_daemon_wake` moved by 2
  while dozens of such moments passed, so nothing woke the thread either:
  it is the proactive compaction `kcompactd` does on its own timer, with
  `vm.compaction_proactiveness` at its default of 20, on a board whose
  4 GB are eight NUMA nodes of 512 MB (`numa=fake=8` on the kernel's
  command line). The pages it moves are also the stream's. A pass that
  runs meanwhile takes minor faults (5 at the median, 18 at most), each on
  a page that is on its way, and waits for it; both threads are slow in
  the same pass, and in 10 of the 68 passes during which pages were
  migrated one of them also waited more than a millisecond for its core.
  67 of the 68 took more than 1.5 ms longer than the median: 7.1 ms longer
  at the median, 17 ms at most.
- **Another process at work: 75 (34 %).** Three or four passes in a row
  take 2 to 6 ms longer while the system counts 400 to 520 page faults that
  are not the stream's within one frame period: a metrics exporter is
  scraped every 30 s, another service is busy every few seconds. The
  stream's threads mostly keep their cores (they wait 0.03 ms for one at
  the median of these passes, 1.7 ms at most) and run slower: the four
  cores share 2 MB of cache and the memory behind it, which is what a frame
  is made of.
- **Neither: 15 (7 %)**, after passes that were 1.5 to 5 ms slow.
- **One slow pass makes several late results.** The next frame waits while
  it finishes, and each pass after it catches up by what a pass is shorter
  than the period: 2.3 ms for NV12 and YUYV, 4.4 ms for RGB8. A pass of
  27 to 30 ms was followed by five late results in YUYV and by two in
  RGB8. The queue of one drops nothing as long as one frame waits; it
  gives late results for lost ones.

What it is not, each of them measured: the frequency (all four cores at
2400 MHz at every frame, governor `ondemand`); the temperature (below
60 °C, `vcgencmd get_throttled` unchanged); a core taken away (0.1 to 0.3
involuntary switches a frame; the wait for a core 0 at the median and
0.08 ms at the 99th percentile of all passes); the second thread's
wake-up (2 to 20 µs); the camera thread's timer (it offers 57 µs
late at the median, and that is not in the latency); major faults (none);
the bench's own frames (16 of them, 4 to 6 MB each, in `malloc` memory,
which a camera does not hand out: with the frames in `/dev/dma_heap/system`,
which is where capture buffers come from and which compaction does not
move, three runs had 4, 0 and 31 late results against 7, 11 and 23; the
stream's 14 MB of tables and its converted tiles are still moved).

What was tried against it. `--camera 60 --frames 600`, two threads,
1920 x 1080, five runs per row and format, the rows of a session in turn:
the median of p50 in ms, the range of p99 over the five runs (its median
in brackets), and the late results of 3000. "Bound" is
`OMP_NUM_THREADS=2 OMP_PROC_BIND=true OMP_PLACES=cores taskset -c 2,3`:

| | RGB8 | NV12 | YUYV |
| :-- | --: | --: | --: |
| first session: the third step's tree | 12.2, 12.8 to 22.0 (15.7), 20 late | 14.3, 16.2 to 21.2 (20.0), 48 late | 14.2, 14.8 to 18.0 (15.1), 18 late |
| kernels by spans | 11.4, 11.8 to 13.2 (13.0), 4 late | 13.8, 14.3 to 20.1 (16.5), 28 late | 13.9, 14.4 to 19.9 (15.3), 13 late |
| kernels by spans, `taskset -c 2,3` | 11.3, 11.7 to 15.7 (13.3), 10 late | 13.7, 15.7 to 18.1 (17.5), 29 late | 13.8, 14.2 to 15.5 (14.5), 12 late |
| kernels by spans, bound | 11.2, 11.6 to 14.9 (12.5), 5 late | 13.1, 13.6 to 16.9 (15.7), 17 late | 13.3, 13.8 to 16.6 (14.3), 8 late |
| second session: the third step's tree, bound | 11.6, 12.4 to 16.5 (14.2), 10 late | 14.2, 14.7 to 24.6 (15.6), 22 late | 14.4, 14.8 to 19.9 (15.2), 22 late |
| kernels by spans, bound | 11.2, 11.7 to 13.8 (12.1), 9 late | 13.1, 13.6 to 21.4 (14.8), 37 late | 13.4, 14.0 to 16.4 (14.4), 8 late |

No frame was dropped in any of these 90 runs, and the worst single result
of each row was between 21.5 and 29.4 ms.

- **Kernels by spans** (see "Integer crops") make the pass shorter, by 0.3
  to 0.8 ms at the median here and by 1.0 ms for NV12 and YUYV once the
  threads are bound, so a slow pass is caught up with sooner and a small
  one fits. The median of p99 fell in five of the six comparisons (by 0.8
  to 3.5 ms) and stayed in one. How many results are late is the board's
  doing more than the pass's: 86 and then 45 of 9000 in the first session,
  54 and 54 in the second.
- **Binding the two threads to two cores**
  (`OMP_NUM_THREADS=2 OMP_PROC_BIND=true OMP_PLACES=cores taskset -c 2,3`;
  without `OMP_NUM_THREADS` the second thread is not used, see "Two
  threads"): the median 0.2 to 0.7 ms lower in the first session, the tail
  as before. `taskset` alone, which keeps the process on two cores and lets
  the threads change places, changes nothing.
- **`OMP_WAIT_POLICY=active`**: five runs with it had 80 late results of
  9000, five without it in the session before 70. Nothing.
- **Locking the memory** (`mlockall`) is not open to a process without
  privileges here (8 MB may be locked), and would not keep compaction away
  by itself: `vm.compact_unevictable_allowed` is 1. A real-time priority is
  not open either, and no core was taken away in the first place.

**The fifth step: a shorter pass.** The slow passes are the board's, so
what the library can add is room. With NV12 and YUYV converted 16 pixels at
a time, and then per slot and packed (see "Where the conversion's 2 ms
went"), a later session, the three trees in turn run by run, bound, 10 runs
of 600 frames per row and format: the median of p50 in ms, the range of
p99 over the ten runs (its median in brackets), the runs whose p99 is below
a frame period, and the late results of 6000:

| | RGB8 | NV12 | YUYV |
| :-- | --: | --: | --: |
| the fourth step's tree | 11.2, 11.1 to 15.0 (12.0), 10 of 10, 9 late | 13.1, 13.2 to 20.3 (13.7), 9 of 10, 21 late | 13.4, 13.9 to 23.4 (17.2), 4 of 10, 67 late |
| + converted 16 pixels at a time | 11.2, 11.7 to 15.1 (11.9), 10 of 10, 11 late | 12.8, 13.1 to 15.8 (13.8), 10 of 10, 16 late | 13.1, 13.7 to 17.2 (13.8), 9 of 10, 16 late |
| + per slot, packed | 11.2, 11.4 to 15.2 (12.0), 10 of 10, 10 late | 11.6, 11.9 to 14.1 (12.8), 10 of 10, 1 late | 11.9, 12.2 to 12.7 (12.5), 10 of 10, 9 late |

A pass now catches up by 5.0 ms (NV12) and 4.7 ms (YUYV) of what a slow one
before it lost, where it was 3.6 and 3.2, and 5.5 for RGB8. That session's
board also had one pass of 30 to 45 ms in 5 of the 10 RGB8 runs and in 4 of
the 10 YUYV runs of every tree, in the same rounds for all three (a round is
the three trees within two minutes), so it follows the clock and not the
tree: it makes one to three results late, which is all of the late results
RGB8 has in the table and all of YUYV's in its last row, and it mostly drops
a frame (of the 26 frames dropped in these 90 runs, 25 were dropped in the
27 runs that had such a pass); the 15 ms in the range of RGB8's p99 are
those runs. `--trace` on two later runs per format of the tree as it is
caught two such passes, of 24.6 and 33.8 ms: each with 7 minor faults of the
process while the kernel migrated 2652 and 2492 pages, which is the
compaction of the fourth step moving more pages at once than it did then
(160 to 1750); and one stretch of 30 passes of 13 to 19 ms with up to 23
involuntary switches in a pass and no page migrated, which is another
process on the stream's two cores.

What `--trace` prints for such a run (YUYV, the fourth step's tree; three
passes during a migration, and a group next to another process):

```
1920x1080  yuyv  stream  camera at 60.0 frames/s: 620 frames, 0 dropped (warm-up included), offer to result p50  13.621 ms   p99  19.247 ms   worst  26.970 ms, 8 later than a frame period
    frame  286 at  4.433 s: pass  19.18 ms after  0.02 ms in the queue, 3 minor faults, 0 involuntary switches, 372 pages migrated
    frame  314 at  4.900 s: pass  15.70 ms after  0.03 ms in the queue, 0 minor faults, 0 involuntary switches, 0 pages migrated
    frame  315 at  4.917 s: pass  15.72 ms after  0.01 ms in the queue, 0 minor faults, 0 involuntary switches, 0 pages migrated
    frame  316 at  4.933 s: pass  16.41 ms after  0.02 ms in the queue, 0 minor faults, 0 involuntary switches, 0 pages migrated
    frame  317 at  4.950 s: pass  15.97 ms after  0.10 ms in the queue, 0 minor faults, 0 involuntary switches, 0 pages migrated
    frame  409 at  6.483 s: pass  25.27 ms after  0.09 ms in the queue, 10 minor faults, 0 involuntary switches, 935 pages migrated
    frame  423 at  6.717 s: pass  15.10 ms after  0.06 ms in the queue, 0 minor faults, 0 involuntary switches, 0 pages migrated
    frame  532 at  8.533 s: pass  26.88 ms after  0.09 ms in the queue, 6 minor faults, 2 involuntary switches, 1214 pages migrated
    8 of 600 passes took more than 1.1 x the median 13.56 ms: 3 while the kernel migrated pages, 3 with minor faults of this process
```

The board's settings were not touched (it is shared), so what follows is
read from the counters and not measured: `vm.compaction_proactiveness=0`
is the setting that turns the proactive compaction off, which is behind
59 % of the late results; `mlockall` with
`vm.compact_unevictable_allowed=0` would keep the stream's pages where
they are whatever else starts a compaction. The second group goes with
the other services. A camera whose scene mostly stands still has the
margin anyway: gated, a frame is 5 ms.

### On a GPU

#533 kept a GPU for last: "only if the CPU path misses the target; the
frame upload usually costs more than these small nets". The sixth step
measured it where a GPU was in reach, on an Apple M1 Max through Metal, and
landed what the numbers there carried. **Nothing here is a number of the
Raspberry Pi 5's GPU**, which was not reachable; see "Not covered".

```c
struct geist_backend *metal = nullptr;
if (geist_backend_create("metal", nullptr, nullptr, &metal) == GEIST_OK &&
    geist_decknet_stream_set_backend(s, metal) == GEIST_OK) {
    /* frames that are not gated have their integer part on the GPU */
}
/* ... */
(void) geist_decknet_stream_set_backend(s, nullptr); /* the CPU again */
```

**The GPU takes what is integer, and the results are the same bits.** A
crop pixel is an integer, and a kernel is the sum of its pixels per
distinct weight (see "Integer crops"): 32-bit arithmetic that wraps, which
a shader computes as the CPU does. So the conversion of an NV12 or YUYV
frame, the 0.88 M bilinear samples and the pixel sums of both kernels go to
the GPU, and what comes back is integers. Everything behind them is F32 and
F64 in an order that is part of the result (the gains, the 1x1 units, the
weight of a kernel part times its sum, the pooling), and Metal has no F64:
that stays on the CPU, in the functions the CPU path computes it with
(`decknet_plane`, which is handed kernel sums as a gated stream hands it
kept ones). There is no tolerance to state: `test_decknet_stream_unit`
runs its frames once more on the GPU and compares with `memcmp`, and
`bench_stream --backend metal` hashes every output of every frame against
the still path.

A frame on the GPU (`push_gpu` in `stream.c`, `src/backends/metal/decknet.c`):

1. **The frame's rows are copied** into memory the GPU reads: the rows a
   sample reads, without their padding. The frame is the caller's again
   when the call returns, as on the CPU.
2. **Two command buffers are committed, one behind the other**, before
   anything is waited for. The first has the samples the CPU needs first:
   the worktable strip and the floor plane of every slot. The second has
   the samples of the other four planes and, behind them, the kernels: per
   pixel a kernel is evaluated at, and per part of the kernel, the sum of
   the crop pixels of its taps.
3. **The CPU waits for the first**, computes the gains and the floor planes
   (the 1x1 units and their region pools) **while the GPU runs the second**,
4. waits for the second, turns the part sums into the kernel sums (weight
   times sum, in F64, in the kernel's order) and pools the units.

An NV12 or YUYV pixel is converted where it is fetched, by the integer
formula under "Frames", so the three formats cost the same. Every buffer is
shared memory, which is the only kind an Apple GPU has: nothing is uploaded
but the frame's rows, and the results are read where the GPU wrote them.

- **A gated frame is the CPU's**, whatever backend is set: most of it is
  kept from the frame before, and what is left is less than a round trip to
  the GPU (with the bench's block moving, a gated frame takes 1.9 ms on
  one thread here and 2.1 ms behind the camera, where a frame on the GPU
  takes 2.6 and 2.9).
- **Two threads** (`geist_decknet_stream_set_threads(s, 2)`) share the
  CPU's part, slot by slot as on the CPU.
- **Allocation.** `set_backend` allocates the GPU's buffers (for the bench's
  network 14 MB of sample tables and 3.5 MB of destinations, 24 MB of crop
  pixels, 3 MB of sums, 6 MB for a frame) and compiles two kernels; a frame
  allocates nothing in the library (`test_decknet_stream_unit` counts).
  Metal itself makes two command buffers and their encoders per frame, as
  for every submission; they go with an autorelease pool that a frame
  opens and closes.
- **A device that fails a frame** gives `GEIST_E_BACKEND`, and the stream
  has no result until a frame succeeds; `set_backend(s, nullptr)` goes on
  on the CPU.
- **One user of the backend at a time.** While a call on the stream runs,
  no other thread uses that backend: the Metal backend's bookkeeping is
  one thread's, for a model's sessions as well.

The interface between the stream and a backend is
`include/geist_decknet_gpu.h`: a plan (how many samples, which pixel each
writes, the kernels as lists of taps, where they are evaluated), four
pieces of host memory (the sample table and the frame to write, the pixels
and the sums to read) and `submit` / `wait(stage)`. All tables are 32-bit
words: a sample is its three weights and `x | y << 16`, as on the CPU.

#### Measured on an M1 Max

Apple M1 Max (8 performance and 2 efficiency cores, 32 GPU cores, macOS
27.2, Apple clang 21), on mains power, `tools/bench_stream` with the
bench's network at 1920 x 1080, under the host lock. The machine is shared
and was not idle (load average 1.5 to 8 during the runs behind the camera,
the display in use), which is what the tails below are made of; no
temperature could be read. Four arms, run in turn round by round, the order moved on by one
with each round: the stream on the CPU on one thread and on two, and on
the Metal backend with one thread and with two for the CPU's part.

**Behind a camera at 60 frames per second** (`--camera 60 --frames 600
--only stream`), ten rounds, RGB8 / NV12 / YUYV: the median of p50 from the
offer of a frame to its result in ms, the median of p99 over the ten runs
and its range, the results of 6000 later than a frame period, and the
processor time of the process for 600 frames in seconds, all threads
(of which the bench's looking for a frame is 0.25 to 0.31 s in every arm):

| | p50 | p99, median (range) | late of 6000 | processor time, s |
| :-- | --: | --: | --: | --: |
| CPU, one thread | 5.25 / 5.31 / 5.38 | 7.3 (6.0 to 18.6) / 6.8 (6.0 to 21.1) / 7.4 (6.0 to 33.7) | 8 / 12 / 11 | 3.42 / 3.45 / 3.48 |
| CPU, two threads | 2.85 / 2.85 / 2.92 | 4.8 (4.3 to 8.6) / 4.6 (3.2 to 10.4) / 5.7 (3.7 to 24.0) | 2 / 4 / 13 | 3.57 / 3.60 / 3.70 |
| Metal, one thread | 2.95 / 2.90 / 2.87 | 5.0 (3.9 to 39.6) / 5.8 (4.7 to 15.6) / 5.4 (4.1 to 21.5) | 21 / 7 / 8 | 1.74 / 1.72 / 1.72 |
| Metal, two threads | 2.54 / 2.52 / 2.51 | 5.0 (3.7 to 17.7) / 4.4 (3.5 to 25.7) / 4.5 (3.9 to 9.3) | 6 / 11 / 2 | 1.80 / 1.76 / 1.77 |

**Back to back** (`--frames 600 --only stream`), ten rounds of the same
four arms, with the machine busier than before (load average 4.7 to 11.5):
the median of p50 of a frame in ms, the median of p99, and the processor
time for 600 frames in seconds:

| | p50 | p99, median | processor time, s |
| :-- | --: | --: | --: |
| CPU, one thread | 5.19 / 5.22 / 5.35 | 5.8 / 5.8 / 6.9 | 3.12 / 3.14 / 3.22 |
| CPU, two threads | 2.76 / 2.75 / 2.82 | 3.7 / 4.2 / 3.9 | 3.19 / 3.21 / 3.28 |
| Metal, one thread | 2.66 / 2.60 / 2.60 | 4.4 / 4.1 / 4.0 | 1.38 / 1.35 / 1.34 |
| Metal, two threads | 2.33 / 2.27 / 2.30 | 3.9 / 3.9 / 3.9 | 1.53 / 1.52 / 1.52 |

Every row of these forty runs hashed every output of every frame against
the still path's: the same bits. What the numbers say:

- **Against one CPU thread** the GPU path takes 45 % off a frame (2.9 ms
  for 5.3) and half of the processor time: 1.4 s for 600 frames where the
  CPU path takes 3.2, the bench's looking taken off both, which is 2.3 ms
  and 5.3 ms a frame.
- **Against two CPU threads** it is within 0.1 ms with one thread for the
  CPU's part (2.9 ms both; back to back 2.6 for 2.8) and 0.3 to 0.5 ms
  ahead with two (2.5 ms, back to back 2.3), at less than half the
  processor time either way (1.4 to 1.5 s for 3.3).
- **The formats cost the same** on the GPU, 2.87 to 2.95 ms; on the CPU a
  YUYV frame is 0.13 ms behind an RGB8 one.
- **The tail is the machine's, in every arm.** Each arm had runs with a
  pass of 23 to 120 ms and results later than a frame period; p99 stayed
  below a frame period in 27, 29, 28 and 28 of the 30 runs of the four
  arms. Nothing in these runs tells a slow pass of the GPU's from one of
  the machine's: the GPU is also the display's.
- **What the processor time leaves out** is the GPU, which works 1.3 ms of
  each frame (below). The time is the half of the energy a program can see;
  the watts were not measured.

Where a frame on the GPU goes, behind the camera, one thread for the CPU's
part unless the column says two: p50 of each part in ms over 600 frames,
the median of three runs (timers that are not in the tree; the GPU's own
time is the command buffers' `GPUStartTime` to `GPUEndTime`, read in the
prototype below):

| | RGB8 | NV12 | YUYV | RGB8, two threads |
| :-- | --: | --: | --: | --: |
| the frame's rows copied | 0.14 | 0.07 | 0.09 | 0.14 |
| both command buffers encoded and committed | 0.05 | 0.05 | 0.05 | 0.05 |
| waiting for the first; the GPU works 0.07 ms on it | 0.61 | 0.60 | 0.54 | 0.63 |
| gains and floor planes, on the CPU | 1.16 | 1.16 | 1.16 | 0.65 |
| waiting for the second; the GPU works 1.27 ms on it | 0.15 | 0.14 | 0.14 | 0.67 |
| the other planes: kernel sums from their parts, pooling, units | 0.82 | 0.82 | 0.82 | 0.44 |
| the frame, offer to result | 3.02 | 2.93 | 2.87 | |

- **The round trip of a command buffer is half a millisecond.** The first
  one is waited for 0.6 ms (0.39 to 0.70 between runs) for 0.07 ms of work:
  the rest is its way to the GPU and the way of its completion back, and
  nothing in the library shortens it. It is paid once: the second command
  buffer is under way meanwhile.
- **The GPU's 1.27 ms** are 0.2 ms for all 0.88 M samples and 0.9 ms for the
  8.0 M taps of the kernels (the prototype with one of the two at a time).
  With one thread for the CPU's part the GPU is done 0.15 ms after the CPU
  has the floor planes; with two the CPU waits 0.67 ms for it.
- **The CPU's part is 2.0 ms of the 2.9**, and is what the processor time
  of a GPU frame is made of.
- **The upload is 2 to 5 % of a frame**: the rows a sample reads, 3 to
  6 MB, copied in 0.07 to 0.14 ms. "The frame upload usually costs more than these small nets"
  does not hold where the GPU's memory is the CPU's.

**What was built to measure, before anything was written into the tree:**
a program outside it (Objective-C, 400 lines, linked against the library's
internals for the stream's tables) with the two kernels, one command buffer
or two, the three ways to wait, the caller's frame wrapped or copied, and
every frame compared with the CPU stream bit for bit. Five rounds in turn,
RGB8, 300 frames at 60 per second, p50 in ms:

| | p50 of a frame (range of five runs) |
| :-- | --: |
| one command buffer for everything, waited for | 4.10 (3.97 to 4.24) |
| two command buffers, `waitUntilCompleted` | 2.89 (2.86 to 3.03) |
| two, a completion handler that signals a semaphore | 2.79 (2.76 to 2.91) |
| two, the status polled in a loop | 2.85 (2.83 to 2.95) |
| two, waited for, the frame copied instead of wrapped | 3.04 (3.01 to 3.23) |

In it the GPU's own time was 1.32 ms for one command buffer and 0.07 and
1.27 ms for two, and a copy of the whole frame took 0.14 ms for RGB8
(6.2 MB), 0.07 ms for NV12 and 0.10 ms for YUYV where wrapping the caller's
memory took 0.001.

What it settled:

- **All of the integer part goes to the GPU, in two command buffers.** With
  one, the CPU waits 1.9 ms and then computes for 2.1; with two it computes
  the floor planes while the GPU sums the kernels, which takes 1.2 ms off
  a frame. Sampling alone on the GPU was not built: the round trip is paid
  in either case, and the kernels add 0.9 ms of GPU time that runs while
  the CPU is busy anyway.
- **The frame is copied, not wrapped.** Wrapping the caller's memory
  (`newBufferWithBytesNoCopy`) saves the 0.07 to 0.14 ms, and has
  conditions a stream's frames do not have: the memory has to start on a
  page and be whole pages (16 KB here); a wrapper per frame is an object
  allocated per frame; and wrappers kept per address rest on the caller
  never having other memory at an address the stream has seen, which
  nothing in a stream's contract says (what Metal does then was not
  tried). A copy needs none of that and leaves the frame the caller's when
  the call returns.
- **`waitUntilCompleted`.** A completion handler was 0.1 ms ahead at the
  median with ranges that overlap; it needs a block, and the library is C
  without blocks. Polling the status was no faster and cost 0.2 s more
  processor time per 300 frames.
- **What it cannot show** is a camera's own buffers; see "Not covered".

So on this machine the GPU path gains in latency and in processor time,
which is the part of the energy that was measured, and it is in the tree.
What that says about the board #533 is about is below.

#### Vulkan, and the Raspberry Pi 5

Not written, because it could not be run: the Pi 5 was not reachable, and
Vulkan on this Mac goes through MoltenVK, where the backend does not run.
What a port has to do is small, and what it has to show is not:

- **What carries over.** The two kernels are integer arithmetic on buffers
  of 32-bit words, with one thread per sample and one per kernel position;
  GLSL has the same unsigned arithmetic, so the shader logic and the tables
  (`include/geist_decknet_gpu.h`) stay as they are, and so do the stream's
  side and the tests, which would run against a second
  `geist_decknet_gpu_ops`. A frame's bytes would be read as words (a std430
  block has no bytes without an extension); the frame memory is whole words
  for that.
- **What does not.** Shared memory. A discrete GPU needs the frame's rows
  uploaded and the results read back: about 7 MB for the bench's network
  (the floor planes' pixels, the pixels of the white unit, the sums), not
  the 24 MB of crop memory. The interface has the two stages and the
  `samples_written` call for that, but no list of the pixels the CPU reads;
  a port for a discrete GPU adds it.
- **What must be measured on the Pi 5** (V3D 7.1, memory shared with the
  CPU): whether the round trip of a command buffer, which is
  half a millisecond here on a GPU that does this work in 1.3 ms, leaves
  anything of the 11 to 12 ms two Cortex-A76 cores take; how long V3D takes
  for 0.88 M samples and 8.0 M taps; whether the CPU's part (2 ms here on a
  core that is three to four times a Pi's) fits beside it; and what the
  board draws, which is the half of the target the Mac says nothing about.
  The 1x1 units and the pooling are a quarter of a CPU frame there (see
  "Where the frame went") and would stay on the CPU.
- **And on a desktop GPU**: the same, with the upload and the read-back in
  the time.

### Not covered

- **A capture loop.** The queue of one is two calls; the thread that
  captures, what wakes the thread that computes and the buffers of the
  camera are the caller's. No camera was connected: the bench's "camera" is
  a thread with a clock.
- **A board of its own.** The slow passes behind a camera come from memory
  compaction and from the board's other processes (see "The tail behind a
  camera"). The board is shared, so its settings stayed as they are: no run
  with `vm.compaction_proactiveness=0`, with locked memory, with isolated
  cores or without the other services.
- **The Raspberry Pi 5's GPU, and Vulkan.** The GPU path was measured on
  an Apple M1 Max, the one GPU in reach; the board was not reachable, and
  no Vulkan code was written because none could be run. Whether a GPU
  brings the board closer to the target of #533 or further from it is open;
  "Vulkan, and the Raspberry Pi 5" has what a port needs and what only the
  board can say.
- **Energy in watts.** The processor time of the process is what a program
  sees without privileges, and it does not include what the GPU draws for
  its part of a frame. No power was measured (`powermetrics` needs
  privileges), on the Mac or anywhere.
- **A camera's own buffers.** On Apple a camera hands out pixel buffers
  that lie on IOSurfaces. A stream takes a pointer to bytes, so a caller
  locks such a buffer and the stream copies its rows; binding the surface
  to the GPU as a texture would save that copy and was not built. The
  bench's frames are plain memory and its camera is a thread with a clock:
  it says what a copy costs and nothing about a surface.
- **A gated frame on the GPU.** A gated stream computes on the CPU, also
  the frame it computes whole because nearly every tile changed.
- **A model and a stream on one backend at once.** Not tried. The Metal
  backend's bookkeeping belongs to one thread at a time.
- **Shorter kernels on the GPU.** The GPU sums a kernel tap by tap, 81
  reads where the CPU's spans take 38. With one thread for the CPU's part
  the GPU is done when the CPU is; with two it is what a frame waits for.
- **The CPU's part of a GPU frame.** The 1x1 units, the pooling and the
  units are 2 ms of such a frame here. They are F32 and F64 in the
  reference's order; whether a shader could give the same bits (F64
  emulated in integers, F32 without contraction) was not tried.
- **A GPU that is busy, cold or hot.** The frames were measured at a steady
  60 per second and back to back, with the display's compositor as the
  GPU's only other user that is known of. The first frames after a pause,
  a GPU shared with another program's work and a throttled one were not
  measured, and the temperature cannot be read without privileges.
- **More of the kernels.** A stream sums a row of equal weights from two
  prefix sums: 38 reads instead of 81 for the bench's 9 x 9 kernel. The
  weight with the most taps could be the whole window less the others (28
  reads); sums down the columns as well (an integral image) would make a
  rectangle four reads, which a disc is not; and the still path and a gated
  stream's kept crops still sum tap by tap. None of it measured.
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
- **The rest of the conversion.** An NV12 or YUYV frame still takes 0.4
  to 0.7 ms longer than an RGB8 one on two threads. Each thread spends
  1.7 ms converting what its three slots read, where one thread alone
  converts all six in 1.8 to 2.0 ms, and a plain copy of the same bytes
  costs as much (see "Where the conversion's 2 ms went"): it is the 0.8 MB
  per slot going through a cache that two threads share, not the
  arithmetic. Smaller pieces were not tried: the five planes of a slot read
  the same pixels, so converting per plane converts each pixel up to five
  times, and converting a band of a slot for all five planes at once needs
  their five crops alive together, which is what made "the prefix sums next
  to the crop" slow. The arithmetic itself could be shorter with a
  platform's instructions (NEON has a narrowing shift that saturates and a
  store that interleaves three vectors, which the vector types cannot
  say); on one thread, where it is what counts, that was not tried either.
- **RGB8 on the second thread.** The second thread samples its three slots
  of an RGB8 frame 0.65 ms slower than the first samples its own, and the
  first waits. Packed images do not have that, but for RGB8 the copy costs
  more than it gains. Handing out slots one by one instead of in halves
  would not change three against three; splitting below a slot was not
  tried.
- **A gated stream's conversion.** A gated stream converts the tiles it
  takes on the calling thread, into the image of the whole frame, as
  before; it has the 16 pixels at a time and not the packing. With every
  tile changing that is 14.3 to 15.1 ms for NV12 and YUYV on two threads
  where a stream that is not gated takes 11.5 to 12.4.
- **The conversion on x86-64.** The 16 pixels at a time build and give the
  right bytes with gcc 14 and `-march=x86-64-v3` (all 2^24 inputs, and the
  unit tests, in an emulated container); how fast they are there was not
  measured.
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

- **A stream sums a row of equal weights from two prefix sums.** The loader
  also writes each part of a kernel as spans, the neighbours of one kernel
  row (`decknet_kernel.span`), and keeps them where the kernel has at least
  three taps per span: a 9 x 9 kernel of a disc and a rim is 19 spans for 81
  taps. A stream that samples a plane whole writes, per run of crop pixels
  the plane reads, the sum of the run's pixels before each pixel instead of
  the pixel (`decknet_work.prefix`; in 32 bits that wrap, in the buffer the
  crop would take), and a span's sum is the difference of two of them: the
  integer the taps give, since the sum itself fits 32 bits. A pixel is the
  difference of the two around it, which is what the taps at the crop's
  edge and the white unit read. The still path and a gated stream's kept
  crops stay pixels, summed tap by tap, so the parity tests compare the two
  ways with each other. A kernel whose weights are all different has no
  spans, and nothing changes for it.

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
  of interest, tiles, frame formats, gating, prefix sums for the kernels,
  the second thread, the queue of one, and the CPU's part of a frame whose
  integer part a GPU computed). One
  implementation in portable C, built without `-ffast-math` and without
  contraction. Its vectors are the compiler's (vector types for a crop pixel,
  for the sample that computes one and for four rows of the 1x1 layer,
  plain loops for the colour conversion): no intrinsics, no platform `#if`. No backend dispatches it,
  so there is no `cpu_scalar` / `cpu_neon` / `cpu_x86` variant to keep in
  step: the same code runs on every platform, and `test_decknet_unit` is its
  parity check there.
- `include/geist_decknet_gpu.h`: what a backend implements to take the
  integer part of a stream's frame (five functions and the layout of the
  tables, see "On a GPU"), and `src/backends/metal/decknet.c`, which
  implements it for Metal: two compute kernels in integer arithmetic. Built
  with `DECKNET=1` where the Metal backend is built; the arch knows the
  interface and no backend, the backend the interface and no arch.
- `src/engine/decknet.c`: the public entry points, and the composition point
  for the arch and for the backends that have a decknet path
  (`tools/layers.json`).
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
| `tests/test_decknet_stream_unit.c` | a stream against the still path, every output bit for bit: RGB8, NV12 and YUYV, padded rows, replaced homographies, across a frame that is not checkable, on one thread and on two; NV12 and YUYV frames of arbitrary bytes (the conversion over its whole range); a gated stream against the still path on the image of its kept tiles, which the test builds with its own tiles and its own sum of absolute differences, at threshold 0 and above it; gating at its edges (a kernel window across a cell of the crop, a tile exactly at the threshold and one above it, each tile changed on its own, one sample at a time across a seam of the sample tables, a frame that is not checkable between two that are); kernels with spans, which a stream sums from prefix sums and the still path tap by tap, at the corners and edges of a crop, one pixel inside them and in the middle, on one thread and two, and in a gated stream that goes both ways; the fixed-point sample itself (weights that never exceed the whole, a flat image sampled to itself, any four bytes to within 2/65536 of the exact value) and a kernel whose pixel sum passes 32 bits; no allocation after the create and the first `set_gate`, the first frame included, the queue's calls among them; 227 frame descriptions refused, exact-fit frames accepted; samples at the last pixels of a frame with 0 to 20 bytes behind it; the queue of one, step by step on one thread, with a consumer that takes one pass per three offers of a second thread, and with that thread offering without waiting (every frame comes back once, dropped or computed, the computed ones in order and with the bits of `push_frame`); state and argument checks; every allocation of a create and of a `set_gate` failing once. The gating tests assert which frames were computed from what was kept (they sample less than a whole frame). **And on a GPU**, where the `metal` backend is built in and has a device (the macOS legs; elsewhere this part is skipped and says so): every test above that pushes a frame once more with its streams on the backend, a stream that goes between the CPU and the GPU from frame to frame, in every format, gated in between, on one thread and two; what `set_backend` refuses (a CPU backend, a null stream); each of its allocations failing once; and that the GPU's memory holds the samples the result was computed from |
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
