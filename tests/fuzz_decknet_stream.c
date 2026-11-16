/*
 * fuzz_decknet_stream — fuzz the frame geometry of a decknet stream (#533).
 *
 * A frame is the caller's memory described by the caller: format, width,
 * height, stride and extent. geist_decknet_stream_push_frame checks that
 * description once and then indexes the buffer through the sample tables
 * without a further check, so a description it wrongly accepts is a read
 * behind the buffer. The input is such a description plus pixel bytes:
 *
 *   [0]     format: 0..2 as they are, anything else a value the enum does
 *           not name
 *   [1] [2] width and height, mostly the model's
 *   [3..4]  stride, [5..6] extent in bytes (little endian)
 *   [7]     bit 0: replace the homographies by bytes of the input first, so
 *           that the tables point anywhere; bit 1: the frame is the
 *           reference as well; bit 2: the gating sequence below, with the
 *           threshold from bits 3 to 7
 *   [8..]   pixel bytes, repeated to fill the extent
 *
 * The buffer handed over is exactly `extent` bytes long, so ASan sees a read
 * one byte behind it. And "no crash" is not the assertion: the harness works
 * out for itself whether the description fits the buffer, and aborts when
 * the stream accepts a frame that does not fit or refuses one that does.
 *
 * Gating (geist_decknet_stream_set_gate) keeps tiles and what was computed
 * from them, so what it can get wrong is a value that is stale. The gating
 * sequence pushes four frames (one, the same again, two with bytes changed)
 * through a gated stream and through one without gating, on a 48 x 40 image
 * of nine tiles, and aborts when at threshold 0 the two differ in a bit, or
 * when at any threshold the same frame twice gives two results.
 *
 * The model is tests/decknet_tiny.h (16 x 12, two slots). Built only with
 * DECKNET=1 (mk/common.mk).
 *
 * Two modes, as fuzz_gguf:
 *   libFuzzer  LLVMFuzzerTestOneInput, coverage-guided (clang).
 *   standalone GEIST_FUZZ_STANDALONE, deterministic PRNG mutations of three
 *              valid frames; prints how many were accepted and run, and
 *              fails when either side is hardly reached.
 */
#include "decknet_tiny.h"

#include <geist.h>
#include <geist_decknet.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

constexpr size_t HEADER = 8;

/* How far the inputs got. External linkage, because under libFuzzer nothing
 * reads them and clang rejects a static that is only written. */
extern size_t fuzz_stream_accepted, fuzz_stream_refused, fuzz_stream_gated, fuzz_stream_partial;
size_t        fuzz_stream_accepted, fuzz_stream_refused, fuzz_stream_gated, fuzz_stream_partial;

static volatile float sink;

/* The harness's own answer to "does this frame lie inside its buffer", for
 * the tiny model. All quantities are below 2^16, so nothing overflows. */
static bool fits(uint32_t format, uint64_t w, uint64_t h, uint64_t stride, uint64_t n_bytes) {
    if (format > 2 || w != TINY_IMG_W || h != TINY_IMG_H) {
        return false;
    }
    const uint64_t row  = format == 0 ? w * 3 : format == 1 ? w : w * 2;
    const uint64_t rows = format == 1 ? h + h / 2 : h;
    return stride >= row && (rows - 1) * stride + row <= n_bytes;
}

static void read_outputs(const struct geist_decknet_stream *s) {
    struct geist_decknet_stream_result res;
    if (geist_decknet_stream_result(s, &res) != GEIST_OK) {
        abort(); /* an accepted frame has a result */
    }
    float acc = res.image.gain[0] + res.image.bg_changed;
    for (size_t i = 0; i < 2; i++) {
        struct geist_decknet_slot slot;
        if (geist_decknet_stream_slot(s, i, &slot) != GEIST_OK) {
            continue; /* the pair was not checkable */
        }
        acc += slot.occ_front + slot.rack + slot.tips[0] + slot.tubes[0] + slot.tubes_rot[0];
    }
    sink = acc;
}

/* ---- gating ----------------------------------------------------------- */

constexpr size_t GATE_W       = 48;
constexpr size_t GATE_H       = 40;
constexpr size_t GATE_OUTPUTS = 64;

/* Every output of the last frame, as bits. Returns how many. */
static size_t outputs(const struct geist_decknet_stream *s, uint32_t out[static GATE_OUTPUTS]) {
    struct geist_decknet_stream_result res;
    if (geist_decknet_stream_result(s, &res) != GEIST_OK) {
        abort();
    }
    float  v[GATE_OUTPUTS] = {res.image.checkable ? 1.0f : 0.0f,
                              res.image.gain[0],
                              res.image.gain[1],
                              res.image.gain[2],
                              res.image.bg_changed};
    size_t n               = 5;
    for (size_t i = 0; res.image.checkable && i < 2; i++) {
        struct geist_decknet_slot slot;
        if (geist_decknet_stream_slot(s, i, &slot) != GEIST_OK) {
            abort();
        }
        const float one[13] = {slot.occ_front,
                               slot.occ_rear,
                               slot.wall_front,
                               slot.rack,
                               slot.plate,
                               slot.deep,
                               slot.tube_n,
                               slot.tube_r,
                               slot.tips[0],
                               slot.tips[1],
                               slot.tubes[0],
                               slot.tubes_rot[0]};
        memcpy(v + n, one, sizeof one);
        n += 13;
    }
    memcpy(out, v, sizeof v);
    return n;
}

static void gate_sequence(const uint8_t *data, size_t size) {
    const uint8_t *pixels    = data + HEADER;
    const size_t   n_pixels  = size - HEADER;
    const uint32_t format    = data[0] % 3;
    const uint32_t wish      = data[7] >> 3U;
    const uint32_t threshold = wish % 4 == 0 ? 0 : wish * 8; /* 0 for a quarter of the inputs */

    /* The tiny network on a larger image and crop: nine tiles, and a crop
     * of several cells. The homographies spread the crop over the image. */
    struct tiny m;
    tiny_default(&m);
    m.img_w  = (uint32_t) GATE_W;
    m.img_h  = (uint32_t) GATE_H;
    m.crop_w = 40;
    m.crop_h = 36;
    float h[11 * 9];
    for (size_t i = 0; i < 11; i++) {
        const float spread[9] = {1.1f, 0.3f, 0.5f + (float) i * 0.2f, 0.2f, 9.0f, 0.5f, 0, 0, 1};
        memcpy(h + i * 9, spread, sizeof spread);
    }
    if (data[7] & 1) {
        for (size_t i = 0; i < 11 * 9; i++) {
            uint8_t raw[4];
            for (size_t k = 0; k < 4; k++) {
                raw[k] = pixels[(i * 4 + k) % n_pixels];
            }
            memcpy(&h[i], raw, 4);
        }
    }
    tiny_set(&m.t[TINY_T_HOMOGRAPHY], "decknet.homography", 3, 11, 3, 3, 99, h);
    static struct tiny_buf file;
    tiny_write(&m, &file);

    const size_t row    = format == 0 ? GATE_W * 3 : format == 1 ? GATE_W : GATE_W * 2;
    const size_t rows   = format == 1 ? GATE_H + GATE_H / 2 : GATE_H;
    const size_t stride = row + data[3] % 8;
    const size_t n      = (rows - 1) * stride + row;
    uint8_t     *buf    = malloc(n);
    if (buf == nullptr) {
        abort();
    }
    for (size_t i = 0; i < n; i++) {
        buf[i] = pixels[i % n_pixels];
    }
    const struct geist_decknet_frame frame = {.format  = (enum geist_decknet_format) format,
                                              .width   = GATE_W,
                                              .height  = GATE_H,
                                              .stride  = stride,
                                              .n_bytes = n,
                                              .data    = buf};

    /* [0] without gating, [1] gated. The frame is its own empty deck, so the
     * pair is checkable until a change reaches the worktable strip. A file
     * the loader refuses (homographies that are not finite) ends here. */
    struct geist_decknet        *d[2] = {nullptr, nullptr};
    struct geist_decknet_stream *s[2] = {nullptr, nullptr};
    bool                         ok   = true;
    for (size_t i = 0; ok && i < 2; i++) {
        ok = geist_decknet_load_from_memory(file.n, file.b, &d[i]) == GEIST_OK;
        if (ok && (geist_decknet_stream_create(d[i], &s[i]) != GEIST_OK ||
                   geist_decknet_stream_set_gate(s[i], i == 1, threshold) != GEIST_OK ||
                   geist_decknet_stream_set_reference(s[i], &frame) != GEIST_OK)) {
            abort();
        }
    }
    uint32_t first[GATE_OUTPUTS] = {0};
    bool     partial             = false;
    for (size_t step = 0; ok && step < 4; step++) {
        if (step == 2) { /* a stretch of bytes, somewhere */
            const size_t at = ((size_t) data[5] | (size_t) data[6] << 8) % n;
            for (size_t i = 0; i < 1 + (size_t) data[4] % 64; i++) {
                buf[(at + i) % n] ^= 0x5A;
            }
        } else if (step == 3) { /* and one byte by one */
            buf[((size_t) data[4] * 131 + data[5]) % n] += 1;
        }
        uint32_t out[2][GATE_OUTPUTS] = {{0}, {0}};
        for (size_t i = 0; i < 2; i++) {
            if (geist_decknet_stream_push_frame(s[i], &frame) != GEIST_OK) {
                abort();
            }
            (void) outputs(s[i], out[i]);
        }
        struct geist_decknet_stream_result res;
        (void) geist_decknet_stream_result(s[1], &res);
        partial = partial || (res.image.checkable && step > 0 && res.n_tiles_changed > 0 &&
                              res.n_tiles_changed < res.n_tiles);
        if (step == 0) {
            memcpy(first, out[1], sizeof first);
        }
        if (threshold == 0 && memcmp(out[0], out[1], sizeof out[0]) != 0) {
            abort(); /* threshold 0 is not the stream without gating */
        }
        if (step == 1 && memcmp(first, out[1], sizeof first) != 0) {
            abort(); /* the same frame twice, two results */
        }
    }
    fuzz_stream_gated += ok;
    fuzz_stream_partial += partial;
    for (size_t i = 0; i < 2; i++) {
        geist_decknet_stream_destroy(s[i]);
        geist_decknet_destroy(d[i]);
    }
    free(buf);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < HEADER + 1) {
        return 0;
    }
    if (data[7] & 4) {
        gate_sequence(data, size);
    }
    static const size_t widths[8] = {
            TINY_IMG_W, TINY_IMG_W, TINY_IMG_W, TINY_IMG_W, TINY_IMG_W, 14, 18, 0};
    static const size_t heights[8] = {
            TINY_IMG_H, TINY_IMG_H, TINY_IMG_H, TINY_IMG_H, TINY_IMG_H, 10, 14, 0};
    const uint32_t format   = data[0] < 3 ? data[0] : 0x100u + data[0];
    const size_t   stride   = (size_t) data[3] | (size_t) data[4] << 8;
    const size_t   n_bytes  = (size_t) data[5] | (size_t) data[6] << 8;
    const uint8_t  flags    = data[7];
    const uint8_t *pixels   = data + HEADER;
    const size_t   n_pixels = size - HEADER;

    struct tiny m;
    tiny_default(&m);
    static struct tiny_buf file;
    tiny_write(&m, &file);
    struct geist_decknet        *d = nullptr;
    struct geist_decknet_stream *s = nullptr;
    if (geist_decknet_load_from_memory(file.n, file.b, &d) != GEIST_OK ||
        geist_decknet_stream_create(d, &s) != GEIST_OK) {
        abort();
    }

    if (flags & 1) {
        /* Any finite bit patterns as homographies: the tables clamp them
         * into the image, wherever they point. */
        float h[11 * 9];
        for (size_t i = 0; i < 11 * 9; i++) {
            uint8_t raw[4];
            for (size_t k = 0; k < 4; k++) {
                raw[k] = pixels[(i * 4 + k) % n_pixels];
            }
            memcpy(&h[i], raw, 4);
        }
        (void) geist_decknet_set_homographies(d, 11, GEIST_DECKNET_IMAGE_CURRENT, h);
        (void) geist_decknet_set_homographies(d, 11, GEIST_DECKNET_IMAGE_EMPTY, h);
    }

    /* Exactly n_bytes, so a read behind the extent is a read behind the
     * allocation. */
    uint8_t *buf = malloc(n_bytes ? n_bytes : 1);
    if (buf == nullptr) {
        abort();
    }
    for (size_t i = 0; i < n_bytes; i++) {
        buf[i] = pixels[i % n_pixels];
    }
    const struct geist_decknet_frame frame = {
            .format    = (enum geist_decknet_format) format,
            .width     = widths[data[1] % 8],
            .height    = heights[data[2] % 8],
            .stride    = stride,
            .n_bytes   = n_bytes,
            .data      = buf,
            .timestamp = size,
    };
    const bool fit = fits(format, frame.width, frame.height, stride, n_bytes);

    /* A reference that is always valid, unless the input asks for its own. */
    uint8_t                    grey[TINY_IMG_W * TINY_IMG_H * 3];
    struct geist_decknet_frame ref = {.format  = GEIST_DECKNET_FORMAT_RGB8,
                                      .width   = TINY_IMG_W,
                                      .height  = TINY_IMG_H,
                                      .stride  = TINY_IMG_W * 3,
                                      .n_bytes = sizeof grey,
                                      .data    = grey};
    memset(grey, 128, sizeof grey);
    if (geist_decknet_stream_set_reference(s, &ref) != GEIST_OK) {
        abort();
    }
    if (flags & 2) {
        if (geist_decknet_stream_set_reference(s, &frame) !=
            (fit ? GEIST_OK : GEIST_E_INVALID_ARG)) {
            abort(); /* the reference is checked as a frame is */
        }
    }

    const enum geist_status st = geist_decknet_stream_push_frame(s, &frame);
    if (st != (fit ? GEIST_OK : GEIST_E_INVALID_ARG)) {
        abort(); /* accepted a frame that does not fit, or refused one that does */
    }
    if (st == GEIST_OK) {
        read_outputs(s);
        fuzz_stream_accepted++;
    } else {
        fuzz_stream_refused++;
    }
    free(buf);
    geist_decknet_stream_destroy(s);
    geist_decknet_destroy(d);
    return 0;
}

/* ---- standalone driver ------------------------------------------------ */

/* libFuzzer reads the corpus file that `--seed` wrote, so under
 * -fsanitize=fuzzer the seeds would be dead code, an error under -Werror. */
#ifdef GEIST_FUZZ_STANDALONE
constexpr size_t SEED_MAX = HEADER + TINY_IMG_W * TINY_IMG_H * 3;

/* A valid frame of `format` with rows padded by `pad`, as an input. */
static size_t seed(uint8_t format, size_t pad, uint8_t out[static SEED_MAX]) {
    const size_t row     = format == 0 ? TINY_IMG_W * 3 : format == 1 ? TINY_IMG_W : TINY_IMG_W * 2;
    const size_t rows    = format == 1 ? TINY_IMG_H + TINY_IMG_H / 2 : TINY_IMG_H;
    const size_t stride  = row + pad;
    const size_t n_bytes = (rows - 1) * stride + row;
    out[0]               = format;
    out[1]               = 0;
    out[2]               = 0;
    out[3]               = (uint8_t) stride;
    out[4]               = (uint8_t) (stride >> 8);
    out[5]               = (uint8_t) n_bytes;
    out[6]               = (uint8_t) (n_bytes >> 8);
    out[7]               = 0;
    for (size_t i = HEADER; i < SEED_MAX; i++) {
        out[i] = (uint8_t) (110 + i % 40);
    }
    return SEED_MAX;
}

/* splitmix64 — deterministic, so a failing run reproduces from its seed. */
static uint64_t prng(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int main(int argc, char **argv) {
    static uint8_t input[SEED_MAX];
    if (argc == 2 && strcmp(argv[1], "--seed") == 0) {
        fwrite(input, 1, seed(1, 3, input), stdout);
        return 0;
    }
    const long runs = (argc == 2) ? strtol(argv[1], nullptr, 10) : 3000;
    uint64_t   s    = 0x533ull;

    /* The unmutated frames first: if a valid frame is not accepted, the
     * mutations below test only the refusals. */
    for (uint8_t format = 0; format < 3; format++) {
        LLVMFuzzerTestOneInput(input, seed(format, 5, input));
    }
    if (fuzz_stream_accepted != 3) {
        fprintf(stderr, "fuzz_decknet_stream: a valid frame was not accepted\n");
        return 1;
    }
    for (long i = 0; i < runs; i++) {
        const size_t size = seed((uint8_t) (prng(&s) % 3), (size_t) (prng(&s) % 9), input);
        if (i % 3 == 0) { /* a third with the gating sequence, any threshold */
            input[7] = (uint8_t) (4u | (uint8_t) (prng(&s) & 0xF8u) | (uint8_t) (prng(&s) & 1u));
            input[4] = (uint8_t) prng(&s);
            input[5] = (uint8_t) prng(&s);
            input[6] = (uint8_t) prng(&s);
            for (size_t k = HEADER; k < size; k += 1 + (size_t) (prng(&s) % 7)) {
                input[k] = (uint8_t) (input[k] + (uint8_t) (prng(&s) % 9));
            }
            LLVMFuzzerTestOneInput(input, size);
            continue;
        }
        /* One to three pokes, two in three of them into the description. */
        const unsigned pokes = 1u + (unsigned) (prng(&s) % 3u);
        for (unsigned p = 0; p < pokes; p++) {
            const size_t at =
                    prng(&s) % 3 != 0 ? (size_t) (prng(&s) % HEADER) : (size_t) (prng(&s) % size);
            /* Half of the pokes move a value by a little: around the exact
             * fit is where an off-by-one would be. */
            input[at] = prng(&s) % 2 == 0 ? (uint8_t) prng(&s)
                                          : (uint8_t) (input[at] + (uint8_t) (prng(&s) % 5) - 2);
        }
        LLVMFuzzerTestOneInput(input, size);
    }
    printf("fuzz_decknet_stream: %ld runs, no crash and no wrong verdict; %zu frames accepted and "
           "run, %zu refused; %zu gating sequences, %zu of them with a part of the tiles taken\n",
           runs,
           fuzz_stream_accepted - 3,
           fuzz_stream_refused,
           fuzz_stream_gated,
           fuzz_stream_partial);
    if (runs >= 1000 && (fuzz_stream_accepted < 100 || fuzz_stream_refused < 100 ||
                         fuzz_stream_gated < 100 || fuzz_stream_partial < 100)) {
        fprintf(stderr, "fuzz_decknet_stream: one side of the check is hardly reached\n");
        return 1;
    }
    return 0;
}
#endif
