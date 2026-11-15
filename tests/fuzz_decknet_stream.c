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
 *           reference as well
 *   [8..]   pixel bytes, repeated to fill the extent
 *
 * The buffer handed over is exactly `extent` bytes long, so ASan sees a read
 * one byte behind it. And "no crash" is not the assertion: the harness works
 * out for itself whether the description fits the buffer, and aborts when
 * the stream accepts a frame that does not fit or refuses one that does.
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
extern size_t fuzz_stream_accepted, fuzz_stream_refused;
size_t        fuzz_stream_accepted, fuzz_stream_refused;

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

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < HEADER + 1) {
        return 0;
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
           "run, %zu refused\n",
           runs,
           fuzz_stream_accepted - 3,
           fuzz_stream_refused);
    if (runs >= 1000 && (fuzz_stream_accepted < 100 || fuzz_stream_refused < 100)) {
        fprintf(stderr, "fuzz_decknet_stream: one side of the check is hardly reached\n");
        return 1;
    }
    return 0;
}
#endif
