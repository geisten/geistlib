/*
 * fuzz_decknet — fuzz the decknet loader and what it lets through (#532).
 *
 * A decknet file is a whole network an attacker may have written: sizes that
 * become allocations, F32 values that become array indices (the sparse
 * pooling rows), homographies that become sampling coordinates. The loader
 * (src/archs/decknet/load.c) has to refuse everything outside the contract,
 * and what it accepts has to be safe to RUN: an index it let through shows
 * as a bad read in the forward pass, not in the loader. So an accepted
 * input is run on two images, its slots are read, and its homographies are
 * replaced by bytes of the input and run again.
 *
 * geist_decknet_load_from_memory is the entry point: no file, no network.
 * Built only with DECKNET=1 (mk/common.mk).
 *
 * Two modes, as fuzz_gguf:
 *   libFuzzer  LLVMFuzzerTestOneInput, coverage-guided (clang).
 *   standalone GEIST_FUZZ_STANDALONE, deterministic PRNG mutations of the
 *              seed; prints how many inputs the loader accepted and ran,
 *              because a harness whose every input is refused at the magic
 *              proves nothing about the pass.
 */
#include <geist.h>
#include <geist_decknet.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* Images larger than this are not run: the file names the size, and a
 * 32768 x 32768 image would only test malloc. */
constexpr size_t MAX_IMAGE_BYTES = (size_t) 1 << 20U;

/* How far the inputs got: accepted by the loader, and run. The standalone
 * driver reports and checks them. External linkage, because under libFuzzer
 * nothing reads them and clang rejects a static that is only written. */
extern size_t fuzz_decknet_accepted, fuzz_decknet_ran;
size_t        fuzz_decknet_accepted, fuzz_decknet_ran;

/* Keeps the outputs read below from being optimized away. */
static volatile float sink;

static void read_slots(const struct geist_decknet *d, const struct geist_decknet_info *info) {
    float acc = 0.0f;
    for (size_t s = 0; s < info->n_slots; s++) {
        struct geist_decknet_slot slot;
        if (geist_decknet_slot(d, s, &slot) != GEIST_OK) {
            continue; /* the pair was not checkable */
        }
        acc += slot.occ_front + slot.occ_rear + slot.wall_front + slot.rack + slot.plate +
               slot.deep + slot.tube_n + slot.tube_r;
        for (size_t i = 0; i < info->n_tips; i++) {
            acc += slot.tips[i];
        }
        for (size_t i = 0; i < info->n_tubes; i++) {
            acc += slot.tubes[i] + slot.tubes_rot[i];
        }
    }
    sink = acc;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* Exact-size copy so a read one byte past the input lands in an ASan
     * redzone. */
    uint8_t *copy = malloc(size ? size : 1);
    if (copy == nullptr) {
        return 0;
    }
    memcpy(copy, data, size);
    struct geist_decknet   *d  = nullptr;
    const enum geist_status st = geist_decknet_load_from_memory(size, copy, &d);
    free(copy); /* the loader keeps nothing of the input */
    if (st != GEIST_OK) {
        if (d != nullptr) {
            abort(); /* a failed load handed out a handle */
        }
        return 0; /* refusing a malformed file is the expected outcome */
    }
    fuzz_decknet_accepted++;

    struct geist_decknet_info info;
    if (geist_decknet_info(d, &info) != GEIST_OK || strlen(info.slots) != info.n_slots) {
        abort();
    }
    const size_t n_bytes = info.image_width * info.image_height * 3;
    if (n_bytes <= MAX_IMAGE_BYTES) {
        uint8_t *cur = malloc(n_bytes);
        uint8_t *ref = malloc(n_bytes);
        float   *h   = malloc(info.n_homographies * 9 * sizeof(float));
        if (cur != nullptr && ref != nullptr && h != nullptr) {
            /* Two images out of the input's bytes; the current one a little
             * off the empty one, so that some pairs stay checkable. */
            for (size_t i = 0; i < n_bytes; i++) {
                ref[i] = data[i % size];
                cur[i] = (uint8_t) (ref[i] ^ (i % 97 == 0 ? data[(i / 97) % size] & 0x1f : 0));
            }
            struct geist_decknet_result res;
            if (geist_decknet_run(d, n_bytes, cur, ref, &res) != GEIST_OK) {
                abort();
            }
            read_slots(d, &info);
            /* Homographies out of the input: any bit pattern. The setter
             * takes the finite ones; the pass must survive those, however
             * far outside the image they point. */
            for (size_t i = 0; i < info.n_homographies * 9; i++) {
                uint8_t raw[4];
                for (size_t k = 0; k < 4; k++) {
                    raw[k] = data[(i * 4 + k + 7) % size];
                }
                memcpy(&h[i], raw, 4);
            }
            if (geist_decknet_set_homographies(
                        d, info.n_homographies, GEIST_DECKNET_IMAGE_CURRENT, h) == GEIST_OK) {
                if (geist_decknet_run(d, n_bytes, cur, ref, &res) != GEIST_OK) {
                    abort();
                }
                read_slots(d, &info);
            }
            fuzz_decknet_ran++;
        }
        free(cur);
        free(ref);
        free(h);
    }
    geist_decknet_destroy(d);
    return 0;
}

/* ---- standalone driver ------------------------------------------------ */

/* The seed builder serves the standalone driver only: libFuzzer reads the
 * corpus file that `--seed` wrote, so under -fsanitize=fuzzer it would be
 * dead code, an error under -Werror. */
#ifdef GEIST_FUZZ_STANDALONE
#include "decknet_tiny.h"

/* splitmix64 — deterministic, so a failing run reproduces from its seed. */
static uint64_t prng(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int main(int argc, char **argv) {
    struct tiny m;
    tiny_default(&m);
    static struct tiny_buf seed;
    tiny_write(&m, &seed);

    if (argc == 2 && strcmp(argv[1], "--seed") == 0) {
        fwrite(seed.b, 1, seed.n, stdout);
        return 0;
    }

    const long runs = (argc == 2) ? strtol(argv[1], nullptr, 10) : 3000;
    uint64_t   s    = 0x5EEDull;

    /* The unmutated seed first: if the valid file does not load and run,
     * the mutations below test nothing. */
    LLVMFuzzerTestOneInput(seed.b, seed.n);
    if (fuzz_decknet_ran != 1) {
        fprintf(stderr, "fuzz_decknet: the seed itself was not accepted and run\n");
        return 1;
    }

    for (long i = 0; i < runs; i++) {
        static uint8_t input[sizeof seed.b];
        /* One in four inputs is cut short; the others keep their length, so
         * a poke lands in a header, a metadata value or a tensor payload. */
        const size_t size = (prng(&s) % 4 == 0) ? 1 + (size_t) (prng(&s) % seed.n) : seed.n;
        memcpy(input, seed.b, size);
        const unsigned pokes = 1u + (unsigned) (prng(&s) % 4u);
        for (unsigned p = 0; p < pokes; p++) {
            input[prng(&s) % size] = (uint8_t) prng(&s);
        }
        LLVMFuzzerTestOneInput(input, size);
    }
    printf("fuzz_decknet: %ld runs, no crash; %zu inputs accepted by the loader, %zu of them run\n",
           runs,
           fuzz_decknet_accepted - 1,
           fuzz_decknet_ran - 1);
    /* A harness that only ever sees refusals has stopped testing the pass. */
    if (runs >= 1000 && fuzz_decknet_ran < 50) {
        fprintf(stderr, "fuzz_decknet: too few accepted inputs to exercise the forward pass\n");
        return 1;
    }
    return 0;
}
#endif
