/*
 * fuzz_gguf — fuzz the GGUF reader, the one parser that reads a whole file
 * an attacker may have written.
 *
 * gguf_reader.c:153 says it plainly: model metadata is attacker-controlled.
 * The reader walks a header, a metadata table of variable-length strings and
 * arrays, and a tensor table, then hands out pointers into the mapping. Every
 * rule in AGENT.md §3 and §4 — ckd_mul before an allocation, bounds by
 * subtraction before the pointer moves — is an invariant this harness tries to
 * break with bytes instead of with review.
 *
 * gguf_open_memory is the entry point, so no file, no model and no network:
 * the input IS the GGUF. Accessors run after a successful parse, because a
 * reader that accepts a malformed file and then hands out a bad pointer is the
 * bug that matters, and it only shows on use.
 *
 * Two modes, following geist-memory's fuzz_store.c:
 *   libFuzzer  LLVMFuzzerTestOneInput, coverage-guided (clang).
 *   standalone GEIST_FUZZ_STANDALONE, deterministic PRNG mutations of the
 *              seed, so the gate also runs where CI has only gcc.
 */
#include "gguf_reader.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The seed builder serves the standalone driver only: libFuzzer reads the
 * corpus file that `--seed` wrote, so under -fsanitize=fuzzer these helpers
 * are dead code and clang says so under -Werror. */
#ifdef GEIST_FUZZ_STANDALONE

/* ---- append-only byte buffer for building the seed -------------------- */

struct buf {
    uint8_t b[1024];
    size_t  n;
};

static void put_bytes(struct buf *o, const void *p, size_t n) {
    if (o->n + n > sizeof(o->b)) {
        fprintf(stderr, "fuzz_gguf: seed buffer overflow\n");
        exit(1);
    }
    memcpy(o->b + o->n, p, n);
    o->n += n;
}
static void put_u32(struct buf *o, uint32_t v) {
    put_bytes(o, &v, 4);
}
static void put_u64(struct buf *o, uint64_t v) {
    put_bytes(o, &v, 8);
}
static void put_f32(struct buf *o, float v) {
    put_bytes(o, &v, 4);
}
/* GGUF string: u64 length + raw bytes, no NUL. */
static void put_gstr(struct buf *o, const char *s) {
    const size_t len = strlen(s);
    put_u64(o, (uint64_t) len);
    put_bytes(o, s, len);
}

/* GGUF metadata value types (mirrors the reader's private enum). */
enum { VT_U32 = 4, VT_F32 = 6, VT_BOOL = 7, VT_STRING = 8, VT_ARRAY = 9 };

#define GGUF_MAGIC 0x46554747u

/* One valid GGUF carrying a value of every metadata type the reader
 * understands plus one F32 tensor — a seed that reaches every branch a
 * mutation can then corrupt. */
static void seed_gguf(struct buf *o) {
    o->n = 0;
    put_u32(o, GGUF_MAGIC);
    put_u32(o, 3); /* version */
    put_u64(o, 1); /* n_tensors */
    put_u64(o, 5); /* n_meta */

    put_gstr(o, "general.alignment");
    put_u32(o, VT_U32);
    put_u32(o, 32);
    put_gstr(o, "general.architecture");
    put_u32(o, VT_STRING);
    put_gstr(o, "llama");
    put_gstr(o, "fuzz.f32");
    put_u32(o, VT_F32);
    put_f32(o, 1.5f);
    put_gstr(o, "fuzz.flag");
    put_u32(o, VT_BOOL);
    put_bytes(o, "\1", 1);
    put_gstr(o, "fuzz.arr");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_U32);
    put_u64(o, 3);
    put_u32(o, 1);
    put_u32(o, 2);
    put_u32(o, 3);

    put_gstr(o, "w");
    put_u32(o, 1);  /* n_dims */
    put_u64(o, 32); /* dim 0 */
    put_u32(o, 0);  /* GGUF F32 */
    put_u64(o, 0);  /* offset */

    while (o->n % 32 != 0) {
        const uint8_t z = 0;
        put_bytes(o, &z, 1);
    }
    for (int i = 0; i < 128; i++) {
        const uint8_t v = (uint8_t) i;
        put_bytes(o, &v, 1);
    }
}

#endif /* GEIST_FUZZ_STANDALONE */

/* ---- the target ------------------------------------------------------- */

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* Keeps the payload read below from being optimized away. */
static volatile uint8_t sink;

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* Exact-size copy so a read one byte past the input lands in an ASan
     * redzone instead of in whatever the caller's buffer happens to own. */
    uint8_t *copy = malloc(size ? size : 1);
    if (copy == nullptr)
        return 0;
    memcpy(copy, data, size);

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(copy, size, &err);
    if (ctx == nullptr) {
        free(copy);
        return 0; /* rejecting malformed input is the expected outcome */
    }

    /* A parse that succeeded promises usable metadata. Walk everything the
     * reader hands out — this is where a bad offset surfaces. */
    const size_t n = gguf_tensor_count(ctx);
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(ctx, i);
        if (t == nullptr)
            continue;
        (void) gguf_tensor_elem_count(t);
        (void) gguf_dtype_name(t->dtype);
        (void) gguf_get_tensor(ctx, t->name);
        /* The payload pointer is what a kernel will read. If the reader let a
         * bad offset through, this touch is where ASan says so — the check the
         * accessors alone cannot make. */
        if (t->data != nullptr && t->nbytes > 0) {
            const uint8_t *p   = t->data;
            uint8_t        acc = 0;
            for (size_t k = 0; k < t->nbytes; k++)
                acc = (uint8_t) (acc ^ p[k]);
            sink = (uint8_t) (sink ^ acc);
        }
    }
    /* One index past the end must be refused, not served. */
    (void) gguf_tensor_at(ctx, n);

    size_t   slen = 0;
    uint32_t u32  = 0;
    float    f32  = 0.0f;
    bool     flag = false;
    (void) gguf_get_meta_string(ctx, "general.architecture", &slen);
    (void) gguf_get_meta_u32(ctx, "general.alignment", &u32);
    (void) gguf_get_meta_f32(ctx, "fuzz.f32", &f32);
    (void) gguf_get_meta_bool(ctx, "fuzz.flag", &flag);

    uint32_t       elem_vt = 0;
    uint64_t       count   = 0;
    const uint8_t *payload = nullptr;
    (void) gguf_get_meta_array_info(ctx, "fuzz.arr", &elem_vt, &count, &payload);
    /* Keys the input is unlikely to carry: the miss path is a bounds path too. */
    (void) gguf_get_meta_string(ctx, "tokenizer.ggml.model", &slen);

    gguf_close(ctx);
    free(copy);
    return 0;
}

/* ---- standalone driver ------------------------------------------------ */

#ifdef GEIST_FUZZ_STANDALONE
/* splitmix64 — deterministic, so a failing run reproduces from its seed. */
static uint64_t prng(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int main(int argc, char **argv) {
    struct buf seed;
    seed_gguf(&seed);

    if (argc == 2 && strcmp(argv[1], "--seed") == 0) {
        fwrite(seed.b, 1, seed.n, stdout);
        return 0;
    }

    const long runs = (argc == 2) ? strtol(argv[1], nullptr, 10) : 3000;
    uint64_t   s    = 0x5EEDull;

    /* The unmutated seed first: if the valid case is broken, say so before
     * spending the budget on random bytes. */
    LLVMFuzzerTestOneInput(seed.b, seed.n);

    for (long i = 0; i < runs; i++) {
        uint8_t      input[sizeof seed.b];
        const size_t size = 1 + (size_t) (prng(&s) % seed.n);
        memcpy(input, seed.b, size);
        /* One to eight byte pokes: enough to break a length or a type tag,
         * little enough that the buffer still reaches the parser's depths. */
        const unsigned pokes = 1u + (unsigned) (prng(&s) % 8u);
        for (unsigned p = 0; p < pokes; p++)
            input[prng(&s) % size] = (uint8_t) prng(&s);
        LLVMFuzzerTestOneInput(input, size);
    }
    printf("fuzz_gguf: %ld runs, no crash\n", runs);
    return 0;
}
#endif
