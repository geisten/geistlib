/* tests/test_io_malformed_unit.c — hermetic malformed-input coverage for
 * src/io (issue #219): the parser-security surface of gguf_reader and
 * safetensors_reader, with NO model fixture.
 *
 * GGUF cases build byte buffers in memory and go through gguf_open_memory
 * (plus one gguf_open round-trip via a temp file to cover the mmap path).
 * safetensors has no from-memory entry point, so each case writes a
 * few-bytes temp file — a valid header + a handful of data bytes is enough
 * to exercise the whole parse path, malformed variants exercise every
 * error exit.
 *
 * Every rejection asserts BOTH the nullptr result and that an error
 * message was set; every acceptance asserts the parsed values. */
#include "gguf_reader.h"
#include "quant.h"
#include "safetensors_reader.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail = 1;                                                     \
        }                                                                   \
    } while (0)

/* ---- tiny append-only byte buffer for building GGUF files ------------- */

struct buf {
    uint8_t b[4096];
    size_t  n;
};

static void put_bytes(struct buf *o, const void *p, size_t n) {
    if (o->n + n > sizeof(o->b)) {
        fprintf(stderr, "test buffer overflow\n");
        exit(GEIST_TEST_ERROR);
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
    size_t len = strlen(s);
    put_u64(o, (uint64_t) len);
    put_bytes(o, s, len);
}
static void pad_to(struct buf *o, size_t alignment) {
    while (o->n % alignment != 0) {
        uint8_t z = 0;
        put_bytes(o, &z, 1);
    }
}

/* GGUF metadata value types (mirrors the reader's private enum). */
enum { VT_U32 = 4, VT_F32 = 6, VT_BOOL = 7, VT_STRING = 8, VT_ARRAY = 9 };

#define GGUF_MAGIC 0x46554747u

/* Header for `n_meta` metadata KVs and `n_tensors` tensor infos; the caller
 * appends those records and then the data section. */
static void gguf_header(struct buf *o, uint32_t version, uint64_t n_tensors, uint64_t n_meta) {
    o->n = 0;
    put_u32(o, GGUF_MAGIC);
    put_u32(o, version);
    put_u64(o, n_tensors);
    put_u64(o, n_meta);
}

/* One F32 tensor "w" of 32 elements at offset 0 (nbytes = 128). */
static void valid_gguf(struct buf *o) {
    gguf_header(o, 3, 1, 5);
    /* metadata */
    put_gstr(o, "general.alignment");
    put_u32(o, VT_U32);
    put_u32(o, 32);
    put_gstr(o, "general.architecture");
    put_u32(o, VT_STRING);
    put_gstr(o, "llama");
    put_gstr(o, "test.f32");
    put_u32(o, VT_F32);
    put_f32(o, 1.5f);
    put_gstr(o, "test.flag");
    put_u32(o, VT_BOOL);
    put_bytes(o, "\1", 1);
    put_gstr(o, "test.arr");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_U32); /* element type */
    put_u64(o, 3);      /* count */
    put_u32(o, 1);
    put_u32(o, 2);
    put_u32(o, 3);
    /* tensor info */
    put_gstr(o, "w");
    put_u32(o, 1); /* n_dims */
    put_u64(o, 32);
    put_u32(o, GGUF_TYPE_F32);
    put_u64(o, 0); /* offset */
    /* data section, aligned */
    pad_to(o, 32);
    for (int i = 0; i < 128; i++) {
        uint8_t z = (uint8_t) i;
        put_bytes(o, &z, 1);
    }
}

static void check_gguf_rejected(const struct buf *o, const char *what) {
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(o->b, o->n, &err);
    if (ctx != nullptr) {
        fprintf(stderr, "FAIL: %s was accepted\n", what);
        g_fail = 1;
        gguf_close(ctx);
        return;
    }
    CHECK(err != nullptr);
}

static void test_gguf_valid(void) {
    struct buf o;
    valid_gguf(&o);
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(o.b, o.n, &err);
    CHECK(ctx != nullptr);
    if (!ctx) {
        fprintf(stderr, "valid GGUF rejected: %s\n", err ? err : "?");
        return;
    }
    CHECK(gguf_tensor_count(ctx) == 1);
    const struct gguf_tensor_t *t = gguf_get_tensor(ctx, "w");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->nbytes == 128);
        CHECK(gguf_tensor_elem_count(t) == 32);
        CHECK(strcmp(gguf_dtype_name(t->dtype), "F32") == 0);
        CHECK(t->data != nullptr);
    }
    CHECK(gguf_get_tensor(ctx, "missing") == nullptr);
    CHECK(gguf_tensor_at(ctx, 1) == nullptr);
    CHECK(gguf_tensor_at(ctx, 0) == t);

    size_t      slen = 0;
    const char *arch = gguf_get_meta_string(ctx, "general.architecture", &slen);
    CHECK(arch != nullptr && slen == 5 && strncmp(arch, "llama", 5) == 0);
    CHECK(gguf_get_meta_string(ctx, "missing.key", &slen) == nullptr && slen == 0);

    uint32_t u = 0;
    CHECK(gguf_get_meta_u32(ctx, "general.alignment", &u) && u == 32);
    CHECK(!gguf_get_meta_u32(ctx, "general.architecture", &u)); /* type mismatch */
    float f = 0;
    CHECK(gguf_get_meta_f32(ctx, "test.f32", &f) && f == 1.5f);
    CHECK(!gguf_get_meta_f32(ctx, "test.flag", &f));
    bool b = false;
    CHECK(gguf_get_meta_bool(ctx, "test.flag", &b) && b);
    uint32_t       elem_vt = 0;
    uint64_t       count   = 0;
    const uint8_t *payload = nullptr;
    CHECK(gguf_get_meta_array_info(ctx, "test.arr", &elem_vt, &count, &payload));
    CHECK(elem_vt == VT_U32 && count == 3 && payload != nullptr);
    CHECK(!gguf_get_meta_array_info(ctx, "test.f32", &elem_vt, &count, &payload));

    gguf_close(ctx);
}

/* Same valid bytes through the file-path entry point (mmap + advice). */
static void test_gguf_open_file(void) {
    struct buf o;
    valid_gguf(&o);
    char tmpl[] = "/tmp/geist_io_gguf_XXXXXX";
    int  fd     = mkstemp(tmpl);
    CHECK(fd >= 0);
    if (fd < 0)
        return;
    FILE *f = fdopen(fd, "wb");
    CHECK(f != nullptr);
    if (!f)
        return;
    xfwrite(o.b, 1, o.n, f);
    fclose(f); /* flushed and closed BEFORE gguf_open mmaps it */
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(tmpl, &err);
    CHECK(ctx != nullptr);
    if (ctx) {
        CHECK(gguf_tensor_count(ctx) == 1);
        gguf_close(ctx);
    }
    unlink(tmpl);

    CHECK(gguf_open("/nonexistent/geist_io_test.gguf", &err) == nullptr);
    CHECK(err != nullptr);
}

static void test_gguf_malformed(void) {
    struct buf o;

    /* null / too-small buffer */
    const char *err = nullptr;
    CHECK(gguf_open_memory(nullptr, 100, &err) == nullptr);
    CHECK(gguf_open_memory("GGUF", 4, &err) == nullptr);

    /* bad magic */
    valid_gguf(&o);
    o.b[0] = 'X';
    check_gguf_rejected(&o, "bad magic");

    /* unsupported version */
    gguf_header(&o, 2, 0, 0);
    check_gguf_rejected(&o, "version 2");

    /* header truncated after version */
    gguf_header(&o, 3, 0, 0);
    o.n = 12;
    check_gguf_rejected(&o, "truncated header");

    /* metadata promised but absent */
    gguf_header(&o, 3, 0, 1);
    check_gguf_rejected(&o, "truncated metadata");

    /* metadata with an invalid value type */
    gguf_header(&o, 3, 0, 1);
    put_gstr(&o, "k");
    put_u32(&o, 99); /* not a GGUF_VT_* */
    check_gguf_rejected(&o, "invalid metadata value type");

    /* metadata string overruns the buffer */
    gguf_header(&o, 3, 0, 1);
    put_gstr(&o, "k");
    put_u32(&o, VT_STRING);
    put_u64(&o, 1000); /* string length beyond EOF */
    check_gguf_rejected(&o, "metadata string past EOF");

    /* tensor info promised but absent */
    gguf_header(&o, 3, 1, 0);
    check_gguf_rejected(&o, "truncated tensor name");

    /* n_dims over GGUF_MAX_DIMS */
    gguf_header(&o, 3, 1, 0);
    put_gstr(&o, "w");
    put_u32(&o, GGUF_MAX_DIMS + 1);
    check_gguf_rejected(&o, "n_dims > max");

    /* elem count not divisible by the dtype's block size */
    gguf_header(&o, 3, 1, 0);
    put_gstr(&o, "w");
    put_u32(&o, 1);
    put_u64(&o, 10); /* Q4_K needs multiples of 256 */
    put_u32(&o, GGUF_TYPE_Q4_K);
    put_u64(&o, 0);
    pad_to(&o, 32);
    check_gguf_rejected(&o, "elem count not block-divisible");

    /* tensor data past EOF */
    gguf_header(&o, 3, 1, 0);
    put_gstr(&o, "w");
    put_u32(&o, 1);
    put_u64(&o, 32); /* F32 x32 = 128 bytes... */
    put_u32(&o, GGUF_TYPE_F32);
    put_u64(&o, 0);
    pad_to(&o, 32);
    put_u32(&o, 0); /* ...but only 4 bytes present */
    check_gguf_rejected(&o, "tensor data past EOF");

    /* ---- arithmetic-abuse seeds (issue #332) --------------------------
     * Each of these used to reach a pointer add, a modulo, or a product on
     * an attacker-chosen value BEFORE anything validated it. The assertion
     * is the same for all of them — rejected, with a message — but the
     * point is that they run clean under ASan/UBSan. */

    /* String length within one byte of UINT64_MAX. `c->p + len` for this
     * value is undefined; `len > (size_t)(end - p)` is not. */
    gguf_header(&o, 3, 0, 1);
    put_gstr(&o, "k");
    put_u32(&o, VT_STRING);
    put_u64(&o, UINT64_MAX - 8u);
    check_gguf_rejected(&o, "metadata string length near UINT64_MAX");

    /* Array element count of UINT64_MAX: no element type is zero bytes, so
     * this cannot fit however many bytes remain. Must be rejected up front
     * rather than walked 2^64 times. */
    gguf_header(&o, 3, 0, 1);
    put_gstr(&o, "k");
    put_u32(&o, VT_ARRAY);
    put_u32(&o, VT_U32);
    put_u64(&o, UINT64_MAX);
    check_gguf_rejected(&o, "metadata array count UINT64_MAX");

    /* general.alignment = 0 — reached `info_end % alignment`. */
    gguf_header(&o, 3, 0, 1);
    put_gstr(&o, "general.alignment");
    put_u32(&o, VT_U32);
    put_u32(&o, 0);
    check_gguf_rejected(&o, "zero alignment");

    /* general.alignment = 24 — not a power of two; the padding it implies
     * is not the padding the rest of the loader computes. */
    gguf_header(&o, 3, 0, 1);
    put_gstr(&o, "general.alignment");
    put_u32(&o, VT_U32);
    put_u32(&o, 24);
    check_gguf_rejected(&o, "non-power-of-two alignment");

    /* Dimensions whose product overflows size_t: 2^32 x 2^32 x 2^32 x 2^32. */
    gguf_header(&o, 3, 1, 0);
    put_gstr(&o, "w");
    put_u32(&o, 4);
    put_u64(&o, 1ull << 32);
    put_u64(&o, 1ull << 32);
    put_u64(&o, 1ull << 32);
    put_u64(&o, 1ull << 32);
    put_u32(&o, GGUF_TYPE_F32);
    put_u64(&o, 0);
    pad_to(&o, 32);
    check_gguf_rejected(&o, "overflowing tensor dimensions");

    /* A single dimension larger than the byte count can ever be: the
     * element count fits, but elems/block * block_bytes overflows. */
    gguf_header(&o, 3, 1, 0);
    put_gstr(&o, "w");
    put_u32(&o, 1);
    put_u64(&o, (UINT64_MAX / 2u) + 1u); /* x4 bytes/elem overflows */
    put_u32(&o, GGUF_TYPE_F32);
    put_u64(&o, 0);
    pad_to(&o, 32);
    check_gguf_rejected(&o, "overflowing tensor byte count");

    /* Tensor offset near UINT64_MAX: data_offset + offset + nbytes wrapped
     * to a small sum that passed the EOF test. */
    gguf_header(&o, 3, 1, 0);
    put_gstr(&o, "w");
    put_u32(&o, 1);
    put_u64(&o, 32);
    put_u32(&o, GGUF_TYPE_F32);
    put_u64(&o, UINT64_MAX - 64u);
    pad_to(&o, 32);
    for (int i = 0; i < 128; i++) {
        uint8_t z = 0;
        put_bytes(&o, &z, 1);
    }
    check_gguf_rejected(&o, "overflowing tensor offset");

    /* ---- I2_S trailing per-tensor scale (issue #334) -------------------
     * I2_S stores 256 elements in 64 packed bytes with NO per-block scale
     * and ONE f32 for the whole tensor, sitting immediately after the last
     * packed byte. The reader used to size the tensor from the blocks
     * alone, so a file that stopped four bytes early validated fine and the
     * kernel then read the scale from whatever followed. */
    {
        /* Exact size: 256 elems -> 64 packed bytes + 4 scale bytes. */
        gguf_header(&o, 3, 1, 0);
        put_gstr(&o, "w");
        put_u32(&o, 1);
        put_u64(&o, 256);
        put_u32(&o, GGUF_TYPE_I2_S);
        put_u64(&o, 0);
        pad_to(&o, 32);
        for (int i = 0; i < 64; i++) {
            uint8_t z = (uint8_t) i;
            put_bytes(&o, &z, 1);
        }
        put_f32(&o, 0.125f);
        const size_t exact_n = o.n;

        struct gguf_ctx *i2s = gguf_open_memory(o.b, o.n, &err);
        CHECK(i2s != nullptr);
        if (i2s) {
            const struct gguf_tensor_t *t = gguf_get_tensor(i2s, "w");
            CHECK(t != nullptr && t->nbytes == 68); /* 64 packed + 4 scale */
            if (t) {
                /* The scale is inside the tensor's own extent, which is the
                 * whole point: the arena copy in beta mode copies nbytes. */
                float sc = 0.0f;
                memcpy(&sc, (const uint8_t *) t->data + i2_s_scale_offset(256), sizeof sc);
                CHECK(sc == 0.125f);
            }
            gguf_close(i2s);
        }

        /* One byte short of the scale: must be rejected, not accepted with
         * a three-byte scale and a one-byte over-read. */
        o.n = exact_n - 1;
        check_gguf_rejected(&o, "I2_S tensor one byte short of its scale");

        /* Packed blocks complete, scale entirely absent. */
        o.n = exact_n - 4;
        check_gguf_rejected(&o, "I2_S tensor missing its trailing scale");
    }

    /* unknown dtype id: accepted, but surfaced with nbytes=0/data=null */
    gguf_header(&o, 3, 1, 0);
    put_gstr(&o, "w");
    put_u32(&o, 1);
    put_u64(&o, 32);
    put_u32(&o, 99); /* unknown dtype */
    put_u64(&o, 0);
    pad_to(&o, 32);
    struct gguf_ctx *ctx = gguf_open_memory(o.b, o.n, &err);
    CHECK(ctx != nullptr);
    if (ctx) {
        const struct gguf_tensor_t *t = gguf_get_tensor(ctx, "w");
        CHECK(t && t->nbytes == 0 && t->data == nullptr);
        CHECK(strcmp(gguf_dtype_name(t->dtype), "?") == 0);
        gguf_close(ctx);
    }
}

/* ---- safetensors ------------------------------------------------------- */

/* Write header-size prefix + `header` + `data_len` zero bytes to a temp
 * file; returns a heap path the caller unlinks+frees. `header_size_override`
 * < 0 means "use strlen(header)". */
static char *write_st(const char *header, long header_size_override, size_t data_len) {
    char *path = strdup("/tmp/geist_io_st_XXXXXX");
    int   fd   = mkstemp(path);
    if (fd < 0) {
        fprintf(stderr, "mkstemp failed\n");
        exit(GEIST_TEST_ERROR);
    }
    FILE *f = fdopen(fd, "wb");
    if (!f) {
        fprintf(stderr, "fdopen failed\n");
        exit(GEIST_TEST_ERROR);
    }
    uint64_t hsize = (header_size_override < 0) ? strlen(header) : (uint64_t) header_size_override;
    xfwrite(&hsize, 8, 1, f);
    if (strlen(header) > 0)
        xfwrite(header, 1, strlen(header), f);
    for (size_t i = 0; i < data_len; i++)
        fputc(0, f);
    fclose(f);
    return path;
}

static void
check_st_rejected(const char *header, long hsize_override, size_t data_len, const char *what) {
    char          *path = write_st(header, hsize_override, data_len);
    const char    *err  = nullptr;
    struct st_ctx *ctx  = st_open(path, &err);
    if (ctx != nullptr) {
        fprintf(stderr, "FAIL: safetensors %s was accepted\n", what);
        g_fail = 1;
        st_close(ctx);
    } else {
        CHECK(err != nullptr);
    }
    unlink(path);
    free(path);
}

static void test_st_valid(void) {
    const char    *hdr  = "{\"__metadata__\":{\"format\":\"pt\",\"n\":[1,{\"x\":2}]},"
                          "\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16],"
                          "\"extra\":true},"
                          "\"s\":{\"dtype\":\"U8\",\"shape\":[],\"data_offsets\":[16,17]}}";
    char          *path = write_st(hdr, -1, 17);
    const char    *err  = nullptr;
    struct st_ctx *ctx  = st_open(path, &err);
    CHECK(ctx != nullptr);
    if (!ctx) {
        fprintf(stderr, "valid safetensors rejected: %s\n", err ? err : "?");
    } else {
        CHECK(st_count(ctx) == 2);
        const struct st_tensor_t *t = st_get(ctx, "a");
        CHECK(t != nullptr);
        if (t) {
            CHECK(t->dtype == ST_DTYPE_F32);
            CHECK(t->rank == 2 && t->shape[0] == 2 && t->shape[1] == 2);
            CHECK(t->nbytes == 16);
            CHECK(t->data != nullptr);
            CHECK(strcmp(st_dtype_name(t->dtype), "F32") == 0);
            CHECK(st_dtype_bytes(t->dtype) == 4);
        }
        const struct st_tensor_t *s = st_get(ctx, "s"); /* rank-0 scalar */
        CHECK(s && s->rank == 0 && s->nbytes == 1);
        CHECK(st_get(ctx, "missing") == nullptr);
        st_close(ctx);
    }
    unlink(path);
    free(path);

    /* Unknown dtype string parses (reported as UNKNOWN, 0 bytes/elem). */
    path = write_st("{\"u\":{\"dtype\":\"X99\",\"shape\":[1],\"data_offsets\":[0,4]}}", -1, 4);
    ctx  = st_open(path, &err);
    CHECK(ctx != nullptr);
    if (ctx) {
        const struct st_tensor_t *t = st_get(ctx, "u");
        CHECK(t && t->dtype == ST_DTYPE_UNKNOWN);
        CHECK(st_dtype_bytes(ST_DTYPE_UNKNOWN) == 0);
        CHECK(strcmp(st_dtype_name(ST_DTYPE_UNKNOWN), "UNKNOWN") == 0);
        st_close(ctx);
    }
    unlink(path);
    free(path);

    /* Missing file. */
    CHECK(st_open("/nonexistent/geist_io_test.safetensors", &err) == nullptr);
    CHECK(err != nullptr);
}

static void test_st_malformed(void) {
    /* File smaller than the 8-byte header-size prefix: write raw bytes. */
    {
        char *path = strdup("/tmp/geist_io_st_XXXXXX");
        int   fd   = mkstemp(path);
        CHECK(fd >= 0);
        if (fd >= 0) {
            CHECK(write(fd, "abc", 3) == 3);
            close(fd);
            const char *err = nullptr;
            CHECK(st_open(path, &err) == nullptr);
            CHECK(err != nullptr);
        }
        unlink(path);
        free(path);
    }

    /* header_size exceeds the file */
    check_st_rejected("{}", 4096, 0, "oversized header_size");
    /* not a JSON object */
    check_st_rejected("[]", -1, 0, "non-object header");
    /* escape sequences unsupported by design */
    check_st_rejected("{\"a\\n\":{}}", -1, 0, "escaped key");
    /* unterminated string */
    check_st_rejected("{\"abc", -1, 0, "unterminated string");
    /* missing colon */
    check_st_rejected("{\"a\" 1}", -1, 0, "missing colon");
    /* EOF mid-header */
    check_st_rejected("{", -1, 0, "EOF in header");
    /* tensor entry not an object */
    check_st_rejected("{\"a\":42}", -1, 0, "non-object tensor entry");
    /* required fields missing */
    check_st_rejected("{\"a\":{\"dtype\":\"F32\"}}", -1, 0, "missing shape/offsets");
    /* rank beyond ST_MAX_RANK (8) */
    check_st_rejected("{\"a\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,1,1,1,1,1],"
                      "\"data_offsets\":[0,4]}}",
                      -1,
                      4,
                      "rank > ST_MAX_RANK");
    /* non-numeric shape entry */
    check_st_rejected("{\"a\":{\"dtype\":\"F32\",\"shape\":[x],\"data_offsets\":[0,4]}}",
                      -1,
                      4,
                      "non-numeric dim");
    /* reversed data_offsets */
    check_st_rejected("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[16,0]}}",
                      -1,
                      16,
                      "reversed offsets");
    /* tensor data outside the file */
    check_st_rejected("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4096]}}",
                      -1,
                      4,
                      "offsets past EOF");
    /* junk between entries */
    check_st_rejected("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]} x}",
                      -1,
                      4,
                      "junk after entry");
}

int main(void) {
    test_gguf_valid();
    test_gguf_open_file();
    test_gguf_malformed();
    test_st_valid();
    test_st_malformed();

    if (g_fail) {
        fprintf(stderr, "test_io_malformed_unit: FAIL\n");
        return 1;
    }
    printf("test_io_malformed_unit: PASS\n");
    return 0;
}
