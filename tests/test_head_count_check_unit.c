/*
 * test_head_count_check_unit — a GGUF whose KV head count does not divide
 * its query head count is refused at load, not run.
 *
 * Every attention path maps query head h to KV head
 * h / (n_q_heads / n_kv_heads), and both counts are model metadata. More
 * KV heads than query heads made that group size 0: a synthetic llama GGUF
 * with 4 query and 8 KV heads (tools/gen_synth_gguf.py --heads 4
 * --kv-heads 8) loaded and then died on its first prefill with SIGFPE, in
 * the INT8 and the FP32 KV attention, on cpu_x86 and cpu_scalar. A count
 * that does not divide (15 query, 4 KV) sent query heads 12-14 to KV head
 * 4, which does not exist: they read the next position's rows, and at the
 * last position one row past the ones written. The loader now refuses both,
 * and a zero count, and says why.
 *
 * Minimal in-memory llama GGUFs (metadata plus one dummy tensor, like
 * test_head_dim_limit_unit), head_dim 64: the bad counts must be refused
 * naming them; valid GQA and MHA counts must get past the check (they then
 * fail for the missing weights, which is not what this test is about).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static size_t put_u32(uint8_t *p, uint32_t v) {
    memcpy(p, &v, 4);
    return 4;
}

static size_t put_u64(uint8_t *p, uint64_t v) {
    memcpy(p, &v, 8);
    return 8;
}

static size_t put_str(uint8_t *p, const char *s) {
    const uint64_t n = strlen(s);
    size_t         o = put_u64(p, n);
    memcpy(p + o, s, n);
    return o + n;
}

static size_t kv_u32(uint8_t *p, const char *key, uint32_t v) {
    size_t o = put_str(p, key);
    o += put_u32(p + o, 4); /* GGUF value type: uint32 */
    return o + put_u32(p + o, v);
}

/* One-layer llama, head_dim 64 (d_model = 64 * n_q_heads, or 64 when the
 * query count is 0). */
static size_t build_gguf(uint8_t *buf, uint32_t n_q_heads, uint32_t n_kv_heads) {
    size_t o = 0;
    memcpy(buf + o, "GGUF", 4);
    o += 4;
    o += put_u32(buf + o, 3); /* version */
    o += put_u64(buf + o, 1); /* n_tensors */
    o += put_u64(buf + o, 8); /* n_kv */
    o += put_str(buf + o, "general.architecture");
    o += put_u32(buf + o, 8); /* string */
    o += put_str(buf + o, "llama");
    o += kv_u32(buf + o, "llama.block_count", 1);
    o += kv_u32(buf + o, "llama.embedding_length", 64 * (n_q_heads > 0 ? n_q_heads : 1));
    o += kv_u32(buf + o, "llama.attention.head_count", n_q_heads);
    o += kv_u32(buf + o, "llama.attention.head_count_kv", n_kv_heads);
    o += kv_u32(buf + o, "llama.feed_forward_length", 256);
    o += kv_u32(buf + o, "llama.vocab_size", 32);
    o += kv_u32(buf + o, "llama.context_length", 64);
    o += put_str(buf + o, "t"); /* tensor info: name, dims, dtype, offset */
    o += put_u32(buf + o, 1);
    o += put_u64(buf + o, 1);
    o += put_u32(buf + o, 0); /* f32 */
    o += put_u64(buf + o, 0);
    while (o % 32 != 0) {
        buf[o++] = 0;
    }
    memset(buf + o, 0, 4); /* the tensor's 4 data bytes */
    return o + 4;
}

/* Loads the counts; returns the error text (nullptr when there is none). */
static const char *load(struct geist_backend *be, uint32_t n_q, uint32_t n_kv, bool *loaded) {
    uint8_t             buf[1024];
    const size_t        n = build_gguf(buf, n_q, n_kv);
    struct geist_model *m = nullptr;
    *loaded               = geist_model_load_from_memory(buf, n, be, &m) == GEIST_OK;
    if (m != nullptr) {
        geist_model_destroy(m);
    }
    return geist_backend_errmsg(be);
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_scalar backend did not register\n");
        return GEIST_TEST_SKIP;
    }
    static const struct {
        uint32_t    n_q, n_kv;
        const char *named; /* the counts as the error must name them */
    } BAD[] = {
            {15, 4, "15 (query) and 4 (KV)"}, /* does not divide */
            {4, 8, "4 (query) and 8 (KV)"},   /* more KV than query heads */
            {4, 0, "4 (query) and 0 (KV)"},
            {0, 4, "0 (query) and 4 (KV)"},
    };
    static const struct {
        uint32_t n_q, n_kv;
    } GOOD[] = {{16, 4}, {15, 5}, {8, 8}, {8, 1}};

    int fails = 0;
    for (size_t i = 0; i < sizeof BAD / sizeof *BAD; i++) {
        bool        loaded = false;
        const char *e      = load(be, BAD[i].n_q, BAD[i].n_kv, &loaded);
        char        what[96];
        snprintf(what, sizeof what, "%u/%u heads are refused naming them", BAD[i].n_q, BAD[i].n_kv);
        fails += geist_expect(!loaded && e != nullptr && strstr(e, BAD[i].named) != nullptr, what);
        if (e != nullptr && strstr(e, BAD[i].named) == nullptr) {
            fprintf(stderr, "  error was: %s\n", e);
        }
    }
    for (size_t i = 0; i < sizeof GOOD / sizeof *GOOD; i++) {
        bool        loaded = false;
        const char *e      = load(be, GOOD[i].n_q, GOOD[i].n_kv, &loaded);
        char        what[96];
        snprintf(what, sizeof what, "%u/%u heads pass the check", GOOD[i].n_q, GOOD[i].n_kv);
        /* No weights in this GGUF: the load fails later, for that reason. */
        fails += geist_expect(e == nullptr || strstr(e, "head counts") == nullptr, what);
    }

    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: head counts that do not divide are refused at load\n");
    return GEIST_TEST_PASS;
}
