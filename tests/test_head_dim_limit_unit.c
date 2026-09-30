/*
 * test_head_dim_limit_unit — a GGUF asking for head_dim > 512 is refused at
 * load, not run.
 *
 * head_dim is model metadata (llama: embedding_length / head_count). The
 * forward pass holds one attention head in stack arrays of
 * TRANSFORMER_HEAD_DIM_MAX (512) elements and sizes its scratch for it; a
 * model with head_dim 1024 used to load and then write past int8_t
 * q_q8[512] in the INT8 KV attention on its first prefill (ASan:
 * stack-buffer-overflow, forward/attention.c). The loader now refuses it
 * and says why. (The engine reports any failed arch state create as
 * GEIST_E_IO; the backend's error message carries the reason.)
 *
 * Minimal in-memory llama GGUFs (metadata plus one dummy tensor, like
 * test_arch_gate_unit): head_dim 1024 must be refused naming head_dim;
 * head_dim 512, the limit itself, must get past that check (it then fails
 * for the missing weights, which is not what this test is about).
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

/* One-layer llama with d_model = head_dim (one query head, one KV head). */
static size_t build_gguf(uint8_t *buf, uint32_t head_dim) {
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
    o += kv_u32(buf + o, "llama.embedding_length", head_dim);
    o += kv_u32(buf + o, "llama.attention.head_count", 1);
    o += kv_u32(buf + o, "llama.attention.head_count_kv", 1);
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

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_scalar backend did not register\n");
        return GEIST_TEST_SKIP;
    }
    int     fails = 0;
    uint8_t buf[1024];

    {
        const size_t            n = build_gguf(buf, 1024);
        struct geist_model     *m = nullptr;
        const enum geist_status s = geist_model_load_from_memory(buf, n, be, &m);
        const char             *e = geist_backend_errmsg(be);
        fails += geist_expect(s != GEIST_OK, "head_dim 1024 is refused");
        fails += geist_expect(m == nullptr, "head_dim 1024 leaves *out null");
        fails += geist_expect(e != nullptr && strstr(e, "head_dim 1024") != nullptr,
                              "the error names head_dim 1024");
        if (m != nullptr) {
            geist_model_destroy(m);
        }
    }
    {
        const size_t            n = build_gguf(buf, 512);
        struct geist_model     *m = nullptr;
        const enum geist_status s = geist_model_load_from_memory(buf, n, be, &m);
        const char             *e = geist_backend_errmsg(be);
        /* No weights in this GGUF: the load fails later, for that reason. */
        fails += geist_expect(e == nullptr || strstr(e, "head_dim") == nullptr,
                              "head_dim 512 passes the head_dim check");
        if (s == GEIST_OK && m != nullptr) {
            geist_model_destroy(m);
        }
    }

    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: head_dim above the supported maximum is refused at load\n");
    return GEIST_TEST_PASS;
}
