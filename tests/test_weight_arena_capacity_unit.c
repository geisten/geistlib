/*
 * test_weight_arena_capacity_unit — the weight arena counts the F32 copy of a
 * small half-precision matrix on every backend, not only on one that sets
 * caps.weights_need_backend_arena (#561).
 *
 * load_layer_proj widens a small F16/BF16 matrix to F32 whenever the
 * backend's resolve_weight refuses it, and in arena mode that stages the
 * source a second time and adds the F32 copy. Metal refuses F16, leaves the
 * cap false (it defaults to mmap-alias) and still runs the arena under
 * GEIST_WEIGHT_MMAP=0: llama-3.2-3B with F16 attn_k ran ~500 MB past the
 * arena at blk.22. compute_weight_arena_capacity only counted the widen when
 * the cap was set.
 *
 * A two-tensor in-memory GGUF (a 32x64 F16 matrix, a 16-element F32 vector)
 * and a descriptor carrying nothing but caps: the capacity is the same with
 * and without the cap.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "src/archs/transformer/weight_load.h"

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

static size_t put_tensor_info(uint8_t       *p,
                              const char    *name,
                              uint32_t       n_dims,
                              const uint64_t dims[static n_dims],
                              uint32_t       dtype,
                              uint64_t       offset) {
    size_t o = put_str(p, name);
    o += put_u32(p + o, n_dims);
    for (uint32_t i = 0; i < n_dims; i++) {
        o += put_u64(p + o, dims[i]);
    }
    o += put_u32(p + o, dtype);
    return o + put_u64(p + o, offset);
}

enum { F16_ELEMS = 64 * 32, F16_BYTES = F16_ELEMS * 2, F32_BYTES = 16 * 4 };

static size_t build_gguf(uint8_t *buf) {
    size_t o = 0;
    memcpy(buf + o, "GGUF", 4);
    o += 4;
    o += put_u32(buf + o, 3); /* version */
    o += put_u64(buf + o, 2); /* n_tensors */
    o += put_u64(buf + o, 0); /* n_kv */
    const uint64_t mat[2] = {64, 32};
    const uint64_t vec[1] = {16};
    o += put_tensor_info(buf + o, "blk.0.attn_k.weight", 2, mat, GGUF_TYPE_F16, 0);
    o += put_tensor_info(buf + o, "blk.0.attn_norm.weight", 1, vec, GGUF_TYPE_F32, F16_BYTES);
    while (o % 32 != 0) {
        buf[o++] = 0;
    }
    memset(buf + o, 0, F16_BYTES + F32_BYTES);
    return o + F16_BYTES + F32_BYTES;
}

static size_t capacity(struct gguf_ctx *g, bool need_backend_arena) {
    struct geist_backend_descriptor desc = {.name = "caps-only"};
    desc.caps.weights_need_backend_arena = need_backend_arena;
    struct geist_backend be              = {.desc = &desc};
    size_t               cap             = 0;
    if (compute_weight_arena_capacity(&be, g, &cap) != GEIST_OK) {
        return 0;
    }
    return cap;
}

int main(void) {
    static uint8_t   buf[16384];
    const size_t     n   = build_gguf(buf);
    const char      *err = nullptr;
    struct gguf_ctx *g   = gguf_open_memory(buf, n, &err);
    if (g == nullptr) {
        fprintf(stderr, "gguf_open_memory: %s\n", err != nullptr ? err : "(null)");
        return GEIST_TEST_FAIL;
    }

    /* Matrix once from the file, once staged again, once as F32; the vector;
     * the 64 MB headroom. Every size here is already a multiple of 64. */
    const size_t want = 2 * F16_BYTES + F16_ELEMS * sizeof(float) + 64 + (64u << 20);

    int fails = 0;
    fails += geist_expect(capacity(g, true) == want, "arena counts the widen with the cap set");
    fails += geist_expect(capacity(g, false) == want,
                          "arena counts the widen without the cap (metal, #561)");
    gguf_close(g);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
