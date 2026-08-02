/*
 * test_arch_gate_unit — the fail-closed architecture gate.
 *
 * geist_model_load* used to map every GGUF onto the transformer descriptor
 * regardless of general.architecture, and the family layer silently fell
 * back to Gemma-4 — an unknown model "loaded" and then failed obscurely at
 * tensor-shape mismatch, or worse, ran with the wrong geometry. The gate
 * now rejects unknown or missing architecture names with
 * GEIST_E_UNSUPPORTED before any weight is touched. Pin the reject path
 * here; the accept paths are covered by the gemma4/llama/bitnet int tests.
 *
 * Uses hand-crafted minimal GGUF v3 blobs (header + one string kv, no
 * tensors) through geist_model_load_from_memory — no fixture file needed,
 * and it exercises the from-memory path that previously hardcoded its
 * lookup.
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

/* Minimal GGUF v3: magic, version, one dummy 1-element f32 tensor (the
 * reader refuses 0-tensor files), and general.architecture = `arch` as
 * the only metadata kv (omitted entirely when arch is nullptr). Data
 * section is 32-byte aligned per spec. */
static size_t build_gguf(uint8_t *buf, const char *arch) {
    size_t o = 0;
    memcpy(buf + o, "GGUF", 4);
    o += 4;
    o += put_u32(buf + o, 3);                       /* version */
    o += put_u64(buf + o, 1);                       /* n_tensors */
    o += put_u64(buf + o, arch != nullptr ? 1 : 0); /* n_kv */
    if (arch != nullptr) {
        o += put_str(buf + o, "general.architecture");
        o += put_u32(buf + o, 8); /* GGUF value type: string */
        o += put_str(buf + o, arch);
    }
    o += put_str(buf + o, "t"); /* tensor info: name, dims, dtype, offset */
    o += put_u32(buf + o, 1);
    o += put_u64(buf + o, 1);
    o += put_u32(buf + o, 0); /* dtype 0 = f32 */
    o += put_u64(buf + o, 0);
    while (o % 32 != 0) {
        buf[o++] = 0;
    }
    memset(buf + o, 0, 4); /* the tensor's 4 data bytes */
    return o + 4;
}

int main(void) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    int     fails = 0;
    uint8_t buf[256];

    /* Unknown architecture name -> rejected, no model handle. */
    {
        size_t              n = build_gguf(buf, "qwen3-definitely-not-supported");
        struct geist_model *m = nullptr;
        s                     = geist_model_load_from_memory(buf, n, be, &m);
        fails += geist_expect(s == GEIST_E_UNSUPPORTED,
                              "unknown general.architecture -> GEIST_E_UNSUPPORTED");
        fails += geist_expect(m == nullptr, "unknown arch leaves *out null");
    }

    /* Missing general.architecture key -> rejected, no silent default. */
    {
        size_t              n = build_gguf(buf, nullptr);
        struct geist_model *m = nullptr;
        s                     = geist_model_load_from_memory(buf, n, be, &m);
        fails += geist_expect(s == GEIST_E_UNSUPPORTED,
                              "missing general.architecture -> GEIST_E_UNSUPPORTED");
        fails += geist_expect(m == nullptr, "missing arch leaves *out null");
    }

    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("arch gate: fail-closed rejection pass\n");
    return GEIST_TEST_PASS;
}
