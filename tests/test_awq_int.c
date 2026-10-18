/*
 * test_awq_int — AWQ scales are applied at model load, change the forward
 * pass, and a malformed scales file is refused (#66).
 *
 * AWQ folds into the model's weights, so the scales go to
 * geist_model_load_with_opts; geist_session_create does not read
 * awq_scales_path. The scales file is synthesised here for the model under
 * test (one "blk.N.attn_norm.out" / "blk.N.ffn_norm.out" entry per layer,
 * sized from the GGUF's norm tensors), so the test needs no AWQ fixture and
 * runs wherever the GGUF is present.
 *
 * The gammas are folded in place, so the weights must be copied: in
 * mmap-alias mode (the CPU default) they are the read-only GGUF pages and
 * the load refuses AWQ. The test checks that refusal, then sets
 * GEIST_WEIGHT_MMAP=0.
 *
 *   1. Load without AWQ: greedy decode N tokens, keep the first logits.
 *   2. Load with the synthesised scales: the first logits must differ.
 *   3. Bad magic, truncated, wrong channel count, missing file: the load
 *      fails and leaves *out nullptr.
 */
#include "test_helpers.h"

#include "gguf_reader.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N_DECODE 6

/* Greedy decode after BOS. Copies the logits of the first step into
 * `logits` (n_logits capacity) and returns their count, or 0 on failure. */
static size_t decode_tokens(struct geist_model   *model,
                            struct geist_backend *be,
                            size_t                n_logits_cap,
                            float                *logits,
                            geist_token_t         out_tokens[static N_DECODE]) {
    const struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session           *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        return 0;
    }
    const geist_token_t bos[1]   = {2};
    size_t              n_logits = 0;
    if (geist_session_prefill_tokens(sess, 1, bos) == GEIST_OK) {
        const float *l = geist_session_peek_logits(&n_logits, sess);
        if (l == nullptr || n_logits == 0 || n_logits > n_logits_cap) {
            n_logits = 0;
        } else {
            memcpy(logits, l, n_logits * sizeof(float));
        }
    }
    for (int i = 0; n_logits != 0 && i < N_DECODE; i++) {
        if (geist_session_decode_step(sess, &out_tokens[i]) != GEIST_OK || out_tokens[i] < 0) {
            n_logits = 0;
        }
    }
    geist_session_destroy(sess);
    return n_logits;
}

static bool put_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof v, 1, f) == 1;
}

/* Write an AWQS file (format: src/formats/ptqtp/ptqtp_awq.h) with a
 * non-uniform scale vector for every blk.N.{attn,ffn}_norm.weight the GGUF
 * holds. `extra` is added to every channel count, so a non-zero value makes
 * a well-formed file the model must refuse. Returns the entry count, 0 on
 * failure. */
static uint32_t write_scales(const char *path, const struct gguf_ctx *g, uint32_t extra) {
    static const char *const NORMS[] = {"attn_norm", "ffn_norm"};
    uint32_t                 n       = 0;
    char                     name[64];
    for (int layer = 0;; layer++) {
        snprintf(name, sizeof name, "blk.%d.attn_norm.weight", layer);
        if (gguf_get_tensor(g, name) == nullptr) {
            break;
        }
        for (size_t k = 0; k < 2; k++) {
            snprintf(name, sizeof name, "blk.%d.%s.weight", layer, NORMS[k]);
            n += gguf_get_tensor(g, name) != nullptr;
        }
    }
    FILE *f = fopen(path, "wb");
    if (f == nullptr) {
        return 0;
    }
    bool ok = fwrite("AWQS", 1, 4, f) == 4 && put_u32(f, 1) && put_u32(f, n);
    for (int layer = 0; ok; layer++) {
        snprintf(name, sizeof name, "blk.%d.attn_norm.weight", layer);
        if (gguf_get_tensor(g, name) == nullptr) {
            break;
        }
        for (size_t k = 0; ok && k < 2; k++) {
            snprintf(name, sizeof name, "blk.%d.%s.weight", layer, NORMS[k]);
            const struct gguf_tensor_t *t = gguf_get_tensor(g, name);
            if (t == nullptr) {
                continue;
            }
            const uint32_t n_in = (uint32_t) gguf_tensor_elem_count(t) + extra;
            char           key[64];
            const int      klen = snprintf(key, sizeof key, "blk.%d.%s.out", layer, NORMS[k]);
            ok = put_u32(f, (uint32_t) klen) && fwrite(key, 1, (size_t) klen, f) == (size_t) klen &&
                 put_u32(f, n_in);
            for (uint32_t j = 0; ok && j < n_in; j++) {
                const float s = 0.5f + 0.25f * (float) ((j + (uint32_t) layer) % 7u);
                ok            = fwrite(&s, sizeof s, 1, f) == 1;
            }
        }
    }
    ok = fclose(f) == 0 && ok;
    return ok ? n : 0;
}

static bool write_bytes(const char *path, size_t n, const void *bytes) {
    FILE *f = fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fwrite(bytes, 1, n, f) == n;
    return fclose(f) == 0 && ok;
}

/* The load with this scales file must fail and leave *out nullptr. */
static int expect_refused(const char           *model_path,
                          struct geist_backend *be,
                          const char           *awq_path,
                          const char           *what) {
    const struct geist_session_opts opts  = {.max_seq_len = 1024, .awq_scales_path = awq_path};
    struct geist_model             *model = nullptr;
    const enum geist_status         s = geist_model_load_with_opts(model_path, be, &opts, &model);
    if (s == GEIST_OK || model != nullptr) {
        fprintf(stderr, "FAIL: %s was accepted at load\n", what);
        geist_model_destroy(model);
        return 1;
    }
    printf("refused (%s): status %d: %s\n", what, (int) s, geist_last_create_error());
    return 0;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    char dir[] = "/tmp/geist_awq_XXXXXX";
    if (mkdtemp(dir) == nullptr) {
        fprintf(stderr, "mkdtemp failed\n");
        return GEIST_TEST_ERROR;
    }
    char good[64], bad_n[64], bad_magic[64], trunc[64], missing[64];
    snprintf(good, sizeof good, "%s/good.bin", dir);
    snprintf(bad_n, sizeof bad_n, "%s/bad_n.bin", dir);
    snprintf(bad_magic, sizeof bad_magic, "%s/bad_magic.bin", dir);
    snprintf(trunc, sizeof trunc, "%s/trunc.bin", dir);
    snprintf(missing, sizeof missing, "%s/missing.bin", dir);

    const char      *gerr = nullptr;
    struct gguf_ctx *g    = gguf_open(model_path, &gerr);
    if (g == nullptr) {
        fprintf(stderr, "gguf_open(%s): %s\n", model_path, gerr != nullptr ? gerr : "?");
        return GEIST_TEST_ERROR;
    }
    const uint32_t n_entries = write_scales(good, g, 0);
    const bool     wrote_bad = write_scales(bad_n, g, 1) != 0;
    gguf_close(g);
    /* "AWQS" v1 claiming one entry whose name runs past the end. */
    static const uint8_t TRUNC[] = {'A', 'W', 'Q', 'S', 1, 0, 0, 0, 1, 0, 0, 0, 200, 0, 0, 0, 'b'};
    static const uint8_t MAGIC[] = {'S', 'Q', 'W', 'A', 1, 0, 0, 0, 0, 0, 0, 0};
    if (n_entries == 0 || !wrote_bad || !write_bytes(trunc, sizeof TRUNC, TRUNC) ||
        !write_bytes(bad_magic, sizeof MAGIC, MAGIC)) {
        fprintf(stderr, "cannot write the scales files under %s (or the GGUF has no norms)\n", dir);
        return GEIST_TEST_ERROR;
    }
    printf("synthesised %u AWQ entries in %s\n", n_entries, good);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = GEIST_E_UNSUPPORTED;
    for (const char *const *n =
                 (const char *const[]) {"cpu_neon", "cpu_x86", "cpu_scalar", nullptr};
         s != GEIST_OK && *n != nullptr;
         n++) {
        s = geist_backend_create(*n, nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    /* mmap-alias mode: refused, not a write into the read-only mmap. */
    int fails = 0;
    setenv("GEIST_WEIGHT_MMAP", "1", 1);
    fails += expect_refused(model_path, be, good, "AWQ in mmap-alias mode");
    setenv("GEIST_WEIGHT_MMAP", "0", 1);

    constexpr size_t CAP    = (size_t) 1 << 20;
    float           *base   = calloc(CAP, sizeof(float));
    float           *awq    = calloc(CAP, sizeof(float));
    size_t           n_base = 0;
    size_t           n_awq  = 0;
    geist_token_t    t_base[N_DECODE];
    geist_token_t    t_awq[N_DECODE];
    if (base == nullptr || awq == nullptr) {
        fprintf(stderr, "out of memory\n");
        fails++;
        goto done;
    }

    for (int pass = 0; pass < 2; pass++) {
        const struct geist_session_opts opts = {
                .max_seq_len     = 1024,
                .awq_scales_path = pass == 0 ? nullptr : good,
        };
        struct geist_model *model = nullptr;
        s                         = geist_model_load_with_opts(model_path, be, &opts, &model);
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "FAIL: load %s AWQ: status %d (%s)\n",
                    pass == 0 ? "without" : "with",
                    (int) s,
                    geist_last_create_error());
            fails++;
            goto done;
        }
        const size_t n =
                decode_tokens(model, be, CAP, pass == 0 ? base : awq, pass == 0 ? t_base : t_awq);
        geist_model_destroy(model);
        if (n == 0) {
            fprintf(stderr, "FAIL: decode %s AWQ failed\n", pass == 0 ? "without" : "with");
            fails++;
            goto done;
        }
        *(pass == 0 ? &n_base : &n_awq) = n;
        printf("%s AWQ tokens:", pass == 0 ? "without" : "with");
        for (int i = 0; i < N_DECODE; i++) {
            printf(" %d", (pass == 0 ? t_base : t_awq)[i]);
        }
        printf("\n");
    }

    /* Effect: the scales divide the attn/ffn norm gammas, so the first
     * logits move. Equal logits mean AWQ was silently ignored. */
    if (n_base != n_awq) {
        fprintf(stderr, "FAIL: vocab differs (%zu vs %zu)\n", n_base, n_awq);
        fails++;
    } else {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n_base; i++) {
            max_diff = fmaxf(max_diff, fabsf(base[i] - awq[i]));
        }
        printf("max |logit(AWQ) - logit(no AWQ)| = %g\n", (double) max_diff);
        if (!(max_diff > 1e-3f)) {
            fprintf(stderr, "FAIL: AWQ had no effect on the logits\n");
            fails++;
        }
    }

    fails += expect_refused(model_path, be, bad_n, "channel count != the norm width");
    fails += expect_refused(model_path, be, bad_magic, "bad magic");
    fails += expect_refused(model_path, be, trunc, "truncated file");
    fails += expect_refused(model_path, be, missing, "missing file");

done:
    free(base);
    free(awq);
    geist_backend_destroy(be);
    unlink(good);
    unlink(bad_n);
    unlink(bad_magic);
    unlink(trunc);
    rmdir(dir);
    if (fails == 0) {
        printf("PASS: AWQ scales applied at load change the logits; malformed files are refused\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
