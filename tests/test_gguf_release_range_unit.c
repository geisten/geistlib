/*
 * test_gguf_release_range_unit — GGUF file pages behind weights a backend
 * copied to its device do not stay resident (#468 item 3).
 *
 * With caps.weights_device_copy (Vulkan) the big matrices alias the GGUF
 * mmap and are uploaded at resolve; left alone, the file pages stay resident
 * beside the device copy (on a unified-memory GPU, the model twice).
 * gguf_release_range drops them. Checked here:
 *
 *   - gguf_release_range on a file-backed context drops the touched pages
 *     (the mapping's Rss falls, counted per path: on tmpfs the pages are
 *     shmem and never show as RssFile) and the bytes still read back
 *     unchanged; a range outside the mapping, a null or empty range and a
 *     context over caller memory leave the bytes as they are;
 *   - loading a model from a file on Vulkan drops the file pages of the
 *     matrices uploaded straight from the mapping, and the model decodes
 *     the same tokens as the same GGUF loaded from memory (where nothing is
 *     released).
 *
 * The residency checks read /proc/self/smaps and run on Linux only.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include "src/io/gguf_reader.h"

#include <geist.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Resident KiB of this process's mappings of the file at `path` (summed
 * over /proc/self/smaps), or -1 where /proc is not there. */
static long rss_of_path_kib(const char *path) {
    FILE *f = fopen("/proc/self/smaps", "r");
    if (f == nullptr) {
        return -1;
    }
    const size_t pl = strlen(path);
    char         line[512];
    long         total = 0, kib = 0;
    bool         in = false;
    while (fgets(line, sizeof line, f) != nullptr) {
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[--len] = '\0';
        }
        unsigned long lo = 0, hi = 0;
        if (sscanf(line, "%lx-%lx ", &lo, &hi) == 2) {
            /* A mapping header: "start-end perms offset dev inode path". */
            in = len >= pl && strcmp(line + len - pl, path) == 0;
        } else if (in && sscanf(line, "Rss: %ld kB", &kib) == 1) {
            total += kib;
        }
    }
    fclose(f);
    return total;
}

/* Write `g` to a fresh temporary file; returns its path (caller frees and
 * unlinks) or nullptr. */
static char *write_temp(const struct tf_buf *g) {
    char *path = strdup("/tmp/geist_release_XXXXXX");
    int   fd   = path != nullptr ? mkstemp(path) : -1;
    if (fd < 0) {
        free(path);
        return nullptr;
    }
    FILE *f  = fdopen(fd, "wb");
    bool  ok = f != nullptr && fwrite(g->b, 1, g->n, f) == g->n;
    if (f != nullptr) {
        ok = fclose(f) == 0 && ok;
    }
    if (!ok) {
        unlink(path);
        free(path);
        return nullptr;
    }
    return path;
}

/* Sum of every byte, which faults each page of [p, p + n) in. */
static uint64_t touch(size_t n, const uint8_t *p) {
    uint64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += p[i];
    }
    return sum;
}

static int check_release(const struct tf_buf *g, const char *path) {
    int              fails = 0;
    const char      *err   = nullptr;
    struct gguf_ctx *ctx   = gguf_open(path, &err);
    if (ctx == nullptr) {
        return geist_expect(false, "gguf_open of the temporary file");
    }
    /* The largest tensor: several MiB of whole pages. */
    const struct gguf_tensor_t *big = nullptr;
    for (size_t i = 0; i < gguf_tensor_count(ctx); i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(ctx, i);
        if (big == nullptr || t->nbytes > big->nbytes) {
            big = t;
        }
    }
    const uint8_t *p   = big->data;
    const size_t   n   = big->nbytes;
    const uint64_t ref = touch(n, p);
    const long     in  = rss_of_path_kib(path);
    gguf_release_range(ctx, p, n);
    const long out = rss_of_path_kib(path);
#if defined(__linux__)
    char msg[192];
    snprintf(msg,
             sizeof msg,
             "releasing %zu KiB drops the file pages (resident %ld -> %ld KiB)",
             n >> 10,
             in,
             out);
    fails += geist_expect(in >= 0 && out >= 0 && in - out >= (long) (n >> 10) * 3 / 4, msg);
#else
    (void) in;
    (void) out;
#endif
    fails += geist_expect(touch(n, p) == ref, "the released bytes read back unchanged");

    /* No-ops: nothing to check but that they return and leave the bytes. */
    gguf_release_range(ctx, nullptr, 16);
    gguf_release_range(ctx, p, 0);
    gguf_release_range(ctx, p + n, SIZE_MAX); /* runs past the mapping */
    gguf_release_range(ctx, g->b, g->n);      /* not this mapping */
    gguf_release_range(nullptr, p, n);
    fails += geist_expect(touch(n, p) == ref, "out-of-range releases leave the bytes");
    char name[128];
    snprintf(name, sizeof name, "%s", big->name); /* owned by ctx */
    gguf_close(ctx);

    /* A context over caller memory: the caller's pages are never touched. */
    struct gguf_ctx *mem = gguf_open_memory(g->b, g->n, &err);
    if (mem == nullptr) {
        return fails + geist_expect(false, "gguf_open_memory");
    }
    const struct gguf_tensor_t *mbig = gguf_get_tensor(mem, name);
    const uint64_t              mref = touch(mbig->nbytes, mbig->data);
    gguf_release_range(mem, mbig->data, mbig->nbytes);
    fails += geist_expect(touch(mbig->nbytes, mbig->data) == mref,
                          "a memory-backed context keeps the caller's bytes");
    gguf_close(mem);
    return fails;
}

/* Prefill + 6 greedy steps; false on any failure. */
static bool decode(struct geist_model *m, struct geist_backend *be, geist_token_t out[6]) {
    struct geist_session_opts  o      = {.kv_mode = GEIST_KV_F16, .top_p = 1.0f, .max_seq_len = 64};
    struct geist_session      *s      = nullptr;
    static const geist_token_t prompt = 7;
    bool                       ok     = geist_session_create(m, be, &o, &s) == GEIST_OK &&
                                        geist_session_prefill_tokens(s, 1, &prompt) == GEIST_OK;
    for (int i = 0; ok && i < 6; i++) {
        ok = geist_session_decode_step(s, &out[i]) == GEIST_OK;
    }
    if (s != nullptr) {
        geist_session_destroy(s);
    }
    return ok;
}

static int check_vulkan_load(const struct tf_buf *g, const char *path, size_t big_bytes) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK) {
        printf("  vulkan not available: model check skipped\n");
        return 0;
    }
    int                 fails = 0;
    struct geist_model *mf = nullptr, *mm = nullptr;
    if (geist_model_load(path, be, &mf) != GEIST_OK) {
        fails += geist_expect(false, "vulkan: load from file");
    }
    const long resident = rss_of_path_kib(path);
#if defined(__linux__)
    char msg[192];
    snprintf(msg,
             sizeof msg,
             "vulkan: %zu KiB of uploaded matrices, %ld KiB of the file resident",
             big_bytes >> 10,
             resident);
    /* Without the release every page of the file is resident (the loader
     * reads it all). Llama's attn_q / attn_k reach the device through a
     * row-permuted host copy, so their file pages are not the uploaded
     * ones; at least half of the matrices' pages must be gone. */
    const long limit = (long) (g->n >> 10) - (long) (big_bytes >> 10) / 2;
    fails += geist_expect(mf == nullptr || (resident >= 0 && resident <= limit), msg);
#else
    (void) resident;
    (void) big_bytes;
#endif
    if (geist_model_load_from_memory(g->b, g->n, be, &mm) != GEIST_OK) {
        fails += geist_expect(false, "vulkan: load from memory");
    }
    geist_token_t a[6] = {0}, b[6] = {0};
    if (mf != nullptr && mm != nullptr) {
        fails += geist_expect(decode(mf, be, a) && decode(mm, be, b) && memcmp(a, b, sizeof a) == 0,
                              "vulkan: the file-loaded model decodes as the in-memory one");
    }
    if (mf != nullptr) {
        geist_model_destroy(mf);
    }
    if (mm != nullptr) {
        geist_model_destroy(mm);
    }
    geist_backend_destroy(be);
    return fails;
}

int main(void) {
    /* d_model 512 x ffn 1024 F32: 2 MiB FFN matrices, above the 1 MiB
     * device-copy threshold. */
    struct tf_buf g    = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                           .d_model  = 512,
                                                           .heads    = 8,
                                                           .kv_heads = 4,
                                                           .ffn      = 1024,
                                                           .vocab    = 256,
                                                           .context  = 256,
                                                           .seed     = 3});
    char         *path = write_temp(&g);
    if (path == nullptr) {
        free(g.b);
        fprintf(stderr, "ERROR: could not write the temporary GGUF\n");
        return GEIST_TEST_ERROR;
    }
    int fails = check_release(&g, path);

    /* Bytes of the matrices at least 1 MiB (the ones a device copies). */
    size_t           big_bytes = 0;
    const char      *err       = nullptr;
    struct gguf_ctx *ctx       = gguf_open(path, &err);
    for (size_t i = 0; ctx != nullptr && i < gguf_tensor_count(ctx); i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(ctx, i);
        if (t->n_dims == 2 && t->nbytes >= ((size_t) 1 << 20)) {
            big_bytes += t->nbytes;
        }
    }
    if (ctx != nullptr) {
        gguf_close(ctx);
    }
    fails += check_vulkan_load(&g, path, big_bytes);

    unlink(path);
    free(path);
    free(g.b);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("gguf release range: pass\n");
    return GEIST_TEST_PASS;
}
