/*
 * test_moe_profile_unit - the expert profile counted at run time and kept
 * with the model (#165, moe_profile.h).
 *
 * The cache is opt-in: no variable or "0" means none, "1" the XDG cache
 * directory, anything else that directory.
 * The module on its own, in a temporary cache directory:
 *   - counts are written when the profile closes and read back by the next
 *     open with the same key, which adds its own on top; tokens likewise;
 *   - a layer past the table (an MTP block) and ids past the routed experts
 *     (a shared expert) are not counted;
 *   - nothing is written when nothing was counted, or without a directory;
 *   - a file made for another key, layer count or expert count, and a file
 *     whose counts do not parse, are ignored (one line on stderr);
 *   - the file is the USAGE format analyze.py and GEIST_MOE_PROFILE read.
 * The qwen3moe fixture on cpu_scalar (GEIST_MOE_PROFILE_CACHE = the
 * directory): a prefill of N tokens counts N x used selections per layer,
 * written when the model goes, and the next load reads them. With a GPU
 * backend in the build and GEIST_MOE_HOT = 4, the four experts the cache
 * counts most often are the ones that get a copy.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/archs/transformer/moe_gpu.h"
#include "../src/archs/transformer/moe_profile.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { L = 3, E = 5, VOCAB = 96, N_PROMPT = 21 };

static char g_dir[256];

static void path_of(uint64_t key, size_t n, char path[static n]) {
    snprintf(path, n, "%s/%016" PRIx64 ".txt", g_dir, key);
}

static void write_text(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (f != nullptr) {
        fputs(text, f);
        fclose(f);
    }
}

static int check_module(void) {
    int            fails = 0;
    const uint64_t key   = 0x1234abcdULL;
    char           path[512];
    path_of(key, sizeof path, path);

    /* Nothing counted: nothing written. */
    struct transformer_moe_profile *p = moe_profile_open(L, E, key, g_dir);
    fails += geist_expect(p != nullptr && !moe_profile_loaded(p), "a first open has no counts");
    moe_profile_close(p);
    fails += geist_expect(access(path, F_OK) != 0, "nothing counted, no file");

    /* Two tokens, stride 3 (two routed + a shared expert id E). */
    const uint32_t ids[] = {1, 4, E, 1, 0, E};
    p                    = moe_profile_open(L, E, key, g_dir);
    moe_profile_count(p, 0, 2, 3, 2, ids);
    moe_profile_count(p, 2, 1, 3, 2, ids);
    moe_profile_count(p, L, 2, 3, 2, ids); /* an MTP block: past the table */
    moe_profile_close(p);
    fails += geist_expect(access(path, F_OK) == 0, "the counts are written on close");

    uint64_t want[L * E] = {0};
    want[0 * E + 1] = 2, want[0 * E + 4] = 1, want[0 * E + 0] = 1;
    want[2 * E + 1] = 1, want[2 * E + 4] = 1;
    uint64_t got[L * E], tok             = 0;
    p = moe_profile_open(L, E, key, g_dir);
    fails += geist_expect(p != nullptr && moe_profile_loaded(p), "the next open reads them");
    if (p != nullptr) {
        moe_profile_counts(p, L * E, got, &tok);
        fails += geist_expect(memcmp(got, want, sizeof got) == 0, "counts read back exactly");
        fails += geist_expect(tok == 2, "tokens: layer 0's");
        moe_profile_count(p, 1, 1, 3, 2, ids);
        moe_profile_counts(p, L * E, got, &tok);
        fails += geist_expect(got[1 * E + 1] == 1 && got[0 * E + 1] == 2 && tok == 2,
                              "the counts since the load add to the file's");
        moe_profile_count(p, 0, 1, 3, 2, ids);
    }
    moe_profile_close(p);
    p = moe_profile_open(L, E, key, g_dir);
    if (p != nullptr) {
        moe_profile_counts(p, L * E, got, &tok);
        fails += geist_expect(got[0 * E + 1] == 3 && got[1 * E + 4] == 1 && tok == 3,
                              "a second run accumulates");
    }
    moe_profile_close(p);

    /* The USAGE format: one line per layer, profile set, layer in place. */
    FILE *f = fopen(path, "r");
    char  line[512];
    int   usage = 0;
    while (f != nullptr && fgets(line, sizeof line, f) != nullptr) {
        char         set[16], file[16];
        unsigned int layer = 0;
        if (sscanf(line, "USAGE %*s %15s %15s %*s %u", set, file, &layer) == 3 &&
            strcmp(set, "profile") == 0 && layer == (unsigned) usage)
            usage++;
    }
    if (f != nullptr)
        fclose(f);
    fails += geist_expect(usage == L, "one USAGE profile line per layer, in order");

    /* Foreign: another key, layer or expert count. */
    p = moe_profile_open(L, E + 1, key, g_dir);
    fails += geist_expect(p != nullptr && !moe_profile_loaded(p), "another expert count: ignored");
    moe_profile_close(p);
    p = moe_profile_open(L + 1, E, key, g_dir);
    fails += geist_expect(p != nullptr && !moe_profile_loaded(p), "another layer count: ignored");
    moe_profile_close(p);
    const uint64_t other = 0x55ULL;
    char           opath[512];
    path_of(other, sizeof opath, opath);
    f       = fopen(path, "r");
    FILE *o = fopen(opath, "w");
    while (f != nullptr && o != nullptr && fgets(line, sizeof line, f) != nullptr)
        fputs(line, o);
    if (f != nullptr)
        fclose(f);
    if (o != nullptr)
        fclose(o);
    p = moe_profile_open(L, E, other, g_dir);
    fails += geist_expect(p != nullptr && !moe_profile_loaded(p), "another key: ignored");
    moe_profile_close(p);

    /* Corrupt: a count that does not parse, a truncated file, no header. */
    char text[512];
    snprintf(text,
             sizeof text,
             "# geist moe profile v1 key %016" PRIx64 " layers 3 experts 5\n"
             "USAGE %016" PRIx64 " profile run-time 4 0 1,2,x,4,5\n",
             key,
             key);
    write_text(path, text);
    p = moe_profile_open(L, E, key, g_dir);
    fails += geist_expect(p != nullptr && !moe_profile_loaded(p), "a corrupt count: ignored");
    if (p != nullptr) {
        moe_profile_counts(p, L * E, got, &tok);
        uint64_t zero[L * E] = {0};
        fails += geist_expect(memcmp(got, zero, sizeof got) == 0 && tok == 0,
                              "an ignored file leaves no counts behind");
    }
    moe_profile_close(p);
    write_text(path, "USAGE x profile run-time 1 0 1,1,1,1,1\n");
    p = moe_profile_open(L, E, key, g_dir);
    fails += geist_expect(p != nullptr && !moe_profile_loaded(p), "no header: ignored");
    moe_profile_close(p);

    /* No directory: counts, writes nothing. */
    p = moe_profile_open(L, E, 0x77ULL, nullptr);
    moe_profile_count(p, 0, 2, 3, 2, ids);
    fails += geist_expect(p != nullptr && moe_profile_path(p)[0] == '\0', "no directory, no path");
    moe_profile_close(p);
    remove(path);
    remove(opath);
    return fails;
}

static const struct mf_qwen3moe GEOM = {.layers    = 2,
                                        .d_model   = 64,
                                        .heads     = 4,
                                        .kv_heads  = 2,
                                        .head_dim  = 16,
                                        .experts   = 8,
                                        .used      = 2,
                                        .expert_ff = 32,
                                        .vocab     = VOCAB,
                                        .seed      = 11};

static const struct geist_session_opts OPTS = {.top_p = 1.0f, .m_max = 8, .max_seq_len = 256};

static int check_model(void) {
    int                   fails = 0;
    struct geist_backend *be    = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return geist_expect(false, "cpu_scalar");
    struct tf_buf         g = mf_qwen3moe_gguf(&GEOM);
    struct geist_model   *m = nullptr;
    struct geist_session *s = nullptr;
    geist_token_t         ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    uint64_t key = 0;
    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK ||
        geist_session_prefill_tokens(s, N_PROMPT, ids) != GEIST_OK) {
        fails += geist_expect(false, "the qwen3moe fixture runs");
    } else {
        const struct transformer_arch_state *st = geist_model_internal_arch_meta(m);
        fails += geist_expect(st->moe_prof != nullptr && !moe_profile_loaded(st->moe_prof),
                              "a first load has a profile, empty");
        uint64_t c[2 * 8], tok = 0, sum = 0;
        moe_profile_counts(st->moe_prof, 2 * 8, c, &tok);
        for (size_t i = 0; i < 2 * 8; i++)
            sum += c[i];
        printf("fixture: %llu tokens, %llu selections counted\n",
               (unsigned long long) tok,
               (unsigned long long) sum);
        fails += geist_expect(tok == N_PROMPT && sum == 2ull * N_PROMPT * 2,
                              "every token's routed experts are counted in every layer");
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    s = nullptr;
    m = nullptr;

    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK) {
        fails += geist_expect(false, "the fixture loads again");
    } else {
        const struct transformer_arch_state *st  = geist_model_internal_arch_meta(m);
        uint64_t                             tok = 0, c[2 * 8];
        fails += geist_expect(moe_profile_loaded(st->moe_prof), "the next load reads the counts");
        moe_profile_counts(st->moe_prof, 2 * 8, c, &tok);
        fails += geist_expect(tok == N_PROMPT, "with the first run's tokens");
        char path[512];
        snprintf(path, sizeof path, "%s", moe_profile_path(st->moe_prof));
        key = strtoull(strrchr(path, '/') + 1, nullptr, 16);
    }
    geist_model_destroy(m);
    m = nullptr;

    /* The placement follows the cache: layer 0's experts 3 and 5, layer 1's
     * 6 and 7 counted most. */
    struct geist_backend *gbe    = nullptr;
    const char *const     gpus[] = {"vulkan", "metal"};
    for (size_t b = 0; b < 2 && gbe == nullptr; b++) {
        if (geist_backend_create(gpus[b], nullptr, nullptr, &gbe) != GEIST_OK)
            gbe = nullptr;
    }
    if (gbe == nullptr) {
        printf("skip placement by the cache: no GPU backend in this build\n");
    } else {
        char path[512], text[1024];
        path_of(key, sizeof path, path);
        snprintf(text,
                 sizeof text,
                 "# geist moe profile v1 key %016" PRIx64 " layers 2 experts 8\n"
                 "USAGE %016" PRIx64 " profile run-time 9 0 1,1,1,90,1,80,1,1\n"
                 "USAGE %016" PRIx64 " profile run-time 9 1 1,1,1,1,1,1,70,60\n",
                 key,
                 key,
                 key);
        write_text(path, text);
        setenv("GEIST_MOE_HOT", "4", 1);
        const struct geist_session_opts lo = {.split_backend = gbe, .split_layers = 1};
        if (geist_model_load_from_memory_with_opts(g.b, g.n, be, &lo, &m) != GEIST_OK ||
            geist_session_create(m, be, &OPTS, &s) != GEIST_OK) {
            fails += geist_expect(false, "the fixture loads with a GPU split backend");
        } else {
            const struct transformer_arch_session *a  = geist_session_internal_arch_session(s);
            bool                                   ok = true;
            for (size_t l = 0; l < 2; l++) {
                for (size_t e = 0; e < 8; e++) {
                    const bool want = l == 0 ? (e == 3 || e == 5) : (e == 6 || e == 7);
                    ok              = ok && moe_gpu_hot(a, l, e) == want;
                }
            }
            fails += geist_expect(ok, "the experts the cache counts most get the GPU copies");
        }
        unsetenv("GEIST_MOE_HOT");
        geist_session_destroy(s);
        geist_model_destroy(m);
        geist_backend_destroy(gbe);
        remove(path);
    }
    char left[512];
    path_of(key, sizeof left, left);
    remove(left);
    free(g.b);
    geist_backend_destroy(be);
    return fails;
}

static int check_dir(void) {
    int  fails = 0;
    char dir[512];
    unsetenv("GEIST_MOE_PROFILE_CACHE");
    fails += geist_expect(!moe_profile_dir(sizeof dir, dir), "no variable: no cache (opt-in)");
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    fails += geist_expect(!moe_profile_dir(sizeof dir, dir), "0: no cache");
    setenv("GEIST_MOE_PROFILE_CACHE", "1", 1);
    setenv("XDG_CACHE_HOME", "/x", 1);
    fails += geist_expect(moe_profile_dir(sizeof dir, dir) &&
                                  strcmp(dir, "/x/geist/moe-profile") == 0,
                          "1: the XDG cache directory");
    unsetenv("XDG_CACHE_HOME");
    setenv("GEIST_MOE_PROFILE_CACHE", "/some/dir", 1);
    fails += geist_expect(moe_profile_dir(sizeof dir, dir) && strcmp(dir, "/some/dir") == 0,
                          "a path: that directory");
    return fails;
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    snprintf(g_dir, sizeof g_dir, "%s/geist_moe_profile_XXXXXX", tmp != nullptr ? tmp : "/tmp");
    if (mkdtemp(g_dir) == nullptr) {
        perror("mkdtemp");
        return 1;
    }
    int fails = check_dir();
    setenv("GEIST_MOE_PROFILE_CACHE", g_dir, 1);
    unsetenv("GEIST_MOE_PROFILE");
    fails += check_module();
    fails += check_model();
    rmdir(g_dir);
    if (fails == 0)
        printf("PASS: moe profile counted, kept, read back, foreign and corrupt files ignored\n");
    return fails == 0 ? 0 : 1;
}
