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
 * Prefill rows (#239 follow-up), the module on its own:
 *   - a chunk's rows per expert land in the bucket of floor(log2 m), the
 *     last from 512; a shared expert and a layer past the table are not
 *     counted; a threshold reads the buckets from its own on;
 *   - they and the staging threshold go to the v2 file and come back, the
 *     threshold carried over by a run that measured none;
 *   - a v1 file (USAGE lines only) still loads, with no prefill rows, and
 *     is rewritten as v2; a PREFILL line that does not parse: ignored;
 *   - moe_stage_order: the most rows per byte at the threshold, then at a
 *     quarter of it, then the most selections, then the lower index;
 *     with no rows, the resident ranking's order.
 * The qwen3moe fixture on cpu_scalar (GEIST_MOE_PROFILE_CACHE = the
 * directory): a prefill of N tokens counts N x used selections per layer,
 * written when the model goes, and the next load reads them. With a GPU
 * backend in the build and GEIST_MOE_HOT = 4, the four experts the cache
 * counts most often are the ones that get a copy.
 * Placement determinism (#48), with a GPU backend in the build: three loads
 * of the fixture with the same GEIST_MOE_PROFILE (counts with ties) and
 * GEIST_MOE_HOT, and three with no profile, place the same experts, the
 * ones the ranking defines, and give the same logits bit for bit; the
 * model on the GPU and the model on cpu_scalar with a GPU split backend.
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

/* Prefill rows: counting, buckets, thresholds, the v2 file, a v1 file. */
static int check_prefill(void) {
    int            fails = 0;
    const uint64_t key   = 0x2468aceULL;
    char           path[512];
    path_of(key, sizeof path, path);
    uint64_t got[L * E], tok = 0;

    /* m = 3, 0, 1, 40, 600 and a shared expert's 6 past n_exp. */
    const size_t                    rs[E + 2] = {0, 3, 3, 4, 44, 644, 650};
    const uint32_t                  ids[]     = {1, 4, E};
    struct transformer_moe_profile *p         = moe_profile_open(L, E, key, g_dir);
    fails += geist_expect(p != nullptr && !moe_profile_prefill_rows(p, 1, L * E, got),
                          "no prefill counted: no rows");
    if (p == nullptr)
        return fails;
    moe_profile_count_chunk(p, 1, E, rs);
    moe_profile_count_chunk(p, L, E, rs); /* an MTP block: past the table */
    moe_profile_count(p, 0, 1, 3, 2, ids);
    struct {
        size_t   m_min;
        uint64_t want[E];
    } const th[] = {{1, {3, 0, 1, 40, 600}},
                    {2, {3, 0, 0, 40, 600}},
                    {4, {0, 0, 0, 40, 600}},
                    {32, {0, 0, 0, 40, 600}},
                    {48, {0, 0, 0, 40, 600}}, /* not a power of two: from 32 */
                    {64, {0, 0, 0, 0, 600}},
                    {4096, {0, 0, 0, 0, 600}}}; /* the last bucket: from 512 */
    for (size_t t = 0; t < sizeof th / sizeof th[0]; t++) {
        const bool any = moe_profile_prefill_rows(p, th[t].m_min, L * E, got);
        bool       ok  = any;
        for (size_t i = 0; i < L * E; i++)
            ok = ok && got[i] == (i / E == 1 ? th[t].want[i % E] : 0);
        char what[96];
        snprintf(what, sizeof what, "prefill rows at threshold %zu", th[t].m_min);
        fails += geist_expect(ok, what);
    }
    fails += geist_expect(moe_profile_stage_m_min(p) == 0, "no threshold yet");
    moe_profile_set_stage_m_min(p, 32);
    moe_profile_close(p);

    /* The v2 file: rows and the threshold come back; a run without a
     * threshold of its own keeps the file's. */
    p = moe_profile_open(L, E, key, g_dir);
    fails += geist_expect(p != nullptr && moe_profile_loaded(p), "the v2 file loads");
    if (p != nullptr) {
        const bool any = moe_profile_prefill_rows(p, 32, L * E, got);
        fails += geist_expect(any && got[1 * E + 3] == 40 && got[1 * E + 4] == 600 &&
                                      got[1 * E + 0] == 0,
                              "the prefill rows read back");
        fails += geist_expect(moe_profile_stage_m_min(p) == 32, "the threshold reads back");
        moe_profile_counts(p, L * E, got, &tok);
        fails += geist_expect(got[0 * E + 1] == 1 && got[0 * E + 4] == 1 && tok == 1,
                              "the USAGE counts beside them");
        moe_profile_count_chunk(p, 1, E, rs);
        moe_profile_count(p, 0, 1, 3, 2, ids);
    }
    moe_profile_close(p);
    p = moe_profile_open(L, E, key, g_dir);
    if (p != nullptr) {
        const bool any = moe_profile_prefill_rows(p, 1, L * E, got);
        fails += geist_expect(any && got[1 * E + 0] == 6 && got[1 * E + 4] == 1200,
                              "a second run's rows accumulate");
        fails += geist_expect(moe_profile_stage_m_min(p) == 32,
                              "a run that measured no threshold keeps the file's");
    }
    moe_profile_close(p);

    /* A v1 file: the USAGE counts, no prefill rows, no threshold. */
    char text[1024];
    snprintf(text,
             sizeof text,
             "# geist moe profile v1 key %016" PRIx64 " layers 3 experts 5\n"
             "USAGE %016" PRIx64 " profile run-time 7 0 1,2,3,4,5\n"
             "USAGE %016" PRIx64 " profile run-time 7 1 0,0,0,0,9\n"
             "USAGE %016" PRIx64 " profile run-time 7 2 5,0,0,0,0\n",
             key,
             key,
             key,
             key);
    write_text(path, text);
    p = moe_profile_open(L, E, key, g_dir);
    fails += geist_expect(p != nullptr && moe_profile_loaded(p), "a v1 file still loads");
    if (p != nullptr) {
        moe_profile_counts(p, L * E, got, &tok);
        fails += geist_expect(got[0 * E + 4] == 5 && got[1 * E + 4] == 9 && got[2 * E] == 5 &&
                                      tok == 7,
                              "a v1 file's counts");
        fails += geist_expect(!moe_profile_prefill_rows(p, 1, L * E, got) &&
                                      moe_profile_stage_m_min(p) == 0,
                              "a v1 file has no prefill rows and no threshold");
        moe_profile_count(p, 0, 1, 3, 2, ids);
    }
    moe_profile_close(p);
    FILE *f         = fopen(path, "r");
    char  head[128] = {0};
    if (f != nullptr) {
        if (fgets(head, sizeof head, f) == nullptr)
            head[0] = '\0';
        fclose(f);
    }
    fails += geist_expect(strncmp(head, "# geist moe profile v2 ", 23) == 0,
                          "a v1 file is rewritten as v2");
    p = moe_profile_open(L, E, key, g_dir);
    if (p != nullptr) {
        moe_profile_counts(p, L * E, got, &tok);
        fails += geist_expect(moe_profile_loaded(p) && got[1 * E + 4] == 9 && tok == 8,
                              "and reads back with its counts");
    }
    moe_profile_close(p);

    /* A PREFILL line that does not parse: the file is ignored. */
    snprintf(text,
             sizeof text,
             "# geist moe profile v2 key %016" PRIx64 " layers 1 experts 2\n"
             "USAGE %016" PRIx64 " profile run-time 3 0 1,2\n"
             "STAGE %016" PRIx64 " m-min 32\n"
             "PREFILL %016" PRIx64 " rows-from 1 0 4,x\n",
             key,
             key,
             key,
             key);
    write_text(path, text);
    p = moe_profile_open(1, 2, key, g_dir);
    fails += geist_expect(p != nullptr && !moe_profile_loaded(p) &&
                                  !moe_profile_prefill_rows(p, 1, 2, got) &&
                                  moe_profile_stage_m_min(p) == 0,
                          "a corrupt PREFILL line: ignored, nothing left behind");
    moe_profile_close(p);
    remove(path);
    return fails;
}

/* moe_stage_order: rows per byte, then selections, then the index. */
static int check_stage_order(void) {
    struct moe_stage_cand c[] = {
            {.rows = 0, .bytes = 100, .count = 9, .idx = 4},
            {.rows = 0, .bytes = 100, .count = 9, .idx = 1},
            {.rows = 0, .bytes = 100, .count = 12, .idx = 7},
    };
    moe_stage_order(3, c);
    int fails = geist_expect(c[0].idx == 7 && c[1].idx == 1 && c[2].idx == 4,
                             "no prefill rows: the selection counts, ties to the lower index");
    struct moe_stage_cand d[] = {
            {.rows = 0, .bytes = 100, .count = 9, .idx = 4},
            {.rows = 50, .bytes = 100, .count = 0, .idx = 9},
            {.rows = 60, .bytes = 200, .count = 99, .idx = 2}, /* 0.3 per byte */
            {.rows = 0, .bytes = 100, .count = 12, .idx = 7},
            {.rows = 30, .bytes = 60, .count = 0, .idx = 3}, /* 0.5, as idx 9 */
            {.rows = 0, .bytes = 100, .count = 9, .idx = 1},
    };
    moe_stage_order(6, d);
    const uint32_t want[] = {3, 9, 2, 7, 1, 4};
    bool           ok     = true;
    for (size_t i = 0; i < 6; i++)
        ok = ok && d[i].idx == want[i];
    fails += geist_expect(ok, "the most prefill rows per byte first, then the selection counts");
    struct moe_stage_cand q[] = {
            {.rows = 0, .rows_lo = 0, .bytes = 100, .count = 50, .idx = 1},
            {.rows = 0, .rows_lo = 10, .bytes = 100, .count = 0, .idx = 5},
            {.rows = 0, .rows_lo = 10, .bytes = 200, .count = 0, .idx = 2}, /* half per byte */
            {.rows = 5, .rows_lo = 5, .bytes = 100, .count = 0, .idx = 9},
    };
    moe_stage_order(4, q);
    fails += geist_expect(q[0].idx == 9 && q[1].idx == 5 && q[2].idx == 2 && q[3].idx == 1,
                          "rows at a quarter of the threshold break ties before the counts");
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

/* One load of the fixture for check_determinism: on `gbe` itself (on_gpu)
 * or on `cpu` with `gbe` as split backend. Writes the hot set (layer-major,
 * 2 x 8) and the next-token logits, chunked and token by token. */
static bool placed_load(size_t                n,
                        const geist_token_t   ids[static n],
                        const struct tf_buf  *g,
                        bool                  on_gpu,
                        bool                  hot[static 2 * 8],
                        float                 chunked[static VOCAB],
                        float                 single[static VOCAB],
                        struct geist_backend *cpu,
                        struct geist_backend *gbe) {
    const struct geist_session_opts lo = {.split_backend = gbe, .split_layers = 1};
    struct geist_backend           *be = on_gpu ? gbe : cpu;
    struct geist_model             *m  = nullptr;
    struct geist_session           *s  = nullptr;
    bool ok = (on_gpu ? geist_model_load_from_memory(g->b, g->n, gbe, &m)
                      : geist_model_load_from_memory_with_opts(g->b, g->n, cpu, &lo, &m)) ==
                      GEIST_OK &&
              geist_session_create(m, be, &OPTS, &s) == GEIST_OK;
    for (size_t step = 0; ok && step < 2; step++) {
        float *out = step == 0 ? chunked : single;
        size_t nl  = 0;
        ok         = geist_session_reset(s) == GEIST_OK;
        for (size_t i = 0, k = step == 0 ? n : 1; ok && i < n; i += k)
            ok = geist_session_prefill_tokens(s, n - i < k ? n - i : k, ids + i) == GEIST_OK;
        const float *lg = ok ? geist_session_peek_logits(&nl, s) : nullptr;
        ok              = lg != nullptr && nl == VOCAB;
        if (ok)
            memcpy(out, lg, VOCAB * sizeof(float));
    }
    if (ok) {
        const struct transformer_arch_session *a = geist_session_internal_arch_session(s);
        for (size_t i = 0; i < 2 * 8; i++)
            hot[i] = moe_gpu_hot(a, i / 8, i % 8);
    } else {
        fprintf(stderr, "%s\n", geist_backend_errmsg(be));
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    return ok;
}

/* #48: placement is decided once at load and is deterministic. The same
 * model with the same profile (GEIST_MOE_PROFILE, counts with ties) and the
 * same GEIST_MOE_HOT gives the same hot set on every load, the one the
 * ranking defines (count descending, ties to the lower e * n_layers + l),
 * and the same logits bit for bit; likewise with no profile (the same share
 * of every layer). Both GPU modes: the model on the GPU, and the model on
 * the CPU with the GPU as split backend. */
static int check_determinism(void) {
    struct geist_backend *gbe    = nullptr;
    const char *const     gpus[] = {"vulkan", "metal"};
    for (size_t b = 0; b < 2 && gbe == nullptr; b++) {
        if (geist_backend_create(gpus[b], nullptr, nullptr, &gbe) != GEIST_OK)
            gbe = nullptr;
    }
    if (gbe == nullptr) {
        printf("skip placement determinism: no GPU backend in this build\n");
        return 0;
    }
    struct geist_backend *cpu = nullptr;
    int fails = geist_expect(geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu) == GEIST_OK,
                             "cpu_scalar");
    struct tf_buf g = mf_qwen3moe_gguf(&GEOM);
    geist_token_t ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);

    /* Ties on purpose: four experts at 9, then nine at 5; five go hot. By
     * the ranking: 9s (l1,e0) (l0,e3) (l0,e5) (l1,e7), then the 5 with the
     * lowest e * 2 + l, (l0,e0). */
    char path[512];
    snprintf(path, sizeof path, "%s/determinism.txt", g_dir);
    write_text(path,
               "USAGE fixture profile corpus 9 0 5,5,5,9,5,9,1,1\n"
               "USAGE fixture profile corpus 9 1 9,5,5,5,5,1,1,9\n");
    const bool want[2 * 8] = {[0] = true, [3] = true, [5] = true, [8 + 0] = true, [8 + 7] = true};

    /* No profile cache: each load would add its counts for the next (#165). */
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    setenv("GEIST_MOE_HOT", "5", 1);
    for (size_t mode = 0; fails == 0 && mode < 4; mode++) {
        const bool on_gpu = mode % 2 == 0, with_prof = mode < 2;
        if (with_prof)
            setenv("GEIST_MOE_PROFILE", path, 1);
        else
            unsetenv("GEIST_MOE_PROFILE");
        bool  hot[3][2 * 8];
        float ch[3][VOCAB], sg[3][VOCAB];
        bool  ok = true;
        for (size_t r = 0; r < 3 && ok; r++)
            ok = placed_load(N_PROMPT, ids, &g, on_gpu, hot[r], ch[r], sg[r], cpu, gbe);
        size_t n_hot = 0;
        for (size_t i = 0; ok && i < 2 * 8; i++)
            n_hot += hot[0][i];
        const bool same_set    = ok && memcmp(hot[0], hot[1], sizeof hot[0]) == 0 &&
                                 memcmp(hot[0], hot[2], sizeof hot[0]) == 0;
        const bool same_logits = ok && memcmp(ch[0], ch[1], sizeof ch[0]) == 0 &&
                                 memcmp(ch[0], ch[2], sizeof ch[0]) == 0 &&
                                 memcmp(sg[0], sg[1], sizeof sg[0]) == 0 &&
                                 memcmp(sg[0], sg[2], sizeof sg[0]) == 0;
        printf("%s, %s, %s: 3 loads, %zu hot, same set %d, same logits %d\n",
               geist_backend_name(gbe),
               on_gpu ? "model on the GPU" : "model on cpu_scalar, GPU split",
               with_prof ? "profile with ties" : "no profile",
               n_hot,
               same_set,
               same_logits);
        fails += geist_expect(ok, "the fixture loads and runs with a placement");
        fails += geist_expect(n_hot == 5, "GEIST_MOE_HOT experts are placed");
        fails += geist_expect(same_set, "every load places the same experts");
        fails += geist_expect(same_logits, "every load gives the same logits bit for bit");
        if (with_prof)
            fails += geist_expect(ok && memcmp(hot[0], want, sizeof want) == 0,
                                  "the profile's ranking decides, ties to the lower index");
    }
    unsetenv("GEIST_MOE_HOT");
    unsetenv("GEIST_MOE_PROFILE");
    setenv("GEIST_MOE_PROFILE_CACHE", g_dir, 1);
    remove(path);
    free(g.b);
    geist_backend_destroy(cpu);
    geist_backend_destroy(gbe);
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
    fails += check_prefill();
    fails += check_stage_order();
    fails += check_model();
    fails += check_determinism();
    rmdir(g_dir);
    if (fails == 0)
        printf("PASS: moe profile counted, kept, read back, foreign and corrupt files ignored\n");
    return fails == 0 ? 0 : 1;
}
