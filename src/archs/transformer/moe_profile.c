/*
 * src/archs/transformer/moe_profile.c - the expert profile counted at run
 * time and kept with the model (#165). Contract and file format:
 * moe_profile.h.
 */
#include "moe_profile.h"

#include "cache_dir.h"
#include "checked.h"
#include "fnv1a.h"
#include "gguf_reader.h"
#include "heap.h"

#include <errno.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct transformer_moe_profile {
    size_t            n_layers, n_exp;
    uint64_t          key;
    char              path[1024]; /* empty: no cache */
    uint64_t         *base;       /* [n_layers * n_exp], read at load */
    uint64_t          base_tokens;
    bool              loaded;
    _Atomic uint64_t *count; /* [n_layers * n_exp], this process's */
    _Atomic uint64_t  tokens;
};

uint64_t moe_profile_key(const struct gguf_ctx *gguf) {
    uint64_t     h = GEIST_FNV1A_INIT;
    const size_t n = gguf_tensor_count(gguf);
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(gguf, i);
        if (t == nullptr)
            continue;
        const uint64_t fields[] = {(uint64_t) t->dtype, (uint64_t) t->n_dims, t->offset};
        h                       = geist_fnv1a(h, strlen(t->name), t->name);
        h                       = geist_fnv1a(h, sizeof fields, fields);
        h                       = geist_fnv1a(h, sizeof t->dims, t->dims);
    }
    const struct gguf_tensor_t *r = gguf_get_tensor(gguf, "blk.0.ffn_gate_inp.weight");
    if (r != nullptr && r->data != nullptr)
        h = geist_fnv1a(h, r->nbytes, r->data);
    return h;
}

bool moe_profile_dir(size_t n, char dir[static n]) {
    const char *env = getenv("GEIST_MOE_PROFILE_CACHE");
    if (env == nullptr || env[0] == '\0' || strcmp(env, "0") == 0)
        return false; /* opt-in */
    if (strcmp(env, "1") == 0)
        return geist_cache_dir(n, "moe-profile", dir);
    const int w = snprintf(dir, n, "%s", env);
    return w > 0 && (size_t) w < n;
}

/* mkdir -p for the cache directory; a failure shows up when the file is
 * written. */
static void make_dirs(const char *dir) {
    char p[1024];
    if ((size_t) snprintf(p, sizeof p, "%s", dir) >= sizeof p)
        return;
    for (char *c = p + 1; *c != '\0'; c++) {
        if (*c == '/') {
            *c = '\0';
            (void) mkdir(p, 0755);
            *c = '/';
        }
    }
    (void) mkdir(p, 0755);
}

/* Parses the cache file into p->base; false (and base zeroed) when it is
 * foreign or malformed, with `why` set. */
static bool read_file(struct transformer_moe_profile *p, FILE *f, const char **why) {
    const size_t L = p->n_layers, E = p->n_exp;
    char         key[17] = {0};
    size_t       fl = 0, fe = 0;
    int          ver = 0;
    if (fscanf(f,
               "# geist moe profile v%d key %16s layers %zu experts %zu ",
               &ver,
               key,
               &fl,
               &fe) != 4) {
        *why = "no profile header";
        return false;
    }
    char want[17];
    snprintf(want, sizeof want, "%016" PRIx64, p->key);
    if (ver != 1 || strcmp(key, want) != 0 || fl != L || fe != E) {
        *why = "made for another model";
        return false;
    }
    for (size_t l = 0; l < L; l++) {
        unsigned long long tok = 0;
        size_t             fl2 = 0;
        if (fscanf(f, " USAGE %16s profile run-time %llu %zu ", key, &tok, &fl2) != 3 || fl2 != l ||
            strcmp(key, want) != 0) {
            *why = "a USAGE line does not parse";
            return false;
        }
        if (l == 0)
            p->base_tokens = tok;
        for (size_t e = 0; e < E; e++) {
            unsigned long long c = 0;
            if (fscanf(f, e == 0 ? "%llu" : ",%llu", &c) != 1) {
                *why = "a count does not parse";
                return false;
            }
            p->base[l * E + e] = c;
        }
    }
    return true;
}

struct transformer_moe_profile *
moe_profile_open(size_t n_layers, size_t n_exp, uint64_t key, const char *dir) {
    size_t n = 0;
    if (ckd_mul(&n, n_layers, n_exp) || n == 0)
        return nullptr;
    struct transformer_moe_profile *p =
            heap_calloc_aligned(1, sizeof *p, alignof(struct transformer_moe_profile));
    if (p == nullptr)
        return nullptr;
    p->n_layers = n_layers;
    p->n_exp    = n_exp;
    p->key      = key;
    p->base     = heap_calloc_array_aligned(uint64_t, n);
    p->count    = heap_calloc_array_aligned(_Atomic uint64_t, n);
    if (p->base == nullptr || p->count == nullptr) {
        moe_profile_close(p);
        return nullptr;
    }
    if (dir == nullptr)
        return p;
    const int w = snprintf(p->path, sizeof p->path, "%s/%016" PRIx64 ".txt", dir, key);
    if (w <= 0 || (size_t) w >= sizeof p->path) {
        p->path[0] = '\0';
        return p;
    }
    FILE *f = fopen(p->path, "r");
    if (f == nullptr)
        return p; /* first load of this model */
    const char *why = nullptr;
    p->loaded       = read_file(p, f, &why);
    fclose(f);
    if (!p->loaded) {
        memset(p->base, 0, n * sizeof *p->base);
        p->base_tokens = 0;
        fprintf(stderr, "geist: moe: profile cache %s ignored: %s\n", p->path, why);
    }
    return p;
}

void moe_profile_count(struct transformer_moe_profile *p,
                       size_t                          layer,
                       size_t                          seq,
                       size_t                          stride,
                       size_t                          n_routed,
                       const uint32_t                  ids[static seq * stride]) {
    if (p == nullptr || layer >= p->n_layers)
        return;
    _Atomic uint64_t *row = p->count + layer * p->n_exp;
    for (size_t t = 0; t < seq; t++) {
        for (size_t j = 0; j < n_routed; j++) {
            const uint32_t e = ids[t * stride + j];
            if (e < p->n_exp)
                atomic_fetch_add_explicit(&row[e], 1, memory_order_relaxed);
        }
    }
    if (layer == 0)
        atomic_fetch_add_explicit(&p->tokens, seq, memory_order_relaxed);
}

void moe_profile_counts(const struct transformer_moe_profile *p,
                        size_t                                n,
                        uint64_t                              out[static n],
                        uint64_t                             *tokens) {
    const size_t m = p->n_layers * p->n_exp < n ? p->n_layers * p->n_exp : n;
    for (size_t i = 0; i < m; i++)
        out[i] = p->base[i] + atomic_load_explicit(&p->count[i], memory_order_relaxed);
    if (tokens != nullptr)
        *tokens = p->base_tokens + atomic_load_explicit(&p->tokens, memory_order_relaxed);
}

bool moe_profile_loaded(const struct transformer_moe_profile *p) {
    return p != nullptr && p->loaded;
}

const char *moe_profile_path(const struct transformer_moe_profile *p) {
    return p != nullptr ? p->path : "";
}

/* The counts to <path>.tmp.<pid>, then renamed over <path>: a reader never
 * sees half a file. */
static void write_file(const struct transformer_moe_profile *p) {
    char dir[1024], tmp[1100];
    snprintf(dir, sizeof dir, "%s", p->path);
    char *slash = strrchr(dir, '/');
    if (slash != nullptr) {
        *slash = '\0';
        make_dirs(dir);
    }
    if ((size_t) snprintf(tmp, sizeof tmp, "%s.tmp.%ld", p->path, (long) getpid()) >= sizeof tmp)
        return;
    FILE *f = fopen(tmp, "w");
    if (f == nullptr) {
        fprintf(stderr, "geist: moe: profile cache %s not written: %s\n", p->path, strerror(errno));
        return;
    }
    char key[17];
    snprintf(key, sizeof key, "%016" PRIx64, p->key);
    const unsigned long long tok =
            p->base_tokens + atomic_load_explicit(&p->tokens, memory_order_relaxed);
    bool ok = fprintf(f,
                      "# geist moe profile v1 key %s layers %zu experts %zu\n",
                      key,
                      p->n_layers,
                      p->n_exp) > 0;
    for (size_t l = 0; ok && l < p->n_layers; l++) {
        ok = fprintf(f, "USAGE %s profile run-time %llu %zu ", key, tok, l) > 0;
        for (size_t e = 0; ok && e < p->n_exp; e++) {
            const size_t i = l * p->n_exp + e;
            ok = fprintf(f,
                         e == 0 ? "%llu" : ",%llu",
                         (unsigned long long) (p->base[i] +
                                               atomic_load_explicit(&p->count[i],
                                                                    memory_order_relaxed))) > 0;
        }
        ok = ok && fputc('\n', f) != EOF;
    }
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp, p->path) != 0) {
        fprintf(stderr, "geist: moe: profile cache %s not written: %s\n", p->path, strerror(errno));
        (void) remove(tmp);
    }
}

void moe_profile_close(struct transformer_moe_profile *p) {
    if (p == nullptr)
        return;
    if (p->path[0] != '\0' && p->count != nullptr &&
        atomic_load_explicit(&p->tokens, memory_order_relaxed) > 0)
        write_file(p);
    void *b = p->base, *c = (void *) p->count, *s = p;
    safe_free(&b);
    safe_free(&c);
    safe_free(&s);
}
