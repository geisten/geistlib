/* #209 pilot: the full logit vector at the last prompt token, per question.
 *
 * usage: dump_last_logits <model.gguf> <items.bin> <out.bin>
 *
 * items.bin (little-endian): uint32 count, then per item uint32 n_prefix,
 * uint32 n_total, int32 ids[n_total]. Items sharing a prefix (the same
 * n_prefix and ids) follow each other; the prefix is prefilled once, then
 * checkpointed, and each item truncates back to it and prefills its own
 * rest (DeltaNet state restored by the checkpoint).
 * out.bin: uint32 count, uint32 n_vocab, then float32[count][n_vocab].
 * Metal, greedy settings irrelevant (no decoding). Run under with-mac-lock.
 */
#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model items.bin out.bin\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[2], "rb"), *out = fopen(argv[3], "wb");
    uint32_t count = 0;
    if (in == nullptr || out == nullptr || rd(in, &count, 4) != 0)
        return 2;
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK)
        return 1;
    struct geist_session_opts opts = {.max_seq_len = 4096, .temperature = 0.0f};
    struct geist_model       *m    = nullptr;
    struct geist_session     *s    = nullptr;
    if (geist_model_load_with_opts(argv[1], be, &opts, &m) != GEIST_OK ||
        geist_session_create(m, be, &opts, &s) != GEIST_OK) {
        fprintf(stderr, "load: %s\n", geist_last_create_error());
        return 1;
    }
    int32_t *ids = malloc(4096 * sizeof *ids), *pref = malloc(4096 * sizeof *pref);
    uint32_t n_pref_cur = 0, n_vocab = 0;
    bool     have_pref = false;
    double   t0 = now_s();
    for (uint32_t i = 0; i < count; i++) {
        uint32_t np = 0, nt = 0;
        if (rd(in, &np, 4) || rd(in, &nt, 4) || nt > 4096 || np >= nt || rd(in, ids, nt * 4u)) {
            fprintf(stderr, "bad item %u\n", i);
            return 2;
        }
        const bool same = have_pref && np == n_pref_cur && memcmp(ids, pref, np * 4u) == 0;
        if (!same) {
            if (geist_session_reset(s) != GEIST_OK ||
                geist_session_prefill_tokens(s, np, (const geist_token_t *) ids) != GEIST_OK ||
                geist_session_checkpoint(s) != GEIST_OK) {
                fprintf(stderr, "prefix %u: %s\n", i, geist_session_errmsg(s));
                return 1;
            }
            memcpy(pref, ids, np * 4u);
            n_pref_cur = np;
            have_pref  = true;
        } else if (geist_session_truncate(s, np) != GEIST_OK) {
            fprintf(stderr, "truncate %u: %s\n", i, geist_session_errmsg(s));
            return 1;
        }
        if (geist_session_prefill_tokens(s, nt - np, (const geist_token_t *) ids + np) != GEIST_OK) {
            fprintf(stderr, "prefill %u: %s\n", i, geist_session_errmsg(s));
            return 1;
        }
        size_t       nv = 0;
        const float *lg = geist_session_peek_logits(&nv, s);
        if (lg == nullptr || nv == 0)
            return 1;
        if (i == 0) {
            n_vocab = (uint32_t) nv;
            fwrite(&count, 4, 1, out);
            fwrite(&n_vocab, 4, 1, out);
        }
        if (nv != n_vocab || fwrite(lg, sizeof(float), nv, out) != nv)
            return 1;
        if ((i + 1) % 100 == 0) {
            fprintf(stderr, "%u/%u  %.2f s/item\n", i + 1, count, (now_s() - t0) / (i + 1));
        }
    }
    fclose(out);
    fprintf(stderr, "done %u items, %.1f s\n", count, now_s() - t0);
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return 0;
}
