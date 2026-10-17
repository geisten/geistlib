/* Teacher-forced next-token logits at every position, through the decode
 * path: prefill the first n_prompt ids of ids.bin, then for each later id
 * write the logits that predict it (float32, n_vocab each) and feed it.
 *
 * Usage: dump_geist_tf_logits <model.gguf> <ids.bin> <n_prompt> <out.bin>
 *
 * out.bin holds (n_ids - n_prompt) rows of n_vocab floats, in order; an
 * independent runtime (benchmark/dump_llamacpp_tf.cpp) writes the same rows
 * for tools/js_streams.c. GEIST_BACKEND picks the backend.
 */
#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <model.gguf> <ids.bin> <n_prompt> <out.bin>\n", argv[0]);
        return 2;
    }
    const long n_prompt = strtol(argv[3], nullptr, 10);
    FILE      *f        = fopen(argv[2], "rb");
    if (n_prompt <= 0 || f == nullptr) {
        if (f != nullptr)
            fclose(f);
        return 2;
    }
    enum { MAX_IDS = 16384 };
    static geist_token_t ids[MAX_IDS];
    const size_t         n_ids = fread(ids, sizeof ids[0], MAX_IDS, f);
    fclose(f);
    if (n_ids <= (size_t) n_prompt)
        return 2;

    struct geist_backend     *be   = nullptr;
    struct geist_model       *m    = nullptr;
    struct geist_session     *s    = nullptr;
    FILE                     *out  = nullptr;
    int                       rc   = 1;
    struct geist_session_opts opts = {.temperature = 0.0f, .max_seq_len = n_ids + 16};
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK ||
        geist_model_load(argv[1], be, &m) != GEIST_OK ||
        geist_session_create(m, be, &opts, &s) != GEIST_OK ||
        geist_session_prefill_tokens(s, (size_t) n_prompt, ids) != GEIST_OK) {
        fprintf(stderr, "setup failed: %s\n", geist_last_create_error());
        goto done;
    }
    out = fopen(argv[4], "wb");
    if (out == nullptr)
        goto done;
    for (size_t i = (size_t) n_prompt; i < n_ids; i++) {
        size_t       n  = 0;
        const float *lg = geist_session_peek_logits(&n, s);
        if (lg == nullptr || fwrite(lg, sizeof lg[0], n, out) != n ||
            geist_session_prefill_tokens(s, 1, &ids[i]) != GEIST_OK) {
            fprintf(stderr, "step %zu failed: %s\n", i, geist_session_errmsg(s));
            goto done;
        }
    }
    fprintf(stderr, "backend=%s steps=%zu\n", geist_backend_name(be), n_ids - (size_t) n_prompt);
    rc = 0;
done:
    if (out != nullptr)
        fclose(out);
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return rc;
}
