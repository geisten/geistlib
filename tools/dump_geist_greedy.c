/* Greedy-decode n tokens after a token-id prompt and print each step.
 *
 * Usage: dump_geist_greedy <model.gguf> <ids.bin> <n>
 *
 * ids.bin is the int32 prompt written by dump_geist_logits, so an
 * independent runtime (benchmark/dump_llamacpp_greedy.c) decodes from the
 * same input. One line per step: "step token top2 logit1 logit2", where
 * token is the argmax the step emits and top2 the runner-up; the gap
 * logit1 - logit2 says how close a divergence between engines is to a tie.
 * GEIST_DUMP_BACKEND=<name> selects the backend ("auto" by default).
 * GEIST_DUMP_LAST_LOGITS=<path> also writes the logits of the last step
 * (float32), for tools/compare_logits.py.
 */
#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <ids.bin> <n>\n", argv[0]);
        return 2;
    }
    const long n_steps = strtol(argv[3], nullptr, 10);
    FILE      *f       = fopen(argv[2], "rb");
    if (n_steps <= 0 || f == nullptr) {
        fprintf(stderr, "bad arguments\n");
        if (f != nullptr)
            fclose(f);
        return 2;
    }
    enum { MAX_PROMPT_TOKENS = 8192 };
    static geist_token_t ids[MAX_PROMPT_TOKENS];
    const size_t         n_ids = fread(ids, sizeof ids[0], MAX_PROMPT_TOKENS, f);
    fclose(f);

    struct geist_backend *backend = nullptr;
    struct geist_model   *model   = nullptr;
    struct geist_session *session = nullptr;
    int                   rc      = 1;
    const char           *bn      = getenv("GEIST_DUMP_BACKEND");
    if (bn == nullptr || bn[0] == '\0')
        bn = "auto";
    if (n_ids == 0 || geist_backend_create(bn, nullptr, nullptr, &backend) != GEIST_OK ||
        geist_model_load(argv[1], backend, &model) != GEIST_OK) {
        fprintf(stderr, "load failed: %s\n", geist_last_create_error());
        goto out;
    }
    struct geist_session_opts opts = {.temperature = 0.0f};
    if (geist_session_create(model, backend, &opts, &session) != GEIST_OK ||
        geist_session_prefill_tokens(session, n_ids, ids) != GEIST_OK) {
        fprintf(stderr,
                "prefill failed: %s\n",
                session != nullptr ? geist_session_errmsg(session) : "no session");
        goto out;
    }
    for (long step = 0; step < n_steps; step++) {
        size_t       n  = 0;
        const float *lg = geist_session_peek_logits(&n, session);
        if (lg == nullptr || n < 2) {
            fprintf(stderr, "no logits at step %ld\n", step);
            goto out;
        }
        size_t a = lg[1] > lg[0] ? 1 : 0, b = 1 - a;
        for (size_t i = 2; i < n; i++) {
            if (lg[i] > lg[a]) {
                b = a;
                a = i;
            } else if (lg[i] > lg[b]) {
                b = i;
            }
        }
        const char *dump = getenv("GEIST_DUMP_LAST_LOGITS");
        if (step == n_steps - 1 && dump != nullptr && dump[0] != '\0') {
            FILE *o = fopen(dump, "wb");
            if (o == nullptr || fwrite(lg, sizeof lg[0], n, o) != n) {
                fprintf(stderr, "failed to write %s\n", dump);
                if (o != nullptr)
                    fclose(o);
                goto out;
            }
            fclose(o);
        }
        const float   la = lg[a], lb = lg[b];
        geist_token_t t = 0;
        if (geist_session_decode_step(session, &t) != GEIST_OK) {
            fprintf(stderr, "decode failed at step %ld: %s\n", step, geist_session_errmsg(session));
            goto out;
        }
        printf("%ld %d %zu %.6f %.6f\n", step, (int) t, b, (double) la, (double) lb);
    }
    fprintf(stderr,
            "backend=%s prompt=%zu steps=%ld\n",
            geist_backend_name(backend),
            n_ids,
            n_steps);
    rc = 0;
out:
    geist_session_destroy(session);
    geist_model_destroy(model);
    geist_backend_destroy(backend);
    return rc;
}
