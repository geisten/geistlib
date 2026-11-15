#include <geist.h>
#include <geist_util.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3)
        return 2;

    geist_token_t ids[91];
    int32_t       selected;
    FILE         *input = fopen(argv[2], "rb");
    if (!input)
        return 2;
    const bool valid = fread(ids, sizeof(ids[0]), 91, input) == 91 &&
                       fread(&selected, sizeof(selected), 1, input) == 1 && fgetc(input) == EOF &&
                       selected >= 0;
    fclose(input);
    if (!valid)
        return 2;

    struct geist_backend     *backend      = nullptr;
    struct geist_model       *model        = nullptr;
    struct geist_session     *session      = nullptr;
    struct geist_backend_opts backend_opts = {.max_threads = 8};
    struct geist_session_opts session_opts = {
            .max_seq_len = 34816,
            .temperature = 0,
            .kv_mode     = GEIST_KV_FP32,
            .m_max       = 1,
    };
    int rc = 1;
    if (geist_backend_create("metal", &backend_opts, nullptr, &backend) != GEIST_OK ||
        geist_model_load_with_opts(argv[1], backend, &session_opts, &model) != GEIST_OK ||
        geist_session_create(model, backend, &session_opts, &session) != GEIST_OK)
        goto done;
    for (size_t pos = 0; pos < 91; pos++) {
        if (geist_session_prefill_tokens(session, 1, &ids[pos]) != GEIST_OK) {
            fprintf(stderr, "Prefill at %zu: %s\n", pos, geist_session_errmsg(session));
            goto done;
        }
        size_t       vocab  = 0;
        const float *logits = geist_session_peek_logits(&vocab, session);
        if (!logits || (size_t) selected >= vocab)
            goto done;
        uint64_t             hash  = UINT64_C(14695981039346656037);
        const unsigned char *bytes = (const unsigned char *) logits;
        for (size_t i = 0; i < vocab * sizeof(*logits); i++) {
            hash ^= bytes[i];
            hash *= UINT64_C(1099511628211);
        }
        printf("logits %zu %016" PRIx64 " %.9g\n", pos, hash, (double) logits[selected]);
    }
    rc = 0;
done:
    geist_session_destroy(session);
    geist_model_destroy(model);
    geist_backend_destroy(backend);
    return rc;
}
