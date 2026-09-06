/* Dump the final next-token logits for a text prompt.
 *
 * Usage: dump_geist_logits <model.gguf> <prompt.txt> <ids.bin> <logits.bin>
 *
 * The token IDs are written so an independent runtime can consume exactly
 * the same input, avoiding tokenizer or chat-template differences.
 */
#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void *read_file(const char *path, size_t *size, int nul_terminate) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr)
        return nullptr;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return nullptr;
    }
    long end = ftell(f);
    if (end < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return nullptr;
    }
    size_t bytes = (size_t) end;
    void  *data  = malloc(bytes + (nul_terminate ? 1u : 0u));
    if (data == nullptr || fread(data, 1, bytes, f) != bytes) {
        free(data);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    if (nul_terminate)
        ((char *) data)[bytes] = '\0';
    *size = bytes;
    return data;
}

static int write_file(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (f == nullptr)
        return -1;
    int rc = fwrite(data, 1, bytes, f) == bytes ? 0 : -1;
    if (fclose(f) != 0)
        rc = -1;
    return rc;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt.txt> <ids.bin> <logits.bin>\n", argv[0]);
        return 2;
    }

    size_t prompt_bytes = 0;
    char  *prompt       = read_file(argv[2], &prompt_bytes, 1);
    if (prompt == nullptr || prompt_bytes == 0) {
        fprintf(stderr, "failed to read non-empty prompt: %s\n", argv[2]);
        free(prompt);
        return 1;
    }

    struct geist_backend *backend = nullptr;
    struct geist_model   *model   = nullptr;
    struct geist_session *session = nullptr;
    geist_token_t        *ids     = nullptr;
    int                   rc      = 1;

    if (geist_backend_create("auto", nullptr, nullptr, &backend) != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        goto out;
    }
    if (geist_model_load(argv[1], backend, &model) != GEIST_OK) {
        fprintf(stderr, "model load failed: %s\n", geist_last_create_error());
        goto out;
    }
    struct geist_session_opts opts = {.temperature = 0.0f};
    if (geist_session_create(model, backend, &opts, &session) != GEIST_OK) {
        fprintf(stderr, "session create failed\n");
        goto out;
    }

    enum { MAX_PROMPT_TOKENS = 8192 };
    size_t n_ids = 0;
    ids          = malloc(MAX_PROMPT_TOKENS * sizeof(*ids));
    if (ids == nullptr ||
        geist_session_tokenize(session, prompt, MAX_PROMPT_TOKENS, ids, &n_ids) != GEIST_OK ||
        n_ids == 0) {
        fprintf(stderr, "tokenization failed: %s\n", geist_session_errmsg(session));
        goto out;
    }
    if (geist_session_prefill_tokens(session, n_ids, ids) != GEIST_OK) {
        fprintf(stderr, "prefill failed: %s\n", geist_session_errmsg(session));
        goto out;
    }

    size_t       n_logits = 0;
    const float *logits   = geist_session_peek_logits(&n_logits, session);
    if (logits == nullptr || n_logits == 0) {
        fprintf(stderr, "no logits: %s\n", geist_session_errmsg(session));
        goto out;
    }
    if (write_file(argv[3], ids, n_ids * sizeof(*ids)) != 0 ||
        write_file(argv[4], logits, n_logits * sizeof(*logits)) != 0) {
        fprintf(stderr, "failed to write output files\n");
        goto out;
    }
    fprintf(stderr,
            "backend=%s tokens=%zu logits=%zu top1=",
            geist_backend_name(backend),
            n_ids,
            n_logits);
    size_t top1 = 0;
    for (size_t i = 1; i < n_logits; i++)
        if (logits[i] > logits[top1])
            top1 = i;
    const char *piece = geist_session_token_to_str(session, (geist_token_t) top1);
    fprintf(stderr, "%zu piece=%s\n", top1, piece != nullptr ? piece : "<decode-error>");
    rc = 0;

out:
    free(ids);
    if (session != nullptr)
        geist_session_destroy(session);
    if (model != nullptr)
        geist_model_destroy(model);
    if (backend != nullptr)
        geist_backend_destroy(backend);
    free(prompt);
    return rc;
}
