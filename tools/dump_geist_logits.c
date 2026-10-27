/* Dump the final next-token logits for a text prompt.
 *
 * Usage: dump_geist_logits <model.gguf> <prompt.txt> <ids.bin> <logits.bin>
 *
 * GEIST_DUMP_BACKEND=<name> selects the backend ("auto" by default) — e.g. to
 * compare vulkan against the cpu_scalar numerical oracle on the same ids.
 *
 * The token IDs are written so an independent runtime can consume exactly
 * the same input, avoiding tokenizer or chat-template differences.
 */
#include "tool_common.h"

#include <stdint.h>

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
    char  *prompt       = tool_read_file(argv[2], &prompt_bytes);
    if (prompt == nullptr || prompt_bytes == 0) {
        fprintf(stderr, "failed to read non-empty prompt: %s\n", argv[2]);
        free(prompt);
        return 1;
    }

    struct tool_engine e   = {0};
    geist_token_t     *ids = nullptr;
    int                rc  = 1;

    const char *backend_name = getenv("GEIST_DUMP_BACKEND");
    if (backend_name == nullptr || backend_name[0] == '\0') {
        backend_name = "auto";
    }
    struct geist_session_opts opts = {.temperature = 0.0f};
    if (!tool_open(backend_name, nullptr, argv[1], nullptr, &opts, &e))
        goto out;
    struct geist_session *session = e.sess;

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
            geist_backend_name(e.be),
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
    tool_close(&e);
    free(prompt);
    return rc;
}
