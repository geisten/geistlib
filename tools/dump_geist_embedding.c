/* Dump pooled sentence embeddings for a set of prompts.
 *
 * Usage: dump_geist_embedding <model.gguf> <prompts.txt> <out.gemb>
 *
 * `prompts.txt` holds one prompt per line; blank lines are skipped. All of
 * them run against ONE loaded model — the point of taking a file rather
 * than a single prompt, since the 0.6B is ~700 MB and reloading it per
 * prompt would dominate everything.
 *
 * This is the parity oracle's geistlib side: `tools/eval_embedding_fidelity.py`
 * compares the .gemb this writes against the same prompts embedded by
 * upstream's llama-embedding, and gates on cosine similarity. See
 * docs/BITNET_EMBEDDINGS_PLAN.md phase 3.
 *
 * Output format (.gemb), little-endian, self-describing so the Python side
 * needs no out-of-band shape:
 *
 *   magic   char[4]   "GEMB"
 *   version uint32    1
 *   count   uint32    number of prompts
 *   dim     uint32    embedding dimension
 *   data    float32[count * dim]   row-major, already L2-normalised
 */
#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_PROMPT_TOKENS = 8192 };

static char *read_file_z(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        return nullptr;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return nullptr;
    }
    const long end = ftell(f);
    if (end < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return nullptr;
    }
    const size_t bytes = (size_t) end;
    char        *data  = malloc(bytes + 1u);
    if (data == nullptr || fread(data, 1, bytes, f) != bytes) {
        free(data);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    data[bytes] = '\0';
    *size       = bytes;
    return data;
}

/* Split `text` in place into NUL-terminated lines, skipping blank ones.
 * Returns the count and fills `out` up to `cap` entries. */
static size_t split_lines(size_t cap, char *text, char *out[]) {
    size_t n    = 0;
    char  *line = text;
    while (*line != '\0' && n < cap) {
        char *nl = strchr(line, '\n');
        if (nl != nullptr) {
            *nl = '\0';
        }
        /* Tolerate CRLF inputs — a stray \r would otherwise become part of
         * the prompt and silently change the embedding. */
        const size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\r') {
            line[len - 1] = '\0';
        }
        if (line[0] != '\0') {
            out[n++] = line;
        }
        if (nl == nullptr) {
            break;
        }
        line = nl + 1;
    }
    return n;
}

static int write_header(FILE *f, uint32_t count, uint32_t dim) {
    const uint32_t version = 1;
    return (fwrite("GEMB", 1, 4, f) == 4 && fwrite(&version, sizeof version, 1, f) == 1 &&
            fwrite(&count, sizeof count, 1, f) == 1 && fwrite(&dim, sizeof dim, 1, f) == 1)
                   ? 0
                   : -1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompts.txt> <out.gemb>\n", argv[0]);
        return 2;
    }

    size_t text_bytes = 0;
    char  *text       = read_file_z(argv[2], &text_bytes);
    if (text == nullptr || text_bytes == 0) {
        fprintf(stderr, "failed to read non-empty prompts file: %s\n", argv[2]);
        free(text);
        return 1;
    }

    enum { MAX_PROMPTS = 4096 };
    char **prompts = malloc(MAX_PROMPTS * sizeof(*prompts));
    if (prompts == nullptr) {
        free(text);
        return 1;
    }
    const size_t n_prompts = split_lines(MAX_PROMPTS, text, prompts);
    if (n_prompts == 0) {
        fprintf(stderr, "no non-blank prompts in %s\n", argv[2]);
        free(prompts);
        free(text);
        return 1;
    }

    struct geist_backend *backend = nullptr;
    struct geist_model   *model   = nullptr;
    struct geist_session *session = nullptr;
    geist_token_t        *ids     = nullptr;
    FILE                 *out     = nullptr;
    int                   rc      = 1;

    if (geist_backend_create("auto", nullptr, nullptr, &backend) != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        goto done;
    }
    if (geist_model_load(argv[1], backend, &model) != GEIST_OK) {
        fprintf(stderr, "model load failed: %s\n", geist_last_create_error());
        goto done;
    }
    struct geist_session_opts opts = {.temperature = 0.0f};
    if (geist_session_create(model, backend, &opts, &session) != GEIST_OK) {
        fprintf(stderr, "session create failed\n");
        goto done;
    }
    ids = malloc(MAX_PROMPT_TOKENS * sizeof(*ids));
    if (ids == nullptr) {
        goto done;
    }

    /* The dimension is not known until the first embedding comes back, so
     * the header is written after it and the file is opened for update. */
    out = fopen(argv[3], "w+b");
    if (out == nullptr) {
        fprintf(stderr, "cannot open %s for writing\n", argv[3]);
        goto done;
    }
    if (write_header(out, (uint32_t) n_prompts, 0) != 0) {
        fprintf(stderr, "failed to write header\n");
        goto done;
    }

    size_t dim = 0;
    for (size_t i = 0; i < n_prompts; i++) {
        if (geist_session_reset(session) != GEIST_OK) {
            fprintf(stderr, "session reset failed at prompt %zu\n", i);
            goto done;
        }
        size_t n_ids = 0;
        if (geist_session_tokenize(session, prompts[i], MAX_PROMPT_TOKENS, ids, &n_ids) !=
                    GEIST_OK ||
            n_ids == 0) {
            fprintf(stderr,
                    "tokenization failed at prompt %zu: %s\n",
                    i,
                    geist_session_errmsg(session));
            goto done;
        }
        if (geist_session_prefill_tokens(session, n_ids, ids) != GEIST_OK) {
            fprintf(stderr, "prefill failed at prompt %zu: %s\n", i, geist_session_errmsg(session));
            goto done;
        }
        size_t       n_dims = 0;
        const float *emb    = geist_session_peek_embedding(&n_dims, session);
        if (emb == nullptr || n_dims == 0) {
            fprintf(stderr,
                    "no embedding at prompt %zu — is this an embedding model? (%s)\n",
                    i,
                    geist_session_errmsg(session));
            goto done;
        }
        /* Every row must be the same width, or the file's single `dim`
         * would be a lie about all but the first. */
        if (dim == 0) {
            dim = n_dims;
        } else if (n_dims != dim) {
            fprintf(stderr, "embedding dim changed: %zu then %zu\n", dim, n_dims);
            goto done;
        }
        if (fwrite(emb, sizeof(float), n_dims, out) != n_dims) {
            fprintf(stderr, "short write at prompt %zu\n", i);
            goto done;
        }
    }

    /* Backfill the now-known dimension. */
    if (fseek(out, 0, SEEK_SET) != 0 ||
        write_header(out, (uint32_t) n_prompts, (uint32_t) dim) != 0) {
        fprintf(stderr, "failed to backfill header\n");
        goto done;
    }
    fprintf(stderr,
            "backend=%s prompts=%zu dim=%zu -> %s\n",
            geist_backend_name(backend),
            n_prompts,
            dim,
            argv[3]);
    rc = 0;

done:
    if (out != nullptr && fclose(out) != 0) {
        rc = 1;
    }
    free(ids);
    if (session != nullptr) {
        geist_session_destroy(session);
    }
    if (model != nullptr) {
        geist_model_destroy(model);
    }
    if (backend != nullptr) {
        geist_backend_destroy(backend);
    }
    free(prompts);
    free(text);
    return rc;
}
