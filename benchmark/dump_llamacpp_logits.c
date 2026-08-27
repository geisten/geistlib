/* Independent llama.cpp adapter used by controlled logit comparisons.
 *
 * Build with:
 *   cc -O2 $(pkg-config --cflags llama ggml) -o dump_llamacpp_logits \
 *      benchmark/dump_llamacpp_logits.c $(pkg-config --libs llama ggml)
 * Usage: dump_llamacpp_logits <model.gguf> <ids.bin> <logits.bin>
 */
#include <llama.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <ids.bin> <logits.bin>\n", argv[0]);
        return 2;
    }
    FILE *input = fopen(argv[2], "rb");
    if (input == NULL || fseek(input, 0, SEEK_END) != 0) {
        perror("ids");
        return 1;
    }
    long bytes = ftell(input);
    if (bytes <= 0 || bytes % (long) sizeof(int32_t) != 0 || fseek(input, 0, SEEK_SET) != 0) {
        fprintf(stderr, "invalid ids file\n");
        fclose(input);
        return 1;
    }
    int32_t *ids = malloc((size_t) bytes);
    if (ids == NULL || fread(ids, 1, (size_t) bytes, input) != (size_t) bytes) {
        fprintf(stderr, "failed to read ids\n");
        free(ids);
        fclose(input);
        return 1;
    }
    fclose(input);
    int n_tokens = (int) ((size_t) bytes / sizeof(*ids));

    ggml_backend_load_all();
    llama_backend_init();
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers              = 999;
    struct llama_model *model              = llama_model_load_from_file(argv[1], model_params);
    if (model == NULL) {
        fprintf(stderr, "model load failed\n");
        free(ids);
        return 1;
    }
    const struct llama_vocab *vocab   = llama_model_get_vocab(model);
    int                       n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                       = (uint32_t) (n_tokens + 16);
    ctx_params.n_batch                     = (uint32_t) n_tokens;
    ctx_params.n_ubatch                    = (uint32_t) n_tokens;
    ctx_params.n_threads                   = 1;
    ctx_params.n_threads_batch             = 1;
    struct llama_context *ctx              = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "context init failed\n");
        llama_model_free(model);
        free(ids);
        return 1;
    }

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = (llama_token) ids[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i == n_tokens - 1;
    }
    batch.n_tokens = n_tokens;
    int rc         = 1;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        goto out;
    }
    float *logits = llama_get_logits_ith(ctx, n_tokens - 1);
    FILE  *output = fopen(argv[3], "wb");
    if (logits == NULL || output == NULL ||
        fwrite(logits, sizeof(*logits), (size_t) n_vocab, output) != (size_t) n_vocab) {
        fprintf(stderr, "failed to write logits\n");
        if (output != NULL)
            fclose(output);
        goto out;
    }
    fclose(output);
    int top1 = 0;
    for (int i = 1; i < n_vocab; i++)
        if (logits[i] > logits[top1])
            top1 = i;
    char    piece[256];
    int32_t piece_len =
            llama_token_to_piece(vocab, (llama_token) top1, piece, sizeof(piece) - 1, 0, true);
    if (piece_len >= 0)
        piece[piece_len] = '\0';
    fprintf(stderr,
            "backend=Metal tokens=%d logits=%d top1=%d piece=%s\n",
            n_tokens,
            n_vocab,
            top1,
            piece_len >= 0 ? piece : "<decode-error>");
    rc = 0;

out:
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(ids);
    return rc;
}
