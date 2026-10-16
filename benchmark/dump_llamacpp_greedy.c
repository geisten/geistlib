/* Greedy-decode n tokens after a token-id prompt with llama.cpp, printing
 * each step like tools/dump_geist_greedy: "step token top2 logit1 logit2".
 *
 * Build against a llama.cpp checkout L:
 *   cc -std=c11 -O2 -I$L/include -I$L/ggml/include -o dump_llamacpp_greedy \
 *      benchmark/dump_llamacpp_greedy.c -L$L/build-cpu/bin -lllama -lggml -lggml-base
 * Usage: dump_llamacpp_greedy <model.gguf> <ids.bin> <n>
 */
#include <llama.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <ids.bin> <n>\n", argv[0]);
        return 2;
    }
    int   n_steps = atoi(argv[3]);
    FILE *f       = fopen(argv[2], "rb");
    if (n_steps <= 0 || f == NULL)
        return 2;
    static int32_t ids[8192];
    int            n_ids = (int) fread(ids, sizeof ids[0], 8192, f);
    fclose(f);
    if (n_ids <= 0)
        return 2;

    ggml_backend_load_all();
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers              = 0;
    struct llama_model *model    = llama_model_load_from_file(argv[1], mp);
    if (model == NULL)
        return 1;
    const struct llama_vocab   *vocab   = llama_model_get_vocab(model);
    const int                   n_vocab = llama_vocab_n_tokens(vocab);
    struct llama_context_params cp      = llama_context_default_params();
    cp.n_ctx                            = (uint32_t) (n_ids + n_steps + 16);
    cp.n_batch                          = (uint32_t) n_ids;
    cp.n_ubatch                         = (uint32_t) n_ids;
    struct llama_context *ctx           = llama_init_from_model(model, cp);
    if (ctx == NULL)
        return 1;
    if (llama_decode(ctx, llama_batch_get_one(ids, n_ids)) != 0)
        return 1;
    for (int step = 0; step < n_steps; step++) {
        const float *lg = llama_get_logits_ith(ctx, -1);
        int          a = lg[1] > lg[0] ? 1 : 0, b = 1 - a;
        for (int i = 2; i < n_vocab; i++) {
            if (lg[i] > lg[a]) {
                b = a;
                a = i;
            } else if (lg[i] > lg[b]) {
                b = i;
            }
        }
        printf("%d %d %d %.6f %.6f\n", step, a, b, (double) lg[a], (double) lg[b]);
        llama_token t = a;
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0)
            return 1;
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
