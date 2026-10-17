/* Teacher-forced log-probabilities with llama.cpp: prefill the first
 * n_prompt ids of ids.bin, then for each later id print log p(id | prefix)
 * and feed it, one token at a time (the decode shapes). One number per line.
 *
 * Build against a llama.cpp checkout L:
 *   cc -std=c11 -O2 -I$L/include -I$L/ggml/include -o dump_llamacpp_nll \
 *      benchmark/dump_llamacpp_nll.c -L$L/build-cpu/bin -lllama -lggml -lggml-base -lm
 * Usage: dump_llamacpp_nll <model.gguf> <ids.bin> <n_prompt>
 */
#include <llama.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4)
        return 2;
    const int n_prompt = atoi(argv[3]);
    FILE     *f        = fopen(argv[2], "rb");
    if (f == NULL || n_prompt <= 0)
        return 2;
    static int32_t ids[16384];
    const int      n = (int) fread(ids, sizeof ids[0], 16384, f);
    fclose(f);
    if (n <= n_prompt)
        return 2;
    ggml_backend_load_all();
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers              = 0;
    struct llama_model *model    = llama_model_load_from_file(argv[1], mp);
    if (model == NULL)
        return 1;
    const int                   nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx                       = (uint32_t) (n + 16);
    cp.n_batch = cp.n_ubatch       = (uint32_t) n_prompt;
    struct llama_context *ctx      = llama_init_from_model(model, cp);
    if (ctx == NULL || llama_decode(ctx, llama_batch_get_one(ids, n_prompt)) != 0)
        return 1;
    for (int i = n_prompt; i < n; i++) {
        const float *lg = llama_get_logits_ith(ctx, -1);
        double       mx = lg[0], z = 0.0;
        for (int v = 1; v < nv; v++)
            mx = lg[v] > mx ? lg[v] : mx;
        for (int v = 0; v < nv; v++)
            z += exp(lg[v] - mx);
        printf("%.6f\n", lg[ids[i]] - mx - log(z));
        if (llama_decode(ctx, llama_batch_get_one(&ids[i], 1)) != 0)
            return 1;
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
