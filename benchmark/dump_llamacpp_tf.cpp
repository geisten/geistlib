// dump_llamacpp_tf — llama.cpp's side of the #48 divergence study.
// Teacher-forced decode, one token at a time: prefill the first n_prompt ids
// of ids.bin, then for each later id write the logits that predict it
// (float32 rows, as tools/dump_geist_tf_logits does) and feed it. With a
// trace path, also append one routing record per (layer, position) in the
// format of GEIST_MOE_TRACE: int32 layer, int32 position, the k expert ids
// ascending (uint32) and a margin field written as NaN. (Reading it back
// from ffn_moe_probs gave margins that contradict the chosen ids, so it is
// not reported rather than reported wrong.)
//
// Build against a llama.cpp checkout L:
//   c++ -std=c++17 -O2 -I$L/include -I$L/ggml/include benchmark/dump_llamacpp_tf.cpp \
//       -L$L/build-cpu/bin -lllama -lggml -lggml-base -o dump_llamacpp_tf
// Usage: dump_llamacpp_tf <model.gguf> <ids.bin> <n_prompt> <logits.bin> [trace.bin]
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

static FILE                     *g_trace = nullptr;
static std::map<int, int>        g_pos;   // layer -> next position

static bool cb(struct ggml_tensor *t, bool ask, void *) {
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0)
        return false;
    if (ask)
        return g_trace != nullptr;
    const int     layer = atoi(t->name + 13);
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
    std::vector<int32_t> ids(ne0 * ne1);
    for (int64_t r = 0; r < ne1; r++)
        ggml_backend_tensor_get(t, ids.data() + r * ne0, r * t->nb[1], ne0 * sizeof(int32_t));
    for (int64_t r = 0; r < ne1; r++) {
        std::vector<uint32_t> s(ids.begin() + r * ne0, ids.begin() + (r + 1) * ne0);
        std::sort(s.begin(), s.end());
        const int32_t hdr[2] = {layer, g_pos[layer]++};
        const float   margin = NAN;
        fwrite(hdr, sizeof hdr, 1, g_trace);
        fwrite(s.data(), sizeof(uint32_t), s.size(), g_trace);
        fwrite(&margin, sizeof margin, 1, g_trace);
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 5)
        return 2;
    const int n_prompt = atoi(argv[3]);
    FILE     *f        = fopen(argv[2], "rb");
    if (f == nullptr || n_prompt <= 0)
        return 2;
    std::vector<int32_t> ids(16384);
    ids.resize(fread(ids.data(), sizeof(int32_t), ids.size(), f));
    fclose(f);
    if ((int) ids.size() <= n_prompt)
        return 2;
    if (argc > 5)
        g_trace = fopen(argv[5], "wb");
    llama_backend_init();
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr)
        return 1;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto      cp = llama_context_default_params();
    cp.n_ctx     = (uint32_t) ids.size() + 16;
    cp.n_batch = cp.n_ubatch = (uint32_t) n_prompt;
    cp.cb_eval               = cb;
    cp.cb_eval_user_data     = nullptr;
    llama_context *ctx       = llama_init_from_model(model, cp);
    if (ctx == nullptr)
        return 1;
    // Logits for every prompt token, so the trace's last layer covers them all.
    llama_batch b = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = n_prompt;
    if (llama_decode(ctx, b) != 0)
        return 1;
    FILE *out = fopen(argv[4], "wb");
    for (size_t i = (size_t) n_prompt; i < ids.size(); i++) {
        const float *lg = llama_get_logits_ith(ctx, -1);
        fwrite(lg, sizeof(float), (size_t) nv, out);
        llama_token t = ids[i];
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0)
            return 1;
    }
    fclose(out);
    if (g_trace != nullptr)
        fclose(g_trace);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
