// moe_usage — per-layer expert selection counts of a llama.cpp MoE model (#48 phase 0).
// Usage: moe_usage <model.gguf> <out.json> <text files...>
// Build against a llama.cpp checkout L (benchmark/results/MOE-EXPERT-USAGE.md):
//   clang++ -std=c++17 -O2 -I$L/include -I$L/ggml/include moe_usage.cpp \
//       -L$L/build-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build-cpu/bin -o moe_usage
// Each file is tokenized and evaluated in chunks of CHUNK tokens with a cleared KV cache per
// file; the router's chosen experts (tensor "ffn_moe_topk-<layer>", int32 [n_used, n_tokens])
// are counted per layer and per file.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::map<int, std::vector<long>> g_counts; // layer -> expert -> count (current file)

static bool cb(struct ggml_tensor *t, bool ask, void *) {
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) return false;
    if (ask) return true;
    const int layer = atoi(t->name + 13);
    // ffn_moe_topk is a view of the argsort over all experts: ne0 = n_used ids per token,
    // rows nb[1] apart. Reading it as contiguous memory counts whole permutations.
    const int64_t n_used = t->ne[0], n_tok = t->ne[1];
    std::vector<int32_t> ids(n_used * n_tok);
    for (int64_t i = 0; i < n_tok; i++)
        ggml_backend_tensor_get(t, ids.data() + i * n_used, i * t->nb[1], n_used * sizeof(int32_t));
    auto &c = g_counts[layer];
    for (int32_t e : ids) {
        if (e < 0) continue;
        if ((size_t) e >= c.size()) c.resize(e + 1, 0);
        c[e]++;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf out.json files...\n", argv[0]); return 2; }
    const int CHUNK = 512;
    llama_backend_init();
    auto mp = llama_model_default_params();
    // MOE_NGL layers on the GPU backend (0 = CPU only); the routing is the same arithmetic.
    mp.n_gpu_layers = getenv("MOE_NGL") ? atoi(getenv("MOE_NGL")) : 0;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = CHUNK; cp.n_ubatch = CHUNK;
    cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::ofstream out(argv[2]);
    out << "{\"model\": \"" << argv[1] << "\", \"files\": [\n";
    for (int f = 3; f < argc; f++) {
        std::ifstream in(argv[f]); std::stringstream ss; ss << in.rdbuf(); std::string text = ss.str();
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
        if (n < 0) { fprintf(stderr, "tokenize failed: %s\n", argv[f]); return 1; }
        toks.resize(n);
        if (toks.size() > 4000) toks.resize(4000); // one context window per file
        g_counts.clear();
        llama_memory_clear(llama_get_memory(ctx), true);
        // Logits for every token: llama.cpp computes the last layer only for the tokens whose
        // logits are requested, so llama_batch_get_one (last token only) undercounts it.
        llama_batch batch = llama_batch_init(CHUNK, 0, 1);
        for (size_t i = 0; i < toks.size(); i += CHUNK) {
            int m = (int) std::min<size_t>(CHUNK, toks.size() - i);
            batch.n_tokens = m;
            for (int j = 0; j < m; j++) {
                batch.token[j] = toks[i + j]; batch.pos[j] = (llama_pos) (i + j);
                batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = 1;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        }
        llama_batch_free(batch);
        out << (f > 3 ? ",\n" : "") << " {\"file\": \"" << argv[f] << "\", \"tokens\": " << toks.size() << ", \"counts\": {";
        bool first = true;
        for (auto &[l, c] : g_counts) {
            out << (first ? "" : ", ") << "\"" << l << "\": [";
            for (size_t e = 0; e < c.size(); e++) out << (e ? "," : "") << c[e];
            out << "]"; first = false;
        }
        out << "}}";
        fprintf(stderr, "%s: %zu tokens, %zu layers\n", argv[f], toks.size(), g_counts.size());
    }
    out << "\n]}\n";
    llama_free(ctx); llama_model_free(model);
    return 0;
}
