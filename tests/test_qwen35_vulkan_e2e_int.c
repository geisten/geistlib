/*
 * test_qwen35_vulkan_e2e_int — qwen35 (hybrid Gated-DeltaNet / attention) end
 * to end on the Vulkan backend, checked against a CPU backend on the same
 * GGUF.
 *
 *   1. Known answer: greedy continuation of "The capital of France is"
 *      contains "Paris" on Vulkan.
 *   2. CPU equivalence: for a short prompt (prefill m = 5) and a long one
 *      (prefill m ~ 60, exercising the batched GEMMs, the token-sequential
 *      DeltaNet kernel and the causal conv over a multi-row window) the whole
 *      greedy continuation equals the CPU backend's. The KV cache is pinned to
 *      FP32 on both sides: the CPU backends default to an INT8-quantized cache
 *      (~1 % relative error in the attention output, which flips near-tied
 *      greedy choices) and Vulkan to F16, so the default configurations are
 *      not comparable token for token — the FP32 configuration is, and there
 *      the logits agree to ~1e-7.
 *   3. Reset equivalence: decode, session_reset, decode the same prompt again
 *      — identical tokens. The DeltaNet conv/delta state lives in VRAM here
 *      (not host-mappable), so this pins the upload-based zeroing path.
 *
 * SKIPs when there is no Vulkan device or the fixture is absent
 * (make fetch-qwen35-model, or set GEIST_QWEN35_GGUF_PATH).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* With an FP32 KV cache on both sides the backends agree to the last bit of
 * the logits, so the whole greedy continuation must match. */
enum { N_GEN = 12, KMIN = N_GEN };

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_QWEN35_GGUF_PATH");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    static const char *candidates[] = {
            "gguf_artifacts/qwen3.5-0.8b-q8_0.gguf",
            "./qwen3.5-0.8b-q8_0.gguf",
            nullptr,
    };
    for (size_t i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return candidates[i];
        }
    }
    return nullptr;
}

struct run {
    geist_token_t tok[N_GEN];
    int           n;
    char          text[512];
};

/* Greedy continuation of `prompt` on `backend`; twice with a reset in
 * between when `again` is non-null (reset equivalence). */
static bool generate(const char *backend,
                     const char *path,
                     const char *prompt,
                     struct run *out,
                     struct run *again) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create(backend, nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend %s: %s\n", backend, geist_last_create_error());
        return false;
    }
    struct geist_model *model = nullptr;
    s                         = geist_model_load(path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "%s: model_load: %s — %s\n",
                backend,
                geist_status_to_string(s),
                geist_last_create_error());
        geist_backend_destroy(be);
        return false;
    }
    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    bool ok                        = s == GEIST_OK;
    for (int pass = 0; ok && pass < (again != nullptr ? 2 : 1); pass++) {
        struct run *r = pass == 0 ? out : again;
        if (pass == 1) {
            geist_session_reset(sess);
        }
        memset(r, 0, sizeof *r);
        s  = geist_session_set_prompt(sess, prompt);
        ok = s == GEIST_OK;
        if (!ok) {
            fprintf(stderr, "%s: set_prompt: %s\n", backend, geist_session_errmsg(sess));
        }
        size_t used = 0;
        for (int i = 0; ok && i < N_GEN; i++) {
            if (geist_session_decode_step(sess, &r->tok[i]) != GEIST_OK) {
                fprintf(stderr,
                        "%s: decode_step %d failed: %s\n",
                        backend,
                        i,
                        geist_session_errmsg(sess));
                ok = false;
                break;
            }
            r->n++;
            const char  *piece = geist_session_token_to_str(sess, r->tok[i]);
            const size_t plen  = piece != nullptr ? strlen(piece) : 0;
            if (used + plen < sizeof r->text - 1) {
                memcpy(r->text + used, piece, plen);
                used += plen;
            }
        }
    }
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return ok;
}

static int first_diff(const struct run *a, const struct run *b) {
    for (int i = 0; i < N_GEN; i++) {
        if (a->tok[i] != b->tok[i]) {
            return i;
        }
    }
    return N_GEN;
}

static const char *cpu_backend(void) {
    /* GEIST_E2E_REF picks the reference backend (cpu_scalar is the f32
     * numerical oracle; the SIMD backends quantize activations to int8). */
    const char *forced = getenv("GEIST_E2E_REF");
    if (forced != nullptr && forced[0] != '\0') {
        return forced;
    }
    static const char *names[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(names[i], nullptr, nullptr, &be) == GEIST_OK) {
            geist_backend_destroy(be);
            return names[i];
        }
    }
    return nullptr;
}

int main(void) {
    /* Comparable numerics: FP32 KV cache on every backend (see above). */
    setenv("GEIST_KV_INT8", "0", 1);
    setenv("GEIST_KV_F16", "0", 1);
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no qwen35 GGUF. Run `make fetch-qwen35-model`, or set "
                           "GEIST_QWEN35_GGUF_PATH");
    }
    struct geist_backend *probe = nullptr;
    enum geist_status     ps    = geist_backend_create("vulkan", nullptr, nullptr, &probe);
    if (ps == GEIST_E_NOT_FOUND || ps == GEIST_E_UNSUPPORTED) {
        GEIST_SKIP("Vulkan backend unavailable");
    }
    if (ps != GEIST_OK) {
        fprintf(stderr, "vulkan create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    geist_backend_destroy(probe);
    const char *cpu = cpu_backend();
    if (cpu == nullptr) {
        GEIST_SKIP("no CPU backend to compare against");
    }

    static const char *prompts[] = {
            "The capital of France is",
            "In the early days of computing, programmers wrote instructions directly in machine "
            "code, one numeric opcode at a time. Later, assemblers introduced mnemonics, and "
            "then compilers let people describe what they wanted in a language closer to "
            "mathematics. The most important consequence of this history is that",
    };
    int fails = 0;
    for (size_t p = 0; p < 2; p++) {
        struct run vk = {0}, vk2 = {0}, ref = {0};
        if (!generate("vulkan", path, prompts[p], &vk, p == 0 ? &vk2 : nullptr) ||
            !generate(cpu, path, prompts[p], &ref, nullptr)) {
            fprintf(stderr, "FAIL: generation failed (prompt %zu)\n", p);
            fails++;
            continue;
        }
        printf("prompt %zu\n  vulkan: \"%s\"\n  %-6s: \"%s\"\n", p, vk.text, cpu, ref.text);
        const int d = first_diff(&vk, &ref);
        if (d < KMIN) {
            fprintf(stderr,
                    "FAIL: vulkan and %s diverge at token %d (need %d equal)\n",
                    cpu,
                    d,
                    KMIN);
            fails++;
        } else {
            printf("  CPU equivalence: all %d tokens equal\n", N_GEN);
        }
        if (p == 0) {
            if (strstr(vk.text, "Paris") == nullptr) {
                fprintf(stderr, "FAIL: vulkan continuation lacks \"Paris\"\n");
                fails++;
            }
            if (memcmp(vk.tok, vk2.tok, sizeof vk.tok) != 0) {
                fprintf(stderr, "FAIL: reset not equivalent (stale DeltaNet state in VRAM?)\n");
                fails++;
            } else {
                printf("  reset equivalence: PASS\n");
            }
        }
    }
    if (fails > 0) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: qwen35 on Vulkan matches %s\n", cpu);
    return GEIST_TEST_PASS;
}
