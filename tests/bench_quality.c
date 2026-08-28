/*
 * bench_quality — quick model-quality eyeball via geist_session_* API.
 *
 * Runs a handful of standard prompts through the model and prints the
 * greedy completions plus a couple of stochastic-temperature samples
 * with a fixed seed. The prompts are chosen to give a coarse signal on
 * fact recall, simple reasoning, format-following, and multilinguality
 * — enough to spot gross quality regressions between quantizations
 * without needing a full MMLU/HellaSwag harness.
 *
 *   ./bench_quality                    — uses default GGUF search path
 *   ./bench_quality <model.gguf>       — explicit path
 *   GEIST_GGUF_PATH=... ./bench_quality
 *
 * Pure pass/print — no automated quality threshold, the user inspects
 * the output. Exits 77 (SKIP) if no model is reachable.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MAX_DECODE 48

/* Gemma 4 is an instruction-tuned chat model; raw-text completion
 * produces gibberish or short loops. Wrap each prompt in the local
 * Gemma chat template before sending. The tokenizer treats <|turn>
 * and <turn|> as structural tokens; <start_of_turn>/<end_of_turn>
 * are plain text for this model family. */
static const char *const PROMPTS[] = {
        "What is the capital of France?",
        "What is 1 + 1?",
        "Complete this poem: 'Roses are red, violets are'",
        "Write a haiku about the ocean.",
        "What does 'Bonjour' mean in English?",
        "State the Pythagorean theorem in one sentence.",
        "Return only code for a recursive Python function fibonacci(n).",
        "Translate 'Good morning.' to German.",
};
#define N_PROMPTS (sizeof PROMPTS / sizeof PROMPTS[0])

#define MAX_PROMPT_BYTES 1024
static int env_int_or_default(const char *name, int fallback, int min, int max) {
    const char *raw = getenv(name);
    if (raw == nullptr || raw[0] == '\0')
        return fallback;
    int value = atoi(raw);
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

/* Per-architecture chat template. Gemma 4 uses its turn markers; Llama-family
 * (BitNet 2B-4T, …) uses the <|user|> … <|end|> <|assistant|> format (the GGUF
 * ships no chat_template, so we hardcode per general.architecture). */
static void format_chat(char *buf, size_t buf_size, const char *user_prompt, const char *arch) {
    if (arch != nullptr && strncmp(arch, "qwen", 4) == 0) {
        /* Qwen3/3.5 use ChatML; the Llama-style markers below make the
         * 0.8B+ instruct models loop on template tokens. */
        snprintf(buf,
                 buf_size,
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                 user_prompt);
    } else if (arch != nullptr && strcmp(arch, "gemma4") != 0) {
        snprintf(buf, buf_size, "<|user|>\n%s<|end|>\n<|assistant|>\n", user_prompt);
    } else {
        snprintf(buf, buf_size, "<bos><|turn>user\n%s<turn|>\n<|turn>model\n", user_prompt);
    }
}

static int run_one(struct geist_model              *model,
                   struct geist_backend            *be,
                   const struct geist_session_opts *opts,
                   const char                      *user_prompt,
                   const char                      *label,
                   int                              max_decode) {
    char chat_buf[MAX_PROMPT_BYTES];
    format_chat(chat_buf, sizeof chat_buf, user_prompt, geist_model_arch(model));

    struct geist_session *sess = nullptr;
    enum geist_status     s    = geist_session_create(model, be, opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "  [%s] session_create: %s\n", label, geist_status_to_string(s));
        return 1;
    }
    s = geist_session_set_prompt(sess, chat_buf);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "  [%s] set_prompt: %s — %s\n",
                label,
                geist_status_to_string(s),
                geist_session_errmsg(sess));
        geist_session_destroy(sess);
        return 1;
    }
    printf("  [%s] Q: %s\n         A: ", label, user_prompt);
    fflush(stdout);
    for (int i = 0; i < max_decode; i++) {
        geist_token_t tok;
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK) {
            printf(" [error %s]", geist_status_to_string(s));
            break;
        }
        const char *txt = geist_session_token_to_str(sess, tok);
        if (txt == nullptr) {
            printf("<%d>", (int) tok);
            continue;
        }
        /* Skip start / BOS markers without printing or stopping. Use strstr
         * since GPT-2 byte-level BPE pieces often carry a leading space. */
        if (strstr(txt, "<bos>") != nullptr || strstr(txt, "<|begin_of_text|>") != nullptr) {
            continue;
        }
        /* Stop on end-of-turn / end-of-stream markers (Gemma + Llama/BitNet). */
        if (tok == 1 || tok == 106 || strstr(txt, "<eos>") != nullptr ||
            strstr(txt, "<end_of_turn>") != nullptr || strstr(txt, "<turn") != nullptr ||
            strstr(txt, "<|end|>") != nullptr || strstr(txt, "<|eot_id|>") != nullptr) {
            break;
        }
        /* The SentencePiece marker '▁' (U+2581) represents a space.
         * Convert for readability in the bench output. */
        for (const char *p = txt; *p;) {
            if ((unsigned char) p[0] == 0xE2 && (unsigned char) p[1] == 0x96 &&
                (unsigned char) p[2] == 0x81) {
                putchar(' ');
                p += 3;
            } else {
                putchar(*p);
                p++;
            }
        }
    }
    printf("\n");
    geist_session_destroy(sess);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : geist_test_find_gguf();
    GEIST_SKIP_IF(model_path == nullptr, "no GGUF model found — pass path or set GEIST_GGUF_PATH");

    const char *backend_name = getenv("GEIST_BENCH_BACKEND");
    if (backend_name == nullptr || backend_name[0] == '\0') {
        backend_name = "cpu_neon";
    }
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create(backend_name, nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "model_load(%s): %s — %s\n",
                model_path,
                geist_status_to_string(s),
                geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("model: %s\n", model_path);
    printf("backend: %s\n", geist_backend_name(be));
    const int max_decode =
            env_int_or_default("GEIST_BENCH_QUALITY_MAX_DECODE", DEFAULT_MAX_DECODE, 1, 256);
    const int max_prompts = env_int_or_default(
            "GEIST_BENCH_QUALITY_MAX_PROMPTS", (int) N_PROMPTS, 1, (int) N_PROMPTS);
    const bool run_sampled = getenv("GEIST_BENCH_QUALITY_SAMPLED") == nullptr ||
                             getenv("GEIST_BENCH_QUALITY_SAMPLED")[0] != '0';
    printf("max_decode: %d tokens per completion\n", max_decode);
    printf("max_prompts: %d / %zu\n\n", max_prompts, N_PROMPTS);

    /* ---- Greedy completions ---- */
    printf("=== Greedy ===\n");
    struct geist_session_opts greedy = {.temperature = 0.0f};
    for (int i = 0; i < max_prompts; i++) {
        run_one(model, be, &greedy, PROMPTS[i], "greedy", max_decode);
    }

    /* ---- One temperature-sampled completion for variety ---- */
    if (run_sampled) {
        printf("\n=== Sampled (temp=0.7, top_k=40, seed=0xC0FFEE) ===\n");
        struct geist_session_opts sampled = {
                .temperature = 0.7f,
                .top_k       = 40,
                .top_p       = 0.95f,
                .random_seed = 0xC0FFEEULL,
        };
        for (int i = 0; i < max_prompts; i++) {
            run_one(model, be, &sampled, PROMPTS[i], "sample", max_decode);
        }
    }

    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;
}
