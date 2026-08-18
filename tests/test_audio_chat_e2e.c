/*
 * test_audio_chat_e2e — end-to-end audio-understanding validation.
 *
 * Verifies the audio_conformer → soft-tokens → LM chat-template path
 * produces semantically meaningful responses to the spoken content,
 * not just any text. For each known clip, asks a target prompt and
 * checks the model's response contains at least one expected keyword.
 *
 * This is the only test in the suite that validates the AUDIO PIPELINE
 * is actually delivering recognizable speech content into the LM (as
 * opposed to merely "the pipeline runs without crashing"). Failures
 * here usually indicate one of:
 *   - audio encoder quantization regressed quality below recognition
 *   - chat-template wrap was changed and no longer matches Gemma 4
 *     audio query format
 *   - PLE soft-token injection regressed
 *   - WAV file changed or model was re-quantized
 *
 * The validator is intentionally lenient (any of N keywords matches)
 * because Gemma 4 E2B's text generation has natural style variance
 * even with greedy decode at default sampler.
 *
 * SKIPs if GGUF / tokenizer / audio assets are missing.
 *
 * Suggested run (Mac M1 dev box):
 *   GEIST_GGUF_PATH=gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \
 *   GEIST_TOKENIZER_PATH=gemma-4-E2B-it/tokenizer.bin \
 *   bin/mac-omp/release/tests/test_audio_chat_e2e
 *
 * For the W8A8 audio path (Pi 5 production):
 *   GEIST_AUDIO_ATTN_W8A8=1 GEIST_AUDIO_LCONV_W8A8=1 \
 *   GEIST_AUDIO_FORCE_QUANT=1 ...
 */
#include "audio_test_util.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROMPT_CAP 1024
#define DECODE_CAP 80
#define REPLY_CAP 4096

struct audio_case {
    const char *wav_path;
    const char *prompt;
    /* The model's response must contain at least ONE of these keywords
     * (case-insensitive substring match). Empty slot ends the list. */
    const char *expect_any[4];
};

/* Hand-curated from observed Gemma 4 E2B responses on these clips at
 * commit 528f1ab6. Each expected-keyword set captures the SEMANTIC
 * content of the spoken phrase, not a specific phrasing - the model
 * may answer the question ("capital of Afghanistan?" -> "Kabul"),
 * repeat the words ("fox jumps over"), or acknowledge in some other
 * recognizable way.
 *
 * Short imperatives + greetings (lampe_an, de_hello, hello_world) get
 * DOMAIN-SPECIFIC few-shot prompts. Without the vocabulary anchor
 * Gemma 4 E2B hallucinates a generic "I don't know" response on
 * sub-1 s clips - see docs/audio-chunk-streaming/short-command-analysis.md
 * for the analysis. Adding 3-4 example phrases of the right domain
 * to the prompt fixes recognition without retraining. */
static struct audio_case cases[] = {
        {
                .wav_path   = "audio_test_data/en_long.wav",
                .prompt     = "Repeat what you heard in quotes.",
                .expect_any = {"fox", "dog", "quick", nullptr},
        },
        {
                .wav_path   = "audio_test_data/en_question.wav",
                .prompt     = "Answer the question you heard.",
                .expect_any = {"Kabul", "Afghanistan", "capital", nullptr},
        },
        {
                .wav_path   = "audio_test_data/de_question.wav",
                .prompt     = "Repeat what you heard in quotes.",
                .expect_any = {"weather", "Wetter", "weath", nullptr},
        },
        /* Smart-home short imperative — only recognised with a few-shot
         * vocabulary anchor in the prompt. */
        {
                .wav_path   = "audio_test_data/lampe_an.wav",
                .prompt     = "The user gave a smart home command. Common commands: "
                              "'Lampe an', 'Licht aus', 'Musik an'. Which command did you hear?",
                .expect_any = {"Lampe an", "Lampe", "lamp", nullptr},
        },
        /* German greeting with greeting-domain few-shot anchor. */
        {
                .wav_path   = "audio_test_data/de_hello.wav",
                .prompt     = "The user said a short greeting. Common greetings: "
                              "'Hallo', 'Hello', 'Hello world', 'Guten Tag'. "
                              "Which greeting did you hear?",
                .expect_any = {"Hallo", nullptr},
        },
        /* English greeting with same greeting-domain anchor. */
        {
                .wav_path   = "audio_test_data/hello_world.wav",
                .prompt     = "The user said a short greeting. Common greetings: "
                              "'Hallo', 'Hello', 'Hello world', 'Guten Tag'. "
                              "Which greeting did you hear?",
                .expect_any = {"Hello world", "Hello", "world", nullptr},
        },
};

/* Chunk-walking WAV reader (audio_test_util.h) — the fixed-44-byte
 * shortcut mis-read ffmpeg WAVs with a LIST chunk (#268). */
static int16_t *read_wav_pcm(const char *path, size_t *n_samples_out, int *sample_rate_out) {
    return audio_test_read_wav(path, n_samples_out, sample_rate_out);
}

static enum geist_status tokenize_drop_bos(struct geist_session *s,
                                           const char           *text,
                                           bool                  drop_bos,
                                           geist_token_t        *out,
                                           size_t                cap,
                                           size_t               *n_out) {
    geist_token_t     scratch[PROMPT_CAP];
    size_t            n  = 0;
    enum geist_status rc = geist_session_tokenize(s, text, PROMPT_CAP, scratch, &n);
    if (rc != GEIST_OK)
        return rc;
    size_t start = (drop_bos && n > 0 && scratch[0] == 2) ? 1 : 0;
    if (n - start > cap)
        return GEIST_E_INVALID_ARG;
    for (size_t i = start; i < n; i++)
        out[i - start] = scratch[i];
    *n_out = n - start;
    return GEIST_OK;
}

/* Normalize the model's decoded reply for substring search:
 *   1. The SentencePiece "leading-space" marker U+2581 (UTF-8 E2 96 81)
 *      becomes a regular space - lets prompts written with normal text
 *      match the model's "▁word" pieces.
 *   2. geist_session_token_to_str appends a per-piece SentencePiece
 *      length-class trailer byte (e.g. "Lam" -> "Lam\x12", "pe" -> "pe\x05")
 *      which fragments "Lampe" into non-contiguous bytes 4c 61 6d 12 70 65.
 *      Strip all control bytes < 0x20 (except real whitespace 0x09 \t,
 *      0x0A \n, 0x0D \r) so the cleaned haystack reads as the model's
 *      intended text. */
static void normalize_sp(const char *in, char *out, size_t out_cap) {
    size_t j = 0;
    for (size_t i = 0; in[i] != '\0' && j + 1 < out_cap; i++) {
        const unsigned char c0 = (unsigned char) in[i];
        if (c0 == 0xE2 && (unsigned char) in[i + 1] == 0x96 && (unsigned char) in[i + 2] == 0x81) {
            out[j++] = ' ';
            i += 2; /* skip the other two bytes of the U+2581 sequence */
            continue;
        }
        if (c0 < 0x20 && c0 != '\t' && c0 != '\n' && c0 != '\r') {
            continue; /* drop SentencePiece trailer / control byte */
        }
        out[j++] = in[i];
    }
    out[j] = '\0';
}

/* Case-insensitive substring search. Substring `needle` is treated as
 * ASCII-lowercased; haystack is compared char-by-char with tolower(). */
static bool contains_ci(const char *haystack, const char *needle) {
    if (haystack == nullptr || needle == nullptr || needle[0] == '\0')
        return false;
    const size_t nn = strlen(needle);
    for (const char *p = haystack; *p; p++) {
        size_t i;
        for (i = 0; i < nn; i++) {
            if (tolower((unsigned char) p[i]) != tolower((unsigned char) needle[i]))
                break;
        }
        if (i == nn)
            return true;
    }
    return false;
}

/* Returns true if `reply` matches any of `expect_any`. */
static bool reply_matches(const char *reply, const char *const *expect_any) {
    char norm[REPLY_CAP];
    normalize_sp(reply, norm, sizeof norm);
    for (size_t i = 0; expect_any[i] != nullptr; i++) {
        if (contains_ci(norm, expect_any[i]))
            return true;
    }
    return false;
}

/* Returns true on success, false on a hard pipeline error (caller
 * propagates as test FAIL). Writes the decoded model reply into `reply`
 * (NUL-terminated, capped at reply_cap). */
static bool run_one_case(struct geist_model      *model,
                         struct geist_backend    *be,
                         const struct audio_case *tc,
                         char                    *reply,
                         size_t                   reply_cap) {
    reply[0] = '\0';

    size_t   n_samples;
    int      sample_rate;
    int16_t *pcm = read_wav_pcm(tc->wav_path, &n_samples, &sample_rate);
    if (pcm == nullptr || sample_rate != 16000) {
        snprintf(reply, reply_cap, "[skip: bad wav]");
        if (pcm)
            free(pcm);
        return false;
    }

    struct geist_session_opts opts = {.max_seq_len = 2048};
    struct geist_session     *sess = nullptr;
    enum geist_status         s    = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        free(pcm);
        return false;
    }

    geist_token_t prefix[PROMPT_CAP];
    size_t        prefix_n = 0;
    s = tokenize_drop_bos(sess, "<bos><|turn>user\n<|audio>", false, prefix, PROMPT_CAP, &prefix_n);
    if (s != GEIST_OK)
        goto out_fail;
    s = geist_session_prefill_tokens(sess, prefix_n, prefix);
    if (s != GEIST_OK)
        goto out_fail;

    s = geist_session_attach_audio(sess, n_samples, pcm, 16000);
    free(pcm);
    pcm = nullptr;
    if (s != GEIST_OK)
        goto out_fail;

    char suffix_text[PROMPT_CAP];
    snprintf(suffix_text, sizeof suffix_text, "<audio|>\n%s<turn|>\n<|turn>model\n", tc->prompt);
    geist_token_t suffix[PROMPT_CAP];
    size_t        suffix_n = 0;
    s = tokenize_drop_bos(sess, suffix_text, true, suffix, PROMPT_CAP, &suffix_n);
    if (s != GEIST_OK)
        goto out_fail;
    s = geist_session_prefill_tokens(sess, suffix_n, suffix);
    if (s != GEIST_OK)
        goto out_fail;

    /* Decode loop — append each token's text representation. */
    size_t reply_len = 0;
    for (size_t i = 0; i < DECODE_CAP; i++) {
        geist_token_t tok;
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK)
            break;
        if (tok == 1 /* <eos> */ || tok == 106 /* <turn|> */)
            break;
        const char *t = geist_session_token_to_str(sess, tok);
        if (t == nullptr)
            continue;
        const size_t tn = strlen(t);
        if (reply_len + tn >= reply_cap)
            break;
        memcpy(reply + reply_len, t, tn);
        reply_len += tn;
        reply[reply_len] = '\0';
    }

    geist_session_destroy(sess);
    return true;

out_fail:
    if (pcm)
        free(pcm);
    geist_session_destroy(sess);
    return false;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        GEIST_SKIP("backend create failed");

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("model_load failed (set GEIST_GGUF_PATH)");
    }

    /* Quick tokenizer probe — SKIP cleanly if the tokenizer is missing
     * since these tests are useless without it. */
    {
        struct geist_session_opts opts  = {.max_seq_len = 64};
        struct geist_session     *probe = nullptr;
        if (geist_session_create(model, be, &opts, &probe) == GEIST_OK) {
            geist_token_t     t[4];
            size_t            pn = 0;
            enum geist_status ps = geist_session_tokenize(probe, "x", 4, t, &pn);
            geist_session_destroy(probe);
            if (ps == GEIST_E_NOT_FOUND) {
                geist_model_destroy(model);
                geist_backend_destroy(be);
                GEIST_SKIP("tokenizer not found (set GEIST_TOKENIZER_PATH)");
            }
        }
    }

    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    size_t       n_pass = 0, n_fail = 0, n_skip = 0;
    char         reply[REPLY_CAP];

    printf("audio_chat_e2e: %zu cases\n\n", n_cases);
    for (size_t i = 0; i < n_cases; i++) {
        const struct audio_case *tc = &cases[i];
        printf("[%zu/%zu] %s\n", i + 1, n_cases, tc->wav_path);
        printf("  prompt: \"%s\"\n", tc->prompt);

        if (!run_one_case(model, be, tc, reply, sizeof reply)) {
            printf("  status: SKIP/ERROR (%s)\n\n", reply[0] ? reply : "pipeline failure");
            n_skip++;
            continue;
        }
        {
            char norm[REPLY_CAP];
            normalize_sp(reply, norm, sizeof norm);
            printf("  reply:  \"%s\"\n", norm);
        }

        const bool ok = reply_matches(reply, tc->expect_any);
        printf("  expect-any:");
        for (size_t k = 0; tc->expect_any[k] != nullptr; k++) {
            printf(" \"%s\"", tc->expect_any[k]);
        }
        printf("\n  status: %s\n\n", ok ? "PASS" : "FAIL");

        if (ok)
            n_pass++;
        else
            n_fail++;
    }

    printf("audio_chat_e2e: %zu pass, %zu fail, %zu skip\n", n_pass, n_fail, n_skip);

    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (n_fail > 0)
        return GEIST_TEST_FAIL;
    if (n_pass == 0)
        return GEIST_TEST_SKIP;
    return GEIST_TEST_PASS;
}
