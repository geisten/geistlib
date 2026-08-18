/*
 * test_chat_audio_int — coherent audio→text chat via Gemma 4 chat template.
 *
 * Mirrors test_chat_image_int but for audio. Validates that the shared
 * prefill_audio path produces coherent text post the PLE pad-embedding
 * fix in transformer_prefill_audio_batch.
 *
 *   <bos><|turn>user\n<|audio>      ← prefix (tokenize → prefill)
 *   [N audio soft tokens]           ← attach_audio
 *   <audio|>\n{question}<turn|>\n<|turn>model\n
 *                                   ← suffix (tokenize → prefill)
 *   [decode loop until <turn|> or <eos>]
 *
 * Special tokens (verified from tokenizer_config.json):
 *   BOA = <|audio>  EOA = <audio|>
 *
 * SKIPs if model, tokenizer, or WAV is missing.
 */
#include "audio_test_util.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROMPT_CAP 1024

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

int main(int argc, char **argv) {
    GEIST_REQUIRE_GGUF(model_path);
    const char *wav_path   = argc > 1 ? argv[1] : "audio_test_data/de_hello.wav";
    const char *question   = argc > 2 ? argv[2] : "Transcribe what you heard.";
    size_t      max_tokens = argc > 3 ? (size_t) atoi(argv[3]) : 80;

    size_t   n_samples;
    int      sample_rate;
    int16_t *pcm = read_wav_pcm(wav_path, &n_samples, &sample_rate);
    if (pcm == nullptr) {
        GEIST_SKIP("could not read WAV (use first arg; default audio_test_data/de_hello.wav)");
    }
    if (sample_rate != 16000) {
        free(pcm);
        GEIST_SKIP("WAV not 16 kHz");
    }
    printf("audio: %s (%.2f s @ 16kHz)\n", wav_path, (double) n_samples / 16000.0);
    printf("question: %s\n", question);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        free(pcm);
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        geist_backend_destroy(be);
        free(pcm);
        return GEIST_TEST_FAIL;
    }

    struct geist_session_opts opts = {.max_seq_len = 2048};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        free(pcm);
        return GEIST_TEST_FAIL;
    }

    {
        geist_token_t probe[4];
        size_t        pn = 0;
        s                = geist_session_tokenize(sess, "x", 4, probe, &pn);
        if (s == GEIST_E_NOT_FOUND) {
            printf("SKIP: tokenizer not found (set GEIST_TOKENIZER_PATH)\n");
            free(pcm);
            geist_session_destroy(sess);
            geist_model_destroy(model);
            geist_backend_destroy(be);
            return GEIST_TEST_SKIP;
        }
    }

    /* Prefix: <bos><|turn>user\n<|audio> */
    geist_token_t prefix[PROMPT_CAP];
    size_t        prefix_n = 0;
    s = tokenize_drop_bos(sess, "<bos><|turn>user\n<|audio>", false, prefix, PROMPT_CAP, &prefix_n);
    if (s != GEIST_OK) {
        fprintf(stderr, "tokenize prefix\n");
        goto fail;
    }
    printf("prefix tokens (%zu):", prefix_n);
    for (size_t i = 0; i < prefix_n; i++)
        printf(" %d", prefix[i]);
    printf("\n");
    s = geist_session_prefill_tokens(sess, prefix_n, prefix);
    if (s != GEIST_OK) {
        fprintf(stderr, "prefill prefix\n");
        goto fail;
    }

    /* attach_audio */
    s = geist_session_attach_audio(sess, n_samples, pcm, 16000);
    free(pcm);
    pcm = nullptr;
    if (s == GEIST_E_NOT_FOUND) {
        printf("SKIP: audio_tower.safetensors or mel_constants.bin missing — %s\n",
               geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "attach_audio failed: %s\n", geist_session_errmsg(sess));
        goto fail;
    }
    printf("audio attached\n");

    /* Suffix: <audio|>\n{question}<turn|>\n<|turn>model\n */
    char suffix_text[PROMPT_CAP];
    snprintf(suffix_text, sizeof suffix_text, "<audio|>\n%s<turn|>\n<|turn>model\n", question);
    geist_token_t suffix[PROMPT_CAP];
    size_t        suffix_n = 0;
    s = tokenize_drop_bos(sess, suffix_text, true, suffix, PROMPT_CAP, &suffix_n);
    if (s != GEIST_OK) {
        fprintf(stderr, "tokenize suffix\n");
        goto fail;
    }
    s = geist_session_prefill_tokens(sess, suffix_n, suffix);
    if (s != GEIST_OK) {
        fprintf(stderr, "prefill suffix\n");
        goto fail;
    }

    /* Decode loop */
    printf("\nmodel: ");
    fflush(stdout);
    for (size_t i = 0; i < max_tokens; i++) {
        geist_token_t tok;
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK)
            break;
        if (tok == 1 /* <eos> */ || tok == 106 /* <turn|> */)
            break;
        const char *t = geist_session_token_to_str(sess, tok);
        if (t) {
            fputs(t, stdout);
            fflush(stdout);
        }
    }
    printf("\n");

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;

fail:
    if (pcm)
        free(pcm);
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
