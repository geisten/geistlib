/*
 * test_audio_latency_e2e — measures push-to-talk pipeline timing per stage.
 *
 * Goal: quantify the user-perceived latency of "say something, get a
 * response" with Gemma 4 E2B. Per-stage breakdown so future optimization
 * targets the right bottleneck.
 *
 * Stages timed:
 *   t_wav   : WAV read off disk
 *   t_attach: audio_encoder forward + soft-token KV injection
 *             (this is the 280-405 ms residual measured by bench_audio_*)
 *   t_pre   : chat-template suffix prefill (LM processes 8-10 tokens)
 *   t_first : first decoded token wall time
 *   t_decode: remaining tokens until <eos>/<turn|> or DECODE_CAP
 *
 *   t_total = t_wav + t_attach + t_pre + t_first + t_decode
 *
 * Reports against several short clips so the absolute numbers make
 * sense vs the spoken duration. Tests does NOT enforce a latency
 * budget (pass condition is "pipeline runs and produces tokens") so it
 * is robust across dev (M1) and target (Pi 5) hardware, but the printed
 * numbers feed benchmark/BENCHMARK_PI5.md updates.
 *
 * SKIPs if GGUF / tokenizer / audio assets are missing.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PROMPT_CAP 1024
#define DECODE_CAP 50
#define REPLY_CAP 2048

struct latency_case {
    const char *wav_path;
    const char *prompt;
    const char *label;
};

static struct latency_case cases[] = {
        {"audio_test_data/lampe_an.wav", "Was hat der Sprecher gesagt?", "lampe_an (DE 0.67s)"},
        {"audio_test_data/de_hello.wav", "Was hat der Sprecher gesagt?", "de_hello (DE 0.87s)"},
        {"audio_test_data/hello_world.wav",
         "Repeat what you heard in quotes.",
         "hello_world (EN 0.86s)"},
        {"audio_test_data/de_question.wav",
         "Repeat what you heard in quotes.",
         "de_question (DE 1.69s)"},
        {"audio_test_data/en_question.wav",
         "Answer the question you heard.",
         "en_question (EN 1.95s)"},
        {"audio_test_data/en_long.wav", "Repeat what you heard in quotes.", "en_long (EN 5.37s)"},
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

static int16_t *read_wav_pcm(const char *path, size_t *n_samples_out, int *sample_rate_out) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr)
        return nullptr;
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, f) != 44 || memcmp(hdr, "RIFF", 4) != 0 ||
        memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return nullptr;
    }
    unsigned short ch = (unsigned short) (hdr[22] | (hdr[23] << 8));
    unsigned int sr = (unsigned int) (hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24));
    unsigned short bps = (unsigned short) (hdr[34] | (hdr[35] << 8));
    if (ch != 1 || bps != 16) {
        fclose(f);
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    long file_bytes = ftell(f);
    fseek(f, 44, SEEK_SET);
    size_t   n   = (size_t) (file_bytes - 44) / 2;
    int16_t *pcm = malloc(n * sizeof(int16_t));
    if (pcm == nullptr) {
        fclose(f);
        return nullptr;
    }
    size_t got = fread(pcm, sizeof(int16_t), n, f);
    fclose(f);
    if (got != n) {
        free(pcm);
        return nullptr;
    }
    *n_samples_out   = n;
    *sample_rate_out = (int) sr;
    return pcm;
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

struct stage_timings {
    double audio_s;  /* spoken length, for reference */
    double t_wav;    /* WAV read */
    double t_attach; /* audio encode + KV inject */
    double t_pre;    /* suffix prefill */
    double t_first;  /* first-token decode */
    double t_decode; /* remaining decode */
    size_t n_decoded;
    char   reply[REPLY_CAP];
};

static bool run_one_case(struct geist_model        *model,
                         struct geist_backend      *be,
                         const struct latency_case *tc,
                         struct stage_timings      *out) {
    memset(out, 0, sizeof(*out));

    const double t0 = now_ms();
    size_t       n_samples;
    int          sample_rate;
    int16_t     *pcm = read_wav_pcm(tc->wav_path, &n_samples, &sample_rate);
    if (pcm == nullptr || sample_rate != 16000) {
        if (pcm)
            free(pcm);
        return false;
    }
    out->audio_s = (double) n_samples / 16000.0;
    out->t_wav   = now_ms() - t0;

    struct geist_session_opts opts = {.max_seq_len = 2048};
    struct geist_session     *sess = nullptr;
    enum geist_status         s    = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        free(pcm);
        return false;
    }

    /* Prefix (small, lump into attach measurement). */
    geist_token_t prefix[PROMPT_CAP];
    size_t        prefix_n = 0;
    s = tokenize_drop_bos(sess, "<bos><|turn>user\n<|audio>", false, prefix, PROMPT_CAP, &prefix_n);
    if (s != GEIST_OK)
        goto out_fail;
    s = geist_session_prefill_tokens(sess, prefix_n, prefix);
    if (s != GEIST_OK)
        goto out_fail;

    /* attach_audio — this is where the audio encoder runs. */
    const double t1 = now_ms();
    s               = geist_session_attach_audio(sess, n_samples, pcm, 16000);
    out->t_attach   = now_ms() - t1;
    free(pcm);
    pcm = nullptr;
    if (s != GEIST_OK)
        goto out_fail;

    /* Suffix prefill — LM processes the chat template tail. */
    const double t2 = now_ms();
    char         suffix_text[PROMPT_CAP];
    snprintf(suffix_text, sizeof suffix_text, "<audio|>\n%s<turn|>\n<|turn>model\n", tc->prompt);
    geist_token_t suffix[PROMPT_CAP];
    size_t        suffix_n = 0;
    s = tokenize_drop_bos(sess, suffix_text, true, suffix, PROMPT_CAP, &suffix_n);
    if (s != GEIST_OK)
        goto out_fail;
    s = geist_session_prefill_tokens(sess, suffix_n, suffix);
    if (s != GEIST_OK)
        goto out_fail;
    out->t_pre = now_ms() - t2;

    /* First token + the rest. */
    const double  t3 = now_ms();
    geist_token_t tok;
    s = geist_session_decode_step(sess, &tok);
    if (s != GEIST_OK)
        goto out_fail;
    out->t_first = now_ms() - t3;

    size_t reply_len = 0;
    if (tok != 1 && tok != 106) {
        const char *t = geist_session_token_to_str(sess, tok);
        if (t != nullptr) {
            const size_t tn = strlen(t);
            if (reply_len + tn < REPLY_CAP - 1) {
                memcpy(out->reply + reply_len, t, tn);
                reply_len += tn;
                out->reply[reply_len] = '\0';
            }
        }
    }
    out->n_decoded = 1;

    const double t4 = now_ms();
    for (size_t i = 1; i < DECODE_CAP; i++) {
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK)
            break;
        if (tok == 1 /* eos */ || tok == 106 /* turn| */)
            break;
        const char *t = geist_session_token_to_str(sess, tok);
        if (t == nullptr)
            continue;
        const size_t tn = strlen(t);
        if (reply_len + tn >= REPLY_CAP - 1)
            break;
        memcpy(out->reply + reply_len, t, tn);
        reply_len += tn;
        out->reply[reply_len] = '\0';
        out->n_decoded++;
    }
    out->t_decode = now_ms() - t4;

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

    /* GEIST_AUDIO_WAV=<path> runs a single caller-provided clip instead of
     * the checked-in voice set — lets the CI smoke drive this harness with a
     * synthetic WAV (tools/gen_test_wav.py) on machines without the voice
     * fixtures. */
    const struct latency_case *list    = cases;
    size_t                     n_cases = sizeof(cases) / sizeof(cases[0]);
    struct latency_case        env_case;
    const char                *env_wav = getenv("GEIST_AUDIO_WAV");
    if (env_wav != nullptr && env_wav[0] != '\0') {
        env_case = (struct latency_case) {env_wav, "Describe the sound you heard.", "env clip"};
        list     = &env_case;
        n_cases  = 1;
    }
    size_t n_runs = 0;

    printf("audio_latency_e2e: per-stage push-to-talk timing\n");
    printf("\n%-26s %7s | %7s %7s %7s %7s %7s | %7s | tokens\n",
           "clip",
           "audio_s",
           "wav",
           "attach",
           "pre",
           "first",
           "decode",
           "total");
    printf("%-26s %7s + %7s %7s %7s %7s %7s = %7s\n",
           "----",
           "-------",
           "-------",
           "-------",
           "-------",
           "-------",
           "-------",
           "-------");

    for (size_t i = 0; i < n_cases; i++) {
        const struct latency_case *tc = &list[i];
        struct stage_timings       t;
        if (!run_one_case(model, be, tc, &t)) {
            printf("%-26s %7s   skipped/error\n", tc->label, "?");
            continue;
        }
        const double total = t.t_wav + t.t_attach + t.t_pre + t.t_first + t.t_decode;
        printf("%-26s %7.2f | %7.1f %7.1f %7.1f %7.1f %7.1f | %7.1f | %zu\n",
               tc->label,
               t.audio_s,
               t.t_wav,
               t.t_attach,
               t.t_pre,
               t.t_first,
               t.t_decode,
               total,
               t.n_decoded);
        if (t.reply[0] != '\0') {
            printf("%-26s   reply: \"%s\"\n", "", t.reply);
        }
        n_runs++;
    }

    printf("\naudio_latency_e2e: %zu / %zu clips processed\n", n_runs, n_cases);

    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (n_runs == 0)
        return GEIST_TEST_SKIP;
    return GEIST_TEST_PASS;
}
