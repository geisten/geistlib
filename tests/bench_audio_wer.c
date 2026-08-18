/*
 * bench_audio_wer — transcribe a list of WAVs through the full audio
 * pipeline, one model load, one line of output per clip. The other half
 * of the WER measurement lives in tools/eval_audio_wer.py, which compares
 * these transcripts against reference text (e.g. LibriSpeech).
 *
 *   bench_audio_wer <wavlist.txt>       # one 16 kHz mono WAV path per line
 *   GEIST_GGUF_PATH=... (model), tower via the usual audio search paths.
 *
 * Output per clip (tab-separated, machine-readable):
 *   WER\t<wav>\t<attach_ms>\t<decode_ms>\t<n_tok>\t<transcript>
 *
 * Bench, not a test: quality judgement happens in the Python half.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PROMPT_CAP 1024
#define DECODE_CAP 200
#define REPLY_CAP 4096

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

/* Minimal 16-bit mono PCM WAV reader (same as the other audio benches). */
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
    unsigned short channels = (unsigned short) (hdr[22] | (hdr[23] << 8));
    unsigned int   rate =
            (unsigned int) (hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24));
    unsigned short bps = (unsigned short) (hdr[34] | (hdr[35] << 8));
    if (channels != 1 || bps != 16) {
        fclose(f);
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    long data_bytes = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    size_t   n   = (size_t) data_bytes / 2;
    int16_t *pcm = malloc(n * sizeof(int16_t));
    if (pcm == nullptr) {
        fclose(f);
        return nullptr;
    }
    if (fread(pcm, sizeof(int16_t), n, f) != n) {
        free(pcm);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    *n_samples_out   = n;
    *sample_rate_out = (int) rate;
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

/* SentencePiece cleanup: U+2581 -> space, drop control/trailer bytes,
 * collapse to single-line. Same normalization test_audio_chat_e2e uses. */
static void normalize_sp(const char *in, char *out, size_t out_cap) {
    size_t j = 0;
    for (size_t i = 0; in[i] != '\0' && j + 1 < out_cap; i++) {
        const unsigned char c0 = (unsigned char) in[i];
        if (c0 == 0xE2 && (unsigned char) in[i + 1] == 0x96 && (unsigned char) in[i + 2] == 0x81) {
            out[j++] = ' ';
            i += 2;
            continue;
        }
        if (c0 == '\n' || c0 == '\r' || c0 == '\t') {
            out[j++] = ' ';
            continue;
        }
        if (c0 < 0x20)
            continue;
        out[j++] = in[i];
    }
    out[j] = '\0';
}

/* Transcribe one clip. Returns false on pipeline error. */
static bool transcribe(struct geist_model *model, struct geist_backend *be, const char *wav_path) {
    size_t   n_samples;
    int      sample_rate;
    int16_t *pcm = read_wav_pcm(wav_path, &n_samples, &sample_rate);
    if (pcm == nullptr || sample_rate != 16000) {
        fprintf(stderr, "skip (bad wav): %s\n", wav_path);
        free(pcm);
        return true; /* skip, not fatal */
    }

    struct geist_session_opts opts = {.max_seq_len = 2048};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        free(pcm);
        return false;
    }

    bool          ok = false;
    geist_token_t toks[PROMPT_CAP];
    size_t        n_toks = 0;
    if (tokenize_drop_bos(sess, "<bos><|turn>user\n<|audio>", false, toks, PROMPT_CAP, &n_toks) !=
                GEIST_OK ||
        geist_session_prefill_tokens(sess, n_toks, toks) != GEIST_OK)
        goto out;

    double            t0       = now_ms();
    enum geist_status s        = geist_session_attach_audio(sess, n_samples, pcm, 16000);
    double            t_attach = now_ms() - t0;
    free(pcm);
    pcm = nullptr;
    if (s != GEIST_OK) {
        fprintf(stderr, "attach failed (%s): %s\n", geist_session_errmsg(sess), wav_path);
        goto out;
    }

    /* GEIST_WER_PROMPT overrides the transcription instruction — prompt
     * wording moves E2B between transcribing and refusing. */
    const char *instr = getenv("GEIST_WER_PROMPT");
    if (instr == nullptr) {
        instr = "Transcribe the audio exactly. Reply with only the spoken words.";
    }
    char suffix_text[PROMPT_CAP];
    snprintf(suffix_text, sizeof suffix_text, "<audio|>\n%s<turn|>\n<|turn>model\n", instr);
    if (tokenize_drop_bos(sess, suffix_text, true, toks, PROMPT_CAP, &n_toks) != GEIST_OK ||
        geist_session_prefill_tokens(sess, n_toks, toks) != GEIST_OK)
        goto out;

    char   reply[REPLY_CAP];
    size_t reply_len = 0;
    reply[0]         = '\0';
    t0               = now_ms();
    size_t n_dec     = 0;
    /* Anti-loop stop (GEIST_WER_ANTILOOP=0 disables): greedy decode can
     * fall into short repetition cycles ("big, big, big, ...") that burn
     * the whole cap as WER insertions. Stop once the last 8 tokens are a
     * period-1 or period-2 cycle. */
    const char   *al       = getenv("GEIST_WER_ANTILOOP");
    const bool    antiloop = al == nullptr || al[0] != '0';
    geist_token_t hist[8];
    for (size_t i = 0; i < DECODE_CAP; i++) {
        geist_token_t tok;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK)
            break;
        n_dec++;
        if (tok == 1 /* <eos> */ || tok == 106 /* <turn|> */)
            break;
        hist[i % 8] = tok;
        if (antiloop && i >= 7) {
            bool cyc2 = true; /* period-2 covers period-1 too */
            for (size_t k = i - 5; k <= i; k++)
                cyc2 = cyc2 && hist[k % 8] == hist[(k - 2) % 8];
            if (cyc2)
                break;
        }
        const char *t = geist_session_token_to_str(sess, tok);
        if (t == nullptr)
            continue;
        size_t tn = strlen(t);
        if (reply_len + tn >= sizeof reply)
            break;
        memcpy(reply + reply_len, t, tn);
        reply_len += tn;
        reply[reply_len] = '\0';
    }
    double t_decode = now_ms() - t0;

    char norm[REPLY_CAP];
    normalize_sp(reply, norm, sizeof norm);
    printf("WER\t%s\t%.0f\t%.0f\t%zu\t%s\n", wav_path, t_attach, t_decode, n_dec, norm);
    fflush(stdout);
    ok = true;

out:
    free(pcm);
    geist_session_destroy(sess);
    return ok;
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 2, "<wavlist.txt>");
    GEIST_REQUIRE_GGUF(model_path);

    FILE *list = fopen(argv[1], "r");
    GEIST_SKIP_IF(list == nullptr, "cannot open wav list");

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fclose(list);
        return GEIST_TEST_ERROR;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        fclose(list);
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }

    char line[1024];
    int  n_done = 0, n_err = 0;
    while (fgets(line, sizeof line, list) != nullptr) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0' || line[0] == '#')
            continue;
        if (transcribe(model, be, line))
            n_done++;
        else
            n_err++;
    }
    fclose(list);
    fprintf(stderr, "bench_audio_wer: %d clips, %d errors\n", n_done, n_err);

    geist_model_destroy(model);
    geist_backend_destroy(be);
    return n_err == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
