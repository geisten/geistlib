/*
 * dictate — system-wide dictation core over the public geist API.
 *
 * Reads raw 16 kHz mono s16le PCM from stdin, segments utterances with
 * the same energy VAD as push_to_talk, transcribes each one through the
 * Gemma 4 audio path, and prints ONE LINE of clean text per utterance to
 * stdout (all status goes to stderr). That contract makes it a pipeline
 * stage — the OS integration stays out of tree:
 *
 *   # Wayland: type into the focused window
 *   arecord -f S16_LE -r 16000 -c 1 -t raw | ./dictate model.gguf | wtype -
 *   # X11
 *   arecord ... | ./dictate model.gguf | xargs -d'\n' -I{} xdotool type --clearmodifiers {}
 *   # macOS test run
 *   ffmpeg -f avfoundation -i ":1" -ar 16000 -ac 1 -f s16le - | ./dictate model.gguf 1200
 *
 * GEIST_DICTATE_PROMPT overrides the instruction (default:
 * "Transcribe this audio." — the best-measured phrasing, 4.2 % WER on
 * the LibriSpeech harness set; see docs/VOICE.md). Decode carries the
 * #267 anti-loop guard, and a reply that opens a thought channel is
 * dropped instead of typed.
 *
 * VAD: identical to push_to_talk — 20 ms frames, opens after 60 ms above
 * threshold, closes after 0.8 s below, threshold as argv[2] (default 300).
 */
#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SR 16000
#define FRAME 320       /* 20 ms */
#define OPEN_FRAMES 3   /* speech starts after 60 ms above threshold */
#define CLOSE_FRAMES 40 /* ...ends after 800 ms below it */
#define MIN_UTT (SR / 2)
#define MAX_UTT (28 * SR)
#define PROMPT_CAP 512
#define DECODE_CAP 400
#define REPLY_CAP 4096

static double frame_rms(const int16_t *f) {
    double acc = 0.0;
    for (int i = 0; i < FRAME; i++)
        acc += (double) f[i] * (double) f[i];
    return sqrt(acc / FRAME);
}

static geist_token_t g_bos = -1;

static bool feed_text(struct geist_session *s, const char *text, bool drop_bos) {
    geist_token_t toks[PROMPT_CAP];
    size_t        n = 0;
    if (geist_session_tokenize(s, text, PROMPT_CAP, toks, &n) != GEIST_OK)
        return false;
    size_t start = (drop_bos && n > 0 && toks[0] == g_bos) ? 1 : 0;
    return geist_session_prefill_tokens(s, n - start, toks + start) == GEIST_OK;
}

/* Append one decoded piece to line[], SentencePiece-normalized: U+2581 and
 * newlines become spaces, control bytes are dropped. */
static void append_piece(const char *t, char *line, size_t *len) {
    for (const char *p = t; *p != '\0' && *len + 1 < REPLY_CAP; p++) {
        if ((unsigned char) p[0] == 0xE2 && (unsigned char) p[1] == 0x96 &&
            (unsigned char) p[2] == 0x81) {
            line[(*len)++] = ' ';
            p += 2;
        } else if (*p == '\n' || *p == '\r' || *p == '\t') {
            line[(*len)++] = ' ';
        } else if ((unsigned char) *p >= 0x20) {
            line[(*len)++] = *p;
        }
    }
    line[*len] = '\0';
}

/* Audio already streamed into the session; finish the turn and emit the
 * transcript line. Returns false only on session errors. */
static void transcribe_utterance(struct geist_session *sess,
                                 geist_token_t         eos,
                                 geist_token_t         end_of_turn,
                                 geist_token_t         channel_tok,
                                 const char           *instr) {
    char suffix[PROMPT_CAP];
    snprintf(suffix, sizeof suffix, "<audio|>\n%s<turn|>\n<|turn>model\n", instr);
    if (!feed_text(sess, suffix, true)) {
        fprintf(stderr, "dictate: audio turn failed: %s\n", geist_session_errmsg(sess));
        return;
    }

    char          line[REPLY_CAP];
    size_t        len     = 0;
    geist_token_t hist[8] = {0};
    line[0]               = '\0';
    for (size_t i = 0; i < DECODE_CAP; i++) {
        geist_token_t tok;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK)
            break;
        if (tok == eos || tok == end_of_turn)
            break;
        /* A thought-channel reply is meta text, not dictation — drop the
         * whole utterance rather than typing it. */
        if (i == 0 && channel_tok >= 0 && tok == channel_tok) {
            fprintf(stderr, "dictate: model produced meta output, utterance dropped\n");
            return;
        }
        /* #267 anti-loop: stop once the last 8 tokens are a period-1/2
         * cycle — keeps a hard clip from typing "big, big, big, ...". */
        hist[i % 8] = tok;
        if (i >= 7) {
            bool cyc2 = true;
            for (size_t k = i - 5; k <= i; k++)
                cyc2 = cyc2 && hist[k % 8] == hist[(k - 2) % 8];
            if (cyc2)
                break;
        }
        const char *t = geist_session_token_to_str(sess, tok);
        if (t != nullptr)
            append_piece(t, line, &len);
    }

    /* Trim the leading SentencePiece space; emit exactly one line. */
    const char *out = line[0] == ' ' ? line + 1 : line;
    if (out[0] != '\0') {
        printf("%s\n", out);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: arecord -f S16_LE -r 16000 -c 1 -t raw | %s <model.gguf> [rms-threshold]\n",
                argv[0]);
        return 2;
    }
    const double rms_thr = argc > 2 ? atof(argv[2]) : 300.0;
    const char  *instr   = getenv("GEIST_DICTATE_PROMPT");
    if (instr == nullptr)
        instr = "Transcribe this audio.";

    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK &&
        geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return 1;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(argv[1], be, &model) != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    if ((geist_model_modalities(model) & GEIST_MOD_AUDIO) == 0) {
        fprintf(stderr, "this model instance cannot hear (no audio tower found)\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    struct geist_session_opts opts = {.max_seq_len = 2048};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }
    g_bos                           = geist_model_bos_token(model);
    const geist_token_t eos         = geist_model_eos_token(model);
    const geist_token_t end_of_turn = geist_model_token_by_text(model, "<turn|>");
    const geist_token_t channel_tok = geist_model_token_by_text(model, "<|channel>");

    {
        geist_token_t prefix[PROMPT_CAP];
        size_t        n_prefix = 0;
        if (geist_session_tokenize(
                    sess, "<bos><|turn>user\n<|audio>", PROMPT_CAP, prefix, &n_prefix) != GEIST_OK ||
            geist_session_pin_prefix(sess, n_prefix, prefix) != GEIST_OK) {
            fprintf(stderr, "pin_prefix failed\n");
            geist_session_destroy(sess);
            geist_model_destroy(model);
            geist_backend_destroy(be);
            return 1;
        }
    }

    fprintf(stderr, "dictate: listening (VAD threshold %.0f RMS, Ctrl-C to quit)...\n", rms_thr);

    /* Same streaming VAD loop as push_to_talk: push PCM while the user
     * speaks, poll injects ready soft tokens, end() pays only the tail. */
    int16_t frame[FRAME];
    int16_t pending[OPEN_FRAMES][FRAME];
    size_t  pushed = 0;
    int     loud = 0, quiet = 0;
    bool    in_speech = false;

    while (fread(frame, sizeof(int16_t), FRAME, stdin) == FRAME) {
        const bool is_loud = frame_rms(frame) > rms_thr;
        if (!in_speech) {
            memcpy(pending[loud % OPEN_FRAMES], frame, sizeof frame);
            loud = is_loud ? loud + 1 : 0;
            if (loud < OPEN_FRAMES)
                continue;
            in_speech = true;
            quiet     = 0;
            pushed    = 0;
            geist_session_reset(sess);
            if (geist_session_audio_begin(sess) != GEIST_OK) {
                fprintf(stderr, "audio_begin failed: %s\n", geist_session_errmsg(sess));
                break;
            }
            for (int i = 0; i < OPEN_FRAMES - 1; i++) {
                (void) geist_session_audio_push(sess, FRAME, pending[i]);
                pushed += FRAME;
            }
        }
        if (pushed + FRAME <= MAX_UTT && geist_session_audio_push(sess, FRAME, frame) == GEIST_OK) {
            pushed += FRAME;
            (void) geist_session_audio_poll(sess);
        }
        quiet = is_loud ? 0 : quiet + 1;
        if (quiet >= CLOSE_FRAMES || pushed + FRAME > MAX_UTT) {
            in_speech               = false;
            loud                    = 0;
            const size_t speech_len = pushed - (size_t) quiet * FRAME;
            if (geist_session_audio_end(sess) != GEIST_OK) {
                fprintf(stderr, "audio_end failed: %s\n", geist_session_errmsg(sess));
                continue;
            }
            if (speech_len >= MIN_UTT) {
                fprintf(stderr, "dictate: [%.1f s]\n", (double) pushed / SR);
                transcribe_utterance(sess, eos, end_of_turn, channel_tok, instr);
            } else {
                geist_session_reset(sess);
            }
        }
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
