/*
 * push_to_talk — a real-time voice loop over the public geist API.
 *
 * Reads raw 16 kHz mono s16le PCM from stdin (that is exactly what a USB
 * mic delivers through arecord), segments utterances with a simple energy
 * VAD, and answers each one through the Gemma 4 audio path:
 *
 *   arecord -f S16_LE -r 16000 -c 1 -t raw | ./push_to_talk model.gguf   # Linux/Pi
 *   ffmpeg -f avfoundation -i ":1" -ar 16000 -ac 1 -f s16le - | \
 *       ./push_to_talk model.gguf 1200                                   # macOS
 *
 * No mic handy? Any 16 kHz WAV works the same way (skip the 44-byte
 * header, append silence so the VAD closes the utterance):
 *
 *   (tail -c +45 clip.wav; dd if=/dev/zero bs=32000 count=1) | \
 *       ./push_to_talk model.gguf
 *
 * GEIST_PTT_PROMPT overrides the per-utterance instruction (default:
 * answer briefly). Needs audio_tower.safetensors + mel_constants.bin
 * discoverable next to the model or in the usual search paths.
 *
 * VAD: 20 ms frames; speech opens after 3 loud frames, closes after
 * 0.8 s of silence; utterances are capped at 28 s (the encoder buffers
 * at most 30 s). Tune the threshold as argv[2] (default 300 RMS) — a
 * real room, mic and gain always need a knob.
 */
#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SR 16000
#define FRAME 320                     /* 20 ms */
#define OPEN_FRAMES 3                 /* speech starts after 60 ms above threshold */
#define CLOSE_FRAMES 40               /* ...ends after 800 ms below it */
#define MIN_UTT (SR / 2)              /* ignore blips under 0.5 s */
#define MAX_UTT (28 * SR)             /* stay under the 30 s encoder buffer */
#define PROMPT_CAP 512
#define DECODE_CAP 160

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

static double frame_rms(const int16_t *f) {
    double acc = 0.0;
    for (int i = 0; i < FRAME; i++)
        acc += (double) f[i] * (double) f[i];
    return sqrt(acc / FRAME);
}

/* Tokenize helper: the tokenizer prepends the model's BOS; drop it for
 * the suffix so the turn markers stay well-formed. */
static geist_token_t g_bos = -1; /* resolved from model metadata in main */

static bool feed_text(struct geist_session *s, const char *text, bool drop_bos) {
    geist_token_t toks[PROMPT_CAP];
    size_t        n = 0;
    if (geist_session_tokenize(s, text, PROMPT_CAP, toks, &n) != GEIST_OK)
        return false;
    size_t start = (drop_bos && n > 0 && toks[0] == g_bos) ? 1 : 0;
    return geist_session_prefill_tokens(s, n - start, toks + start) == GEIST_OK;
}

/* The audio was already streamed into the session (begin/push/end in the
 * VAD loop below); this finishes the turn: instruction suffix + decode. */
static void answer_utterance(struct geist_session *sess,
                             const geist_token_t   eos,
                             const geist_token_t   end_of_turn,
                             double                t_tail,
                             double                audio_s,
                             const char           *instr) {
    double t0 = now_ms();
    char   suffix[PROMPT_CAP];
    snprintf(suffix, sizeof suffix, "<audio|>\n%s<turn|>\n<|turn>model\n", instr);
    if (!feed_text(sess, suffix, true)) {
        fprintf(stderr, "audio turn failed: %s\n", geist_session_errmsg(sess));
        return;
    }
    double t_attach = t_tail + (now_ms() - t0);

    printf("assistant> ");
    fflush(stdout);
    t0           = now_ms();
    size_t n_tok = 0;
    for (size_t i = 0; i < DECODE_CAP; i++) {
        geist_token_t tok;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK)
            break;
        if (tok == eos || tok == end_of_turn)
            break;
        const char *t = geist_session_token_to_str(sess, tok);
        if (t != nullptr) {
            /* SentencePiece: U+2581 marks a leading space; control bytes
             * are per-piece trailers — print the readable part. */
            for (const char *p = t; *p; p++) {
                if ((unsigned char) p[0] == 0xE2 && (unsigned char) p[1] == 0x96 &&
                    (unsigned char) p[2] == 0x81) {
                    putchar(' ');
                    p += 2;
                } else if ((unsigned char) *p >= 0x20 || *p == '\n') {
                    putchar(*p);
                }
            }
            fflush(stdout);
        }
        n_tok++;
    }
    double t_decode = now_ms() - t0;
    printf("\n  [%.1f s audio | tail %.0f ms after end-of-speech | %zu tokens @ %.1f tok/s]\n\n",
           audio_s,
           t_attach,
           n_tok,
           n_tok / (t_decode / 1000.0));
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: arecord -f S16_LE -r 16000 -c 1 -t raw | %s <model.gguf> [rms-threshold]\n",
                argv[0]);
        return 2;
    }
    const double rms_thr = argc > 2 ? atof(argv[2]) : 300.0;
    const char  *instr   = getenv("GEIST_PTT_PROMPT");
    if (instr == nullptr)
        instr = "Answer the speaker briefly.";

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
        fprintf(stderr,
                "this model instance cannot hear (no audio tower found) — "
                "see docs/MODELS.md\n");
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
    /* Stop tokens from the model's own metadata — no hardcoded ids. */
    g_bos                           = geist_model_bos_token(model);
    const geist_token_t eos         = geist_model_eos_token(model);
    const geist_token_t end_of_turn = geist_model_token_by_text(model, "<turn|>");

    /* Pin the constant turn prefix ONCE: geist_session_reset then
     * restores to it instead of to an empty cache, so every utterance
     * skips re-prefilling these tokens (STABLE agent-runtime API). */
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

    fprintf(stderr, "listening (VAD threshold %.0f RMS, Ctrl-C to quit)...\n", rms_thr);

    /* Streaming turn (#256): PCM is pushed WHILE the user speaks — the
     * encoder overlaps its work with the capture, and audio_end() after
     * end-of-speech pays only the tail (~0.2 s on a Pi 5 vs re-encoding
     * the whole utterance). A tiny ring of pre-open frames is pushed
     * retroactively so the utterance keeps its first 60 ms. */
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
            if (loud < OPEN_FRAMES) {
                continue;
            }
            in_speech = true;
            quiet     = 0;
            pushed    = 0;
            fprintf(stderr, "[speech]\n");
            geist_session_reset(sess); /* back to the pinned prefix */
            if (geist_session_audio_begin(sess) != GEIST_OK) {
                fprintf(stderr, "audio_begin failed: %s\n", geist_session_errmsg(sess));
                break;
            }
            for (int i = 0; i < OPEN_FRAMES - 1; i++) {
                (void) geist_session_audio_push(sess, FRAME, pending[i]);
                pushed += FRAME;
            }
        }
        if (pushed + FRAME <= MAX_UTT &&
            geist_session_audio_push(sess, FRAME, frame) == GEIST_OK) {
            pushed += FRAME;
            /* Phase 2 of #256: inject ready soft tokens into the LM while
             * the user is still speaking — end()'s tail shrinks to the
             * final chunk. Cheap no-op when nothing is ready. */
            (void) geist_session_audio_poll(sess);
        }
        quiet = is_loud ? 0 : quiet + 1;
        if (quiet >= CLOSE_FRAMES || pushed + FRAME > MAX_UTT) {
            in_speech = false;
            loud      = 0;
            /* The closing silence is part of pushed — subtract it, or a
             * 60 ms door slam plus 0.8 s of quiet always beats MIN_UTT. */
            const size_t speech_len = pushed - (size_t) quiet * FRAME;
            double       t0         = now_ms();
            if (geist_session_audio_end(sess) != GEIST_OK) {
                fprintf(stderr, "audio_end failed: %s\n", geist_session_errmsg(sess));
                continue;
            }
            double t_tail = now_ms() - t0;
            if (speech_len >= MIN_UTT) {
                fprintf(stderr, "[%.1f s — thinking]\n", (double) pushed / SR);
                answer_utterance(
                        sess, eos, end_of_turn, t_tail, (double) pushed / SR, instr);
                fprintf(stderr, "listening...\n");
            } else {
                geist_session_reset(sess); /* discard the blip's audio */
            }
        }
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
