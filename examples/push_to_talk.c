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

/* Tokenize helper: Gemma's tokenizer prepends <bos>; drop it for the
 * suffix so the turn markers stay well-formed. */
static bool feed_text(struct geist_session *s, const char *text, bool drop_bos) {
    geist_token_t toks[PROMPT_CAP];
    size_t        n = 0;
    if (geist_session_tokenize(s, text, PROMPT_CAP, toks, &n) != GEIST_OK)
        return false;
    size_t start = (drop_bos && n > 0 && toks[0] == 2) ? 1 : 0;
    return geist_session_prefill_tokens(s, n - start, toks + start) == GEIST_OK;
}

static void answer_utterance(struct geist_model   *model,
                             struct geist_backend *be,
                             const int16_t        *pcm,
                             size_t                n,
                             const char           *instr) {
    struct geist_session_opts opts = {.max_seq_len = 2048};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        return;
    }

    double t0 = now_ms();
    char   suffix[PROMPT_CAP];
    snprintf(suffix, sizeof suffix, "<audio|>\n%s<turn|>\n<|turn>model\n", instr);
    if (!feed_text(sess, "<bos><|turn>user\n<|audio>", false) ||
        geist_session_attach_audio(sess, n, pcm, SR) != GEIST_OK ||
        !feed_text(sess, suffix, true)) {
        fprintf(stderr, "audio turn failed: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess);
        return;
    }
    double t_attach = now_ms() - t0;

    printf("assistant> ");
    fflush(stdout);
    t0           = now_ms();
    size_t n_tok = 0;
    for (size_t i = 0; i < DECODE_CAP; i++) {
        geist_token_t tok;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK)
            break;
        if (tok == 1 /* <eos> */ || tok == 106 /* <turn|> */)
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
    printf("\n  [%.1f s audio | attach %.0f ms | %zu tokens @ %.1f tok/s]\n\n",
           (double) n / SR,
           t_attach,
           n_tok,
           n_tok / (t_decode / 1000.0));
    fflush(stdout);
    geist_session_destroy(sess);
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

    fprintf(stderr, "listening (VAD threshold %.0f RMS, Ctrl-C to quit)...\n", rms_thr);

    static int16_t utt[MAX_UTT];
    int16_t        frame[FRAME];
    size_t         utt_len = 0;
    int            loud = 0, quiet = 0;
    bool           in_speech = false;

    while (fread(frame, sizeof(int16_t), FRAME, stdin) == FRAME) {
        const bool is_loud = frame_rms(frame) > rms_thr;
        if (!in_speech) {
            loud = is_loud ? loud + 1 : 0;
            if (loud >= OPEN_FRAMES) {
                in_speech = true;
                quiet     = 0;
                utt_len   = 0;
                fprintf(stderr, "[speech]\n");
            } else {
                continue;
            }
        }
        if (utt_len + FRAME <= MAX_UTT) {
            memcpy(utt + utt_len, frame, sizeof frame);
            utt_len += FRAME;
        }
        quiet = is_loud ? 0 : quiet + 1;
        if (quiet >= CLOSE_FRAMES || utt_len >= MAX_UTT) {
            in_speech = false;
            loud      = 0;
            if (utt_len >= MIN_UTT) {
                fprintf(stderr, "[%.1f s — thinking]\n", (double) utt_len / SR);
                answer_utterance(model, be, utt, utt_len, instr);
                fprintf(stderr, "listening...\n");
            }
            utt_len = 0;
        }
    }

    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
