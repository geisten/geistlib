/*
 * geist_chat.c — interactive multi-turn chat + a file-based "memory palace".
 *
 * Stage 1+2 of the agent plan: a REPL chat loop over the STABLE core API
 * (set_prompt -> decode_step -> token_to_str) plus the markdown memory palace
 * (tools/mind.h) under ./mind (override with GEIST_MIND_DIR). INDEX.md is fed
 * into the model's context each session so it knows what is stored. No DB, no
 * embeddings — grep + the in-context index. Web fetch/search is a later stage.
 *
 *   geist_chat <model.gguf>           # chat; type /help for commands
 *   geist_chat --selftest             # palace roundtrip check, no model needed
 *
 * REPL commands:
 *   /remember <title> | <text>   write a note + index it
 *   /recall <slug>               load a note into the conversation context
 *   /notes                       print the index (what's stored)
 *   /quit
 * Anything else is a chat turn.
 */
#define _POSIX_C_SOURCE 200809L

#include <geist.h>
#include <geist_util.h>

#include "mind.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LINE_CAP = 8192, NOTE_CAP = 1 << 15, TRANSCRIPT_CAP = 1 << 16, PATH_CAP = 1024 };

static void rstrip(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == ' ' || s[n - 1] == '\t')) {
        s[--n] = '\0';
    }
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) {
        setenv("GEIST_MIND_DIR", "./.mind_selftest", 1);
        char buf[NOTE_CAP];
        int  ok = mind_remember("Test Note", "hello world contents") == 0 &&
                  mind_recall("test-note", buf, sizeof buf) > 0 &&
                  strstr(buf, "hello world contents") != nullptr &&
                  mind_slurp("./.mind_selftest/INDEX.md", buf, sizeof buf) > 0 &&
                  strstr(buf, "[Test Note](test-note.md)") != nullptr;
        remove("./.mind_selftest/test-note.md");
        remove("./.mind_selftest/INDEX.md");
        remove("./.mind_selftest");
        puts(ok ? "geist_chat selftest ok" : "geist_chat selftest FAILED");
        return ok ? 0 : 1;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>   (or --selftest)\n", argv[0]);
        return 2;
    }

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(argv[1], be, &model) != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    struct geist_session_opts opts = {0}; /* greedy */
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }
    geist_token_t eos = geist_model_eos_token(model);
    geist_token_t eot = geist_model_token_by_text(model, "<end_of_turn>");

    /* Seed the transcript with whatever the palace already holds, so the model
     * starts the session aware of its own notes. */
    static char transcript[TRANSCRIPT_CAP];
    static char index_buf[NOTE_CAP];
    char        ipath[PATH_CAP];
    snprintf(ipath, sizeof ipath, "%s/INDEX.md", mind_dir());
    size_t tlen = 0;
    if (mind_slurp(ipath, index_buf, sizeof index_buf) > 0) {
        tlen += (size_t) snprintf(transcript + tlen,
                                  sizeof transcript - tlen,
                                  "You have a memory palace. Stored notes:\n%s\n",
                                  index_buf);
    }

    fprintf(stderr, "loaded %s — /help for commands, /quit to exit\n", geist_model_arch(model));
    static char line[LINE_CAP];
    static char note[NOTE_CAP];
    fputs("> ", stdout);
    fflush(stdout);
    while (fgets(line, sizeof line, stdin)) {
        rstrip(line);
        if (line[0] == '\0') {
            fputs("> ", stdout);
            fflush(stdout);
            continue;
        }

        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) {
            break;
        }
        if (strcmp(line, "/help") == 0) {
            puts("/remember <title> | <text>   /recall <slug>   /notes   /quit");
            fputs("> ", stdout);
            fflush(stdout);
            continue;
        }
        if (strcmp(line, "/notes") == 0) {
            if (mind_slurp(ipath, index_buf, sizeof index_buf) > 0) {
                fputs(index_buf, stdout);
            } else {
                puts("(no notes yet)");
            }
            fputs("> ", stdout);
            fflush(stdout);
            continue;
        }
        if (strncmp(line, "/remember ", 10) == 0) {
            char *bar = strchr(line + 10, '|');
            if (!bar) {
                puts("usage: /remember <title> | <text>");
            } else {
                *bar        = '\0';
                char *title = line + 10, *text = bar + 1;
                rstrip(title);
                while (*text == ' ') {
                    text++;
                }
                puts(mind_remember(title, text) == 0 ? "remembered." : "remember failed.");
            }
            fputs("> ", stdout);
            fflush(stdout);
            continue;
        }
        if (strncmp(line, "/recall ", 8) == 0) {
            const char *slug = line + 8;
            if (mind_recall(slug, note, sizeof note) > 0) {
                /* inject into context so the model can use it next turn */
                tlen += (size_t) snprintf(transcript + tlen,
                                          sizeof transcript - tlen,
                                          "Recalled note %s:\n%s\n",
                                          slug,
                                          note);
                printf("recalled %s into context\n", slug);
            } else {
                printf("no note '%s'\n", slug);
            }
            fputs("> ", stdout);
            fflush(stdout);
            continue;
        }

        /* ---- chat turn: append, reprefill the whole transcript, decode ---- */
        /* ponytail: full-transcript reprefill each turn — O(n^2) over the chat.
         * Switch to incremental geist_session_prefill_tokens (append only the
         * new turn) if sessions get long enough to feel it. */
        int w = snprintf(transcript + tlen,
                         sizeof transcript - tlen,
                         "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n",
                         line);
        if (w < 0 || (size_t) w >= sizeof transcript - tlen) {
            puts("(context full — /quit and start over)");
            break;
        }
        tlen += (size_t) w;

        if (geist_session_reset(sess) != GEIST_OK ||
            geist_session_set_prompt(sess, transcript) != GEIST_OK) {
            fprintf(stderr, "prefill failed: %s\n", geist_session_errmsg(sess));
            break;
        }
        size_t rlen = 0;
        for (int i = 0; i < 512; i++) {
            geist_token_t tok = 0;
            if (geist_session_decode_step(sess, &tok) != GEIST_OK) {
                break;
            }
            if (tok == eos || tok == eot) {
                break;
            }
            const char *piece = geist_session_token_to_str(sess, tok);
            if (!piece) {
                break;
            }
            size_t pl = strlen(piece);
            /* ponytail: stops only on a single-token <marker>. Gemma's
             * <end_of_turn> can arrive as multiple BPE pieces and then leaks
             * into the reply before EOS stops it. Wire the model's real chat
             * template / turn-token ids to end cleanly when targeting an -it model. */
            if (pl >= 2 && piece[0] == '<' && piece[pl - 1] == '>') {
                break; /* control marker */
            }
            fputs(piece, stdout);
            fflush(stdout);
            if (rlen + pl + 1 < sizeof note) {
                memcpy(note + rlen, piece, pl);
                rlen += pl;
            }
        }
        note[rlen] = '\0';
        /* fold the model's reply back into the transcript for the next turn */
        tlen += (size_t) snprintf(
                transcript + tlen, sizeof transcript - tlen, "%s<end_of_turn>\n", note);
        fputs("\n> ", stdout);
        fflush(stdout);
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
