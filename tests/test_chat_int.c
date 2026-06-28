/*
 * test_chat_int — multi-turn chat over the STABLE session API (model-gated).
 *
 * Validates the multi-turn chat transcript pattern: build a turn-marked transcript,
 * reset + set_prompt (which prepends BOS), decode greedily, and stop on EOS /
 * <end_of_turn>. Asserts generation MECHANICS, not content (greedy text varies):
 *   - a turn produces at least one non-control token,
 *   - it terminates on EOS/<end_of_turn> within the cap (does not run away),
 *   - a second turn appended to the transcript still generates.
 *
 * SKIPs cleanly without a GGUF (GEIST_GGUF_PATH). No assert() — exit codes.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <string.h>

enum { TRANSCRIPT_CAP = 1 << 14, DECODE_CAP = 48 };

/* Decode one assistant turn; return #non-control tokens emitted, and set
 * *stopped if it ended on EOS/<eot> rather than hitting the cap. */
static int decode_turn(struct geist_session *s,
                       geist_token_t         eos,
                       geist_token_t         eot,
                       char                 *reply,
                       size_t                reply_cap,
                       int                  *stopped) {
    int    emitted = 0;
    size_t rlen    = 0;
    *stopped       = 0;
    for (int i = 0; i < DECODE_CAP; i++) {
        geist_token_t tok = 0;
        if (geist_session_decode_step(s, &tok) != GEIST_OK) {
            break;
        }
        if (tok == eos || tok == eot) {
            *stopped = 1;
            break;
        }
        const char *piece = geist_session_token_to_str(s, tok);
        if (!piece) {
            break;
        }
        size_t pl = strlen(piece);
        if (pl >= 2 && piece[0] == '<' && piece[pl - 1] == '>') {
            *stopped = 1; /* control marker = end of turn */
            break;
        }
        emitted++;
        if (rlen + pl + 1 < reply_cap) {
            memcpy(reply + rlen, piece, pl);
            rlen += pl;
        }
    }
    reply[rlen] = '\0';
    return emitted;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        GEIST_SKIP("backend_create failed");
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("model_load failed (set GEIST_GGUF_PATH)");
    }
    struct geist_session_opts opts = {0}; /* greedy, deterministic */
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("session_create failed");
    }
    geist_token_t eos = geist_model_eos_token(model);
    geist_token_t eot = geist_model_token_by_text(model, "<end_of_turn>");

    int  rc = GEIST_TEST_PASS;
    char transcript[TRANSCRIPT_CAP];
    char reply[2048];

    /* ---- turn 1 ---- */
    size_t tlen = (size_t) snprintf(
            transcript,
            sizeof transcript,
            "<start_of_turn>user\nName one color.<end_of_turn>\n<start_of_turn>model\n");
    if (geist_session_reset(sess) != GEIST_OK ||
        geist_session_set_prompt(sess, transcript) != GEIST_OK) {
        fprintf(stderr, "FAIL: turn-1 prefill: %s\n", geist_session_errmsg(sess));
        rc = GEIST_TEST_FAIL;
        goto done;
    }
    int stopped1 = 0;
    int n1       = decode_turn(sess, eos, eot, reply, sizeof reply, &stopped1);
    fprintf(stderr, "turn 1: %d tokens, stopped=%d, reply=\"%s\"\n", n1, stopped1, reply);
    if (n1 < 1) {
        fprintf(stderr, "FAIL: turn 1 produced no tokens\n");
        rc = GEIST_TEST_FAIL;
        goto done;
    }
    if (!stopped1) {
        fprintf(stderr, "FAIL: turn 1 ran to the cap without stopping on EOS/<eot>\n");
        rc = GEIST_TEST_FAIL;
        goto done;
    }

    /* ---- turn 2: append reply + a new user turn, reprefill, decode again ---- */
    tlen += (size_t) snprintf(transcript + tlen,
                              sizeof transcript - tlen,
                              "%s<end_of_turn>\n<start_of_turn>user\nAnd another?"
                              "<end_of_turn>\n<start_of_turn>model\n",
                              reply);
    if (geist_session_reset(sess) != GEIST_OK ||
        geist_session_set_prompt(sess, transcript) != GEIST_OK) {
        fprintf(stderr, "FAIL: turn-2 prefill: %s\n", geist_session_errmsg(sess));
        rc = GEIST_TEST_FAIL;
        goto done;
    }
    int stopped2 = 0;
    int n2       = decode_turn(sess, eos, eot, reply, sizeof reply, &stopped2);
    fprintf(stderr, "turn 2: %d tokens, stopped=%d\n", n2, stopped2);
    if (n2 < 1) {
        fprintf(stderr, "FAIL: turn 2 produced no tokens\n");
        rc = GEIST_TEST_FAIL;
        goto done;
    }

    printf("multi-turn chat over the session API works\n");

done:
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return rc;
}
