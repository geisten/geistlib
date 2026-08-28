/* Real-weight coverage for the isolated Qwen3.5 MTP forward primitive.
 * The public CI fixture has no nextn block, so this test skips unless the
 * 27B artifact (or GEIST_QWEN35_MTP_GGUF_PATH) is available. */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/arch_state.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *resolve_model(void) {
    const char *env = getenv("GEIST_QWEN35_MTP_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/qwen3.8-27b-q4_0.gguf",
            nullptr,
    };
    for (size_t i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return candidates[i];
        }
    }
    return nullptr;
}

int main(void) {
    const char *path = resolve_model();
    if (path == nullptr)
        GEIST_SKIP_FIXTURE("no Qwen3.5/3.8 GGUF with an MTP block");

    /* PR 4 keeps MTP opt-in so ordinary decode has no synchronization cost. */
    if (setenv("GEIST_MTP", "1", 1) != 0)
        return GEIST_TEST_ERROR;

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend_create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    const struct geist_session_opts opts = {
            .max_seq_len = 16,
            .m_max       = 2,
            .temperature = 0.0f,
    };
    struct transformer_arch_state *st = nullptr;
    s                                 = transformer_state_create(be, path, &opts, &st);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "state_create: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    if (st->n_mtp_layers != 1) {
        transformer_state_destroy(st);
        geist_backend_destroy(be);
        GEIST_SKIP("selected model has no single MTP block");
    }

    struct transformer_arch_session *sess   = transformer_default_session(st);
    const size_t                     H      = st->d_model;
    float                           *hidden = calloc(2 * H, sizeof(float));
    float                           *out_a  = calloc(2 * H, sizeof(float));
    float                           *out_b  = calloc(2 * H, sizeof(float));
    if (hidden == nullptr || out_a == nullptr || out_b == nullptr) {
        free(hidden);
        free(out_a);
        free(out_b);
        transformer_state_destroy(st);
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }
    for (size_t i = 0; i < 2 * H; i++)
        hidden[i] = (float) ((int) (i % 29) - 14) / 29.0f;

    const geist_token_t ids[2]   = {1, 42};
    geist_token_t       tok_a[2] = {-1, -1};
    geist_token_t       tok_b[2] = {-1, -1};
    int                 fails    = 0;

    /* Non-zero sentinels prove that the isolated head preserves an existing
     * target transaction rather than merely leaving fresh zero state alone. */
    sess->kv_len             = 3;
    sess->logits_valid       = true;
    sess->next_token_pending = 123;
    size_t dn_sentinel_layer = st->n_layers;
    for (size_t li = 0; li < st->n_layers; li++) {
        if (sess->dn_conv_state[li] != nullptr && sess->dn_S[li] != nullptr) {
            dn_sentinel_layer          = li;
            sess->dn_conv_state[li][0] = 1.25f;
            sess->dn_S[li][0]          = -2.5f;
            break;
        }
    }

    s = transformer_mtp_forward(sess, 2, ids, hidden, tok_a, out_a);
    fails += geist_expect(s == GEIST_OK, "first MTP batch succeeds");
    fails += geist_expect(sess->mtp_kv_len == 2, "MTP cache advances by batch width");
    fails += geist_expect(sess->kv_len == 3, "MTP forward does not advance target KV");
    fails += geist_expect(sess->logits_valid && sess->next_token_pending == 123,
                          "MTP forward preserves pending target logits");
    fails += geist_expect(dn_sentinel_layer < st->n_layers &&
                                  sess->dn_conv_state[dn_sentinel_layer][0] == 1.25f &&
                                  sess->dn_S[dn_sentinel_layer][0] == -2.5f,
                          "MTP forward preserves target recurrent state");
    fails += geist_expect(tok_a[0] >= 0 && (size_t) tok_a[0] < st->vocab_size && tok_a[1] >= 0 &&
                                  (size_t) tok_a[1] < st->vocab_size,
                          "MTP argmax tokens are in vocabulary");

    transformer_mtp_reset(sess);
    fails += geist_expect(sess->mtp_kv_len == 0, "MTP reset clears only its logical cache");
    s = transformer_mtp_forward(sess, 2, ids, hidden, tok_b, out_b);
    fails += geist_expect(s == GEIST_OK, "repeated MTP batch succeeds");
    fails += geist_expect(memcmp(tok_a, tok_b, sizeof tok_a) == 0,
                          "MTP argmax is deterministic after reset");
    fails += geist_expect(memcmp(out_a, out_b, 2 * H * sizeof(float)) == 0,
                          "MTP hidden rows are deterministic after reset");
    bool  finite_hidden = true;
    float max_abs       = 0.0f;
    for (size_t i = 0; i < 2 * H; i++) {
        finite_hidden = finite_hidden && isfinite(out_a[i]);
        const float a = fabsf(out_a[i]);
        if (a > max_abs)
            max_abs = a;
    }
    fails += geist_expect(finite_hidden && max_abs > 0.0f,
                          "MTP returns finite, non-zero hidden rows");

    /* PR 5 routes recursive one-row drafts through the i8 sketch head. The
     * dense fallback remains available, fast candidates are deterministic,
     * and shared scratch must not leak sparse-logit metadata into the
     * authoritative target state. End-to-end target-token equality is covered
     * by test_speculative_loop_int because drafts may intentionally differ. */
    if (st->spec_state == 1) {
        geist_token_t dense_tok = -1, fast_tok = -1, fast_tok_2 = -1;
        float        *fast_h = calloc(H, sizeof(float));
        if (fast_h == nullptr) {
            fails++;
        } else {
            transformer_mtp_reset(sess);
            if (setenv("GEIST_SPEC_HEAD", "0", 1) != 0)
                fails++;
            if (setenv("GEIST_MTP_SPEC_HEAD", "0", 1) != 0)
                fails++;
            s = transformer_mtp_forward(sess, 1, ids, hidden, &dense_tok, nullptr);
            fails += geist_expect(s == GEIST_OK, "dense one-row MTP reference succeeds");
            fails += geist_expect(dense_tok >= 0 && (size_t) dense_tok < st->vocab_size,
                                  "dense one-row MTP token is in vocabulary");

            transformer_mtp_reset(sess);
            sess->logits_sparse     = false;
            sess->logits_softcapped = true;
            if (setenv("GEIST_SPEC_HEAD", "1", 1) != 0)
                fails++;
            if (setenv("GEIST_MTP_SPEC_HEAD", "1", 1) != 0)
                fails++;
            s = transformer_mtp_forward(sess, 1, ids, hidden, &fast_tok, fast_h);
            fails += geist_expect(s == GEIST_OK, "sketch one-row MTP forward succeeds");
            fails += geist_expect(fast_tok >= 0 && (size_t) fast_tok < st->vocab_size,
                                  "27B sketch MTP token is in vocabulary");
            fails += geist_expect(!sess->logits_sparse && sess->logits_softcapped,
                                  "MTP sketch preserves target logit metadata");
            transformer_mtp_reset(sess);
            s = transformer_mtp_forward(sess, 1, ids, hidden, &fast_tok_2, nullptr);
            fails += geist_expect(s == GEIST_OK && fast_tok_2 == fast_tok,
                                  "27B sketch MTP token is deterministic");
            free(fast_h);
        }
    }

    const size_t        before_bad = sess->mtp_kv_len;
    const geist_token_t bad        = (geist_token_t) st->vocab_size;
    s                              = transformer_mtp_forward(sess, 1, &bad, hidden, tok_b, nullptr);
    fails += geist_expect(s == GEIST_E_INVALID_ARG, "invalid MTP token is rejected");
    fails += geist_expect(sess->mtp_kv_len == before_bad,
                          "invalid input does not mutate MTP cache length");

    transformer_session_reset(sess);
    fails +=
            geist_expect(sess->mtp_kv_len == 0, "session reset also clears the isolated MTP cache");

    /* Target catch-up uses the one-position hidden shift expected by Qwen's
     * nextn block. Drafting must then leave its provisional cache writes at
     * the target boundary so batched verification can overwrite them. */
    s = transformer_prefill_text_batch(sess, 2, ids);
    fails += geist_expect(s == GEIST_OK, "target prefill with MTP synchronization succeeds");
    fails += geist_expect(sess->kv_len == 2 && sess->mtp_kv_len == 2,
                          "target and MTP cache lengths stay synchronized");
    geist_token_t drafts[2] = {-1, -1};
    size_t        n_drafts  = 0;
    s = transformer_mtp_draft(sess, 2, sess->next_token_pending, drafts, &n_drafts);
    fails += geist_expect(s == GEIST_OK && n_drafts == 2,
                          "native MTP produces the requested greedy draft chain");
    fails += geist_expect(sess->mtp_kv_len == sess->kv_len && sess->kv_len == 2,
                          "drafting restores the provisional MTP cache boundary");
    for (size_t i = 0; i < n_drafts; i++) {
        fails += geist_expect(drafts[i] >= 0 && (size_t) drafts[i] < st->vocab_size,
                              "native draft token is in vocabulary");
    }

    transformer_session_reset(sess);

    printf("MTP 27B: tokens=%d,%d hidden-bytes=%zu max|h|=%.4f drafts=%zu "
           "advanced-cache=%zu reset-cache=%zu\n",
           tok_a[0],
           tok_a[1],
           2 * H * sizeof(float),
           max_abs,
           n_drafts,
           before_bad,
           sess->mtp_kv_len);
    free(hidden);
    free(out_a);
    free(out_b);
    transformer_state_destroy(st);
    geist_backend_destroy(be);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
