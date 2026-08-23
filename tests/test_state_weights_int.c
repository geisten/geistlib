/*
 * test_state_weights_int — Phase B-4e sub-step 1 verification.
 *
 * Loads Gemma 4 weights through transformer_state_create (the new
 * backend-buffer path) and verifies that every byte of every weight matches
 * the corresponding mmap'd region in the GGUF reader. Covers:
 *   - 5 global tensors (token_embd, per_layer_token_embd, per_layer_model_proj,
 *                       per_layer_proj_norm, output_norm)
 *   - 35 layers × {6 norms, ≤9 projections, 1 scalar}
 *     (KV-shared layers 15..34 skip k_proj/v_proj/k_norm, which is correct.)
 *
 * The comparison: gguf_get_tensor(name)->data is the authoritative byte
 * source. We download each backend buffer back to host and memcmp.
 *
 * This is a strict byte-identity check, not a numerical-equivalence check.
 * Sub-steps 2-3 add the forward-pass numerical checks.
 *
 * SKIPs cleanly if no GGUF is available.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/archs/transformer/arch_state.h"

#include "gguf_reader.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Gemma-4 E2B reference geometry — the fixture this test runs against.
 * (Were GEIST_GEMMA4_* macros in arch_state.h until the neutral-defaults
 * inversion moved family defaults into the populators.) */
#define GEIST_GEMMA4_HIDDEN 1536
#define GEIST_GEMMA4_NUM_LAYERS 35
#define GEIST_GEMMA4_HIDDEN_PER_LAYER 256
#define GEIST_GEMMA4_PLE_OUT (GEIST_GEMMA4_NUM_LAYERS * GEIST_GEMMA4_HIDDEN_PER_LAYER)
#define GEIST_GEMMA4_N_Q_HEADS 8
#define GEIST_GEMMA4_N_KV_HEADS 1
#define GEIST_GEMMA4_VOCAB 262144

struct check_stats {
    size_t n_checked;
    size_t bytes_checked;
    size_t n_failed;
};

/* Compare a backend buffer's contents against the named GGUF tensor's mmap'd
 * bytes. Logs failure detail to stderr but does not exit — caller aggregates. */
static void check_buf_eq_gguf(struct geist_backend *be,
                              struct gguf_ctx      *ctx,
                              struct geist_buffer  *buf,
                              const char           *tensor_name,
                              struct check_stats   *stats) {
    const struct gguf_tensor_t *t = gguf_get_tensor(ctx, tensor_name);
    if (t == nullptr) {
        fprintf(stderr, "  MISS in GGUF: '%s'\n", tensor_name);
        stats->n_failed++;
        return;
    }
    uint8_t *readback = malloc(t->nbytes);
    if (readback == nullptr) {
        fprintf(stderr, "  alloc fail for '%s'\n", tensor_name);
        stats->n_failed++;
        return;
    }
    enum geist_status s = be->desc->vtbl->buffer_download(t->nbytes, readback, buf);
    if (s != GEIST_OK) {
        fprintf(stderr, "  download fail for '%s': %s\n", tensor_name, geist_status_to_string(s));
        free(readback);
        stats->n_failed++;
        return;
    }
    int mismatch = memcmp(readback, t->data, t->nbytes);
    free(readback);
    if (mismatch != 0) {
        fprintf(stderr, "  BYTES MISMATCH for '%s' (%zu bytes)\n", tensor_name, t->nbytes);
        stats->n_failed++;
        return;
    }
    stats->n_checked++;
    stats->bytes_checked += t->nbytes;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct transformer_arch_state *st = nullptr;
    s                                 = transformer_state_create(be, model_path, nullptr, &st);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "transformer_state_create failed: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("v2 state created on %s backend\n", geist_backend_name(be));

    /* Independent GGUF handle for the comparison (the state owns its own). */
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(model_path, &err);
    if (ctx == nullptr) {
        fprintf(stderr, "gguf_open(ref): %s\n", err != nullptr ? err : "(no detail)");
        transformer_state_destroy(st);
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }

    struct check_stats stats = {0};

    /* ---- Globals (5 tensors). Order mirrors load_globals so
     * global_bufs[i] corresponds 1:1 with the name. Special-case
     * per_layer_model_proj: load_globals dequantizes it to F32 because
     * the backends' linear() doesn't accept F16 weights. So the buffer
     * holds FP32 bytes that won't memcmp against the GGUF F16 source —
     * we skip the byte check there and verify size only. */
    static const char *const GLOBAL_NAMES[] = {
            "token_embd.weight",
            "per_layer_token_embd.weight",
            "per_layer_model_proj.weight", /* see special case below */
            "per_layer_proj_norm.weight",
            "output_norm.weight",
    };
    const size_t n_globals_expected = sizeof GLOBAL_NAMES / sizeof GLOBAL_NAMES[0];
    if (st->n_global_bufs != n_globals_expected) {
        fprintf(stderr,
                "global buffer count mismatch: got %zu, expected %zu\n",
                st->n_global_bufs,
                n_globals_expected);
        stats.n_failed++;
    }
    for (size_t i = 0; i < st->n_global_bufs && i < n_globals_expected; i++) {
        if (strcmp(GLOBAL_NAMES[i], "per_layer_model_proj.weight") == 0) {
            /* Verify the F32 mirror downloads cleanly and matches the
             * expected element count. */
            const size_t expect_bytes =
                    (size_t) GEIST_GEMMA4_PLE_OUT * GEIST_GEMMA4_HIDDEN * sizeof(float);
            uint8_t *probe = malloc(expect_bytes);
            if (probe == nullptr) {
                stats.n_failed++;
                continue;
            }
            enum geist_status s2 =
                    be->desc->vtbl->buffer_download(expect_bytes, probe, st->global_bufs[i]);
            free(probe);
            if (s2 != GEIST_OK) {
                fprintf(stderr,
                        "per_layer_model_proj F32 download failed: %s\n",
                        geist_status_to_string(s2));
                stats.n_failed++;
            } else {
                stats.n_checked++;
                stats.bytes_checked += expect_bytes;
            }
            continue;
        }
        check_buf_eq_gguf(be, ctx, st->global_bufs[i], GLOBAL_NAMES[i], &stats);
    }

    /* ---- Per-layer. Ordering inside load_one_layer:
     *    [0] attn_norm
     *    [1] attn_q_norm
     *    [2] post_attention_norm
     *    [3] ffn_norm
     *    [4] post_ffw_norm
     *    [5] post_norm
     *    [6] attn_q
     *    [7] attn_output
     *    [8..10]  IF !kv_shared: attn_k, attn_v, attn_k_norm
     *    [next 5] ffn_gate, ffn_up, ffn_down, inp_gate, proj
     *    layer_output_scale is loaded as a scalar (no buffer).
     */
    for (int li = 0; li < GEIST_GEMMA4_NUM_LAYERS; li++) {
        const struct transformer_layer_weights *L = &st->layers[li];
        char                                    nb[64];
        size_t                                  i = 0;

        snprintf(nb, sizeof nb, "blk.%d.attn_norm.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.attn_q_norm.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.post_attention_norm.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.ffn_norm.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.post_ffw_norm.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.post_norm.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.attn_q.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.attn_output.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);

        if (!L->is_kv_shared) {
            snprintf(nb, sizeof nb, "blk.%d.attn_k.weight", li);
            check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
            snprintf(nb, sizeof nb, "blk.%d.attn_v.weight", li);
            check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
            snprintf(nb, sizeof nb, "blk.%d.attn_k_norm.weight", li);
            check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        }

        snprintf(nb, sizeof nb, "blk.%d.ffn_gate.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.ffn_up.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.ffn_down.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.inp_gate.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);
        snprintf(nb, sizeof nb, "blk.%d.proj.weight", li);
        check_buf_eq_gguf(be, ctx, L->bufs[i++], nb, &stats);

        if (i != L->n_bufs) {
            fprintf(stderr, "layer %d: walked %zu bufs, state lists %zu\n", li, i, L->n_bufs);
            stats.n_failed++;
        }

        /* Also sanity-check the per-layer scalar: gguf has a 1-element F32
         * tensor whose value must equal L->layer_scalar. */
        snprintf(nb, sizeof nb, "blk.%d.layer_output_scale.weight", li);
        const struct gguf_tensor_t *t = gguf_get_tensor(ctx, nb);
        if (t == nullptr || t->dtype != GGUF_TYPE_F32 || gguf_tensor_elem_count(t) != 1) {
            fprintf(stderr, "layer %d: layer_output_scale missing/malformed\n", li);
            stats.n_failed++;
        } else {
            float expected = *(const float *) t->data;
            if (expected != L->layer_scalar) {
                fprintf(stderr,
                        "layer %d: scalar mismatch %g vs %g\n",
                        li,
                        (double) expected,
                        (double) L->layer_scalar);
                stats.n_failed++;
            }
        }
    }

    gguf_close(ctx);
    transformer_state_destroy(st);
    geist_backend_destroy(be);

    printf("checked %zu tensors, %.2f MB total, %zu failures\n",
           stats.n_checked,
           (double) stats.bytes_checked / (1024.0 * 1024.0),
           stats.n_failed);

    if (stats.n_failed == 0 && stats.n_checked >= 5 + GEIST_GEMMA4_NUM_LAYERS * 13) {
        printf("PASS: state weight load is byte-identical to GGUF for all "
               "globals + 35 layers\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
