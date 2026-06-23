/*
 * test_state_layer_fwd_int — Phase B-4e sub-step 2a verification.
 *
 * Drives one Gemma 4 layer through transformer_forward_one_layer (pure
 * backend->vtbl path) and checks the output against a direct-kernel
 * reference that calls gemma4_kernels.c primitives in the same order
 * lm.c::forward_layer_kv does. Both paths use the same dequantized FP32
 * mirrors of the layer's projection weights — the comparison isolates
 * the vtable-composition logic from kernel correctness (which has its
 * own cross-reference tests).
 *
 * Coverage in this commit:
 *   - layer 0 (sliding, non-shared, head_dim=256, window=512, theta=1e4)
 *   - layer 4 (full,    non-shared, head_dim=512, window=0,   theta=1e6)
 *   - layer 15 (sliding, KV-shared from layer 13) — primes layer 13's
 *     cache first, then runs layer 15 against the shared cache.
 *   - layer 19 (full, KV-shared from layer 14) — same pattern.
 * PLE input is set to zero so the PLE block contributes nothing and the
 * layer output reduces to h_post_ff * layer_scalar — same in both paths.
 *
 * Also exercises transformer_compute_per_layer_input: dequant the PLE
 * row, run the model_proj linear, scale, rmsnorm, add, and final scale —
 * compared against a host-side reference that uses the same kernels
 * directly. PLE-precompute is independent of per-layer math, so it gets
 * its own assertion.
 *
 * SKIPs cleanly if no GGUF is available.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/archs/transformer/arch_state.h"

#include "quant.h"
#include "gguf_reader.h"
#include "gemma4_kernels.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN GEIST_GEMMA4_HIDDEN
#define N_Q_HEADS GEIST_GEMMA4_N_Q_HEADS
#define N_KV_HEADS GEIST_GEMMA4_N_KV_HEADS
/* RMS eps moved to struct geist_arch_config (P1.4.a); the value is a
 * Gemma-4 family default. Test mirrors the hardcoded reference. */
#define RMS_EPS 1e-6f

/* Deterministic-seeded uniform random in [-0.5, 0.5]. */
static void fill_random(float *p, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        p[i] = (float) ((int32_t) s) * (1.0f / (float) INT32_MAX) * 0.5f;
    }
}

/* Read a 1D F32 tensor's bytes (norm weights) into a host buffer. */
static void
read_norm(struct geist_backend *be, const struct geist_tensor *t, float *dst, size_t expected) {
    size_t            bytes = expected * sizeof(float);
    enum geist_status s     = be->desc->vtbl->buffer_download(bytes, (uint8_t *) dst, t->buffer);
    if (s != GEIST_OK) {
        fprintf(stderr, "buffer_download (norm) failed: %s\n", geist_status_to_string(s));
        exit(GEIST_TEST_ERROR);
    }
}

/* Dequantize a 2D weight tensor's bytes (whatever GGUF dtype) to a host
 * FP32 buffer using the gguf_quant.c row helpers — same way lm.c builds
 * its FP32 mirror. n_out * n_in floats written. The caller frees. */
static float *dequant_proj_to_fp32(struct geist_backend      *be,
                                   const struct geist_tensor *t,
                                   size_t                     n_out,
                                   size_t                     n_in) {
    /* struct geist_buffer is backend-private; buffer_map gives us the
     * raw host pointer to its bytes. That's all we need for dequant. */
    const uint8_t *raw = (const uint8_t *) be->desc->vtbl->buffer_map(t->buffer);
    if (raw == nullptr) {
        fprintf(stderr, "buffer_map for projection weight failed\n");
        exit(GEIST_TEST_ERROR);
    }
    float *fp32 = aligned_alloc(64, n_out * n_in * sizeof(float));
    if (fp32 == nullptr) {
        exit(GEIST_TEST_ERROR);
    }

    /* The geist_tensor's dtype tells us which dequant helper to call. */
    switch (t->dtype) {
    case GEIST_DTYPE_F32:
        memcpy(fp32, raw, n_out * n_in * sizeof(float));
        break;
    case GEIST_DTYPE_Q3_K:
        for (size_t j = 0; j < n_out; j++) {
            dequant_q3_K_row(
                    raw + j * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES, fp32 + j * n_in, n_in);
        }
        break;
    case GEIST_DTYPE_Q4_K:
        for (size_t j = 0; j < n_out; j++) {
            dequant_q4_K_row(
                    raw + j * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES, fp32 + j * n_in, n_in);
        }
        break;
    case GEIST_DTYPE_Q5_K:
        for (size_t j = 0; j < n_out; j++) {
            dequant_q5_K_row(
                    raw + j * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, fp32 + j * n_in, n_in);
        }
        break;
    case GEIST_DTYPE_Q6_K:
        for (size_t j = 0; j < n_out; j++) {
            dequant_q6_K_row(
                    raw + j * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES, fp32 + j * n_in, n_in);
        }
        break;
    case GEIST_DTYPE_Q8_0:
        for (size_t j = 0; j < n_out; j++) {
            dequant_q8_0_row(
                    raw + j * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, fp32 + j * n_in, n_in);
        }
        break;
    default:
        fprintf(stderr, "unsupported projection dtype %d\n", (int) t->dtype);
        exit(GEIST_TEST_ERROR);
    }
    be->desc->vtbl->buffer_unmap(t->buffer);
    return fp32;
}

/* Reference path: forward_layer_kv equivalent, PLE-input=0. The math
 * mirrors lm.c forward_layer_kv exactly — same kernel call sequence,
 * same eps, same q_norm/k_norm/v_norm ordering. */
static void reference_layer_forward(
        /* Per-layer geometry. */
        bool   is_full,
        size_t head_dim,
        size_t q_out,
        size_t kv_out,
        size_t intermediate,
        size_t sliding_window,
        float  rope_theta,
        size_t n_rotated_dims,
        float  layer_scalar,
        /* Weights (FP32 dequantized mirrors). */
        const float *attn_norm,
        const float *q_proj,
        const float *k_proj,
        const float *v_proj,
        const float *o_proj,
        const float *q_norm,
        const float *k_norm,
        const float *post_attn_norm,
        const float *ffn_norm,
        const float *gate_proj,
        const float *up_proj,
        const float *down_proj,
        const float *post_ffw_norm,
        /* Activations. */
        const float *h_in,
        size_t       q_position,
        float       *h_out) {

    /* h_pre = h_in copy. */
    float h_pre[HIDDEN];
    memcpy(h_pre, h_in, sizeof(h_pre));

    /* 1. rmsnorm(h_in, attn_norm) → normed */
    float normed[HIDDEN];
    rmsnorm_fp32(h_pre, attn_norm, 1, HIDDEN, RMS_EPS, normed);

    /* 2. Q/K/V projections (1 × HIDDEN) × (n_out × HIDDEN)^T → (1 × n_out). */
    float *q   = aligned_alloc(64, q_out * sizeof(float));
    float *k   = aligned_alloc(64, kv_out * sizeof(float));
    float *vbu = aligned_alloc(64, kv_out * sizeof(float));
    linear_fp32(normed, q_proj, nullptr, 1, HIDDEN, q_out, q);
    linear_fp32(normed, k_proj, nullptr, 1, HIDDEN, kv_out, k);
    linear_fp32(normed, v_proj, nullptr, 1, HIDDEN, kv_out, vbu);

    /* 3-5. q_norm, rope(q), k_norm, v_norm, rope(k). */
    rmsnorm_fp32(q, q_norm, N_Q_HEADS, head_dim, RMS_EPS, q);
    float *cos_b = aligned_alloc(64, head_dim * sizeof(float));
    float *sin_b = aligned_alloc(64, head_dim * sizeof(float));
    rope_compute_at(q_position, 1, head_dim, n_rotated_dims, rope_theta, cos_b, sin_b);
    rope_apply(q, cos_b, sin_b, 1, N_Q_HEADS, head_dim);
    rmsnorm_fp32(k, k_norm, N_KV_HEADS, head_dim, RMS_EPS, k);
    rmsnorm_fp32(vbu, nullptr, N_KV_HEADS, head_dim, RMS_EPS, vbu);
    rope_apply(k, cos_b, sin_b, 1, N_KV_HEADS, head_dim);

    /* 6-7. Build a fresh single-slot KV cache and run attention. */
    const size_t kv_len = q_position + 1;
    float *k_cache = aligned_alloc(64, kv_len * (size_t) N_KV_HEADS * head_dim * sizeof(float));
    float *v_cache = aligned_alloc(64, kv_len * (size_t) N_KV_HEADS * head_dim * sizeof(float));
    /* For this test we only care about the position q_position contributing
     * to its own output. Fill earlier slots with zeros so they don't perturb
     * softmax — but causal+window masking already ensures only s <= q_position
     * is read. */
    memset(k_cache, 0, kv_len * (size_t) N_KV_HEADS * head_dim * sizeof(float));
    memset(v_cache, 0, kv_len * (size_t) N_KV_HEADS * head_dim * sizeof(float));
    memcpy(k_cache + q_position * (size_t) N_KV_HEADS * head_dim,
           k,
           (size_t) N_KV_HEADS * head_dim * sizeof(float));
    memcpy(v_cache + q_position * (size_t) N_KV_HEADS * head_dim,
           vbu,
           (size_t) N_KV_HEADS * head_dim * sizeof(float));

    float *attn_out = aligned_alloc(64, q_out * sizeof(float));
    attention_mqa_causal_kv(q,
                            k_cache,
                            v_cache,
                            1,
                            kv_len,
                            q_position,
                            N_Q_HEADS,
                            N_KV_HEADS,
                            head_dim,
                            sliding_window,
                            attn_out);

    /* 8. O projection. */
    float o[HIDDEN];
    linear_fp32(attn_out, o_proj, nullptr, 1, q_out, HIDDEN, o);

    /* 9. post_attn_norm + residual. */
    float post_attn[HIDDEN], h_post_attn[HIDDEN];
    rmsnorm_fp32(o, post_attn_norm, 1, HIDDEN, RMS_EPS, post_attn);
    add_fp32(h_pre, post_attn, HIDDEN, h_post_attn);

    /* 10. ffn_norm + gate/up + GeGLU + down. */
    float pre_ff[HIDDEN];
    rmsnorm_fp32(h_post_attn, ffn_norm, 1, HIDDEN, RMS_EPS, pre_ff);
    float *gate = aligned_alloc(64, intermediate * sizeof(float));
    float *up   = aligned_alloc(64, intermediate * sizeof(float));
    linear_fp32(pre_ff, gate_proj, nullptr, 1, HIDDEN, intermediate, gate);
    linear_fp32(pre_ff, up_proj, nullptr, 1, HIDDEN, intermediate, up);
    gelu_tanh_fp32(gate, intermediate, gate);
    mul_fp32(gate, up, intermediate, gate);
    float ffn_out[HIDDEN];
    linear_fp32(gate, down_proj, nullptr, 1, intermediate, HIDDEN, ffn_out);

    /* 11. post_ffw_norm + residual. */
    float post_ff[HIDDEN], h_post_ff[HIDDEN];
    rmsnorm_fp32(ffn_out, post_ffw_norm, 1, HIDDEN, RMS_EPS, post_ff);
    add_fp32(h_post_attn, post_ff, HIDDEN, h_post_ff);

    /* 12. PLE skipped (per_layer_input = 0). h_out = h_post_ff. */
    /* 13. Per-layer scalar. */
    for (size_t i = 0; i < HIDDEN; i++) {
        h_out[i] = h_post_ff[i] * layer_scalar;
    }

    free(q);
    free(k);
    free(vbu);
    free(cos_b);
    free(sin_b);
    free(k_cache);
    free(v_cache);
    free(attn_out);
    free(gate);
    free(up);
    (void) is_full;
}

/* Run one layer through transformer_forward_one_layer and download h_out. */
static int run_layer(struct transformer_arch_state *st,
                     int                            layer_idx,
                     size_t                         q_position,
                     const float                   *h_in_host,
                     float                         *h_out_host) {
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    struct geist_buffer *h_in_buf  = nullptr;
    struct geist_buffer *h_out_buf = nullptr;
    enum geist_status    s         = v->buffer_create(
            be, HIDDEN * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &h_in_buf);
    if (s != GEIST_OK) {
        return -1;
    }
    s = v->buffer_create(
            be, HIDDEN * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &h_out_buf);
    if (s != GEIST_OK) {
        v->buffer_destroy(be, h_in_buf);
        return -1;
    }
    s = v->buffer_upload(h_in_buf, HIDDEN * sizeof(float), (const uint8_t *) h_in_host);
    if (s != GEIST_OK) {
        goto cleanup;
    }

    s = transformer_forward_one_layer(st,
                                      layer_idx,
                                      q_position,
                                      /* seq = */ 1,
                                      /* advance_kv = */ false,
                                      h_in_buf,
                                      /* per_layer_input = */ nullptr,
                                      h_out_buf);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "transformer_forward_one_layer (L%d) failed: %s — %s\n",
                layer_idx,
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        goto cleanup;
    }
    s = v->buffer_download(HIDDEN * sizeof(float), (uint8_t *) h_out_host, h_out_buf);

cleanup:
    v->buffer_destroy(be, h_in_buf);
    v->buffer_destroy(be, h_out_buf);
    return s == GEIST_OK ? 0 : -1;
}

static int check_one_layer(struct transformer_arch_state *st, int layer_idx) {
    struct geist_backend                   *be = st->backend;
    const struct transformer_layer_weights *L  = &st->layers[layer_idx];

    if (L->is_kv_shared) {
        printf("layer %d: skip (kv-shared not yet implemented in v2)\n", layer_idx);
        return 0;
    }

    /* Load FP32 mirrors of every weight by downloading + dequantizing. */
    float *attn_norm      = aligned_alloc(64, HIDDEN * sizeof(float));
    float *post_attn_norm = aligned_alloc(64, HIDDEN * sizeof(float));
    float *ffn_norm       = aligned_alloc(64, HIDDEN * sizeof(float));
    float *post_ffw_norm  = aligned_alloc(64, HIDDEN * sizeof(float));
    float *q_norm         = aligned_alloc(64, L->head_dim * sizeof(float));
    float *k_norm         = aligned_alloc(64, L->head_dim * sizeof(float));
    read_norm(be, &L->attn_norm, attn_norm, HIDDEN);
    read_norm(be, &L->post_attn_norm, post_attn_norm, HIDDEN);
    read_norm(be, &L->ffn_norm, ffn_norm, HIDDEN);
    read_norm(be, &L->post_ffw_norm, post_ffw_norm, HIDDEN);
    read_norm(be, &L->q_norm, q_norm, L->head_dim);
    read_norm(be, &L->k_norm, k_norm, L->head_dim);

    float *q_proj    = dequant_proj_to_fp32(be, &L->q_proj, L->q_out, HIDDEN);
    float *k_proj    = dequant_proj_to_fp32(be, &L->k_proj, L->kv_out, HIDDEN);
    float *v_proj    = dequant_proj_to_fp32(be, &L->v_proj, L->kv_out, HIDDEN);
    float *o_proj    = dequant_proj_to_fp32(be, &L->o_proj, HIDDEN, L->q_out);
    float *gate_proj = dequant_proj_to_fp32(be, &L->gate_proj, L->intermediate, HIDDEN);
    float *up_proj   = dequant_proj_to_fp32(be, &L->up_proj, L->intermediate, HIDDEN);
    float *down_proj = dequant_proj_to_fp32(be, &L->down_proj, HIDDEN, L->intermediate);

    /* Random input residual. q_position = layer_idx (arbitrary nonzero). */
    float h_in[HIDDEN];
    fill_random(h_in, HIDDEN, 0xCAFEBABEu ^ (uint32_t) layer_idx);
    const size_t q_position = (size_t) layer_idx;

    /* Reference path. */
    float h_ref[HIDDEN];
    reference_layer_forward(L->is_full,
                            L->head_dim,
                            L->q_out,
                            L->kv_out,
                            L->intermediate,
                            L->sliding_window,
                            L->rope_theta,
                            (size_t) L->n_rotated_dims,
                            L->layer_scalar,
                            attn_norm,
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            q_norm,
                            k_norm,
                            post_attn_norm,
                            ffn_norm,
                            gate_proj,
                            up_proj,
                            down_proj,
                            post_ffw_norm,
                            h_in,
                            q_position,
                            h_ref);

    /* Vtable path. */
    float h_vtable[HIDDEN];
    if (run_layer(st, layer_idx, q_position, h_in, h_vtable) != 0) {
        return 1;
    }

    /* Compare two *different* lossy projection paths: the reference dequants
     * (dequant_qX_K_row → cblas sgemv), the vtable's cpu_neon backend uses the
     * W4A8-style INT8 matmul for Q4_K. They can diverge by more than either's
     * own error from the fp32 truth, and the gap is platform-dependent (Apple
     * clang/Accelerate vs Linux gcc/OpenBLAS, both with -ffast-math): observed
     * up to ~1e-2 on a single intermediate. atol=2e-2 = 2× the documented W4A8
     * budget (geist_fp32_close), with headroom so this isn't a flaky cross-
     * platform gate; the end-to-end token test is the real correctness check. */
    ptrdiff_t bad   = geist_fp32_close_array(h_ref, h_vtable, HIDDEN, 1e-3f, 2e-2f);
    int       fails = 0;
    if (bad >= 0) {
        fprintf(stderr,
                "layer %d FAIL: idx %td ref=%g vtable=%g diff=%g\n",
                layer_idx,
                bad,
                (double) h_ref[bad],
                (double) h_vtable[bad],
                (double) fabsf(h_ref[bad] - h_vtable[bad]));
        /* Diagnostic: print a few neighbors. */
        size_t lo = bad > 4 ? (size_t) bad - 4 : 0;
        size_t hi = lo + 9 < HIDDEN ? lo + 9 : HIDDEN;
        for (size_t i = lo; i < hi; i++) {
            fprintf(stderr,
                    "    [%zu] ref=%.6f vtable=%.6f diff=%g\n",
                    i,
                    (double) h_ref[i],
                    (double) h_vtable[i],
                    (double) fabsf(h_ref[i] - h_vtable[i]));
        }
        fails = 1;
    } else {
        printf("layer %d PASS (head_dim=%zu, %s, window=%zu, scalar=%g)\n",
               layer_idx,
               L->head_dim,
               L->is_full ? "full" : "sliding",
               L->sliding_window,
               (double) L->layer_scalar);
    }

    free(attn_norm);
    free(post_attn_norm);
    free(ffn_norm);
    free(post_ffw_norm);
    free(q_norm);
    free(k_norm);
    free(q_proj);
    free(k_proj);
    free(v_proj);
    free(o_proj);
    free(gate_proj);
    free(up_proj);
    free(down_proj);
    return fails;
}

/* ---- KV-shared layer smoke test --------------------------------------
 *
 * A full numerical cross-reference for shared layers requires running the
 * source layer first (to populate the cache) in both the reference and
 * vtable paths, capturing K/V from the source layer, then verifying that
 * the shared layer's attention reuses those K/V correctly. That depends
 * on the layer-loop (sub-step 2c) for the source-layer priming step.
 *
 * For now we run a smoke test: prime layer 13's cache via the vtable
 * (already verified for non-shared layers), then run layer 15 (kv-shared
 * sliding, sourcing from 13). Confirm the call returns GEIST_OK, produces
 * finite values, and has a non-zero magnitude (catches the obvious
 * regressions like reading the wrong cache or skipping the K/V plumbing
 * entirely). Full cross-ref happens at the end-to-end token gate. */
static int check_kv_shared_layer_smoke(struct transformer_arch_state *st,
                                       int                            src_layer_idx,
                                       int                            shared_layer_idx) {
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    struct geist_buffer *h_src_buf = nullptr, *h_src_out_buf = nullptr;
    struct geist_buffer *h_shared_buf = nullptr, *h_shared_out_buf = nullptr;
    enum geist_status    s;
    s = v->buffer_create(
            be, HIDDEN * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &h_src_buf);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->buffer_create(
            be, HIDDEN * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &h_src_out_buf);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->buffer_create(
            be, HIDDEN * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &h_shared_buf);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->buffer_create(be,
                         HIDDEN * sizeof(float),
                         GEIST_BUFFER_ACTIVATION,
                         GEIST_MEMORY_AUTO,
                         &h_shared_out_buf);
    if (s != GEIST_OK)
        goto cleanup;

    float h_src[HIDDEN], h_shared[HIDDEN];
    fill_random(h_src, HIDDEN, 0x13371337u);
    fill_random(h_shared, HIDDEN, 0x42424242u);
    s = v->buffer_upload(h_src_buf, HIDDEN * sizeof(float), (const uint8_t *) h_src);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->buffer_upload(h_shared_buf, HIDDEN * sizeof(float), (const uint8_t *) h_shared);
    if (s != GEIST_OK)
        goto cleanup;

    const size_t q_position = (size_t) shared_layer_idx;

    /* Prime the source layer's KV cache at q_position. */
    s = transformer_forward_one_layer(
            st, src_layer_idx, q_position, 1, false, h_src_buf, nullptr, h_src_out_buf);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "kv-shared smoke: priming src L%d failed: %s — %s\n",
                src_layer_idx,
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        goto cleanup;
    }

    /* Run the shared layer. */
    s = transformer_forward_one_layer(
            st, shared_layer_idx, q_position, 1, false, h_shared_buf, nullptr, h_shared_out_buf);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "kv-shared smoke: shared L%d failed: %s — %s\n",
                shared_layer_idx,
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        goto cleanup;
    }

    /* Check finite + non-trivial. */
    float h_out[HIDDEN];
    s = v->buffer_download(HIDDEN * sizeof(float), (uint8_t *) h_out, h_shared_out_buf);
    if (s != GEIST_OK)
        goto cleanup;
    int    n_finite = 0;
    double abs_sum  = 0.0;
    for (size_t i = 0; i < HIDDEN; i++) {
        if (isfinite(h_out[i])) {
            n_finite++;
            abs_sum += fabs((double) h_out[i]);
        }
    }
    if (n_finite != HIDDEN || abs_sum < 1e-6) {
        fprintf(stderr,
                "kv-shared L%d smoke FAIL: n_finite=%d/%zu, abs_sum=%g\n",
                shared_layer_idx,
                n_finite,
                (size_t) HIDDEN,
                abs_sum);
        s = GEIST_E_INTERNAL;
    } else {
        printf("layer %d kv-shared smoke PASS (src=L%d, mean|h_out|=%.4g)\n",
               shared_layer_idx,
               src_layer_idx,
               abs_sum / HIDDEN);
    }

cleanup:
    if (h_src_buf)
        v->buffer_destroy(be, h_src_buf);
    if (h_src_out_buf)
        v->buffer_destroy(be, h_src_out_buf);
    if (h_shared_buf)
        v->buffer_destroy(be, h_shared_buf);
    if (h_shared_out_buf)
        v->buffer_destroy(be, h_shared_out_buf);
    return s == GEIST_OK ? 0 : 1;
}

/* ---- PLE precompute cross-reference ---------------------------------- */

/* Host-side reference for transformer_compute_per_layer_input. */
static void reference_ple_precompute(const float               *h,
                                     geist_token_t              token_id,
                                     const struct geist_tensor *ple_table,
                                     const float               *ple_lookup_host_row,
                                     const float               *model_proj_host,
                                     const float               *model_proj_norm_host,
                                     float                     *per_layer_input_out) {
    (void) ple_table;
    (void) token_id;
    /* ple_lookup_host_row is the dequantized ple_table[token_id, :]
     * (8960 floats) — the caller staged it because the dequant helpers
     * aren't directly importable here. */
    float *ple_proj = aligned_alloc(64, GEIST_GEMMA4_PLE_OUT * sizeof(float));
    linear_fp32(
            h, model_proj_host, nullptr, 1, GEIST_GEMMA4_HIDDEN, GEIST_GEMMA4_PLE_OUT, ple_proj);
    for (size_t i = 0; i < (size_t) GEIST_GEMMA4_PLE_OUT; i++) {
        ple_proj[i] *= 0.02551551815399144f;
    }
    rmsnorm_fp32(ple_proj,
                 model_proj_norm_host,
                 GEIST_GEMMA4_NUM_LAYERS,
                 GEIST_GEMMA4_HIDDEN_PER_LAYER,
                 RMS_EPS,
                 ple_proj);
    for (size_t i = 0; i < (size_t) GEIST_GEMMA4_PLE_OUT; i++) {
        per_layer_input_out[i] =
                (ple_proj[i] + ple_lookup_host_row[i] * 16.0f) * 0.7071067811865476f;
    }
    free(ple_proj);
}

static int check_ple_precompute(struct transformer_arch_state *st, geist_token_t token_id) {
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    /* Random residual stream. */
    float h_in[HIDDEN];
    fill_random(h_in, HIDDEN, 0xDEADBEEFu);

    /* Upload h to a buffer. */
    struct geist_buffer *h_buf = nullptr, *ple_buf = nullptr;
    enum geist_status    s = v->buffer_create(
            be, HIDDEN * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &h_buf);
    if (s != GEIST_OK)
        return 1;
    s = v->buffer_create(be,
                         GEIST_GEMMA4_PLE_OUT * sizeof(float),
                         GEIST_BUFFER_ACTIVATION,
                         GEIST_MEMORY_AUTO,
                         &ple_buf);
    if (s != GEIST_OK) {
        v->buffer_destroy(be, h_buf);
        return 1;
    }
    s = v->buffer_upload(h_buf, HIDDEN * sizeof(float), (const uint8_t *) h_in);
    if (s != GEIST_OK)
        goto cleanup;

    /* Vtable path. */
    s = transformer_compute_per_layer_input(st, token_id, h_buf, ple_buf);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "transformer_compute_per_layer_input failed: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        goto cleanup;
    }
    float *ple_vtable = aligned_alloc(64, GEIST_GEMMA4_PLE_OUT * sizeof(float));
    s = v->buffer_download(GEIST_GEMMA4_PLE_OUT * sizeof(float), (uint8_t *) ple_vtable, ple_buf);
    if (s != GEIST_OK) {
        free(ple_vtable);
        goto cleanup;
    }

    /* Reference path: dequant the PLE row + FP32 mirrors of model_proj and
     * model_proj_norm, then compute the same expression. */
    float *ple_lookup_row = aligned_alloc(64, GEIST_GEMMA4_PLE_OUT * sizeof(float));
    float *model_proj_fp32 =
            dequant_proj_to_fp32(be, &st->model_proj, GEIST_GEMMA4_PLE_OUT, GEIST_GEMMA4_HIDDEN);
    float model_proj_norm[GEIST_GEMMA4_HIDDEN_PER_LAYER];
    read_norm(be, &st->model_proj_norm, model_proj_norm, GEIST_GEMMA4_HIDDEN_PER_LAYER);

    /* Dequant the PLE table row using the same helper used internally —
     * mirror that logic in dequant_proj_to_fp32 indirectly by calling the
     * row dequant on the raw bytes. */
    {
        const uint8_t *raw     = (const uint8_t *) v->buffer_map(st->ple_table.buffer);
        const size_t   row_idx = (size_t) token_id;
        const size_t   n_in    = (size_t) GEIST_GEMMA4_PLE_OUT;
        switch (st->ple_table.dtype) {
        case GEIST_DTYPE_F32:
            memcpy(ple_lookup_row, raw + row_idx * n_in * sizeof(float), n_in * sizeof(float));
            break;
        case GEIST_DTYPE_Q3_K:
            dequant_q3_K_row(raw + row_idx * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES,
                             ple_lookup_row,
                             n_in);
            break;
        case GEIST_DTYPE_Q4_K:
            dequant_q4_K_row(raw + row_idx * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES,
                             ple_lookup_row,
                             n_in);
            break;
        case GEIST_DTYPE_Q5_K:
            dequant_q5_K_row(raw + row_idx * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES,
                             ple_lookup_row,
                             n_in);
            break;
        case GEIST_DTYPE_Q6_K:
            dequant_q6_K_row(raw + row_idx * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES,
                             ple_lookup_row,
                             n_in);
            break;
        default:
            fprintf(stderr, "PLE table dtype %d unsupported in test\n", (int) st->ple_table.dtype);
            v->buffer_unmap(st->ple_table.buffer);
            free(ple_vtable);
            free(ple_lookup_row);
            free(model_proj_fp32);
            goto cleanup;
        }
        v->buffer_unmap(st->ple_table.buffer);
    }

    float *ple_ref = aligned_alloc(64, GEIST_GEMMA4_PLE_OUT * sizeof(float));
    reference_ple_precompute(h_in,
                             token_id,
                             &st->ple_table,
                             ple_lookup_row,
                             model_proj_fp32,
                             model_proj_norm,
                             ple_ref);

    /* Compare. */
    ptrdiff_t bad = geist_fp32_close_array(ple_ref, ple_vtable, GEIST_GEMMA4_PLE_OUT, 1e-3f, 1e-3f);
    int       rc  = 0;
    if (bad >= 0) {
        fprintf(stderr,
                "PLE precompute FAIL: idx %td ref=%g vtable=%g\n",
                bad,
                (double) ple_ref[bad],
                (double) ple_vtable[bad]);
        rc = 1;
    } else {
        printf("PLE precompute PASS (token_id=%d, %d outputs matched)\n",
               (int) token_id,
               GEIST_GEMMA4_PLE_OUT);
    }

    free(ple_vtable);
    free(ple_lookup_row);
    free(model_proj_fp32);
    free(ple_ref);
    v->buffer_destroy(be, h_buf);
    v->buffer_destroy(be, ple_buf);
    return rc;

cleanup:
    v->buffer_destroy(be, h_buf);
    v->buffer_destroy(be, ple_buf);
    return 1;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    /* For numerical comparison stability, prefer cpu_scalar (its kernels
     * match the reference more closely; cpu_neon's W4A8 quantized GEMV has
     * its own larger numerical envelope that the per-kernel cross-ref
     * tests already gate at rtol=1e-6). */
    enum geist_status s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct transformer_arch_state *st = nullptr;
    s                                 = transformer_state_create(be, model_path, nullptr, &st);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "state_create failed: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    int fails = 0;
    /* Layer 0: sliding, non-shared (head_dim 256, theta 1e4, window 512). */
    fails += check_one_layer(st, 0);
    /* Layer 4: full, non-shared (head_dim 512, theta 1e6, window 0). */
    fails += check_one_layer(st, 4);
    /* Layer 15 + 19: KV-shared smoke (uses caches from layer 13 and 14). */
    fails += check_kv_shared_layer_smoke(st, 13, 15);
    fails += check_kv_shared_layer_smoke(st, 14, 19);

    /* PLE precompute cross-ref. */
    fails += check_ple_precompute(st, /* token_id = */ 9259);

    transformer_state_destroy(st);
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: all sub-step 2 layer/PLE checks\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
