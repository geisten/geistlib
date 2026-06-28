/*
 * test_transformer_block_via_vtable_int — runs ONE full transformer block
 * through pure backend->vtbl->* calls and verifies it produces the same
 * output as a hand-coded reference using the same kernels directly.
 *
 * The block topology mirrors Gemma 4 (simplified — no q/k_norm, no PLE):
 *
 *     x_in
 *      │
 *      ├──► rmsnorm ──► linear(W_q) ──► reshape ──► rope ─┐
 *      │       │                                          │
 *      │       └──────► linear(W_k) ──► reshape ──► rope ─┤
 *      │       │                                          │
 *      │       └──────► linear(W_v) ──► reshape ──────────┤
 *      │                                                  ▼
 *      │                                       attention(MQA, causal)
 *      │                                                  │
 *      │                                                  ▼
 *      │                                              linear(W_o)
 *      │                                                  │
 *      └──────────────────────► add (residual) ◄──────────┘
 *                                  │
 *                                  ▼
 *                              x_after_attn
 *                                  │
 *                                  ├──► rmsnorm ──► linear(W_gate) ──► gelu ─┐
 *                                  │       │                                │
 *                                  │       └──────► linear(W_up) ──► mul ◄──┘
 *                                  │                                  │
 *                                  │                                  ▼
 *                                  │                            linear(W_down)
 *                                  │                                  │
 *                                  └────────────► add (residual) ◄────┘
 *                                                  │
 *                                                  ▼
 *                                                 y_out
 *
 * Phase B-4e Step 3 verification: proves the backend op-vocab assembled in
 * Steps 1+2 is sufficient to express a full transformer block. Random F32
 * weights (no quantization, no GGUF) — exercises only the vtable.
 *
 * Cross-references the vtable path against a "reference path" that calls
 * the same gemma4_kernels.c / cblas_sgemm functions directly. Bit-identical
 * output expected (same underlying kernels, just different dispatch route).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "gemma4_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward-declare cblas to bypass Apple's vImage chain (same pattern as
 * the cpu_neon backend). */
typedef enum CBLAS_ORDER { CblasRowMajor = 101 } CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 } CBLAS_TRANSPOSE;
extern void cblas_sgemm(CBLAS_ORDER,
                        CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB,
                        int             M,
                        int             N,
                        int             K,
                        float           alpha,
                        const float    *A,
                        int             lda,
                        const float    *B,
                        int             ldb,
                        float           beta,
                        float          *C,
                        int             ldc);

/* Simplified Gemma-style transformer block dims (small for fast test).
 * Matches Gemma 4's MQA pattern: more Q heads than KV heads. */
#define D_MODEL 64
#define N_Q_HEADS 4
#define N_KV_HEADS 2
#define HEAD_DIM 16
#define D_FF 128

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

/* ---- Reference path: same kernels, no vtable ---- */
static void reference_block(const float *x_in,
                            const float *w_q,
                            const float *w_k,
                            const float *w_v,
                            const float *w_o,
                            const float *w_gate,
                            const float *w_up,
                            const float *w_down,
                            const float *w_attn_norm,
                            const float *w_ffn_norm,
                            float        eps,
                            float       *y_out) {
    float x_norm[D_MODEL];
    rmsnorm_fp32(x_in, w_attn_norm, 1, D_MODEL, eps, x_norm);

    /* Q/K/V projections: y = x @ W^T */
    float q_proj[N_Q_HEADS * HEAD_DIM];
    float k_proj[N_KV_HEADS * HEAD_DIM];
    float v_proj[N_KV_HEADS * HEAD_DIM];
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                1,
                N_Q_HEADS * HEAD_DIM,
                D_MODEL,
                1.0f,
                x_norm,
                D_MODEL,
                w_q,
                D_MODEL,
                0.0f,
                q_proj,
                N_Q_HEADS * HEAD_DIM);
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                1,
                N_KV_HEADS * HEAD_DIM,
                D_MODEL,
                1.0f,
                x_norm,
                D_MODEL,
                w_k,
                D_MODEL,
                0.0f,
                k_proj,
                N_KV_HEADS * HEAD_DIM);
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                1,
                N_KV_HEADS * HEAD_DIM,
                D_MODEL,
                1.0f,
                x_norm,
                D_MODEL,
                w_v,
                D_MODEL,
                0.0f,
                v_proj,
                N_KV_HEADS * HEAD_DIM);

    /* RoPE: precompute cos/sin for seq_len=1 starting at position 0. */
    float cos[HEAD_DIM], sin_[HEAD_DIM];
    rope_compute_at(0, 1, HEAD_DIM, HEAD_DIM, 10000.0f, cos, sin_);
    rope_apply(q_proj, cos, sin_, 1, N_Q_HEADS, HEAD_DIM);
    rope_apply(k_proj, cos, sin_, 1, N_KV_HEADS, HEAD_DIM);

    /* Attention with KV cache len=1 (single-token decode). */
    float attn_out[N_Q_HEADS * HEAD_DIM];
    attention_mqa_causal_kv(
            q_proj, k_proj, v_proj, 1, 1, 0, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, 0, attn_out);

    /* O projection. */
    float o_proj[D_MODEL];
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                1,
                D_MODEL,
                N_Q_HEADS * HEAD_DIM,
                1.0f,
                attn_out,
                N_Q_HEADS * HEAD_DIM,
                w_o,
                N_Q_HEADS * HEAD_DIM,
                0.0f,
                o_proj,
                D_MODEL);

    /* Residual. */
    float x_post_attn[D_MODEL];
    add_fp32(x_in, o_proj, D_MODEL, x_post_attn);

    /* FFN. */
    float ffn_norm[D_MODEL];
    rmsnorm_fp32(x_post_attn, w_ffn_norm, 1, D_MODEL, eps, ffn_norm);

    float gate[D_FF], up[D_FF];
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                1,
                D_FF,
                D_MODEL,
                1.0f,
                ffn_norm,
                D_MODEL,
                w_gate,
                D_MODEL,
                0.0f,
                gate,
                D_FF);
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                1,
                D_FF,
                D_MODEL,
                1.0f,
                ffn_norm,
                D_MODEL,
                w_up,
                D_MODEL,
                0.0f,
                up,
                D_FF);
    gelu_tanh_fp32(gate, D_FF, gate);
    mul_fp32(gate, up, D_FF, gate);

    float ffn_out[D_MODEL];
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                1,
                D_MODEL,
                D_FF,
                1.0f,
                gate,
                D_FF,
                w_down,
                D_FF,
                0.0f,
                ffn_out,
                D_MODEL);

    /* Residual. */
    add_fp32(x_post_attn, ffn_out, D_MODEL, y_out);
}

/* Helper to (re-)allocate + upload a tensor's contents. */
static struct geist_buffer *
alloc_and_upload(struct geist_backend *be, const float *data, size_t n) {
    struct geist_buffer *buf = nullptr;
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &buf);
    if (buf == nullptr)
        return nullptr;
    (void) be->desc->vtbl->buffer_upload(buf, n * sizeof(float), (const uint8_t *) data);
    return buf;
}

/* P2-final helper: resolve a F32 DENSE weight once. n_in/n_out follow the
 * row-major (n_out, n_in) layout the resolvers expect. */
static struct geist_weight
resolve_w_f32(struct geist_backend *be, struct geist_buffer *bw, int32_t n_in, int32_t n_out) {
    void               *w_host = be->desc->vtbl->buffer_map(bw);
    struct geist_weight wkr    = {
            .raw   = w_host,
            .n_in  = n_in,
            .n_out = n_out,
            .dtype = (uint16_t) GEIST_DTYPE_F32,
    };
    (void) be->desc->vtbl->resolve_weight(be, &wkr);
    be->desc->vtbl->buffer_unmap(bw);
    return wkr;
}

/* Dispatch a single-token linear via the resolver-installed kernel. */
static enum geist_status linear1_via_resolver(struct geist_backend      *be,
                                              struct geist_buffer       *bx,
                                              const struct geist_weight *wkr,
                                              struct geist_buffer       *by) {
    if (wkr->linear_m1 == nullptr)
        return GEIST_E_UNSUPPORTED;
    void *xh = be->desc->vtbl->buffer_map(bx);
    void *yh = be->desc->vtbl->buffer_map(by);
    wkr->linear_m1((const float *) xh, wkr, be, (float *) yh);
    be->desc->vtbl->buffer_unmap(bx);
    be->desc->vtbl->buffer_unmap(by);
    return GEIST_OK;
}

static struct geist_tensor
make_tensor(struct geist_buffer *buf, int ndim, int64_t s0, int64_t s1, int64_t s2) {
    struct geist_tensor t = {
            .buffer = buf,
            .offset = 0,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = ndim,
    };
    t.shape[0] = s0;
    if (ndim >= 2)
        t.shape[1] = s1;
    if (ndim >= 3)
        t.shape[2] = s2;
    /* DENSE row-major strides: rightmost = 1, leftward = product of inner shapes. */
    if (ndim == 1)
        t.stride[0] = 1;
    if (ndim == 2) {
        t.stride[0] = s1;
        t.stride[1] = 1;
    }
    if (ndim == 3) {
        t.stride[0] = s1 * s2;
        t.stride[1] = s2;
        t.stride[2] = 1;
    }
    return t;
}

/* ---- Vtable path: same math, via backend ops only ---- */
static int vtable_block_via_backend(const char  *backend_name,
                                    const float *x_in,
                                    const float *w_q,
                                    const float *w_k,
                                    const float *w_v,
                                    const float *w_o,
                                    const float *w_gate,
                                    const float *w_up,
                                    const float *w_down,
                                    const float *w_attn_norm,
                                    const float *w_ffn_norm,
                                    float        eps,
                                    float       *y_out) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;

    /* Upload all inputs + weights. */
    struct geist_buffer *bx      = alloc_and_upload(be, x_in, D_MODEL);
    struct geist_buffer *bwq     = alloc_and_upload(be, w_q, N_Q_HEADS * HEAD_DIM * D_MODEL);
    struct geist_buffer *bwk     = alloc_and_upload(be, w_k, N_KV_HEADS * HEAD_DIM * D_MODEL);
    struct geist_buffer *bwv     = alloc_and_upload(be, w_v, N_KV_HEADS * HEAD_DIM * D_MODEL);
    struct geist_buffer *bwo     = alloc_and_upload(be, w_o, D_MODEL * N_Q_HEADS * HEAD_DIM);
    struct geist_buffer *bw_gate = alloc_and_upload(be, w_gate, D_FF * D_MODEL);
    struct geist_buffer *bw_up   = alloc_and_upload(be, w_up, D_FF * D_MODEL);
    struct geist_buffer *bw_down = alloc_and_upload(be, w_down, D_MODEL * D_FF);
    struct geist_buffer *bw_an   = alloc_and_upload(be, w_attn_norm, D_MODEL);
    struct geist_buffer *bw_fn   = alloc_and_upload(be, w_ffn_norm, D_MODEL);

    /* Pre-allocate scratch buffers. */
    struct geist_buffer *b_xnorm = nullptr, *b_q = nullptr, *b_k = nullptr, *b_v = nullptr;
    struct geist_buffer *b_attn = nullptr, *b_o = nullptr, *b_post_attn = nullptr;
    struct geist_buffer *b_ffn_norm = nullptr, *b_gate = nullptr, *b_up = nullptr;
    struct geist_buffer *b_ffn_out = nullptr, *b_y = nullptr;
    (void) v->buffer_create(be, D_MODEL * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_xnorm);
    (void) v->buffer_create(
            be, N_Q_HEADS * HEAD_DIM * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_q);
    (void) v->buffer_create(
            be, N_KV_HEADS * HEAD_DIM * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_k);
    (void) v->buffer_create(
            be, N_KV_HEADS * HEAD_DIM * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_v);
    (void) v->buffer_create(
            be, N_Q_HEADS * HEAD_DIM * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_attn);
    (void) v->buffer_create(be, D_MODEL * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_o);
    (void) v->buffer_create(be, D_MODEL * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_post_attn);
    (void) v->buffer_create(be, D_MODEL * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_ffn_norm);
    (void) v->buffer_create(be, D_FF * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_gate);
    (void) v->buffer_create(be, D_FF * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_up);
    (void) v->buffer_create(be, D_MODEL * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_ffn_out);
    (void) v->buffer_create(be, D_MODEL * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &b_y);

    /* RoPE cos/sin tables — precomputed externally and uploaded. */
    float cos[HEAD_DIM], sin_[HEAD_DIM];
    rope_compute_at(0, 1, HEAD_DIM, HEAD_DIM, 10000.0f, cos, sin_);
    struct geist_buffer *b_cos = alloc_and_upload(be, cos, HEAD_DIM);
    struct geist_buffer *b_sin = alloc_and_upload(be, sin_, HEAD_DIM);

    /* Tensor descriptors for non-linear ops. P2-final: linear is no longer
     * a vtbl op — resolved weights below. */
    struct geist_tensor t_x     = make_tensor(bx, 1, D_MODEL, 0, 0);
    struct geist_tensor t_xnorm = make_tensor(b_xnorm, 1, D_MODEL, 0, 0);
    struct geist_tensor t_w_an  = make_tensor(bw_an, 1, D_MODEL, 0, 0);
    struct geist_tensor t_w_fn  = make_tensor(bw_fn, 1, D_MODEL, 0, 0);

    /* The rope op wants 3D [seq_len, n_heads, head_dim] views. We need an alias
     * onto the same buffer with reshaped descriptor. */
    struct geist_tensor t_q_rope = make_tensor(b_q, 3, 1, N_Q_HEADS, HEAD_DIM);
    struct geist_tensor t_k_rope = make_tensor(b_k, 3, 1, N_KV_HEADS, HEAD_DIM);

    struct geist_tensor t_cos = make_tensor(b_cos, 2, 1, HEAD_DIM, 0);
    struct geist_tensor t_sin = make_tensor(b_sin, 2, 1, HEAD_DIM, 0);

    struct geist_tensor t_v_3d    = make_tensor(b_v, 3, 1, N_KV_HEADS, HEAD_DIM);
    struct geist_tensor t_attn_3d = make_tensor(b_attn, 3, 1, N_Q_HEADS, HEAD_DIM);

    struct geist_tensor t_o_1d       = make_tensor(b_o, 1, D_MODEL, 0, 0);
    struct geist_tensor t_post_attn  = make_tensor(b_post_attn, 1, D_MODEL, 0, 0);
    struct geist_tensor t_ffn_norm   = make_tensor(b_ffn_norm, 1, D_MODEL, 0, 0);
    struct geist_tensor t_gate_1d    = make_tensor(b_gate, 1, D_FF, 0, 0);
    struct geist_tensor t_up_1d      = make_tensor(b_up, 1, D_FF, 0, 0);
    struct geist_tensor t_ffn_out_1d = make_tensor(b_ffn_out, 1, D_MODEL, 0, 0);
    struct geist_tensor t_y_1d       = make_tensor(b_y, 1, D_MODEL, 0, 0);

    /* Resolve every weight once. */
    struct geist_weight wkr_q    = resolve_w_f32(be, bwq, D_MODEL, N_Q_HEADS * HEAD_DIM);
    struct geist_weight wkr_k    = resolve_w_f32(be, bwk, D_MODEL, N_KV_HEADS * HEAD_DIM);
    struct geist_weight wkr_v    = resolve_w_f32(be, bwv, D_MODEL, N_KV_HEADS * HEAD_DIM);
    struct geist_weight wkr_o    = resolve_w_f32(be, bwo, N_Q_HEADS * HEAD_DIM, D_MODEL);
    struct geist_weight wkr_gate = resolve_w_f32(be, bw_gate, D_MODEL, D_FF);
    struct geist_weight wkr_up   = resolve_w_f32(be, bw_up, D_MODEL, D_FF);
    struct geist_weight wkr_down = resolve_w_f32(be, bw_down, D_FF, D_MODEL);

    enum geist_status s;

    /* Attention block. */
    s = v->rmsnorm(be, &t_x, &t_w_an, eps, &t_xnorm);
    if (s != GEIST_OK)
        goto cleanup;
    s = linear1_via_resolver(be, b_xnorm, &wkr_q, b_q);
    if (s != GEIST_OK)
        goto cleanup;
    s = linear1_via_resolver(be, b_xnorm, &wkr_k, b_k);
    if (s != GEIST_OK)
        goto cleanup;
    s = linear1_via_resolver(be, b_xnorm, &wkr_v, b_v);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->rope_apply(be, &t_q_rope, &t_cos, &t_sin);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->rope_apply(be, &t_k_rope, &t_cos, &t_sin);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->attention(be, &t_q_rope, &t_k_rope, &t_v_3d, 0, 0, &t_attn_3d);
    if (s != GEIST_OK)
        goto cleanup;
    s = linear1_via_resolver(be, b_attn, &wkr_o, b_o);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->add(be, &t_x, &t_o_1d, &t_post_attn);
    if (s != GEIST_OK)
        goto cleanup;

    /* FFN block. */
    s = v->rmsnorm(be, &t_post_attn, &t_w_fn, eps, &t_ffn_norm);
    if (s != GEIST_OK)
        goto cleanup;
    s = linear1_via_resolver(be, b_ffn_norm, &wkr_gate, b_gate);
    if (s != GEIST_OK)
        goto cleanup;
    s = linear1_via_resolver(be, b_ffn_norm, &wkr_up, b_up);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->gelu_tanh(be, &t_gate_1d, &t_gate_1d);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->mul(be, &t_gate_1d, &t_up_1d, &t_gate_1d);
    if (s != GEIST_OK)
        goto cleanup;
    s = linear1_via_resolver(be, b_gate, &wkr_down, b_ffn_out);
    if (s != GEIST_OK)
        goto cleanup;
    s = v->add(be, &t_post_attn, &t_ffn_out_1d, &t_y_1d);
    if (s != GEIST_OK)
        goto cleanup;

    /* Download result. */
    s = v->buffer_download(D_MODEL * sizeof(float), (uint8_t *) y_out, b_y);

cleanup:
    /* Free all buffers. */
    struct geist_buffer *all[] = {
            bx,          bwq,        bwk,     bwv,  bwo,       bw_gate, bw_up,  bw_down,
            bw_an,       bw_fn,      b_xnorm, b_q,  b_k,       b_v,     b_attn, b_o,
            b_post_attn, b_ffn_norm, b_gate,  b_up, b_ffn_out, b_y,     b_cos,  b_sin,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        if (all[i] != nullptr)
            v->buffer_destroy(be, all[i]);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

int main(void) {
    /* Allocate weights + input on stack (small dims = ~70 KB total). */
    float x_in[D_MODEL];
    float w_q[N_Q_HEADS * HEAD_DIM * D_MODEL];
    float w_k[N_KV_HEADS * HEAD_DIM * D_MODEL];
    float w_v[N_KV_HEADS * HEAD_DIM * D_MODEL];
    float w_o[D_MODEL * N_Q_HEADS * HEAD_DIM];
    float w_gate[D_FF * D_MODEL];
    float w_up[D_FF * D_MODEL];
    float w_down[D_MODEL * D_FF];
    float w_attn_norm[D_MODEL];
    float w_ffn_norm[D_MODEL];

    fill_random(x_in, D_MODEL, 0xCAFEBABEu);
    fill_random(w_q, N_Q_HEADS * HEAD_DIM * D_MODEL, 0x10001u);
    fill_random(w_k, N_KV_HEADS * HEAD_DIM * D_MODEL, 0x10002u);
    fill_random(w_v, N_KV_HEADS * HEAD_DIM * D_MODEL, 0x10003u);
    fill_random(w_o, D_MODEL * N_Q_HEADS * HEAD_DIM, 0x10004u);
    fill_random(w_gate, D_FF * D_MODEL, 0x10005u);
    fill_random(w_up, D_FF * D_MODEL, 0x10006u);
    fill_random(w_down, D_MODEL * D_FF, 0x10007u);
    /* Norm weights near 1.0 to avoid degenerate cases. */
    for (size_t i = 0; i < D_MODEL; i++) {
        w_attn_norm[i] = 1.0f + 0.05f * (float) i / D_MODEL;
        w_ffn_norm[i]  = 1.0f - 0.05f * (float) i / D_MODEL;
    }

    const float eps = 1e-6f;

    /* Reference path. */
    float y_ref[D_MODEL];
    reference_block(
            x_in, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, w_attn_norm, w_ffn_norm, eps, y_ref);

    /* Vtable path — both backends compiled in. cpu_neon is skipped when not
     * built (e.g. scalar-only x86 builds). */
    int fails = 0;
#if defined(GEIST_BACKEND_CPU_NEON) && GEIST_BACKEND_CPU_NEON
    const char *first_backend = "cpu_neon";
#else
    const char *first_backend = "cpu_scalar";
#endif
    for (const char *backend = first_backend; backend != nullptr;
         backend             = (strcmp(backend, "cpu_neon") == 0) ? "cpu_scalar" : nullptr) {
        float y_vtable[D_MODEL];
        if (vtable_block_via_backend(backend,
                                     x_in,
                                     w_q,
                                     w_k,
                                     w_v,
                                     w_o,
                                     w_gate,
                                     w_up,
                                     w_down,
                                     w_attn_norm,
                                     w_ffn_norm,
                                     eps,
                                     y_vtable) != 0) {
            fprintf(stderr, "vtable path via %s failed\n", backend);
            return GEIST_TEST_FAIL;
        }
        ptrdiff_t bad = geist_fp32_close_array(y_ref, y_vtable, D_MODEL, 1e-4f, 1e-5f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL %s: y[%td]=%f, ref[%td]=%f, diff=%g\n",
                    backend,
                    bad,
                    (double) y_vtable[bad],
                    bad,
                    (double) y_ref[bad],
                    (double) fabsf(y_vtable[bad] - y_ref[bad]));
            fails++;
        } else {
            printf("PASS: %s — full transformer block via vtable matches reference (%d outputs)\n",
                   backend,
                   D_MODEL);
        }
    }

    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
