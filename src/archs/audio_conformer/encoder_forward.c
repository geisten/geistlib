/*
 * src/archs/audio_conformer/encoder_forward.c — the Conformer forward stages.
 *
 * Layer: ARCHITECTURE (audio_conformer). Split from the former
 * monolithic audio_encoder.c; pure moves, no behavior change.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "encoder_internal.h"

#include "audio_linear.h"

/* Quantized matmuls live in audio_linear.c, bound at load time from the
 * hardware probe (#236) — no compile-time ISA selection here. */

/* ----------------------- Conformer per-stage helpers ----------------------- */

/* Apply struct ClippableLinear: y = clamp(linear(clamp(x, in_min, in_max)), out_min, out_max).
 * x: (n, in_dim), y: (n, out_dim). x is mutated by the input-clamp pass. */
void clip_linear_apply(const struct ClippableLinear *cl,
                       float                        *x,
                       size_t                        n,
                       size_t                        in_dim,
                       size_t                        out_dim,
                       float                        *y) {
    clamp_fp32(x, n * in_dim, cl->input_min, cl->input_max);

    const enum audio_linear_prec   prec = cl->prec;
    const struct audio_linear_ops *ops  = audio_linear_bind();
    switch (prec) {
    case AUDIO_PREC_W8A8: {
        /* Static activation scale derived from clip bounds applied above. */
        float amax    = fmaxf(fabsf(cl->input_min), fabsf(cl->input_max));
        float scale_x = amax / 127.0f;
        if (scale_x == 0.0f)
            scale_x = 1.0f;
        ops->w8a8(cl->w_q8, cl->w_scales, x, 1.0f / scale_x, scale_x, n, in_dim, out_dim, y);
        break;
    }
    case AUDIO_PREC_W8A32:
        ops->w8a32(cl->w_q8, cl->w_scales, x, n, in_dim, out_dim, y);
        break;
    default:
        linear_fp32(x, cl->w, nullptr, n, in_dim, out_dim, y);
        break;
    }
    clamp_fp32(y, n * out_dim, cl->output_min, cl->output_max);
}

/* Macaron-style struct FFN: y = norm_post(SiLU(norm_pre(x) @ W1) @ W2) * 0.5 + x */
void ffn_run(const struct FFN *ffn, float *h, size_t n) {
    const size_t hsize    = (size_t) n * AUDIO_HIDDEN;
    float       *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h, hsize * sizeof(float));

    rmsnorm_fp32(h, ffn->pre_norm, n, AUDIO_HIDDEN, RMS_EPS, h);

    float *mid = heap_alloc_array_aligned(float, (size_t) n *FFN_INTER);
    clip_linear_apply(&ffn->ffw1, h, n, AUDIO_HIDDEN, FFN_INTER, mid);
    silu_fp32(mid, (size_t) n * FFN_INTER);
    clip_linear_apply(&ffn->ffw2, mid, n, FFN_INTER, AUDIO_HIDDEN, h);
    safe_free((void **) &mid);

    rmsnorm_fp32(h, ffn->post_norm, n, AUDIO_HIDDEN, RMS_EPS, h);
    for (size_t i = 0; i < hsize; i++)
        h[i] = h[i] * 0.5f + residual[i];
    safe_free((void **) &residual);
}

/* dot_head_fp32 / axpy_head_fp32 / zero_head_fp32 live in audio_kernels.c
 * with the other elementwise kernels (#236). */

/* Chunked self-attention with relative position bias. Reads (n, 1024) hidden,
 * pos_emb (13, 1024), attn_mask (num_blocks, 12, 24) bool. Writes (n, 1024).
 *
 * Per HF Gemma4AudioAttention.forward — naive C, per-block, per-head loops.
 * Performance is not critical: 1024 hidden × 8 heads × ≤2-4 blocks per chunk. */
void attn_run(const struct Attn *attn,
              const float       *h,
              size_t             n,
              const float       *pos_emb,
              const bool        *attn_mask,
              float             *y) {
    const float q_scale = (1.0f / sqrtf((float) HEAD_DIM)) / logf(2.0f);
    /* M_E is glibc-specific (not in C23); pin the constant for portability. */
    const float k_scale    = log1pf(2.71828182845904523536f) / logf(2.0f); /* log(1+e) / log(2) */
    const int   num_blocks = ((int) n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const int   n_padded   = num_blocks * CHUNK_SIZE;

    /* 1. Q/K/V projections (with input/output clip). */
    float *h_clip_in = heap_alloc_array_aligned(float, n *AUDIO_HIDDEN);
    float *q         = heap_calloc_array_aligned(float, (size_t) n_padded *AUDIO_HIDDEN);
    float *k         = heap_alloc_array_aligned(float, (size_t) n *AUDIO_HIDDEN);
    float *v         = heap_alloc_array_aligned(float, (size_t) n *AUDIO_HIDDEN);

    memcpy(h_clip_in, h, n * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->q_proj, h_clip_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, q);
    memcpy(h_clip_in, h, n * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->k_proj, h_clip_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, k);
    memcpy(h_clip_in, h, n * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->v_proj, h_clip_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, v);
    safe_free((void **) &h_clip_in);

    /* 2. Apply scales: q *= q_scale * softplus(per_dim_scale), k *= k_scale.
     *    Layout: q is (n_padded, n_heads, head_dim). Q-pad rows beyond n stay zero.
     *    Fold q_scale into the per-dim table so the inner loop is one mul/lane. */
    float q_pds[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        q_pds[d] = q_scale * log1pf(expf(attn->per_dim_scale[d]));
    }
    for (int t = 0; t < (int) n; t++) {
        for (int head_i = 0; head_i < N_HEADS; head_i++) {
            float *qrow = q + ((size_t) t * N_HEADS + head_i) * HEAD_DIM;
            /* Plain elementwise mul — the compiler vectorizes this at -O3;
             * the former hand-NEON added nothing (#236). */
            for (int d = 0; d < HEAD_DIM; d++)
                qrow[d] *= q_pds[d];
        }
    }
    for (size_t i = 0; i < n * AUDIO_HIDDEN; i++)
        k[i] *= k_scale;

    /* 3. Build context-windowed K/V: (num_blocks, CONTEXT_SIZE=24, n_heads, head_dim).
     *    HF: F.pad(K, (0,0, 0,0, max_past=12, chunk-1=11)), then unfold(size=24, stride=12).
     *    Block b's window covers padded source positions [b*12 .. b*12 + 23]
     *    where the source is K with 12 left zeros + n original + (chunk-1) right zeros.
     *    Mapped back: src_idx = (b*12 + ctx_pos) - 12; valid if src_idx in [0, n). */
    const size_t hd_per_t = (size_t) N_HEADS * HEAD_DIM;
    float *k_ctx = heap_calloc_array_aligned(float, (size_t) num_blocks * CONTEXT_SIZE * hd_per_t);
    float *v_ctx = heap_calloc_array_aligned(float, (size_t) num_blocks * CONTEXT_SIZE * hd_per_t);
    for (int b = 0; b < num_blocks; b++) {
        for (int c = 0; c < CONTEXT_SIZE; c++) {
            int src = b * CHUNK_SIZE + c - MAX_PAST_HORIZON;
            if (src < 0 || src >= (int) n)
                continue;
            memcpy(k_ctx + ((size_t) b * CONTEXT_SIZE + c) * hd_per_t,
                   k + (size_t) src * hd_per_t,
                   hd_per_t * sizeof(float));
            memcpy(v_ctx + ((size_t) b * CONTEXT_SIZE + c) * hd_per_t,
                   v + (size_t) src * hd_per_t,
                   hd_per_t * sizeof(float));
        }
    }
    safe_free((void **) &k);
    safe_free((void **) &v);

    /* 4. Relative-K projection: (POS_LEN=13, 1024) → (POS_LEN, 1024) → reshape (POS_LEN, 8, 128).
     */
    float *rel_k = heap_alloc_array_aligned(float, (size_t) POS_LEN *AUDIO_HIDDEN);
    linear_fp32(
            pos_emb, attn->relative_k_proj, nullptr, POS_LEN, AUDIO_HIDDEN, AUDIO_HIDDEN, rel_k);

    /* Output of attention before post-proj: (n_padded, 1024). */
    float *attn_out = heap_calloc_array_aligned(float, (size_t) n_padded *AUDIO_HIDDEN);

    /* Per-block, per-head attention. (b, hd) pairs are independent — parallelize.
     * Scratch arrays are declared inside the body so each thread has its own. */
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int b = 0; b < num_blocks; b++) {
        for (int hd = 0; hd < N_HEADS; hd++) {
            float scores_ac[CHUNK_SIZE * CONTEXT_SIZE];
            float scores_bd[CHUNK_SIZE * CONTEXT_SIZE];
            float bd_padded[CHUNK_SIZE * (CONTEXT_SIZE + 1)];
            /* Q rows for this block/head: (CHUNK_SIZE, HEAD_DIM). */
            const float *q_bh = q + ((size_t) b * CHUNK_SIZE) * hd_per_t + (size_t) hd * HEAD_DIM;
            /* K rows: (CONTEXT_SIZE, HEAD_DIM). */
            const float *k_bh =
                    k_ctx + ((size_t) b * CONTEXT_SIZE) * hd_per_t + (size_t) hd * HEAD_DIM;
            /* V rows: same layout as K. */
            const float *v_bh =
                    v_ctx + ((size_t) b * CONTEXT_SIZE) * hd_per_t + (size_t) hd * HEAD_DIM;
            /* Relative-K rows: (POS_LEN, HEAD_DIM). */
            const float *rk_h = rel_k + (size_t) hd * HEAD_DIM;
            /* Note: rel_k layout is (POS_LEN, N_HEADS, HEAD_DIM) flat as (POS_LEN, 1024).
             * To pick head `hd`, stride POS_LEN times by AUDIO_HIDDEN. */

            /* matrix_ac[i, j] = sum_d q[i, d] * k[j, d]   for i in chunk, j in context. */
            for (int i = 0; i < CHUNK_SIZE; i++) {
                const float *qi = q_bh + (size_t) i * hd_per_t;
                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    const float *kj                 = k_bh + (size_t) j * hd_per_t;
                    scores_ac[i * CONTEXT_SIZE + j] = dot_head_fp32(qi, kj);
                }
            }

            /* matrix_bd[i, p] = sum_d q[i, d] * rel_k[p, d]   for i in chunk, p in pos_len.
             * Then rel_shift to (CHUNK_SIZE, CONTEXT_SIZE). */
            for (int i = 0; i < CHUNK_SIZE; i++) {
                const float *qi = q_bh + (size_t) i * hd_per_t;
                for (int p = 0; p < POS_LEN; p++) {
                    const float *rp                       = rk_h + (size_t) p * AUDIO_HIDDEN;
                    bd_padded[i * (CONTEXT_SIZE + 1) + p] = dot_head_fp32(qi, rp);
                }
                /* zero-pad the rest of this row (positions POS_LEN..CONTEXT_SIZE) */
                for (int p = POS_LEN; p <= CONTEXT_SIZE; p++) {
                    bd_padded[i * (CONTEXT_SIZE + 1) + p] = 0.0f;
                }
            }
            /* rel_shift: flatten (CHUNK_SIZE, CONTEXT_SIZE+1) → CHUNK_SIZE * (CONTEXT_SIZE+1)
             * then take the first CHUNK_SIZE * CONTEXT_SIZE entries, view as
             * (CHUNK_SIZE, CONTEXT_SIZE). */
            for (int idx = 0; idx < CHUNK_SIZE * CONTEXT_SIZE; idx++) {
                scores_bd[idx] = bd_padded[idx];
            }

            /* Combine + soft-cap + mask + softmax. */
            for (int i = 0; i < CHUNK_SIZE; i++) {
                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    float s = scores_ac[i * CONTEXT_SIZE + j] + scores_bd[i * CONTEXT_SIZE + j];
                    s       = tanhf(s / ATTN_SOFTCAP) * ATTN_SOFTCAP;
                    if (attn_mask && !attn_mask[((size_t) b * CHUNK_SIZE + i) * CONTEXT_SIZE + j]) {
                        s = -1e9f;
                    }
                    scores_ac[i * CONTEXT_SIZE + j] = s; /* reuse scores_ac as combined */
                }
            }
            softmax_fp32(scores_ac, CHUNK_SIZE, CONTEXT_SIZE);

            /* attn @ V → (CHUNK_SIZE, HEAD_DIM) into attn_out at this block/head slot. */
            for (int i = 0; i < CHUNK_SIZE; i++) {
                const int t_out = b * CHUNK_SIZE + i;
                if (t_out >= n_padded)
                    break;
                float *out_row = attn_out + (size_t) t_out * AUDIO_HIDDEN + (size_t) hd * HEAD_DIM;
                zero_head_fp32(out_row);
                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    const float  w  = scores_ac[i * CONTEXT_SIZE + j];
                    const float *vj = v_bh + (size_t) j * hd_per_t;
                    axpy_head_fp32(out_row, w, vj);
                }
            }
        }
    }
    safe_free((void **) &rel_k);
    safe_free((void **) &k_ctx);
    safe_free((void **) &v_ctx);
    safe_free((void **) &q);

    /* 5. Crop attn_out to first n rows, post-proj. */
    float *tmp_in = heap_alloc_array_aligned(float, n *AUDIO_HIDDEN);
    memcpy(tmp_in, attn_out, n * AUDIO_HIDDEN * sizeof(float));
    safe_free((void **) &attn_out);
    clip_linear_apply(&attn->post, tmp_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, y);
    safe_free((void **) &tmp_in);
}

/* LightConv1D: norm → linear_start (×2) → GLU → causal depthwise conv → conv_norm → SiLU →
 * linear_end → +residual */
void lconv_run(const struct LConv *lc, float *h, size_t n) {
    const size_t hsize    = (size_t) n * AUDIO_HIDDEN;
    float       *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h, hsize * sizeof(float));

    rmsnorm_fp32(h, lc->pre_norm, n, AUDIO_HIDDEN, RMS_EPS, h);

    /* linear_start: (n, 1024) → (n, 2048) */
    float *doubled = heap_alloc_array_aligned(float, (size_t) n * 2 * AUDIO_HIDDEN);
    clip_linear_apply(&lc->linear_start, h, n, AUDIO_HIDDEN, 2 * AUDIO_HIDDEN, doubled);

    /* GLU: (n, 2048) → (n, 1024) */
    glu_fp32(doubled, n, AUDIO_HIDDEN, h);
    safe_free((void **) &doubled);

    /* Depthwise conv1d: input is (n, 1024). HF transposes (B, T, C) → (B, C, T),
     * runs conv, transposes back. We use (C, T) layout for the conv directly. */
    float *hct = heap_alloc_array_aligned(float, hsize);
    for (size_t t = 0; t < n; t++)
        for (int c = 0; c < AUDIO_HIDDEN; c++)
            hct[(size_t) c * n + t] = h[(size_t) t * AUDIO_HIDDEN + c];

    float *conv_out = heap_alloc_array_aligned(float, hsize);
    depthwise_conv1d_causal_fp32(hct, lc->depthwise, conv_out, AUDIO_HIDDEN, (int) n, CONV_KERNEL);
    safe_free((void **) &hct);

    /* Transpose back (C, T) → (T, C) */
    for (size_t t = 0; t < n; t++)
        for (int c = 0; c < AUDIO_HIDDEN; c++)
            h[(size_t) t * AUDIO_HIDDEN + c] = conv_out[(size_t) c * n + t];
    safe_free((void **) &conv_out);

    /* conv_norm + SiLU + linear_end + residual */
    rmsnorm_fp32(h, lc->conv_norm, n, AUDIO_HIDDEN, RMS_EPS, h);
    silu_fp32(h, hsize);
    float *end_in = heap_alloc_array_aligned(float, hsize);
    memcpy(end_in, h, hsize * sizeof(float));
    clip_linear_apply(&lc->linear_end, end_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, h);
    safe_free((void **) &end_in);

    for (size_t i = 0; i < hsize; i++)
        h[i] += residual[i];
    safe_free((void **) &residual);
}

void audio_encoder_layer_run(const struct AudioEncoder *a,
                             int                        layer_idx,
                             const float               *h_in,
                             size_t                     n,
                             const float               *pos_emb,
                             const bool                *attn_mask_5d,
                             float                     *h_out) {
    const struct ConformerLayer *L     = &a->layers[layer_idx];
    const size_t                 hsize = (size_t) n * AUDIO_HIDDEN;

    float *h = heap_alloc_array_aligned(float, hsize);
    memcpy(h, h_in, hsize * sizeof(float));

    /* 1. feed_forward1 (with internal residual + 0.5 scale) */
    {
        AE_TIC();
        ffn_run(&L->ff1, h, n);
        AE_TOC(g_ae_ffn1);
    }

    /* 2. residual + norm_pre_attn + self_attn + norm_post_attn + residual */
    float *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h, hsize * sizeof(float));
    {
        AE_TIC();
        rmsnorm_fp32(h, L->norm_pre_attn, n, AUDIO_HIDDEN, RMS_EPS, h);
        AE_TOC(g_ae_norm_pre_attn);
    }

    float *attn_out = heap_alloc_array_aligned(float, hsize);
    {
        AE_TIC();
        attn_run(&L->attn, h, n, pos_emb, attn_mask_5d, attn_out);
        AE_TOC(g_ae_attn);
    }
    memcpy(h, attn_out, hsize * sizeof(float));
    safe_free((void **) &attn_out);

    {
        AE_TIC();
        rmsnorm_fp32(h, L->norm_post_attn, n, AUDIO_HIDDEN, RMS_EPS, h);
        AE_TOC(g_ae_norm_post_attn);
    }
    for (size_t i = 0; i < hsize; i++)
        h[i] += residual[i];
    safe_free((void **) &residual);

    /* 3. lconv1d (internal residual) */
    {
        AE_TIC();
        lconv_run(&L->lconv, h, n);
        AE_TOC(g_ae_lconv);
    }

    /* 4. feed_forward2 (internal residual) */
    {
        AE_TIC();
        ffn_run(&L->ff2, h, n);
        AE_TOC(g_ae_ffn2);
    }

    /* 5. norm_out */
    {
        AE_TIC();
        rmsnorm_fp32(h, L->norm_out, n, AUDIO_HIDDEN, RMS_EPS, h_out);
        AE_TOC(g_ae_norm_out);
    }

    safe_free((void **) &h);
}

float *audio_encoder_compute_pos_emb(const struct AudioEncoder *a) {
    (void) a;
    /* Sinusoidal relative position embedding: (POS_LEN, 1024) with
     * position_ids = [POS_LEN-1, POS_LEN-2, ..., 0]. */
    float      *out            = heap_alloc_array_aligned(float, (size_t) POS_LEN *AUDIO_HIDDEN);
    const int   num_timescales = AUDIO_HIDDEN / 2;
    const float log_inc        = logf(10000.0f) / (float) (num_timescales - 1);
    for (int p = 0; p < POS_LEN; p++) {
        const int pos_id = POS_LEN - 1 - p; /* arange(context_size//2, -1, -1) */
        for (int i = 0; i < num_timescales; i++) {
            const float inv_ts                                  = expf((float) i * -log_inc);
            const float scaled                                  = (float) pos_id * inv_ts;
            out[(size_t) p * AUDIO_HIDDEN + i]                  = sinf(scaled);
            out[(size_t) p * AUDIO_HIDDEN + num_timescales + i] = cosf(scaled);
        }
    }
    return out;
}

bool *audio_encoder_compute_attn_mask(const struct AudioEncoder *a, size_t n) {
    (void) a;
    /* For each query at absolute position q_g = b*CHUNK + i, the visible keys
     * are k_g in [q_g - MAX_PAST_HORIZON, q_g + MAX_FUTURE] ∩ [0, n).
     * Context position c maps to k_g = b*CHUNK + c - MAX_PAST_HORIZON. */
    const int num_blocks = ((int) n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    bool     *m = heap_calloc_array_aligned(bool, (size_t) num_blocks * CHUNK_SIZE * CONTEXT_SIZE);
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < CHUNK_SIZE; i++) {
            const int q_g = b * CHUNK_SIZE + i;
            if (q_g >= (int) n)
                continue; /* invalid query row → mask all False */
            for (int c = 0; c < CONTEXT_SIZE; c++) {
                const int k_g = b * CHUNK_SIZE + c - MAX_PAST_HORIZON;
                if (k_g < 0 || k_g >= (int) n)
                    continue;
                const int off = k_g - q_g;
                /* sliding_window_mask_function((max_past_horizon, max_future)) — the
                 * effective window is `max_past_horizon` positions including the
                 * current one, i.e., off in [-(max_past_horizon-1), max_future]. */
                if (off < -(MAX_PAST_HORIZON - 1) || off > MAX_FUTURE)
                    continue;
                m[((size_t) b * CHUNK_SIZE + i) * CONTEXT_SIZE + c] = true;
            }
        }
    }
    return m;
}

/* HF subsample, B=1:
 *   conv_a(x_NCHW): (1, c_in, H, W) → (1, c_out, H_out, W_out)
 *   then permute (B,C,H,W) → (B,H,W,C), LayerNorm-ws over last axis (C),
 *   permute back → (B,C,H,W), ReLU.
 *
 * For our single-batch case we keep buffers in (C, H, W) layout, which is
 * what conv2d_fp32 expects directly. Permute-to-LayerNorm-and-back is
 * implemented by walking strides without an actual buffer transpose. */
/* Subsample stage = conv2d + LayerNorm (HWC over C) + ReLU.
 * Default start_h=0 runs the whole h-axis (used by audio_encoder_run).
 * The Phase-3 streaming path passes start_h > 0 to skip already-computed
 * h positions; conv2d outputs at h < start_h must already be in `out`
 * (cached from a previous call). */
static void subsample_layer_from(const float *w_conv,
                                 const float *w_norm,
                                 const float *in,
                                 int          c_in,
                                 int          h_in,
                                 int          w_in,
                                 int          c_out,
                                 float       *out,
                                 int         *h_out_p,
                                 int         *w_out_p,
                                 int          start_h) {
    const int kh = 3, kw = 3, sh = 2, sw = 2, ph = 1, pw = 1;
    const int h_out = (h_in + 2 * ph - kh) / sh + 1;
    const int w_out = (w_in + 2 * pw - kw) / sw + 1;
    *h_out_p        = h_out;
    *w_out_p        = w_out;
    if (start_h < 0)
        start_h = 0;
    if (start_h >= h_out)
        return;

    /* 1: conv → (c_out, h_out, w_out), only oh >= start_h written. */
    conv2d_fp32_from(in, w_conv, out, c_in, c_out, h_in, w_in, kh, kw, sh, sw, ph, pw, start_h);

    /* 2+3: LayerNorm over channel axis (c_out), per (h_out * w_out) "pixel".
     * Buffer is (c_out, h_out, w_out) — channels outer. LN normalises along
     * the C axis per (h, w) pixel, independent of other pixels. So we can
     * restrict to new h positions only. */
    const int    new_h     = h_out - start_h;
    const size_t n_pix_new = (size_t) new_h * w_out;
    float       *tmp       = heap_alloc_array_aligned(float, n_pix_new *(size_t) c_out);
    /* CHW(new slice) → HWC */
    for (int h = 0; h < new_h; h++) {
        for (int wpos = 0; wpos < w_out; wpos++) {
            for (int c = 0; c < c_out; c++) {
                tmp[((size_t) h * w_out + wpos) * c_out + c] =
                        out[((size_t) c * h_out + (start_h + h)) * w_out + wpos];
            }
        }
    }
    layernorm_fp32_ws(tmp, w_norm, n_pix_new, c_out, LN_EPS, tmp);
    /* HWC → CHW(new slice) */
    for (int h = 0; h < new_h; h++) {
        for (int wpos = 0; wpos < w_out; wpos++) {
            for (int c = 0; c < c_out; c++) {
                out[((size_t) c * h_out + (start_h + h)) * w_out + wpos] =
                        tmp[((size_t) h * w_out + wpos) * c_out + c];
            }
        }
    }
    safe_free((void **) &tmp);

    /* 4: ReLU on new h positions only. */
    for (int c = 0; c < c_out; c++) {
        float *slot = out + ((size_t) c * h_out + start_h) * w_out;
        for (int i = 0; i < new_h * w_out; i++) {
            if (slot[i] < 0.0f)
                slot[i] = 0.0f;
        }
    }
}

/* Thin wrapper for the existing call sites — full h_out range. */
static void subsample_layer(const float *w_conv,
                            const float *w_norm,
                            const float *in,
                            int          c_in,
                            int          h_in,
                            int          w_in,
                            int          c_out,
                            float       *out,
                            int         *h_out_p,
                            int         *w_out_p) {
    subsample_layer_from(w_conv, w_norm, in, c_in, h_in, w_in, c_out, out, h_out_p, w_out_p, 0);
}

/* Zero out time-positions in a (C, T, W) buffer where mask[t] == false. */
static void apply_time_mask(float *buf, const bool *mask, int c, int t, int w) {
    if (!mask)
        return;
    const size_t plane = (size_t) t * w;
    for (int ti = 0; ti < t; ti++) {
        if (mask[ti])
            continue;
        for (int ci = 0; ci < c; ci++) {
            float *row = buf + (size_t) ci * plane + (size_t) ti * w;
            for (int wi = 0; wi < w; wi++)
                row[wi] = 0.0f;
        }
    }
}

size_t audio_encoder_subsample_run(const struct AudioEncoder *a,
                                   const float               *mel_in,
                                   const bool                *mask_in,
                                   size_t                     n_mel_frames,
                                   float                     *out) {
    /* Treat (n_mel_frames, 128) as a (1, T, 128) image. In our (C, H, W)
     * scratch convention: c_in=1, h_in=T, w_in=128. */
    const int T = (int) n_mel_frames;
    const int W = MEL_DIM;

    /* Optional pre-conv-0 mask: HF multiplies the input by mask before conv.
     * We need a writable copy if a mask is provided. */
    const float *in_for_conv0 = mel_in;
    float       *mel_masked   = nullptr;
    if (mask_in) {
        mel_masked = heap_alloc_array_aligned(float, (size_t) T *W);
        memcpy(mel_masked, mel_in, (size_t) T * W * sizeof(float));
        apply_time_mask(mel_masked, mask_in, 1, T, W);
        in_for_conv0 = mel_masked;
    }

    /* Layer 0 output buffer: (128, T_out0, W_out0). */
    const int T_out0 = (T + 2 - 3) / 2 + 1;
    const int W_out0 = (W + 2 - 3) / 2 + 1;
    float    *l0     = heap_alloc_array_aligned(float, (size_t) 128 * T_out0 * W_out0);
    int       hh, ww;
    subsample_layer(a->l0_conv, a->l0_norm, in_for_conv0, 1, T, W, 128, l0, &hh, &ww);
    safe_free((void **) &mel_masked);

    /* Mask after layer 0 is mask[::2]. Apply before layer 1 conv. */
    if (mask_in) {
        bool *mask1 = heap_alloc_array_aligned(bool, (size_t) hh);
        for (int i = 0; i < hh; i++)
            mask1[i] = (i * 2 < T) ? mask_in[i * 2] : false;
        apply_time_mask(l0, mask1, 128, hh, ww);
        safe_free((void **) &mask1);
    }

    /* Layer 1 output: (32, T_out1, W_out1). */
    const int T_out1 = (hh + 2 - 3) / 2 + 1;
    const int W_out1 = (ww + 2 - 3) / 2 + 1;
    float    *l1     = heap_alloc_array_aligned(float, (size_t) 32 * T_out1 * W_out1);
    int       hh2, ww2;
    subsample_layer(a->l1_conv, a->l1_norm, l0, 128, hh, ww, 32, l1, &hh2, &ww2);
    safe_free((void **) &l0);

    /* Reshape (32, T_out1, 32) → (T_out1, 32, 32) → (T_out1, 1024).
     * HF: permute(0, 2, 3, 1).reshape(B, T_out1, -1)
     * The permute makes channels the LAST axis, then reshape concatenates
     * (W * C) per time step. So per (t, w, c) → flat[t * (W*C) + w * C + c]. */
    const int proj_in_dim = W_out1 * 32; /* = 32 * 32 = 1024 */
    float    *flat        = heap_alloc_array_aligned(float, (size_t) T_out1 *proj_in_dim);
    for (int t = 0; t < T_out1; t++) {
        for (int w = 0; w < W_out1; w++) {
            for (int c = 0; c < 32; c++) {
                flat[(size_t) t * proj_in_dim + w * 32 + c] =
                        l1[((size_t) c * T_out1 + t) * W_out1 + w];
            }
        }
    }
    safe_free((void **) &l1);

    /* Linear projection: (T_out1, 1024) @ in_proj^T (1024, 1024) → (T_out1, 1024). */
    linear_fp32(flat, a->in_proj, nullptr, (size_t) T_out1, AUDIO_HIDDEN, AUDIO_HIDDEN, out);
    safe_free((void **) &flat);

    return (size_t) T_out1;
}

/* Phase-3 incremental subsample. Extends the cached `subs->l0` and
 * `subs->l1` buffers with new time positions and projects the new l1
 * rows into `out_sub_buf` at the right offset. Returns the new total
 * number of sub-tokens (== T_out1 for the current n_mel_total).
 *
 * Conv2d outputs are stable for the time positions already computed
 * (kernel taps are at deterministic offsets from oh and the input
 * grows only at the right edge), so the old cached values for h <
 * subs->n_t_out0 stay correct. The Phase-3 `_from` family writes only
 * new h positions. */
size_t audio_encoder_subsample_run_inc(const struct AudioEncoder *a,
                                       struct subs_cache         *subs,
                                       const float               *mel_in,
                                       const bool                *mask_in,
                                       size_t                     n_mel_total,
                                       float                     *out_sub_buf) {
    if (n_mel_total <= subs->n_mel_seen)
        return subs->n_t_out1;

    const int T          = (int) n_mel_total;
    const int W          = MEL_DIM;
    const int T_out0_new = (T + 2 - 3) / 2 + 1;
    const int W_out0     = (W + 2 - 3) / 2 + 1;
    const int T_out1_new = (T_out0_new + 2 - 3) / 2 + 1;
    const int W_out1     = (W_out0 + 2 - 3) / 2 + 1;

    if ((size_t) T_out0_new > SUBS_T_OUT0_CAP || (size_t) T_out1_new > SUBS_T_OUT1_CAP) {
        /* Fell off the cache cap - shouldn't happen for ≤30 s audio, but
         * fall back to a full re-run as a safety valve. */
        return audio_encoder_subsample_run(a, mel_in, mask_in, n_mel_total, out_sub_buf);
    }

    /* Optional mask. Apply on the entire mel (cheap - just zeroes rows). */
    const float *in_for_conv0 = mel_in;
    float       *mel_masked   = nullptr;
    if (mask_in) {
        mel_masked = heap_alloc_array_aligned(float, (size_t) T *W);
        memcpy(mel_masked, mel_in, (size_t) T * W * sizeof(float));
        apply_time_mask(mel_masked, mask_in, 1, T, W);
        in_for_conv0 = mel_masked;
    }

    /* l0 is dimensioned (128, SUBS_T_OUT0_CAP, 64). conv2d_fp32_from writes
     * outputs at oh ∈ [start_h, T_out0_new). The actual storage stride is
     * SUBS_T_OUT0_CAP, but conv2d expects stride == h_out_new. We resolve
     * by passing a per-channel sliced view = NOT possible with the
     * existing kernel signature. Workaround: write to a temporary buffer
     * sized (128, T_out0_new, 64) for the conv pass, then copy the *new*
     * h rows into the cache.
     *
     * The temporary allocation is bounded by T_out0_new * 128 * 64 * 4 B
     * = at most ~7 MB for 1 s of new audio. Heap-arena reuse keeps this
     * cheap compared to a re-run that would compute ALL h positions. */
    const int new_h0 = T_out0_new - (int) subs->n_t_out0;
    float    *l0_tmp =
            heap_alloc_array_aligned(float, (size_t) SUBS_L0_CHANNELS * T_out0_new * W_out0);
    /* Copy old l0 rows into the temp buffer at h=[0, n_t_out0). */
    for (int c = 0; c < SUBS_L0_CHANNELS; c++) {
        for (size_t h = 0; h < subs->n_t_out0; h++) {
            memcpy(l0_tmp + ((size_t) c * T_out0_new + h) * W_out0,
                   subs->l0 + ((size_t) c * SUBS_T_OUT0_CAP + h) * W_out0,
                   (size_t) W_out0 * sizeof(float));
        }
    }
    int hh, ww;
    subsample_layer_from(a->l0_conv,
                         a->l0_norm,
                         in_for_conv0,
                         1,
                         T,
                         W,
                         SUBS_L0_CHANNELS,
                         l0_tmp,
                         &hh,
                         &ww,
                         (int) subs->n_t_out0);
    safe_free((void **) &mel_masked);

    /* Mask on l0[::2] before conv1. Apply only to NEW rows. */
    if (mask_in) {
        bool *mask1 = heap_alloc_array_aligned(bool, (size_t) hh);
        for (int i = 0; i < hh; i++)
            mask1[i] = (i * 2 < T) ? mask_in[i * 2] : false;
        /* apply_time_mask runs on the full (C, hh, ww) buffer - the new rows
         * we just wrote are at the right end and any unchanged old rows we
         * also re-mask are bit-equivalent to their cached values. */
        apply_time_mask(l0_tmp, mask1, SUBS_L0_CHANNELS, hh, ww);
        safe_free((void **) &mask1);
    }

    /* Copy new l0 rows back into the cache. */
    for (int c = 0; c < SUBS_L0_CHANNELS; c++) {
        for (int h = (int) subs->n_t_out0; h < hh; h++) {
            memcpy(subs->l0 + ((size_t) c * SUBS_T_OUT0_CAP + h) * W_out0,
                   l0_tmp + ((size_t) c * T_out0_new + h) * W_out0,
                   (size_t) W_out0 * sizeof(float));
        }
    }

    /* Same trick for l1: gather full l0 slice, run subsample_layer_from
     * starting at the new T_out1 boundary, then store new rows. */
    float *l1_tmp =
            heap_alloc_array_aligned(float, (size_t) SUBS_L1_CHANNELS * T_out1_new * W_out1);
    /* Old l1 rows. */
    for (int c = 0; c < SUBS_L1_CHANNELS; c++) {
        for (size_t h = 0; h < subs->n_t_out1; h++) {
            memcpy(l1_tmp + ((size_t) c * T_out1_new + h) * W_out1,
                   subs->l1 + ((size_t) c * SUBS_T_OUT1_CAP + h) * W_out1,
                   (size_t) W_out1 * sizeof(float));
        }
    }
    int hh2, ww2;
    subsample_layer_from(a->l1_conv,
                         a->l1_norm,
                         l0_tmp,
                         SUBS_L0_CHANNELS,
                         hh,
                         ww,
                         SUBS_L1_CHANNELS,
                         l1_tmp,
                         &hh2,
                         &ww2,
                         (int) subs->n_t_out1);
    safe_free((void **) &l0_tmp);

    /* Copy new l1 rows into cache. */
    for (int c = 0; c < SUBS_L1_CHANNELS; c++) {
        for (int h = (int) subs->n_t_out1; h < hh2; h++) {
            memcpy(subs->l1 + ((size_t) c * SUBS_T_OUT1_CAP + h) * W_out1,
                   l1_tmp + ((size_t) c * T_out1_new + h) * W_out1,
                   (size_t) W_out1 * sizeof(float));
        }
    }
    (void) new_h0; /* not directly used after the boundary calc above */

    /* Reshape + project NEW l1 rows only into out_sub_buf at the right
     * offset, leaving older sub_buf positions untouched. */
    const int new_t1      = hh2 - (int) subs->n_t_out1;
    const int proj_in_dim = W_out1 * SUBS_L1_CHANNELS;
    float    *flat        = heap_alloc_array_aligned(float, (size_t) new_t1 *proj_in_dim);
    for (int t = 0; t < new_t1; t++) {
        const int abs_t = (int) subs->n_t_out1 + t;
        for (int wp = 0; wp < W_out1; wp++) {
            for (int c = 0; c < SUBS_L1_CHANNELS; c++) {
                flat[(size_t) t * proj_in_dim + wp * SUBS_L1_CHANNELS + c] =
                        l1_tmp[((size_t) c * T_out1_new + abs_t) * W_out1 + wp];
            }
        }
    }
    safe_free((void **) &l1_tmp);

    linear_fp32(flat,
                a->in_proj,
                nullptr,
                (size_t) new_t1,
                AUDIO_HIDDEN,
                AUDIO_HIDDEN,
                out_sub_buf + (size_t) subs->n_t_out1 * AUDIO_HIDDEN);
    safe_free((void **) &flat);

    /* Boundary-stability: with kernel=3, stride=2, pad=1, the last output
     * position reads input[2p+1] which falls in the zero-pad region when
     * h_in is odd. That last position will get DIFFERENT (correct) values
     * once h_in extends past 2p+1, so we must not cache it here - the
     * next push starts conv2d at this boundary and overwrites it. For
     * even h_in all output positions are stable. */
    const size_t stable_l0 = (T % 2 == 0) ? (size_t) hh : (size_t) (hh - 1);
    const size_t stable_l1 = (hh % 2 == 0) ? (size_t) hh2 : (size_t) (hh2 - 1);
    subs->n_t_out0         = stable_l0;
    subs->n_t_out1         = stable_l1;
    subs->n_mel_seen       = n_mel_total;
    return (size_t) hh2;
}

/* Full audio-tower forward: subsample + 12 Conformer layers + output_proj +
 * embed_audio. Caller passes padded mel + mask; output is the soft-token sequence. */
size_t audio_encoder_run(const struct AudioEncoder *a,
                         const float               *mel_in,
                         const bool                *mel_mask_in,
                         size_t                     n_mel_frames,
                         float                     *softtokens_out) {
    /* Step 1: subsample → (T_sub, 1024). */
    float *sub = heap_alloc_array_aligned(float, (n_mel_frames / 4 + 4) * AUDIO_HIDDEN);
    size_t n_sub;
    {
        AE_TIC();
        n_sub = audio_encoder_subsample_run(a, mel_in, mel_mask_in, n_mel_frames, sub);
        AE_TOC(g_ae_subsample);
    }

    /* Step 2: precompute pos_emb + attn_mask (constant across layers). */
    float *pos_emb;
    bool  *attn_mask;
    {
        AE_TIC();
        pos_emb   = audio_encoder_compute_pos_emb(a);
        attn_mask = audio_encoder_compute_attn_mask(a, n_sub);
        AE_TOC(g_ae_pos_emb);
    }

    /* Step 3: run 12 Conformer layers in sequence. Ping-pong two buffers. */
    float *h_a = sub; /* takes ownership of sub */
    float *h_b = heap_alloc_array_aligned(float, n_sub *AUDIO_HIDDEN);
    {
        AE_TIC();
        for (int li = 0; li < N_LAYERS; li++) {
            audio_encoder_layer_run(a, li, h_a, n_sub, pos_emb, attn_mask, h_b);
            float *tmp = h_a;
            h_a        = h_b;
            h_b        = tmp; /* result lands in h_a */
        }
        AE_TOC(g_ae_layer_total);
    }
    safe_free((void **) &h_b);
    safe_free((void **) &attn_mask);
    safe_free((void **) &pos_emb);

    /* Step 4: output_proj (1024 → 1536, with bias). */
    float *op = heap_alloc_array_aligned(float, n_sub *OUTPUT_PROJ_DIMS);
    {
        AE_TIC();
        linear_fp32(
                h_a, a->output_proj_w, a->output_proj_b, n_sub, AUDIO_HIDDEN, OUTPUT_PROJ_DIMS, op);
        AE_TOC(g_ae_output_proj);
    }
    safe_free((void **) &h_a);

    /* Step 5: embed_audio.embedding_pre_projection_norm (RMSNorm with_scale=False)
     *         then embedding_projection (1536 → 1536, no bias). */
    float *normed = heap_alloc_array_aligned(float, n_sub *OUTPUT_PROJ_DIMS);
    rmsnorm_fp32(op, nullptr, n_sub, OUTPUT_PROJ_DIMS, RMS_EPS, normed);
    safe_free((void **) &op);
    {
        AE_TIC();
        linear_fp32(normed,
                    a->embed_audio_proj,
                    nullptr,
                    n_sub,
                    OUTPUT_PROJ_DIMS,
                    TEXT_HIDDEN,
                    softtokens_out);
        AE_TOC(g_ae_embed_proj);
    }
    safe_free((void **) &normed);

    ae_profile_print_and_reset();
    return n_sub;
}
