/*
 * src/archs/audio_conformer/encoder_stream.c — the streaming API and its worker thread.
 *
 * Layer: ARCHITECTURE (audio_conformer). Split from the former
 * monolithic audio_encoder.c.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "encoder_internal.h"

/* Intra-module forward decls (definition order preserved from the split). */
static void audio_stream_state_reset(struct audio_stream_state *ss);

/* === Phase 8 streaming API ===================================================*/

/* Lazily create the mel pipeline. Constants path is a fixed default for now;
 * a future API can let callers override. */
static int ensure_mel(struct AudioEncoder *a) {
    if (a->mel)
        return 0;
    /* Constants live next to the audio_test_data dir of the bringup tree —
     * resolve relative to CWD to keep the C side tooling-agnostic. */
    const char *path = "audio_test_data/mel_constants.bin";
    a->mel           = mel_create(path);
    return a->mel ? 0 : -1;
}

int audio_encoder_push_pcm(struct AudioEncoder *a, const int16_t *samples, size_t n) {
    pthread_mutex_lock(&a->mtx);
    if (a->shutdown_flag || a->end_input_flag || a->pcm_len + n > a->pcm_cap) {
        pthread_mutex_unlock(&a->mtx);
        return -1;
    }
    memcpy(a->pcm_buf + a->pcm_len, samples, n * sizeof(int16_t));
    a->pcm_len += n;

    /* Advance mel pipeline as far as the now-larger PCM buffer allows.
     * Frame k uses padded[k*160 : k*160+320] where padded = 160 zeros +
     * pcm. So frame k requires pcm[0 .. (k+1)*160 - 1] available — i.e.
     * we need at least (k+1)*160 PCM samples to produce frame k. Phase
     * C: this work used to happen all-at-once after end_input; pushing
     * it inside push_pcm overlaps it with capture. */
    if (a->mel == nullptr)
        (void) ensure_mel(a); /* lazy first time */
    if (a->mel != nullptr) {
        float frame_pcm[MEL_FRAME_LENGTH];
        while (a->mel_n_computed < a->mel_cap && (a->mel_n_computed + 1) * 160 <= a->pcm_len) {
            const size_t k = a->mel_n_computed;
            for (size_t i = 0; i < MEL_FRAME_LENGTH; i++) {
                const long pi = (long) (k * 160 + i) - 160; /* unshift by left-pad */
                frame_pcm[i]  = (pi < 0) ? 0.0f : (float) a->pcm_buf[pi] / 32768.0f;
            }
            mel_frame_compute(a->mel, frame_pcm, a->mel_buf + k * MEL_N_MEL);
            a->mel_n_computed++;
        }
    }

    /* Phase 2: if the streaming worker is active and we've crossed a
     * sub-token-block boundary (~48 new mel frames = 12 sub-tokens) since
     * the last fire, kick the worker. This overlaps the Conformer compute
     * with the still-arriving PCM. */
    if (a->stream_enabled && a->worker_active && a->mel_n_computed >= a->worker_last_mel + 48) {
        a->worker_kick = true;
        pthread_cond_signal(&a->cv);
    }

    pthread_mutex_unlock(&a->mtx);
    return 0;
}

void audio_encoder_end_input(struct AudioEncoder *a) {
    pthread_mutex_lock(&a->mtx);
    a->end_input_flag = true;
    if (a->stream_enabled && a->worker_active) {
        a->worker_kick  = true;
        a->worker_final = true;
    }
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->mtx);
}

/* Run subsample + Conformer + projections on the pre-computed mel buffer
 * (populated incrementally by push_pcm — Phase C). Caller holds a->mtx;
 * we drop it during the heavy work and re-acquire afterwards. */
static int compute_segment_locked(struct AudioEncoder *a) {
    /* Snapshot mel frame count under the lock. PCM is needed only as a
     * fallback when something went wrong in the streaming path. */
    const size_t n_frames = a->mel_n_computed;

    if (a->mel == nullptr || n_frames == 0) {
        /* No mel computed (e.g. push_pcm never ran the lazy mel init).
         * Skip — pull will surface zero soft-tokens. */
        a->computed_flag = true;
        pthread_cond_broadcast(&a->cv);
        return 0;
    }

    /* Hand the mel buffer + mask to the heavy stages. The mel buffer is
     * owned by struct AudioEncoder and stable across the unlock — only push_pcm
     * writes to it, and end_input has set end_input_flag which blocks
     * further pushes. */
    float *mel_view = a->mel_buf;
    size_t n_mel;
    bool  *mask = audio_mel_mask_alloc(n_frames, true, &n_mel);
    pthread_mutex_unlock(&a->mtx);

    const size_t max_soft = audio_soft_bound_from_mel(n_mel);
    float       *soft     = heap_alloc_array_aligned(float, max_soft *AUDIO_SOFT_TOKEN_DIM);
    size_t       n_soft   = audio_encoder_run(a, mel_view, mask, n_mel, soft);
    safe_free((void **) &mask);

    pthread_mutex_lock(&a->mtx);
    a->soft_tokens   = soft;
    a->n_soft        = n_soft;
    a->n_emitted     = 0;
    a->computed_flag = true;
    pthread_cond_broadcast(&a->cv);
    return 0;
}

/* Compute absolute deadline `timeout_ms` from now. */
static void make_deadline(struct timespec *ts, int timeout_ms) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    long long total_ns = (long long) tv.tv_usec * 1000 + (long long) timeout_ms * 1000000;
    ts->tv_sec         = tv.tv_sec + total_ns / 1000000000;
    ts->tv_nsec        = total_ns % 1000000000;
}

size_t
audio_encoder_pull_softtokens(struct AudioEncoder *a, float *out, size_t max_out, int timeout_ms) {
    pthread_mutex_lock(&a->mtx);
    while (true) {
        if (a->shutdown_flag) {
            pthread_mutex_unlock(&a->mtx);
            return 0;
        }

        /* Phase 2 streaming-worker path: drain incremental soft tokens from
         * state->soft as the worker produces them. The worker may emit
         * tokens BEFORE end_input arrives, so we can return early without
         * waiting for the full segment. */
        if (a->stream_enabled && a->worker_active && a->stream != nullptr) {
            const size_t avail = a->stream->n_soft - a->stream->n_drained;
            if (avail > 0) {
                const size_t take = avail < max_out ? avail : max_out;
                memcpy(out,
                       a->stream->soft + a->stream->n_drained * AUDIO_SOFT_TOKEN_DIM,
                       take * AUDIO_SOFT_TOKEN_DIM * sizeof(float));
                a->stream->n_drained += take;
                pthread_mutex_unlock(&a->mtx);
                return take;
            }
            /* No new tokens. If the worker has finalised, we're done. */
            if (a->computed_flag) {
                pthread_mutex_unlock(&a->mtx);
                return 0;
            }
            /* Otherwise fall through to wait. */
        } else {
            /* Sync path: drain queue if we have unread soft-tokens. */
            if (a->computed_flag && a->n_emitted < a->n_soft) {
                size_t avail = a->n_soft - a->n_emitted;
                size_t take  = avail < max_out ? avail : max_out;
                memcpy(out,
                       a->soft_tokens + a->n_emitted * AUDIO_SOFT_TOKEN_DIM,
                       take * AUDIO_SOFT_TOKEN_DIM * sizeof(float));
                a->n_emitted += take;
                pthread_mutex_unlock(&a->mtx);
                return take;
            }
            if (a->computed_flag && a->n_emitted >= a->n_soft) {
                pthread_mutex_unlock(&a->mtx);
                return 0;
            }
            /* Trigger sync compute on first pull after end_input. */
            if (!a->computed_flag && a->end_input_flag) {
                compute_segment_locked(a);
                continue;
            }
        }

        /* Nothing available, no end yet → wait or return per timeout. */
        if (timeout_ms == 0) {
            pthread_mutex_unlock(&a->mtx);
            return 0;
        }
        if (timeout_ms < 0) {
            pthread_cond_wait(&a->cv, &a->mtx);
        } else {
            struct timespec dl;
            make_deadline(&dl, timeout_ms);
            int rc = pthread_cond_timedwait(&a->cv, &a->mtx, &dl);
            if (rc == ETIMEDOUT) {
                pthread_mutex_unlock(&a->mtx);
                return 0;
            }
        }
    }
}

bool audio_encoder_segment_done(const struct AudioEncoder *a) {
    /* Read-only snapshot — race is OK; caller polls. */
    return a->end_input_flag && a->computed_flag && a->n_emitted >= a->n_soft;
}

void audio_encoder_reset(struct AudioEncoder *a) {
    pthread_mutex_lock(&a->mtx);
    a->pcm_len        = 0;
    a->mel_n_computed = 0;
    a->end_input_flag = false;
    a->computed_flag  = false;
    a->n_emitted      = 0;
    a->n_soft         = 0;
    safe_free((void **) &a->soft_tokens);
    a->soft_tokens = nullptr;
    if (a->stream)
        audio_stream_state_reset(a->stream);
    /* Phase 2 worker bookkeeping. */
    a->worker_kick     = false;
    a->worker_final    = false;
    a->worker_last_mel = 0;
    pthread_mutex_unlock(&a->mtx);
}

void audio_encoder_shutdown(struct AudioEncoder *a) {
    pthread_mutex_lock(&a->mtx);
    a->shutdown_flag = true;
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->mtx);
}

/* === Phase 8b chunk-streaming scaffolding (impl in subsequent commit). === */

struct audio_stream_state *audio_stream_state_create(void) {
    struct audio_stream_state *s = heap_calloc_array_aligned(struct audio_stream_state, 1);
    if (s == nullptr)
        return nullptr;
    for (int li = 0; li < N_LAYERS; li++) {
        s->attn[li].k     = heap_calloc_array_aligned(float, MAX_SUB_TOKENS *AUDIO_HIDDEN);
        s->attn[li].v     = heap_calloc_array_aligned(float, MAX_SUB_TOKENS *AUDIO_HIDDEN);
        s->lconv[li].hist = heap_calloc_array_aligned(float, (CONV_KERNEL - 1) * AUDIO_HIDDEN);
        if (!s->attn[li].k || !s->attn[li].v || !s->lconv[li].hist) {
            audio_stream_state_destroy(s);
            return nullptr;
        }
    }
    s->sub_buf = heap_calloc_array_aligned(float, MAX_SUB_TOKENS *AUDIO_HIDDEN);
    s->soft    = heap_calloc_array_aligned(float, MAX_SUB_TOKENS *AUDIO_SOFT_TOKEN_DIM);
    /* Phase-3 subsample cache. l0 dominates the allocation - 128 channels
     * × T_out0 × 64 freq positions × 4 B = up to ~49 MB at the 30 s mel
     * cap. Allocated only when the stream worker may run; on Pi 5 we run
     * with 4 GB so this is comfortable. */
    s->subs.l0 = heap_calloc_array_aligned(
            float, (size_t) SUBS_L0_CHANNELS * SUBS_T_OUT0_CAP * SUBS_W_OUT0);
    s->subs.l1 = heap_calloc_array_aligned(
            float, (size_t) SUBS_L1_CHANNELS * SUBS_T_OUT1_CAP * SUBS_W_OUT1);
    if (!s->sub_buf || !s->soft || !s->subs.l0 || !s->subs.l1) {
        audio_stream_state_destroy(s);
        return nullptr;
    }
    return s;
}

void audio_stream_state_destroy(struct audio_stream_state *s) {
    if (s == nullptr)
        return;
    for (int li = 0; li < N_LAYERS; li++) {
        safe_free((void **) &s->attn[li].k);
        safe_free((void **) &s->attn[li].v);
        safe_free((void **) &s->lconv[li].hist);
    }
    safe_free((void **) &s->sub_buf);
    safe_free((void **) &s->soft);
    safe_free((void **) &s->pos_emb);
    safe_free((void **) &s->subs.l0);
    safe_free((void **) &s->subs.l1);
    safe_free((void **) &s);
}

static void audio_stream_state_reset(struct audio_stream_state *s) {
    if (s == nullptr)
        return;
    for (int li = 0; li < N_LAYERS; li++) {
        s->attn[li].n         = 0;
        s->lconv[li].n_filled = 0;
        /* No need to memset — n=0 / n_filled=0 guards readers. */
    }
    s->n_sub_total = 0;
    s->n_soft      = 0;
    s->n_drained   = 0; /* forgotten before: pull after reset underflowed avail */
    /* Phase-3 cache reset: just clear the counters; the conv2d_fp32_from
     * overwrites the relevant cells before any subsequent reader. */
    s->subs.n_t_out0   = 0;
    s->subs.n_t_out1   = 0;
    s->subs.n_mel_seen = 0;
}

/* === Phase 1b: chunk-streaming forward (parity with audio_encoder_run). ===
 *
 * Process one block of new sub-tokens through one Conformer layer, threading
 * K/V cache (attention) and depthwise conv state (LConv) through state.
 *
 * h_chunk_io: (CHUNK_SIZE, AUDIO_HIDDEN) — input on entry, output on return.
 *             Always CHUNK_SIZE rows; if n_valid < CHUNK_SIZE the trailing
 *             rows are padding that must be zero-initialized by the caller.
 * n_valid:    actual sub-tokens in this block (≤ CHUNK_SIZE; last block
 *             of a final push may be shorter).
 * block_idx:  absolute block index (kv->n / CHUNK_SIZE BEFORE this call).
 * pos_emb:    (POS_LEN, AUDIO_HIDDEN) — constant across the whole utterance.
 *
 * Bit-for-bit equivalent to running the monolithic attn_run on the same
 * block's slice when the cache is read from the same positions. The chunked
 * attention layout has MAX_FUTURE=0, so per-block compute is causal and
 * does not depend on yet-unseen sub-tokens. */
static void attn_run_streaming_block(struct audio_stream_state *state,
                                     int                        layer_idx,
                                     const struct Attn         *attn,
                                     const float               *h_chunk_in,
                                     size_t                     n_valid,
                                     size_t                     block_idx,
                                     const float               *pos_emb,
                                     float                     *y_chunk_out) {
    const float           q_scale  = (1.0f / sqrtf((float) HEAD_DIM)) / logf(2.0f);
    const float           k_scale  = log1pf(2.71828182845904523536f) / logf(2.0f);
    const size_t          hd_per_t = (size_t) N_HEADS * HEAD_DIM;
    struct attn_kv_cache *kv       = &state->attn[layer_idx];

    /* 1. Q/K/V projection for the n_valid real rows. Q is padded to CHUNK_SIZE
     *    rows (zero pad) so the per-head attention loop has a stable shape. */
    float *h_clip = heap_alloc_array_aligned(float, (size_t) CHUNK_SIZE *AUDIO_HIDDEN);
    float *q      = heap_calloc_array_aligned(float, (size_t) CHUNK_SIZE *AUDIO_HIDDEN);
    float *k_new  = heap_alloc_array_aligned(float, n_valid *AUDIO_HIDDEN);
    float *v_new  = heap_alloc_array_aligned(float, n_valid *AUDIO_HIDDEN);

    memcpy(h_clip, h_chunk_in, n_valid * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->q_proj, h_clip, n_valid, AUDIO_HIDDEN, AUDIO_HIDDEN, q);
    memcpy(h_clip, h_chunk_in, n_valid * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->k_proj, h_clip, n_valid, AUDIO_HIDDEN, AUDIO_HIDDEN, k_new);
    memcpy(h_clip, h_chunk_in, n_valid * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->v_proj, h_clip, n_valid, AUDIO_HIDDEN, AUDIO_HIDDEN, v_new);
    safe_free((void **) &h_clip);

    /* 2. Scale q and k. */
    float q_pds[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        q_pds[d] = q_scale * log1pf(expf(attn->per_dim_scale[d]));
    }
    for (size_t t = 0; t < n_valid; t++) {
        for (int head_i = 0; head_i < N_HEADS; head_i++) {
            float *qrow = q + ((size_t) t * N_HEADS + head_i) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++)
                qrow[d] *= q_pds[d];
        }
    }
    for (size_t i = 0; i < n_valid * AUDIO_HIDDEN; i++)
        k_new[i] *= k_scale;

    /* 3. Append scaled K, V to the cache. */
    memcpy(kv->k + kv->n * AUDIO_HIDDEN, k_new, n_valid * AUDIO_HIDDEN * sizeof(float));
    memcpy(kv->v + kv->n * AUDIO_HIDDEN, v_new, n_valid * AUDIO_HIDDEN * sizeof(float));
    safe_free((void **) &k_new);
    safe_free((void **) &v_new);
    const size_t kv_n_after = kv->n + n_valid;

    /* 4. Build K/V context window for this block from cache, matching the
     *    monolithic attn_run's b'th iteration: src = block_idx*12 + c - 12. */
    float *k_ctx = heap_calloc_array_aligned(float, (size_t) CONTEXT_SIZE *hd_per_t);
    float *v_ctx = heap_calloc_array_aligned(float, (size_t) CONTEXT_SIZE *hd_per_t);
    for (int c = 0; c < CONTEXT_SIZE; c++) {
        int src = (int) block_idx * CHUNK_SIZE + c - MAX_PAST_HORIZON;
        if (src < 0 || src >= (int) kv_n_after)
            continue;
        memcpy(k_ctx + (size_t) c * hd_per_t,
               kv->k + (size_t) src * hd_per_t,
               hd_per_t * sizeof(float));
        memcpy(v_ctx + (size_t) c * hd_per_t,
               kv->v + (size_t) src * hd_per_t,
               hd_per_t * sizeof(float));
    }

    /* 5. Compute the per-block mask locally. Same formula as
     *    audio_encoder_compute_attn_mask for the relevant block, with
     *    n = kv_n_after (total sub-tokens through layer 0 so far). */
    bool block_mask[CHUNK_SIZE * CONTEXT_SIZE] = {0};
    for (int i = 0; i < (int) n_valid; i++) {
        const int q_g = (int) block_idx * CHUNK_SIZE + i;
        if (q_g >= (int) kv_n_after)
            continue;
        for (int c = 0; c < CONTEXT_SIZE; c++) {
            const int k_g = (int) block_idx * CHUNK_SIZE + c - MAX_PAST_HORIZON;
            if (k_g < 0 || k_g >= (int) kv_n_after)
                continue;
            const int off = k_g - q_g;
            if (off < -(MAX_PAST_HORIZON - 1) || off > MAX_FUTURE)
                continue;
            block_mask[i * CONTEXT_SIZE + c] = true;
        }
    }

    /* 6. Relative-K projection — constant per encoder, recomputed per call
     *    (small: POS_LEN×1024). */
    float *rel_k = heap_alloc_array_aligned(float, (size_t) POS_LEN *AUDIO_HIDDEN);
    linear_fp32(
            pos_emb, attn->relative_k_proj, nullptr, POS_LEN, AUDIO_HIDDEN, AUDIO_HIDDEN, rel_k);

    /* 7. Attention output (CHUNK_SIZE, AUDIO_HIDDEN); only first n_valid used. */
    float *attn_out = heap_calloc_array_aligned(float, (size_t) CHUNK_SIZE *AUDIO_HIDDEN);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int hd = 0; hd < N_HEADS; hd++) {
        float scores_ac[CHUNK_SIZE * CONTEXT_SIZE];
        float scores_bd[CHUNK_SIZE * CONTEXT_SIZE];
        float bd_padded[CHUNK_SIZE * (CONTEXT_SIZE + 1)];

        const float *q_bh = q + (size_t) hd * HEAD_DIM;
        const float *k_bh = k_ctx + (size_t) hd * HEAD_DIM;
        const float *v_bh = v_ctx + (size_t) hd * HEAD_DIM;
        const float *rk_h = rel_k + (size_t) hd * HEAD_DIM;

        for (int i = 0; i < CHUNK_SIZE; i++) {
            const float *qi = q_bh + (size_t) i * hd_per_t;
            for (int j = 0; j < CONTEXT_SIZE; j++) {
                const float *kj                 = k_bh + (size_t) j * hd_per_t;
                scores_ac[i * CONTEXT_SIZE + j] = dot_head_fp32(qi, kj);
            }
        }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            const float *qi = q_bh + (size_t) i * hd_per_t;
            for (int p = 0; p < POS_LEN; p++) {
                const float *rp                       = rk_h + (size_t) p * AUDIO_HIDDEN;
                bd_padded[i * (CONTEXT_SIZE + 1) + p] = dot_head_fp32(qi, rp);
            }
            for (int p = POS_LEN; p <= CONTEXT_SIZE; p++) {
                bd_padded[i * (CONTEXT_SIZE + 1) + p] = 0.0f;
            }
        }
        /* rel_shift — flat copy from bd_padded (stride CONTEXT_SIZE+1) into
         * scores_bd (stride CONTEXT_SIZE). This is a cyclic row shift that
         * aligns the relative-position bias across the chunk; same as the
         * monolithic attn_run. */
        for (int idx = 0; idx < CHUNK_SIZE * CONTEXT_SIZE; idx++) {
            scores_bd[idx] = bd_padded[idx];
        }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            for (int j = 0; j < CONTEXT_SIZE; j++) {
                float s = scores_ac[i * CONTEXT_SIZE + j] + scores_bd[i * CONTEXT_SIZE + j];
                s       = tanhf(s / ATTN_SOFTCAP) * ATTN_SOFTCAP;
                if (!block_mask[i * CONTEXT_SIZE + j])
                    s = -1e9f;
                scores_ac[i * CONTEXT_SIZE + j] = s;
            }
        }
        softmax_fp32(scores_ac, CHUNK_SIZE, CONTEXT_SIZE);
        for (int i = 0; i < CHUNK_SIZE; i++) {
            float *out_row = attn_out + (size_t) i * AUDIO_HIDDEN + (size_t) hd * HEAD_DIM;
            zero_head_fp32(out_row);
            for (int j = 0; j < CONTEXT_SIZE; j++) {
                const float  w  = scores_ac[i * CONTEXT_SIZE + j];
                const float *vj = v_bh + (size_t) j * hd_per_t;
                axpy_head_fp32(out_row, w, vj);
            }
        }
    }

    safe_free((void **) &rel_k);
    safe_free((void **) &k_ctx);
    safe_free((void **) &v_ctx);
    safe_free((void **) &q);

    /* 8. Post-projection on the n_valid real rows only. */
    float *tmp_in = heap_alloc_array_aligned(float, n_valid *AUDIO_HIDDEN);
    memcpy(tmp_in, attn_out, n_valid * AUDIO_HIDDEN * sizeof(float));
    safe_free((void **) &attn_out);
    clip_linear_apply(&attn->post, tmp_in, n_valid, AUDIO_HIDDEN, AUDIO_HIDDEN, y_chunk_out);
    safe_free((void **) &tmp_in);

    /* 9. Advance cache. */
    kv->n = kv_n_after;
}

/* Streaming LConv: like lconv_run but the depthwise conv reads the last
 * (CONV_KERNEL-1) post-GLU rows from state for causal continuity, then
 * stores the last (CONV_KERNEL-1) of the new post-GLU h into state. */
static void lconv_run_streaming(struct audio_stream_state *state,
                                int                        layer_idx,
                                const struct LConv        *lc,
                                float                     *h,
                                size_t                     n) {
    const size_t        hsize = (size_t) n * AUDIO_HIDDEN;
    struct lconv_state *lcs   = &state->lconv[layer_idx];

    float *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h, hsize * sizeof(float));

    rmsnorm_fp32(h, lc->pre_norm, n, AUDIO_HIDDEN, RMS_EPS, h);

    float *doubled = heap_alloc_array_aligned(float, (size_t) n * 2 * AUDIO_HIDDEN);
    clip_linear_apply(&lc->linear_start, h, n, AUDIO_HIDDEN, 2 * AUDIO_HIDDEN, doubled);
    glu_fp32(doubled, n, AUDIO_HIDDEN, h);
    safe_free((void **) &doubled);

    /* Snapshot post-GLU h — both as conv input (with prepended history) and
     * as next call's state. */
    float *h_glu = heap_alloc_array_aligned(float, hsize);
    memcpy(h_glu, h, hsize * sizeof(float));

    /* Extended (T,C) input: zero-pad slot of (CONV_KERNEL-1) rows, with the
     * tail filled by lcs->hist (left-aligned to the boundary so n_filled<K-1
     * shows up as leading zeros — matches the implicit zero-pad of the
     * monolithic depthwise conv at sub-token 0). */
    const size_t prepend = (size_t) CONV_KERNEL - 1;
    const size_t ext_T   = prepend + n;
    float       *ext_in  = heap_calloc_array_aligned(float, ext_T *AUDIO_HIDDEN);
    if (lcs->n_filled > 0) {
        const size_t off = prepend - lcs->n_filled;
        memcpy(ext_in + off * AUDIO_HIDDEN,
               lcs->hist,
               lcs->n_filled * AUDIO_HIDDEN * sizeof(float));
    }
    memcpy(ext_in + prepend * AUDIO_HIDDEN, h_glu, hsize * sizeof(float));

    /* (T,C) → (C,T_ext) — depthwise conv expects per-channel rows. */
    float *ext_ct = heap_alloc_array_aligned(float, AUDIO_HIDDEN *ext_T);
    for (size_t t = 0; t < ext_T; t++) {
        for (int c = 0; c < AUDIO_HIDDEN; c++) {
            ext_ct[(size_t) c * ext_T + t] = ext_in[t * AUDIO_HIDDEN + c];
        }
    }
    safe_free((void **) &ext_in);

    float *conv_out = heap_alloc_array_aligned(float, AUDIO_HIDDEN *ext_T);
    depthwise_conv1d_causal_fp32(
            ext_ct, lc->depthwise, conv_out, AUDIO_HIDDEN, (int) ext_T, CONV_KERNEL);
    safe_free((void **) &ext_ct);

    /* Transpose back, taking only the n outputs corresponding to the new
     * sub-tokens (positions [prepend, prepend+n)). */
    for (size_t t = 0; t < n; t++) {
        for (int c = 0; c < AUDIO_HIDDEN; c++) {
            h[t * AUDIO_HIDDEN + c] = conv_out[(size_t) c * ext_T + (prepend + t)];
        }
    }
    safe_free((void **) &conv_out);

    /* Update lconv state: new hist = last min(K-1, n_filled+n) rows of
     * (lcs->hist || h_glu). Use a temp to avoid aliasing on lcs->hist. */
    const size_t total_avail  = lcs->n_filled + n;
    const size_t new_n_filled = total_avail < prepend ? total_avail : prepend;
    if (n >= new_n_filled) {
        memcpy(lcs->hist,
               h_glu + (n - new_n_filled) * AUDIO_HIDDEN,
               new_n_filled * AUDIO_HIDDEN * sizeof(float));
    } else {
        const size_t take_prior = new_n_filled - n;
        const size_t prior_off  = lcs->n_filled - take_prior;
        float       *tmp        = heap_alloc_array_aligned(float, new_n_filled *AUDIO_HIDDEN);
        memcpy(tmp,
               lcs->hist + prior_off * AUDIO_HIDDEN,
               take_prior * AUDIO_HIDDEN * sizeof(float));
        memcpy(tmp + take_prior * AUDIO_HIDDEN, h_glu, n * AUDIO_HIDDEN * sizeof(float));
        memcpy(lcs->hist, tmp, new_n_filled * AUDIO_HIDDEN * sizeof(float));
        safe_free((void **) &tmp);
    }
    lcs->n_filled = new_n_filled;
    safe_free((void **) &h_glu);

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

/* Streaming per-layer forward. Mirrors audio_encoder_layer_run but threads
 * state through attention (K/V cache) and LConv (conv-state). */
static void audio_encoder_layer_run_streaming(const struct AudioEncoder *a,
                                              struct audio_stream_state *state,
                                              int                        layer_idx,
                                              float                     *h_io,
                                              size_t                     n_valid,
                                              size_t                     block_idx,
                                              const float               *pos_emb) {
    const struct ConformerLayer *L     = &a->layers[layer_idx];
    const size_t                 hsize = n_valid * AUDIO_HIDDEN;

    /* 1. FFN-1 (no state). */
    ffn_run(&L->ff1, h_io, n_valid);

    /* 2. norm_pre_attn + streaming attention + norm_post_attn + residual. */
    float *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h_io, hsize * sizeof(float));
    rmsnorm_fp32(h_io, L->norm_pre_attn, n_valid, AUDIO_HIDDEN, RMS_EPS, h_io);

    float *attn_out = heap_alloc_array_aligned(float, hsize);
    attn_run_streaming_block(
            state, layer_idx, &L->attn, h_io, n_valid, block_idx, pos_emb, attn_out);
    memcpy(h_io, attn_out, hsize * sizeof(float));
    safe_free((void **) &attn_out);

    rmsnorm_fp32(h_io, L->norm_post_attn, n_valid, AUDIO_HIDDEN, RMS_EPS, h_io);
    for (size_t i = 0; i < hsize; i++)
        h_io[i] += residual[i];
    safe_free((void **) &residual);

    /* 3. LConv with conv-state. */
    lconv_run_streaming(state, layer_idx, &L->lconv, h_io, n_valid);

    /* 4. FFN-2 (no state). */
    ffn_run(&L->ff2, h_io, n_valid);

    /* 5. norm_out. */
    rmsnorm_fp32(h_io, L->norm_out, n_valid, AUDIO_HIDDEN, RMS_EPS, h_io);
}

/* Top-level streaming entry: drives chunk processing. Runs incremental
 * subsample on the cached l0/l1 intermediates (Phase 3 - extends the
 * cache with new time positions rather than recomputing from scratch),
 * identifies new sub-tokens, and pushes complete blocks (or the final
 * partial block if is_final) through the 12 streaming layers +
 * output_proj + embed_audio. Returns the number of NEW soft tokens
 * appended to state->soft. */
size_t audio_encoder_stream_push(struct AudioEncoder       *a,
                                 struct audio_stream_state *state,
                                 const float               *mel_full,
                                 const bool                *mel_mask,
                                 size_t                     n_mel_total,
                                 bool                       is_final) {
    if (state == nullptr || n_mel_total == 0)
        return 0;

    /* 1. Subsample. One-shot segments default to the full-mel re-run (the
     *    alloc + memcpy overhead of the Phase-3 cache exceeds the conv2d
     *    savings on a single pass). The live worker calls this per 48-frame
     *    kick, where re-running from frame 0 is O(T²) over the clip — it
     *    defaults to the incremental path. GEIST_AUDIO_SUBSAMPLE_INC=0/1
     *    overrides either way. */
    const char  *inc_env  = getenv("GEIST_AUDIO_SUBSAMPLE_INC");
    const bool   subs_inc = inc_env != nullptr ? inc_env[0] == '1' : a->stream_enabled;
    const size_t n_sub_full =
            subs_inc ? audio_encoder_subsample_run_inc(
                               a, &state->subs, mel_full, mel_mask, n_mel_total, state->sub_buf)
                     : audio_encoder_subsample_run(
                               a, mel_full, mel_mask, n_mel_total, state->sub_buf);

    if (n_sub_full <= state->n_sub_total)
        return 0;

    /* 2. Determine block range to emit. Non-final: only full blocks.
     *    Final: include the partial trailing block (padded to CHUNK_SIZE). */
    const size_t block_start = state->n_sub_total / CHUNK_SIZE;
    const size_t block_end_excl =
            is_final ? (n_sub_full + CHUNK_SIZE - 1) / CHUNK_SIZE : n_sub_full / CHUNK_SIZE;
    if (block_end_excl <= block_start)
        return 0;

    /* 3. Constant pos_emb across blocks — computed once per encoder,
     *    cached on the stream state (freed with it). */
    if (state->pos_emb == nullptr)
        state->pos_emb = audio_encoder_compute_pos_emb(a);
    float *pos_emb = state->pos_emb;

    /* 4. Per-block × per-layer streaming. */
    float  h_chunk[CHUNK_SIZE * AUDIO_HIDDEN];
    size_t n_new_soft_total = 0;

    for (size_t b = block_start; b < block_end_excl; b++) {
        const size_t sub_start = b * CHUNK_SIZE;
        const size_t sub_end =
                (sub_start + CHUNK_SIZE) < n_sub_full ? (sub_start + CHUNK_SIZE) : n_sub_full;
        const size_t n_chunk = sub_end - sub_start;

        memset(h_chunk, 0, sizeof(h_chunk));
        memcpy(h_chunk,
               state->sub_buf + sub_start * AUDIO_HIDDEN,
               n_chunk * AUDIO_HIDDEN * sizeof(float));

        for (int li = 0; li < N_LAYERS; li++) {
            audio_encoder_layer_run_streaming(a, state, li, h_chunk, n_chunk, b, pos_emb);
        }

        /* output_proj + embed_audio on the n_chunk real rows. */
        float *op = heap_alloc_array_aligned(float, n_chunk *OUTPUT_PROJ_DIMS);
        linear_fp32(h_chunk,
                    a->output_proj_w,
                    a->output_proj_b,
                    n_chunk,
                    AUDIO_HIDDEN,
                    OUTPUT_PROJ_DIMS,
                    op);

        float *normed = heap_alloc_array_aligned(float, n_chunk *OUTPUT_PROJ_DIMS);
        rmsnorm_fp32(op, nullptr, n_chunk, OUTPUT_PROJ_DIMS, RMS_EPS, normed);
        safe_free((void **) &op);

        linear_fp32(normed,
                    a->embed_audio_proj,
                    nullptr,
                    n_chunk,
                    OUTPUT_PROJ_DIMS,
                    TEXT_HIDDEN,
                    state->soft + state->n_soft * AUDIO_SOFT_TOKEN_DIM);
        safe_free((void **) &normed);

        /* Publish under the encoder mutex: the pull side reads n_soft and
         * memcpys the rows below it while the worker is still mid-push —
         * without this release the counter can become visible before the
         * soft-token stores on ARM (data race, garbage rows to the LM). */
        pthread_mutex_lock(&a->mtx);
        state->n_soft += n_chunk;
        pthread_mutex_unlock(&a->mtx);
        state->n_sub_total = sub_end;
        n_new_soft_total += n_chunk;
    }

    return n_new_soft_total;
}

/* Public accessor: return the current streaming state pointer (read-only
 * for tests; lifecycle is owned by audio_encoder_create/destroy/reset). */
struct audio_stream_state *audio_encoder_stream_state(struct AudioEncoder *a) {
    return a ? a->stream : nullptr;
}

const float *audio_stream_state_soft(const struct audio_stream_state *s) {
    return s ? s->soft : nullptr;
}

size_t audio_stream_state_n_soft(const struct audio_stream_state *s) {
    return s ? s->n_soft : 0;
}

/* === Phase 2: streaming worker thread driver. === */

/* Run one stream_push iteration. Caller has dropped a->mtx; we re-grab it
 * only to copy out the snapshot info, then drop again for the heavy
 * compute (which mutates state but not a->mel_buf — push_pcm continues
 * to append to mel_buf concurrently, but only the snapshot range is
 * read here so it's stable). */
static void worker_do_push(struct AudioEncoder *a, size_t mel_snap, bool is_final) {
    if (mel_snap == 0)
        return;
    /* Mid-stream pushes see only real frames; the FINAL push closes the
     * segment with the same padded extra frame the monolithic path uses. */
    size_t n_mel;
    bool  *mask = audio_mel_mask_alloc(mel_snap, is_final, &n_mel);
    (void) audio_encoder_stream_push(a, a->stream, a->mel_buf, mask, n_mel, is_final);
    safe_free((void **) &mask);
}

static void *audio_encoder_stream_worker(void *arg) {
    struct AudioEncoder *a = (struct AudioEncoder *) arg;
    pthread_mutex_lock(&a->mtx);
    while (true) {
        while (!a->worker_kick && !a->shutdown_flag) {
            pthread_cond_wait(&a->cv, &a->mtx);
        }
        if (a->shutdown_flag)
            break;
        a->worker_kick        = false;
        const size_t mel_snap = a->mel_n_computed;
        const bool   is_final = a->worker_final;
        a->worker_last_mel    = mel_snap;
        pthread_mutex_unlock(&a->mtx);

        worker_do_push(a, mel_snap, is_final);

        pthread_mutex_lock(&a->mtx);
        if (is_final) {
            a->computed_flag = true;
            a->worker_final  = false;
        }
        pthread_cond_broadcast(&a->cv);
    }
    pthread_mutex_unlock(&a->mtx);
    return nullptr;
}

/* Enable the streaming worker (called from audio_encoder_create when
 * GEIST_AUDIO_STREAM=1). Spawns the worker thread. */
bool stream_worker_start(struct AudioEncoder *a) {
    a->stream_enabled  = true;
    a->worker_active   = true;
    a->worker_kick     = false;
    a->worker_final    = false;
    a->worker_last_mel = 0;
    if (pthread_create(&a->worker_tid, nullptr, audio_encoder_stream_worker, a) != 0) {
        a->worker_active  = false;
        a->stream_enabled = false;
        return false;
    }
    return true;
}

void stream_worker_stop(struct AudioEncoder *a) {
    if (!a->worker_active)
        return;
    pthread_mutex_lock(&a->mtx);
    a->shutdown_flag = true;
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->mtx);
    pthread_join(a->worker_tid, nullptr);
    a->worker_active = false;
}
