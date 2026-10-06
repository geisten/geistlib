/*
 * geist_arch.h — extension API for architecture authors.
 *
 * Include this in addition to <geist.h> when implementing a new
 * architecture (transformer, audio conformer, vision siglip, ...). Defines
 * the three arch_ops vtables the engine dispatches through; each concrete
 * arch exports a descriptor wiring its implementations, registered in
 * src/engine/arch_registry.c. The engine owns the interface, the arch
 * layer implements it (as with geist_backend.h).
 *
 * @stability EXPERIMENTAL — vtable layout may evolve until 1.0.
 */
#ifndef GEIST_ARCH_H
#define GEIST_ARCH_H

#include <geist.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/* Decoder arch_ops vtable — what every decoder-arch must implement.       */
/* ====================================================================== */

/* The vtable operates on two opaque handles:
 *
 *   `void *arch_state` — the MODEL: weights, geometry, precomputed
 *       tables. Immutable after state_create; shared by all sessions.
 *   `void *session`    — ONE inference stream's mutable state (KV cache,
 *       recurrent state, scratch, sampler). Minted by session_alloc. For architectures
 *       WITHOUT session_alloc the engine passes the arch_state itself
 *       as the session handle — such an arch's model is its one
 *       session.
 *
 * THREAD-SAFETY CONTRACT. Setup and teardown are single-threaded:
 * state_create/destroy and session_alloc/free must not run concurrently
 * with anything else on the same model. Steady-state per-session ops
 * (prefill*, decode_step, peek_*, state_reset, pin_prefix, the
 * speculative primitives) may run concurrently across DIFFERENT
 * sessions of one model — one thread per session; a single session is
 * never called from two threads at once. Encoder ops (audio/vision) and
 * the engine's tokenizers are NOT covered by this guarantee — serialize
 * them externally.
 *
 * New optional slots are appended at the end so existing field offsets do
 * not move. */
struct gguf_ctx; /* the engine's GGUF reader, opaque here */

struct geist_arch_ops_decoder {
    const char *name;

    /* state_create: load the GGUF at gguf_path into a new arch_state on
     * `be`; nullptr on failure. opts carries load-time options
     * (geist_model_load_with_opts) and may be nullptr; sampler options
     * arrive later via set_session_opts. */
    void *(*state_create)(struct geist_backend            *be,
                          const char                      *gguf_path,
                          const struct geist_session_opts *opts);

    /* state_create_from_memory: like state_create but the GGUF is already in
     * memory (e.g. embedded in the binary). The buffer is aliased read-only and
     * must outlive the arch_state. No aux files (tokenizer.bin / vision / audio
     * safetensors) are searched — text-only with the GGUF-embedded tokenizer.
     * nullptr if the arch does not support memory loading. */
    void *(*state_create_from_memory)(struct geist_backend            *be,
                                      const void                      *data,
                                      size_t                           size,
                                      const struct geist_session_opts *opts);

    /* state_destroy: tear down arch_state. nullptr is a no-op. */
    void (*state_destroy)(void *arch_state);

    /* Optional: mutable view of the model's ZO-tuning gains — one f32 per
     * linear weight, all 1.0f at load, multiplied into that weight's
     * output. Model-level, so it takes arch_state, not a session; a write
     * is seen by every session on the model. nullptr, or
     * GEIST_E_UNSUPPORTED, when this build or backend has no gain path,
     * which callers must treat as "tuning unavailable", not an error. */
    enum geist_status (*gains)(void *arch_state, float **out, size_t *n);

    /* Optional: apply session opts (sampler config) to the session; called
     * from geist_session_create. nullptr = the architecture ignores them
     * (greedy-only). May fail (sampler workspace allocation); the caller
     * propagates. */
    enum geist_status (*set_session_opts)(void *session, const struct geist_session_opts *opts);

    /* state_reset: drop the session's conversational state (KV, recurrent
     * state), keep weights. Used by geist_session_reset. */
    void (*state_reset)(void *session);

    /* prefill: append `n` tokens to the session's recurrent state.
     * Status is the control flow (KV-window overflow, OOM, backend
     * failure); the backend error slot only carries detail text. */
    enum geist_status (*prefill)(void               *session,
                                 size_t              n,
                                 const geist_token_t ids[GEIST_AT_LEAST(n)]);

    /* decode_step: one autoregressive step. Writes the emitted token to
     * *out on GEIST_OK only — no in-band sentinel values. The forward that
     * appends the token may be deferred to the next call on the session
     * that reads or extends its state (transformer defers it, so the first
     * token after a prefill costs no forward). That call then returns what
     * it would have anyway, kv_len counts the token from this call on, and
     * a failure of the deferred forward is that call's status. */
    enum geist_status (*decode_step)(void *session, geist_token_t *out);

    /* Optional: pin prefix into the session's KV cache so reset()
     * restores to it instead of clearing. nullptr if architecture
     * doesn't support it. `ids` may be nullptr when `n` is 0, as for
     * geist_session_pin_prefix, which forwards it unchanged. */
    enum geist_status (*pin_prefix)(void *session, size_t n, const geist_token_t *ids);

    /* Optional: append audio soft-tokens (1536-dim per token for Gemma 4)
     * to the recurrent state. nullptr if no audio path. */
    enum geist_status (*prefill_audio)(void *session, size_t n, const float *soft_tokens);

    /* Optional: append vision soft-tokens (1536-dim per token for Gemma 4)
     * to the recurrent state. Same wire format as prefill_audio — both
     * modalities feed d_model-dim floats into the residual stream — so
     * the transformer impl is shared. nullptr if no vision path. */
    enum geist_status (*prefill_image)(void *session, size_t n, const float *soft_tokens);

    /* Optional: pointer to the session's cached next-token logits. Writes
     * the vocab size to `*n_logits` on success. Returns nullptr (and sets
     * *n_logits=0) if logits aren't materialized yet. Pointer is valid
     * until the next mutating call on THIS session. CPU-only contract —
     * GPU backends that need a copy should populate this via a
     * session-owned scratch buffer. */
    const float *(*peek_logits)(size_t *n_logits, void *session);

    /* Optional: pointer to the session's pooled sentence embedding, for
     * models that have pooling instead of an LM head. Writes the embedding
     * dimension to `*n_dims` on success; returns nullptr (and sets
     * *n_dims=0) on a generative model or before a prefill has produced
     * one. Same ownership and lifetime contract as peek_logits. */
    const float *(*peek_embedding)(size_t *n_dims, void *session);

    /* Optional: residual-stream width (d_model) of the loaded model. The
     * engine refuses a modality tower whose soft-token width differs (e.g.
     * an E2B tower next to an E4B GGUF). nullptr = unknown; check skipped. */
    size_t (*hidden_dim)(const void *arch_state);

    /* Speculative-decode primitives. Optional — leave nullptr if the
     * architecture has no batched verify path or no truncatable cache.
     * When any of these is nullptr, geist_session_decode_speculative
     * falls back to sequential decode_step.
     *
     * peek_next_token: the architecture's already-computed argmax for the
     *   immediate next position, or -1 if no valid logits are pending.
     *   "Free" — must not run a forward pass of its own; it may run one
     *   decode_step deferred (see there).
     * verify_forward: feed k candidate tokens through the full stack,
     *   advance kv_len by k, write k per-position samples to out_tokens.
     * kv_truncate: shrink recurrent state to new_len. Subsequent prefill
     *   overwrites from new_len onwards.
     * kv_len: current recurrent-state length (positions filled). */
    geist_token_t (*peek_next_token)(void *session);
    enum geist_status (*verify_forward)(void               *session,
                                        size_t              k,
                                        const geist_token_t ids[GEIST_AT_LEAST(k)],
                                        geist_token_t       out_tokens[GEIST_AT_LEAST(k)]);
    enum geist_status (*kv_truncate)(void *session, size_t new_len);
    size_t (*kv_len)(const void *session);

    /* Session lifecycle. Each engine-level geist_session owns one arch
     * session (KV cache, scratch pool, sampler RNG, ...); the model
     * (arch_state) owns the immutable weight set and is shared.
     *
     * session_alloc: mint a fresh session on the model. Returns the
     *   opaque session handle, or nullptr on OOM or unsatisfiable opts
     *   (e.g. a max_seq_len beyond what the model was created with).
     * session_free: tear down a session handle.
     *
     * Both nullptr → architecture is single-session-per-model; the
     * engine passes session == nullptr and the arch uses its default
     * session. There is no attach: per-session ops receive their
     * session explicitly on every call. */
    void *(*session_alloc)(void *arch_state, const struct geist_session_opts *opts);
    void (*session_free)(void *arch_state, void *session);

    /* Optional architecture-native drafter. Returns a candidate chain whose
     * first token is `seed`; GEIST_E_UNSUPPORTED asks the engine to use its generic
     * n-gram drafter instead. */
    enum geist_status (*draft_tokens)(void          *session,
                                      size_t         k_max,
                                      geist_token_t  seed,
                                      geist_token_t *out_tokens,
                                      size_t        *n_out);
    /* Optional model-level capability for numeric decisions. The next-token
     * logits vocabulary, or 0 for an embedding-only/unsupported model.
     * Allows the engine to validate all input IDs before a forward pass. */
    size_t (*logits_vocab_size)(const void *arch_state);

    /* Optional independent decision readout. Capability is resolved at
     * creation; unsupported pairs return GEIST_E_UNSUPPORTED, never dense.
     * The readout owns bounded workspace and borrows its private session.
     * prefill_rows skips ordinary logits finalization and returns only the
     * requested model-conformant logits. Outputs are zero on failure. */
    bool (*decision_rows_supported)(const void *arch_state);
    enum geist_status (*decision_rows_create)(void *session, size_t max_candidates, void **out);
    void (*decision_rows_destroy)(void *readout);
    enum geist_status (*prefill_rows)(size_t              *projected_rows,
                                      size_t              *readback_bytes,
                                      uint64_t            *head_ns,
                                      void                *readout,
                                      size_t               n_prompt,
                                      size_t               n_candidates,
                                      const geist_token_t *prompt_ids,
                                      const geist_token_t *candidate_ids,
                                      float               *out);

    /* Optional: session snapshot / restore (geist_session_snapshot). An
     * image of the session's complete decoding state that restore puts
     * back into any session of the same loaded model. snapshot_size reports
     * the bytes snapshot needs now; snapshot writes them to buf[capacity]
     * and their count to *out_bytes; neither changes the session's state.
     * restore replaces the session's state with an image, GEIST_E_FORMAT
     * if it is not one that fits. nullptr when the architecture cannot. */
    enum geist_status (*snapshot_size)(size_t *out_bytes, const void *session);
    enum geist_status (*snapshot)(size_t *out_bytes, size_t capacity, void *buf, void *session);
    enum geist_status (*restore)(size_t n_bytes, const void *buf, void *session);

    /* Optional: geist_session_truncate. Drop the session's state after
     * its first n positions, so the next prefill continues at n; pending
     * logits are invalid afterwards. GEIST_E_INVALID_ARG past the session's
     * length or below a pinned prefix; GEIST_E_UNSUPPORTED where the state
     * cannot return to n (recurrent layers, an already compressed KV region),
     * never a silently wrong state.
     * kv_bytes_per_token: the KV-cache bytes one more position costs in
     * this session, for its resolved KV mode; fixed buffers and recurrent
     * state, which do not grow with the length, are not included.
     * nullptr when the architecture cannot. */
    enum geist_status (*truncate)(void *session, size_t n);
    enum geist_status (*kv_bytes_per_token)(size_t *out_bytes, const void *session);

    /* Optional: geist_model_plan. From an open GGUF (metadata and
     * tensor table; no weight read, nothing allocated on be): the KV bytes
     * per position a session created with opts would report through
     * kv_bytes_per_token, and the model's bytes per position of
     * max_seq_len. nullptr when the architecture cannot plan. */
    enum geist_status (*plan)(struct geist_backend            *be,
                              struct gguf_ctx                 *gguf,
                              const struct geist_session_opts *opts,
                              size_t                          *kv_bytes_per_token,
                              size_t                          *model_bytes_per_token);
};

/* ====================================================================== */
/* Encoder arch_ops vtable — stateless modality encoders (audio).          */
/* ====================================================================== */

/* Encoder runs are session-independent (no recurrent state across calls);
 * the encoder weights live in encoder_state owned by the model and shared
 * across all sessions that consume the model. Encoder ops are NOT
 * thread-safe (encoder_state holds shared scratch) — serialize calls
 * externally. */
struct geist_arch_ops_encoder {
    const char *name;

    /* state_create: load encoder weights + auxiliary data (mel constants
     * for audio, normalization stats for vision). Returns the encoder
     * state pointer or nullptr on failure. */
    void *(*state_create)(struct geist_backend *be, const char *aux_search_root);

    /* state_destroy: free encoder weights. */
    void (*state_destroy)(void *encoder_state);

    /* encode_pcm: 16 kHz int16 PCM → soft-token sequence. Caller provides
     * out_soft buffer of size (max_soft × soft_token_dim() floats). Returns
     * the number of soft tokens produced (≤ max_soft), or 0 on error. */
    size_t (*encode_pcm)(void          *encoder_state,
                         size_t         n_samples,
                         size_t         max_soft,
                         const int16_t *pcm, /* null-tolerant: impls return 0 */
                         float         *out_soft);

    /* soft_token_dim: dimensionality of each soft-token vector (1536 for
     * Gemma 4 audio tower). */
    size_t (*soft_token_dim)(const void *encoder_state);

    /* Optional streaming encode: begin, push (repeated), end must be
     * equivalent to one encode_pcm over the concatenated PCM. The encoder
     * overlaps the heavy work with the arriving PCM, so end() only encodes
     * the tail. push is safe to call from a capture thread (the encoder
     * serializes internally); begin/end from the inference thread.
     * All three nullptr when the encoder has no streaming path. */
    bool (*stream_begin)(void *encoder_state);
    /* Returns false on overflow (>30 s buffered) or before begin. */
    bool (*stream_push)(void *encoder_state, size_t n, const int16_t *pcm);
    /* Non-blocking: drain whatever soft tokens are ready NOW (0 when
     * none), so the session can inject them while the user is speaking. */
    size_t (*stream_poll)(void *encoder_state, size_t max_soft, float *out_soft);

    /* Finish the tail, write up to max_soft soft tokens, return the
     * count (0 on error). */
    size_t (*stream_end)(void *encoder_state, size_t max_soft, float *out_soft);

    /* Drop an open stream without finishing it: discard buffered audio
     * and any soft tokens the worker has produced, and leave the encoder
     * ready for the next stream_begin; computes nothing. For callers that
     * cannot continue (allocation failure, session teardown). Must be safe
     * with no stream open and when called twice. Optional; without it the
     * caller can only stream_end. */
    void (*stream_abort)(void *encoder_state);

    /* max_soft_tokens: upper bound on soft tokens encode_pcm can produce
     * for n_samples of PCM, used to size the output buffer. Optional;
     * nullptr = the engine uses a conservative default. */
    size_t (*max_soft_tokens)(const void *encoder_state, size_t n_samples);
};

/* ====================================================================== */
/* Vision encoder arch_ops vtable.                                         */
/* ====================================================================== */

/* Parallel to geist_arch_ops_encoder but with image/video signatures that
 * don't fit the PCM-shaped surface. Encoder runs are session-independent;
 * weights live in encoder_state owned by the model and shared across all
 * sessions that consume the model. Like the audio encoder, NOT
 * thread-safe — serialize calls externally. */
struct geist_arch_ops_vision {
    const char *name;

    /* state_create: load tower weights from vision_tower.safetensors.
     * Returns the encoder state pointer or nullptr on failure (missing
     * weight file, OOM, etc.). aux_search_root mirrors the audio path
     * — typically the directory holding the GGUF. */
    void *(*state_create)(struct geist_backend *be, const char *aux_search_root);

    /* state_destroy: free tower weights. */
    void (*state_destroy)(void *encoder_state);

    /* encode_image: RGB uint8 image (H, W, 3) row-major → soft-token
     * sequence. Caller provides out_soft buffer of size (max_soft ×
     * soft_token_dim() floats). Returns the number of soft tokens
     * produced (≤ max_soft), or 0 on error.
     *
     * Image preprocessing (aspect-preserving bicubic resize, patchify,
     * bilinear pos-embed interp) is owned by the encoder — caller hands
     * over already-decoded RGB pixels at whatever native resolution. */
    size_t (*encode_image)(void          *encoder_state,
                           size_t         height,
                           size_t         width,
                           size_t         max_soft,
                           const uint8_t *rgb, /* height*width*3 bytes */
                           float         *out_soft);

    /* encode_video: stack of n_frames RGB uint8 images, each (H, W, 3).
     * Frames are tower-encoded in one batched pass for SGEMM amortization.
     * Soft tokens are concatenated across frames in input order. Returns
     * total soft-token count (≤ max_soft), or 0 on error.
     *
     * Frame sampling (picking n_frames from a longer clip) is the
     * caller's responsibility — geist does not link a video decoder. */
    size_t (*encode_video)(void          *encoder_state,
                           size_t         n_frames,
                           size_t         height,
                           size_t         width,
                           size_t         max_soft,
                           const uint8_t *frames, /* n_frames*height*width*3 bytes */
                           float         *out_soft);

    /* soft_token_dim: dimensionality of each soft-token vector. Projector
     * output dim — matches LM hidden_size so soft tokens splice directly
     * into the residual stream. */
    size_t (*soft_token_dim)(const void *encoder_state);
};

#ifdef __cplusplus
}
#endif

#endif /* GEIST_ARCH_H */
