/*
 * geist_arch.h — extension API for architecture authors.
 *
 * Include this in addition to <geist.h> when implementing a new
 * architecture (transformer, audio conformer, vision siglip, a future
 * Mamba/SSM, etc.). Defines the three arch_ops vtables the engine
 * dispatches through; each concrete arch exports a descriptor wiring its
 * implementations, registered in src/engine/arch_registry.c.
 *
 * Parallel to geist_backend.h: the engine owns the interface here, the
 * arch layer implements it. Keeping the vtable shapes in this neutral
 * header (rather than inside a concrete arch's private header) lets the
 * engine dispatch without including any specific architecture — adding an
 * architecture touches only its own sources plus the registry.
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

/* The vtable operates on two opaque handles to keep arch implementations
 * decoupled from the engine's full session definition:
 *
 *   `void *arch_state` — the MODEL: weights, geometry, precomputed
 *       tables. Immutable after state_create; shared by all sessions.
 *   `void *session`    — ONE inference stream's mutable state: for a
 *       transformer the KV cache, scratch, sampler; for Mamba the SSM
 *       hidden vector. Minted by session_alloc. For architectures
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
 * them externally. */
struct geist_arch_ops_decoder {
    const char *name;

    /* state_create: allocate arch_state on backend. Returns nullptr on
     * failure (engine reports OOM/IO via error path). Caller passes
     * the GGUF path that model_load used to find the file. opts is
     * typically nullptr at model-load time (no session yet); session
     * options arrive later via set_session_opts. */
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

    /* Optional: push session opts into the session. Engine calls this
     * from geist_session_create so per-session sampler config (temperature,
     * top_p, top_k, random_seed) reaches the decode hot path. nullptr
     * means the architecture ignores session opts (greedy-only). May fail
     * (sampler workspace allocation) — the caller must propagate. */
    enum geist_status (*set_session_opts)(void *session, const struct geist_session_opts *opts);

    /* state_reset: drop the session's conversational state (KV / SSM
     * hidden), keep weights. Used by geist_session_reset. */
    void (*state_reset)(void *session);

    /* prefill: append `n` tokens to the session's recurrent state.
     * Status is the control flow (KV-window overflow, OOM, backend
     * failure); the backend error slot only carries detail text. */
    enum geist_status (*prefill)(void *session, size_t n, const geist_token_t ids[static n]);

    /* decode_step: one autoregressive step. Writes the emitted token to
     * *out on GEIST_OK only — no in-band sentinel values. */
    enum geist_status (*decode_step)(void *session, geist_token_t *out);

    /* Optional: pin prefix into the session's KV cache so reset()
     * restores to it instead of clearing. nullptr if architecture
     * doesn't support it. */
    enum geist_status (*pin_prefix)(void *session, size_t n, const geist_token_t ids[static n]);

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
    const float *(*peek_logits)(void *session, size_t *n_logits);

    /* Optional: residual-stream width (d_model) of the loaded model.
     * The engine refuses a modality tower whose soft-token width doesn't
     * match (e.g. an E2B 1536-dim tower next to an E4B 2560-dim GGUF —
     * same family, wrong geometry, #258). nullptr = unknown; the engine
     * then skips the check. */
    size_t (*hidden_dim)(const void *arch_state);

    /* Speculative-decode primitives. Optional — leave nullptr if the
     * architecture has no batched verify path or no truncatable cache.
     * When any of these is nullptr, geist_session_decode_speculative
     * falls back to sequential decode_step.
     *
     * peek_next_token: the architecture's already-computed argmax for the
     *   immediate next position, or -1 if no valid logits are pending.
     *   "Free" — must not run a forward pass.
     * verify_forward: feed k candidate tokens through the full stack,
     *   advance kv_len by k, write k per-position samples to out_tokens.
     * kv_truncate: shrink recurrent state to new_len. Subsequent prefill
     *   overwrites from new_len onwards.
     * kv_len: current recurrent-state length (positions filled). */
    geist_token_t (*peek_next_token)(void *session);
    enum geist_status (*verify_forward)(void               *session,
                                        size_t              k,
                                        const geist_token_t ids[static k],
                                        geist_token_t       out_tokens[static k]);
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

    /* Optional architecture-native drafter. Appended to preserve offsets of
     * the pre-existing decoder ABI. Returns a candidate chain whose first
     * token is `seed`; GEIST_E_UNSUPPORTED asks the engine to use its generic
     * n-gram drafter instead. */
    enum geist_status (*draft_tokens)(void          *session,
                                      size_t         k_max,
                                      geist_token_t  seed,
                                      geist_token_t *out_tokens,
                                      size_t        *n_out);
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
                         const int16_t *pcm,
                         size_t         n_samples,
                         float         *out_soft,
                         size_t         max_soft);

    /* soft_token_dim: dimensionality of each soft-token vector (1536 for
     * Gemma 4 audio tower). */
    size_t (*soft_token_dim)(const void *encoder_state);

    /* Optional streaming encode (#256): begin, push (repeated), end must
     * be equivalent to one encode_pcm over the concatenated PCM — that
     * equivalence is the testable contract. The encoder overlaps the
     * heavy work with the arriving PCM, so end() returns after only the
     * tail. push is safe to call from a capture thread (the encoder
     * serializes internally); begin/end from the inference thread.
     * All three nullptr when the encoder has no streaming path. */
    bool (*stream_begin)(void *encoder_state);
    /* Returns false on overflow (>30 s buffered) or before begin. */
    bool (*stream_push)(void *encoder_state, const int16_t *pcm, size_t n);
    /* Non-blocking: drain whatever soft tokens are ready NOW (0 when
     * none). Lets the session inject tokens into the LM while the user
     * is still speaking — phase 2 of #256. */
    size_t (*stream_poll)(void *encoder_state, float *out_soft, size_t max_soft);

    /* Finish the tail, write up to max_soft soft tokens, return the
     * count (0 on error). */
    size_t (*stream_end)(void *encoder_state, float *out_soft, size_t max_soft);

    /* Drop an open stream without finishing it: discard buffered audio
     * and any soft tokens the worker has produced, and leave the encoder
     * ready for the next stream_begin. Unlike stream_end it computes
     * nothing and returns nothing — it exists so a caller that cannot
     * continue (an allocation failed, the session is being destroyed) can
     * close the stream instead of leaking it. Must be safe to call when
     * no stream is open, and safe to call twice.
     *
     * Optional: an encoder without it leaves the caller only stream_end,
     * which pays for a tail nobody will read. */
    void (*stream_abort)(void *encoder_state);

    /* max_soft_tokens: upper bound on soft tokens encode_pcm can produce
     * for n_samples of PCM — lets the engine size the output buffer from
     * the audio length instead of guessing a fixed cap (#247: a hardcoded
     * 256 silently truncated everything past ~10 s while still paying the
     * full encode). Optional; nullptr means the engine falls back to a
     * conservative default. */
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
                           const uint8_t *rgb,
                           size_t         height,
                           size_t         width,
                           float         *out_soft,
                           size_t         max_soft);

    /* encode_video: stack of n_frames RGB uint8 images, each (H, W, 3).
     * Frames are tower-encoded in one batched pass for SGEMM amortization.
     * Soft tokens are concatenated across frames in input order. Returns
     * total soft-token count (≤ max_soft), or 0 on error.
     *
     * Frame sampling (picking n_frames from a longer clip) is the
     * caller's responsibility — geist does not link a video decoder. */
    size_t (*encode_video)(void          *encoder_state,
                           const uint8_t *frames,
                           size_t         n_frames,
                           size_t         height,
                           size_t         width,
                           float         *out_soft,
                           size_t         max_soft);

    /* soft_token_dim: dimensionality of each soft-token vector. Projector
     * output dim — matches LM hidden_size so soft tokens splice directly
     * into the residual stream. */
    size_t (*soft_token_dim)(const void *encoder_state);
};

#ifdef __cplusplus
}
#endif

#endif /* GEIST_ARCH_H */
