/*
 * geist_util.h — helper and advanced APIs layered on the core <geist.h>.
 *
 * <geist.h> holds the minimal surface to load and run a model (backend →
 * model → session → decode). This header adds everything beyond that happy
 * path: tokenizer / special-token helpers, multimodal soft-token attach,
 * speculative decode, raw-logits / prefill access, KV-prefix pinning,
 * telemetry, and backend-capability queries. A program that only generates
 * text needs only <geist.h>.
 */
#pragma once

#include <geist.h>
#include <geist_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/* Special tokens (chat stop handling / templating)                        */
/* ====================================================================== */

/* Sentinel returned by the token-id accessors when the model has no
 * tokenizer, the metadata field is unset, or a lookup misses. */
#define GEIST_TOKEN_NONE ((geist_token_t) - 1)

/* @stability STABLE since 0.2.0 — special-token ids from the model's
 * tokenizer metadata. Use these for clean stop handling and chat templating
 * by token-id instead of string-matching decoded output. Each returns
 * GEIST_TOKEN_NONE when the model has no tokenizer or the field is unset.
 *
 *   const geist_token_t eos = geist_model_eos_token(model);
 *   ...
 *   geist_session_decode_step(s, &tok);
 *   if (tok == eos) break;            // stop cleanly, no string match
 */
geist_token_t geist_model_eos_token(const struct geist_model *m);
geist_token_t geist_model_bos_token(const struct geist_model *m);

/* @stability STABLE since 0.12.0 — the tokenizer's add_bos_token /
 * add_eos_token metadata (geist-runtime contract, #622).
 *
 * geist_session_tokenize returns content tokens only; these say whether the
 * model expects BOS / EOS around them. Embedding models in particular pool
 * over exactly the sequence passed, so a missing BOS gives a different
 * vector. Both false when no tokenizer is loaded. */
bool geist_model_add_bos(const struct geist_model *m);
bool geist_model_add_eos(const struct geist_model *m);

/* @stability STABLE since 0.2.0 — look up the token id for an exact vocab
 * entry, e.g. "<end_of_turn>". Returns GEIST_TOKEN_NONE if the model has no
 * tokenizer or `text` is not a single vocab token. Lets a chat app discover
 * extra stop tokens / template markers by name (Gemma ends a turn with
 * `<end_of_turn>`, which some GGUFs set as eos and some do not). */
geist_token_t geist_model_token_by_text(const struct geist_model *m, const char *text);

/* ====================================================================== */
/* Tokenization / direct token feeding                                     */
/* ====================================================================== */

/* @stability STABLE since 0.6.0 — agent-runtime contract (docs/API_CONTRACT.md).
 *
 * Tokenize without prefilling: the token IDs set_prompt would produce
 * (e.g. to seed the speculative-decode history or measure a candidate
 * before committing it). Writes up to `out_capacity`
 * IDs to `out_ids` and the actual count to `*n_out`; `out_ids` may be
 * nullptr only when `out_capacity` is 0. Returns GEIST_E_NOT_FOUND if no
 * tokenizer is loaded, GEIST_E_INVALID_ARG on overflow. */
[[nodiscard]] enum geist_status geist_session_tokenize(struct geist_session *s,
                                                       const char           *text,
                                                       size_t                out_capacity,
                                                       geist_token_t        *out_ids,
                                                       size_t               *n_out);

/* @stability STABLE since 0.1.0 — prefill caller-supplied token IDs,
 * bypassing the tokenizer (set_prompt = tokenize + this).
 *
 * Appends `n` tokens to the KV cache. After return the next call to
 * geist_session_decode_step yields the prediction for the position
 * following ids[n-1]. */
[[nodiscard]] enum geist_status
geist_session_prefill_tokens(struct geist_session *s, size_t n, const geist_token_t *ids);

/* ====================================================================== */
/* Multimodal soft-token attach                                            */
/* ====================================================================== */

/* Bits returned by geist_model_modalities. */
enum geist_modality {
    GEIST_MOD_AUDIO  = 1 << 0,
    GEIST_MOD_VISION = 1 << 1,
    GEIST_MOD_VIDEO  = 1 << 2,
};

/* @stability EXPERIMENTAL — bitmask of modalities this loaded model
 * instance can consume beyond text.
 *
 * The mask is a property of the *loaded instance*, not the architecture
 * string: it depends on the encoder weights found next to the GGUF at
 * load time (audio_tower.safetensors + mel_constants.bin,
 * vision_tower.safetensors) and on GEIST_TEXT_ONLY. It mirrors exactly
 * the capability checks the attach calls perform — for WELL-FORMED input
 * (16 kHz mono PCM within the encoder's 30 s limit, valid RGB), a set
 * bit means the corresponding geist_session_attach_* cannot fail with
 * GEIST_E_NOT_FOUND / GEIST_E_UNSUPPORTED; a clear bit means it will.
 * Malformed input (wrong sample rate, oversize clip) still errors with
 * the bit set. Returns 0 for a text-only model or m == nullptr. */
unsigned geist_model_modalities(const struct geist_model *m);

/* @stability EXPERIMENTAL — soft-token injection semantics may change.
 *
 * PCM is consumed as 16-bit signed mono at the indicated `sample_rate`.
 * Only 16 kHz is currently supported (returns GEIST_E_UNSUPPORTED
 * otherwise). */
enum geist_status geist_session_attach_audio(struct geist_session *s,
                                             size_t                n_samples,
                                             const int16_t        *pcm_samples,
                                             int                   sample_rate);

/* @stability EXPERIMENTAL — streaming audio turn (#256): push PCM while
 * the user is still speaking; the encoder overlaps its work with the
 * arriving audio, so end() only has the tail left to encode.
 *
 * Contract: begin → push* → end is equivalent to a single
 * geist_session_attach_audio over the concatenated PCM: same token
 * count, identical greedy prediction, logits within the small numeric
 * noise of the incremental encoder (bit-equality is NOT promised — the
 * overlapped attention reassociates float sums). Same 16 kHz mono s16
 * input, same 30 s limit, same error codes.
 *
 * Threading: push is safe from a capture thread; begin/end belong to
 * the thread driving the session. One streaming turn at a time per
 * model (the audio encoder is model-owned); begin returns
 * GEIST_E_INVALID_STATE if a turn is already open. */
enum geist_status geist_session_audio_begin(struct geist_session *s);
enum geist_status geist_session_audio_push(struct geist_session *s, size_t n, const int16_t *pcm);
/* Optional, from the session thread between pushes: inject the soft
 * tokens that are ready NOW into the LM, so end() has less left to do.
 * Cheap no-op when nothing is ready. Never required for correctness. */
enum geist_status geist_session_audio_poll(struct geist_session *s);
enum geist_status geist_session_audio_end(struct geist_session *s);

/* @stability EXPERIMENTAL — vision-tower soft-token injection.
 *
 * RGB is consumed as height × width × 3 uint8 row-major (HWC, channels
 * innermost), decoded pixels at native resolution; resizing and
 * patchification happen inside the encoder.
 *
 * Returns GEIST_E_NOT_FOUND if vision_tower.safetensors was not found
 * at model-load time. */
enum geist_status geist_session_attach_image(struct geist_session *s,
                                             size_t                height,
                                             size_t                width,
                                             const uint8_t        *rgb);

/* @stability EXPERIMENTAL — vision-tower soft-token injection for video.
 *
 * Frames are consumed as n_frames × height × width × 3 uint8 row-major,
 * with all frames at the same resolution (caller's responsibility).
 * Each frame contributes ≤ 70 soft tokens (per Gemma 4 video-processor
 * default) so the LM context fits all 32 frames at ≈ 2240 soft tokens.
 *
 * Frame sampling (selecting 32 frames from a longer clip) is the
 * caller's responsibility — geist does not link a video decoder.
 *
 * Returns GEIST_E_NOT_FOUND if vision_tower.safetensors was not found
 * at model-load time. */
enum geist_status geist_session_attach_video(struct geist_session *s,
                                             size_t                n_frames,
                                             size_t                height,
                                             size_t                width,
                                             const uint8_t        *frames);

/* ====================================================================== */
/* Advanced decode: KV-prefix pinning, raw logits, speculative             */
/* ====================================================================== */

/* @stability STABLE since 0.6.0 — agent-runtime contract (docs/API_CONTRACT.md).
 *
 * Pin `n` prefix tokens into the KV cache. After pin_prefix returns
 * GEIST_OK, the session's cache holds those tokens' KV state and any
 * subsequent geist_session_reset() truncates the cache back to this
 * prefix length (rather than 0). Use this to amortize a constant system
 * prompt across many chat turns.
 *
 * `ids` may be nullptr when `n` is 0. Returns GEIST_E_UNSUPPORTED if the
 * active architecture does not implement prefix pinning, and for a
 * non-empty prefix on a model with recurrent (DeltaNet) layers such as
 * Qwen3.5: a reset cannot return their state to a prefix (use
 * geist_session_snapshot / geist_session_restore there). n = 0 empties
 * the session and unpins. */
enum geist_status
geist_session_pin_prefix(struct geist_session *s, size_t n, const geist_token_t *ids);

/* @stability EXPERIMENTAL (#622) — going back in a conversation.
 *
 * geist_session_length: the positions in the session's state now (prefilled
 * and decoded tokens; a token decode_step returned counts), 0 if unknown.
 *
 * geist_session_truncate: keep the first n positions and drop the rest, so
 * the next prefill continues at n — a chat runtime rewinds to an earlier
 * message without processing the kept part again. The pending logits are
 * gone: prefill before the next decode_step. n equal to the length changes
 * nothing; n = 0 empties the session like geist_session_reset.
 *   GEIST_E_INVALID_ARG  n is past the length or below a pinned prefix.
 *   GEIST_E_UNSUPPORTED  the state cannot return to n: recurrent (DeltaNet)
 *                        layers such as Qwen3.5 (only n = 0 or n = length;
 *                        use geist_session_snapshot / _restore), positions
 *                        in the compressed KIVI region, MTP drafting, an
 *                        architecture without truncation. The session is
 *                        unchanged; the caller can reset and prefill again.
 *
 * geist_session_kv_bytes_per_token: the KV-cache bytes one more position
 * costs in this session, for its resolved KV mode (kv_mode, the
 * GEIST_KV_* env and the backend decide it at session_create). Buffers that
 * do not grow with the length (scratch, recurrent state, the KIVI residual
 * ring) are not included. With the weights' size this lets a caller choose
 * the longest max_seq_len that fits into memory. */
size_t                          geist_session_length(const struct geist_session *s);
[[nodiscard]] enum geist_status geist_session_truncate(struct geist_session *s, size_t n);
[[nodiscard]] enum geist_status geist_session_kv_bytes_per_token(const struct geist_session *s,
                                                                 size_t *out_bytes);

/* @stability EXPERIMENTAL (#625) — plan a model before loading it.
 *
 * The window a model can hold is fixed at load (max_seq_len sizes its RoPE
 * tables and bounds every session), but what a position costs depends on
 * the model's layers, the KV mode and the backend. geist_model_plan reads
 * only the GGUF's metadata and tensor table — no weight is read, nothing is
 * allocated on be — and reports what a load with these opts would cost:
 *
 *   context_length         the trained window (<arch>.context_length), 0 if
 *                          the GGUF does not say; geist_model_context_length
 *   weight_bytes           the GGUF's tensor bytes
 *   kv_bytes_per_token     per position of one session created with opts on
 *                          be: equal to geist_session_kv_bytes_per_token of
 *                          that session (one formula, one KV-mode resolution)
 *   model_bytes_per_token  the model's own per-position tables (RoPE), per
 *                          position of max_seq_len
 *
 * So one session with window W needs about weight_bytes +
 * W * (kv_bytes_per_token + model_bytes_per_token), plus buffers that do not
 * grow with W (scratch, recurrent state): keep headroom for those. A
 * runtime picks W from the memory it has, then loads with that max_seq_len.
 * Errors as for geist_model_load (geist_last_create_error); *out is zero on
 * failure. */
struct geist_model_plan {
    size_t context_length;
    size_t weight_bytes;
    size_t kv_bytes_per_token;
    size_t model_bytes_per_token;
};
[[nodiscard]] enum geist_status geist_model_plan(const char                      *path,
                                                 struct geist_backend            *be,
                                                 const struct geist_session_opts *opts,
                                                 struct geist_model_plan         *out);
[[nodiscard]] enum geist_status geist_model_plan_from_memory(const void                      *data,
                                                             size_t                           size,
                                                             struct geist_backend            *be,
                                                             const struct geist_session_opts *opts,
                                                             struct geist_model_plan         *out);

/* @stability EXPERIMENTAL — session snapshot / restore (#548).
 *
 * Save a session's complete decoding state into a caller buffer and put it
 * back later, into the same session or into another session of the SAME
 * loaded model (a fork). After a restore the session continues exactly as
 * the source did at snapshot time: the same pending logits, the same next
 * decode_step token (sampling included: the RNG state is part of the
 * image), and prefills append after the same positions.
 *
 * This is the reusable prefix that pin_prefix cannot give a model with
 * recurrent layers (Qwen3.5's Gated DeltaNet): prefill the constant part
 * once, snapshot, and restore before each request instead of re-prefilling.
 * It works for attention-only models too, and also captures a pinned
 * prefix (a reset after the restore truncates to it).
 *
 * The image holds the used KV rows, every recurrent layer's state, the
 * pending logits and a small header; its size grows with the cached
 * positions. It is an in-process format: valid only for the model handle
 * it was taken on, in a session with the same KV-cache mode (kv_mode, and
 * the GEIST_KV_* experiment env vars) and a max_seq_len that holds its
 * positions. No on-disk or cross-version stability.
 *
 *   geist_session_snapshot_size  writes the bytes a snapshot needs now.
 *   geist_session_snapshot       writes the image into buf[capacity] and
 *                                its size to *out_bytes. Neither call
 *                                changes the session's state or runs a
 *                                forward pass.
 *   geist_session_restore        replaces the session's state with the
 *                                image of n_bytes.
 *
 * *out_bytes is 0 on every failure. GEIST_E_INVALID_ARG when capacity is
 * smaller than the image; GEIST_E_FORMAT when restore is handed anything
 * that is not a complete image for this model and session (the session is
 * then untouched); GEIST_E_UNSUPPORTED for the KIVI KV cache, an enabled
 * MTP drafter (GEIST_MTP=1), embedding models and architectures without
 * snapshots; GEIST_E_INVALID_STATE while an audio stream is open. A
 * backend transfer that fails during restore leaves the session reset. */
[[nodiscard]] enum geist_status geist_session_snapshot_size(size_t               *out_bytes,
                                                            struct geist_session *s);
[[nodiscard]] enum geist_status
geist_session_snapshot(size_t *out_bytes, size_t capacity, void *buf, struct geist_session *s);
[[nodiscard]] enum geist_status
geist_session_restore(size_t n_bytes, const void *buf, struct geist_session *s);

/* @stability STABLE since 0.6.0 — agent-runtime contract (docs/API_CONTRACT.md).
 *
 * Raw-logits accessor for evaluation, scoring, and constrained decoding.
 *
 * Returns a pointer to the next-position logits and writes the vocab size
 * to *n_logits. Returns nullptr (and sets *n_logits=0) if no logits are
 * pending — call geist_session_prefill_tokens / set_prompt / decode_step
 * first — or if the active architecture does not implement peek_logits,
 * which callers must treat as "constrained decoding unavailable", not as
 * an error.
 *
 * Ownership: the buffer belongs to the SESSION, not the backend (it may be
 * a staging copy of device memory), and stays valid until the next mutating
 * call on that session. Do not free it and do not hold it across a
 * decode. */
const float *geist_session_peek_logits(size_t *n_logits, struct geist_session *s);

/* @stability EXPERIMENTAL — embedding models.
 *
 * The pooled sentence embedding for everything prefilled into `s` so far:
 * pooled per the model's own pooling metadata, passed through the final
 * norm, and L2-normalised, so a dot product between two of these is their
 * cosine similarity. Writes the dimension to `*n_dims`.
 *
 * Returns nullptr with *n_dims = 0 on a generative model — those have an LM
 * head and no pooling — and before any prefill has produced a vector. An
 * embedding model conversely emits no tokens: geist_session_decode_step
 * returns GEIST_E_UNSUPPORTED on one.
 *
 * Ownership matches geist_session_peek_logits: the buffer belongs to the
 * SESSION, stays valid until the next mutating call on it, and must not be
 * freed. Copy it if you need it past the next prefill.
 *
 * Query instruction prefixes (which these models need for quality),
 * embedding quantization and indexing are the caller's job. */
const float *geist_session_peek_embedding(size_t *n_dims, struct geist_session *s);

/* @stability EXPERIMENTAL — forward-only (zeroth-order) fine-tuning.
 *
 * Mutable view of the model's tuning gains: one f32 per linear weight, all
 * 1.0f after load, each multiplied into that weight's output. The weight
 * bytes are never touched, so a gradient-free optimizer (MeZO/QZO-style)
 * searches `*n` scalars, and a tuned model is the unmodified GGUF plus `*n`
 * floats. Writes take effect on the next forward pass (no reload).
 * The array is MODEL-level: a write is seen by every session on `m`, and
 * concurrent sessions must not race it (tune, then serve).
 *
 * Returns GEIST_E_UNSUPPORTED — which callers must treat as "tuning
 * unavailable", not an error — when geist was built without GEIST_TUNE, or
 * when the backend runs the fused tensor linear path that bypasses the
 * gain apply. On success the pointer is owned by the model and stays valid
 * until geist_model_destroy.
 *
 * Slot order is a pure function of the layer count, so a sidecar written
 * by one process is readable by another on the same model:
 *   layer l -> slots 9*l + 0..8 = q, k, v, o, gate, up, down,
 *              per_layer_gate, per_layer_proj
 *   then 9*n_layers + 0 = lm_head, +1 = model_proj.
 *
 *   float *g; size_t n;
 *   if (geist_model_gains(m, &g, &n) == GEIST_OK) {
 *       FILE *f = fopen(path, "rb");
 *       if (f) { (void) fread(g, sizeof *g, n, f); fclose(f); }
 *   }
 */
[[nodiscard]] enum geist_status
geist_model_gains(struct geist_model *m, float **out_gains, size_t *out_n);

/* @stability EXPERIMENTAL — speculative-decode API.
 *
 * One speculative-decode step: drafts up to k_max candidate tokens via an
 * architecture-native head when available and enabled (Qwen3.5 MTP with
 * GEIST_MTP=1), otherwise via an internal n-gram lookup over `history`, then
 * verifies them in one batched forward pass. Writes the emitted tokens
 * (1..k_max+1) to `out_tokens` and the count to `*n_out`.
 *
 * The drafter's first guess is always the model's own argmax over the
 * already-pending logits (zero cost), so spec_step emits at least 1
 * token per call even when the n-gram drafter has no proposal.
 *
 * `history` should hold every token committed to the cache so far
 * (prompt + previously emitted). The drafter searches it for suffix
 * matches; history_n=0 (`history` may then be nullptr) degrades to
 * single-token decode. `out_capacity` must be at least k_max + 1.
 *
 * Sampler config: each position is sampled through the session's
 * configured sampler (argmax / top_k / top_p / temperature), same as
 * decode_step.
 *
 * Distribution caveat: under greedy decoding (temperature = 0), the
 * emitted stream is numerically equivalent to running decode_step
 * `*n_out` times. Under stochastic decoding (temperature > 0) each token
 * is sampled correctly per position, but the accept/reject step does not
 * preserve the model's joint distribution; use decode_step when that
 * matters.
 *
 * Falls back to single-token decode if the active architecture lacks
 * the speculative primitives. */
[[nodiscard]] enum geist_status geist_session_decode_speculative(struct geist_session *s,
                                                                 size_t                k_max,
                                                                 size_t                history_n,
                                                                 const geist_token_t  *history,
                                                                 size_t                out_capacity,
                                                                 geist_token_t        *out_tokens,
                                                                 size_t               *n_out);

/* ====================================================================== */
/* Stats / Telemetry                                                       */
/* ====================================================================== */

struct geist_session_stats {
    /* Wired (CLOCK_MONOTONIC-based, ~1ns precision). total_decode_ns
     * covers geist_session_decode_step + decode_speculative; the
     * speculative path's verify-forward time counts as decode. */
    uint64_t n_tokens_decoded;
    uint64_t total_decode_ns;
    uint64_t total_prefill_ns;
    uint64_t total_audio_encode_ns;

    /* Always zero: backend-side counters are not implemented. */
    uint64_t buffer_alloc_count;
    uint64_t buffer_alloc_bytes_peak;
    uint64_t buffer_alloc_bytes_current;
};

/* @stability EXPERIMENTAL */
enum geist_status geist_session_get_stats(const struct geist_session *s,
                                          struct geist_session_stats *out);
enum geist_status geist_session_reset_stats(struct geist_session *s);

/* @stability EXPERIMENTAL — optional, observational backend telemetry.
 * Metal reports MTLDevice.currentAllocatedSize from this backend's device:
 * provider resource allocation, NOT physical residency, process RSS or a
 * working-set budget (never add it to RSS on unified memory). No-copy
 * aliases, heaps and driver retention follow the provider's accounting.
 * Safe to call concurrently with inference; takes no lock, submits no work
 * and does not allocate. The backend must stay alive for the call (join
 * observer threads before destroy). Failure zero-initializes out; only
 * GEIST_OK makes a zero a measurement. Backends without a provider (CPU)
 * return GEIST_E_UNSUPPORTED. */
enum geist_resource_source { GEIST_RESOURCE_NONE, GEIST_RESOURCE_METAL_DEVICE };
struct geist_backend_resources {
    uint64_t                   allocated_bytes;
    enum geist_resource_source source;
    bool                       unified_memory;
};
[[nodiscard]] enum geist_status
geist_backend_resources_snapshot(const struct geist_backend     *be,
                                 struct geist_backend_resources *out);

#ifdef __cplusplus
} /* extern "C" */
#endif
