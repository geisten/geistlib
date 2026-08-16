/*
 * geist_util.h — helper and advanced APIs layered on the core <geist.h>.
 *
 * <geist.h> holds the minimal surface to load and run a model (backend →
 * model → session → decode). This header adds everything beyond that happy
 * path: tokenizer / special-token helpers, multimodal soft-token attach,
 * speculative decode, raw-logits / prefill access, KV-prefix pinning,
 * telemetry, and backend-capability queries.
 *
 * Split out of <geist.h> in 0.2.0. A program that only generates text needs
 * only <geist.h>; include this header where you use the functions below.
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
 * Tokenize without prefilling. Lets the caller inspect the token IDs that
 * would be produced by set_prompt, e.g. to seed a speculative-decode
 * drafter's history buffer with the prompt tokens, or to measure a
 * constrained candidate before committing it. Writes up to `out_capacity`
 * IDs to `out_ids` and the actual count to `*n_out`. Returns
 * GEIST_E_NOT_FOUND if no tokenizer is loaded, GEIST_E_INVALID_ARG on
 * overflow. */
[[nodiscard]] enum geist_status geist_session_tokenize(struct geist_session *s,
                                                       const char           *text,
                                                       size_t                out_capacity,
                                                       geist_token_t out_ids[static out_capacity],
                                                       size_t       *n_out);

/* @stability STABLE since 0.1.0 — bypass tokenization: caller supplies
 * token IDs directly. Useful for testing and for integrations that
 * already have a tokenizer (set_prompt is the wrapper that does the
 * tokenize-then-prefill flow when a tokenizer.bin is available).
 *
 * Appends `n` tokens to the KV cache. After return the next call to
 * geist_session_decode_step yields the prediction for the position
 * following ids[n-1]. */
[[nodiscard]] enum geist_status
geist_session_prefill_tokens(struct geist_session *s, size_t n, const geist_token_t ids[static n]);

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
 * instance can consume beyond text, so a host can decide up front whether
 * to offer e.g. microphone input instead of finding out from a failing
 * attach call.
 *
 * The mask is a property of the *loaded instance*, not the architecture
 * string: it depends on the encoder weights found next to the GGUF at
 * load time (audio_tower.safetensors + mel_constants.bin,
 * vision_tower.safetensors) and on GEIST_TEXT_ONLY. It mirrors exactly
 * the capability checks the attach calls perform — a set bit means the
 * corresponding geist_session_attach_* cannot fail with
 * GEIST_E_NOT_FOUND / GEIST_E_UNSUPPORTED; a clear bit means it will.
 * Returns 0 for a text-only model or m == nullptr. */
unsigned geist_model_modalities(const struct geist_model *m);

/* @stability EXPERIMENTAL — soft-token injection semantics may change.
 *
 * PCM is consumed as 16-bit signed mono at the indicated `sample_rate`.
 * Only 16 kHz is currently supported (returns GEIST_E_UNSUPPORTED
 * otherwise). */
enum geist_status geist_session_attach_audio(struct geist_session *s,
                                             size_t                n_samples,
                                             const int16_t         pcm_samples[static n_samples],
                                             int                   sample_rate);

/* @stability EXPERIMENTAL — vision-tower soft-token injection.
 *
 * RGB is consumed as height × width × 3 uint8 row-major (i.e. HWC,
 * channels innermost). The vision encoder owns aspect-preserving
 * resize, patchification, the 16-block ViT, kernel-3 avg-pool, and
 * the multimodal projector — calling code only needs to supply
 * decoded pixels at native resolution.
 *
 * Returns GEIST_E_NOT_FOUND if vision_tower.safetensors was not found
 * at model-load time. */
enum geist_status geist_session_attach_image(struct geist_session *s,
                                             size_t                height,
                                             size_t                width,
                                             const uint8_t         rgb[static height * width * 3]);

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
enum geist_status
geist_session_attach_video(struct geist_session *s,
                           size_t                n_frames,
                           size_t                height,
                           size_t                width,
                           const uint8_t         frames[static n_frames * height * width * 3]);

/* ====================================================================== */
/* Advanced decode: KV-prefix pinning, raw logits, speculative             */
/* ====================================================================== */

/* @stability STABLE since 0.6.0 — agent-runtime contract (docs/API_CONTRACT.md).
 *
 * Pin `n` prefix tokens into the KV cache. After pin_prefix returns
 * GEIST_OK, the session's cache holds those tokens' KV state and any
 * subsequent geist_session_reset() truncates the cache back to this
 * prefix length (rather than 0). Use this to amortize a constant system
 * prompt across many chat turns. The arch decides whether to support
 * pin_prefix at all; transformer (Gemma 4) does, Mamba2 does not.
 *
 * Returns GEIST_E_UNSUPPORTED if the active architecture does not
 * implement prefix pinning. */
enum geist_status
geist_session_pin_prefix(struct geist_session *s, size_t n, const geist_token_t ids[static n]);

/* @stability STABLE since 0.6.0 — agent-runtime contract (docs/API_CONTRACT.md).
 *
 * Raw-logits accessor for evaluation, scoring, and constrained decoding.
 *
 * Returns a pointer to the next-position logits and writes the vocab size
 * to *n_logits. Returns nullptr (and sets *n_logits=0) if no logits are
 * pending — call geist_session_prefill_tokens / set_prompt / decode_step
 * first — or if the active architecture does not implement peek_logits
 * (Mamba2 currently), which callers must treat as "constrained decoding
 * unavailable", not as an error.
 *
 * Ownership: the buffer belongs to the SESSION, not the backend, and stays
 * valid until the next mutating call on that session. It may be a staging
 * copy — an accelerator backend satisfies this contract by copying device
 * memory into session storage, so the signature does not change when such a
 * backend lands. Do not free it and do not hold it across a decode. */
const float *geist_session_peek_logits(struct geist_session *s, size_t *n_logits);

/* @stability EXPERIMENTAL — speculative-decode API.
 *
 * One speculative-decode step: drafts up to k_max candidate tokens via
 * an internal n-gram lookup over `history`, then verifies them in a
 * single batched forward pass. Writes the emitted tokens (1..k_max+1)
 * to `out_tokens` and the count to `*n_out`.
 *
 * The drafter's first guess is always the model's own argmax over the
 * already-pending logits (zero cost), so spec_step emits at least 1
 * token per call even when the n-gram drafter has no proposal.
 *
 * `history` should hold every token committed to the cache so far
 * (prompt + previously emitted). The drafter searches it for suffix
 * matches; passing nullptr or history_n=0 degrades to single-token
 * decode. `out_capacity` must be at least k_max + 1.
 *
 * Sampler config: each position is sampled through the session's
 * configured sampler (argmax / top_k / top_p / temperature), same as
 * decode_step.
 *
 * Distribution caveat: under greedy decoding (temperature = 0), the
 * emitted stream is numerically equivalent to running decode_step
 * `*n_out` times. Under stochastic decoding (temperature > 0) the
 * emitted stream is valid (tokens sampled correctly per position) but
 * not distribution-preserving — the simple argmax-style accept-reject
 * loses the rejection-sampling step that would match the target
 * model's exact joint distribution. For strict stochastic equivalence,
 * use decode_step.
 *
 * Falls back to single-token decode if the active architecture lacks
 * the speculative primitives. */
[[nodiscard]] enum geist_status
geist_session_decode_speculative(struct geist_session *s,
                                 size_t                k_max,
                                 size_t                history_n,
                                 const geist_token_t   history[static history_n],
                                 size_t                out_capacity,
                                 geist_token_t         out_tokens[static out_capacity],
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

    /* Stubbed at zero. Backend-side counters not yet plumbed — these
     * land with an opt-in geist_backend_opts.enable_op_profiling
     * configuration in a future revision. */
    uint64_t buffer_alloc_count;
    uint64_t buffer_alloc_bytes_peak;
    uint64_t buffer_alloc_bytes_current;
};

/* @stability EXPERIMENTAL */
enum geist_status geist_session_get_stats(const struct geist_session *s,
                                          struct geist_session_stats *out);
enum geist_status geist_session_reset_stats(struct geist_session *s);

#ifdef __cplusplus
} /* extern "C" */
#endif
