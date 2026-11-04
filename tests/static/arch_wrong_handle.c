/*
 * tests/static/arch_wrong_handle.c - an arch vtable handle of the wrong type
 * must not compile (#347).
 *
 * Never linked into anything; scripts/check_handles.sh compiles it as is
 * (every handle in its place) and once per GEIST_WRONG_HANDLE case, each of
 * which must fail with an incompatible-pointer diagnostic. Before #347 the
 * handles were void *, and every case compiled.
 */
#include <geist_arch.h>

void arch_wrong_handle(const struct geist_arch_ops_decoder *dec,
                       const struct geist_arch_ops_encoder *audio,
                       const struct geist_arch_ops_vision  *vision,
                       struct geist_arch_state             *model,
                       struct geist_arch_session           *session,
                       struct geist_audio_encoder_state    *audio_state,
                       struct geist_vision_encoder_state   *vision_state);

void arch_wrong_handle(const struct geist_arch_ops_decoder *dec,
                       const struct geist_arch_ops_encoder *audio,
                       const struct geist_arch_ops_vision  *vision,
                       struct geist_arch_state             *model,
                       struct geist_arch_session           *session,
                       struct geist_audio_encoder_state    *audio_state,
                       struct geist_vision_encoder_state   *vision_state) {
    float out[4];
#if !defined(GEIST_WRONG_HANDLE)
    dec->session_free(model, session);
    (void) audio->soft_token_dim(audio_state);
    (void) vision->soft_token_dim(vision_state);
    (void) out;
#elif GEIST_WRONG_HANDLE == 1
    /* model and session swapped */
    dec->session_free(session, model);
#elif GEIST_WRONG_HANDLE == 2
    /* a per-session op handed the model */
    dec->state_reset(model);
    (void) session;
#elif GEIST_WRONG_HANDLE == 3
    /* the audio encoder's state handed to the vision tower */
    (void) vision->encode_image(audio_state, 1, 1, 4, nullptr, out);
#elif GEIST_WRONG_HANDLE == 4
    /* an encoder state where the decoder model belongs */
    (void) dec->hidden_dim(vision_state);
#elif GEIST_WRONG_HANDLE == 5
    /* a session where its decision readout belongs */
    dec->decision_rows_destroy(session);
#endif
}
