/*
 * tests/static/metal_wrong_handle.c - a Metal handle of the wrong type must
 * not compile (#347).
 *
 * Never linked into anything. scripts/check_handles.sh compiles it as is,
 * which must succeed (every handle in its place), and once per case with
 * -DGEIST_WRONG_HANDLE=<n>, which must fail with an incompatible-pointer
 * diagnostic. Before #347 every handle was a void *, and each case below
 * compiled.
 */
#include "../../src/backends/metal/metal_internal.h"

void metal_wrong_handle(struct metal_state *st, struct metal_cmdbuf_h *cmd);

void metal_wrong_handle(struct metal_state *st, struct metal_cmdbuf_h *cmd) {
    struct metal_encoder_h *enc = st->sequence_compute_encoder;
#if !defined(GEIST_WRONG_HANDLE)
    (void) cmd;
    metal_msg_send_set_pipeline(st, enc, st->q4k_n4_pipeline);
    metal_msg_send_set_buffer(st, enc, st->argmax_part_buffer, 0, 0);
    metal_flush_if_referenced(st, st->argmax_result_buffer, 0, SIZE_MAX);
#elif GEIST_WRONG_HANDLE == 1
    /* a buffer where the pipeline belongs */
    metal_msg_send_set_pipeline(st, enc, st->argmax_part_buffer);
#elif GEIST_WRONG_HANDLE == 2
    /* the function, not the pipeline built from it */
    metal_msg_send_set_pipeline(st, enc, st->q4k_n4_function);
#elif GEIST_WRONG_HANDLE == 3
    /* the command buffer where its encoder belongs */
    metal_msg_send_set_buffer(st, cmd, st->argmax_part_buffer, 0, 0);
#elif GEIST_WRONG_HANDLE == 4
    /* a pipeline bound as a buffer */
    metal_msg_send_set_buffer(st, enc, st->q4k_n4_pipeline, 0, 0);
#elif GEIST_WRONG_HANDLE == 5
    /* the geist_buffer wrapper, not its MTLBuffer */
    metal_flush_if_referenced(st, st->attn_dec_partials_buffer, 0, SIZE_MAX);
#endif
}
