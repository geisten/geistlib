/*
 * src/backends/metal/sequence.c — batched-submit command sequencing and flush tracking.
 *
 * Layer: BACKEND (metal). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "metal_internal.h"

bool metal_ranges_overlap(size_t a_offset, size_t b_offset, size_t n_bytes) {
    if (n_bytes == 0) {
        return false;
    }
    return a_offset < b_offset + n_bytes && b_offset < a_offset + n_bytes;
}

/* Drain the pipelined (committed, unwaited) command buffers: optionally
 * surface their errors, always release. Buffers execute in commit order,
 * so when the caller has waited on the FINAL buffer these are complete. */
static void metal_sequence_drain_pending(struct metal_state *st, bool *out_failed) {
    for (uint32_t i = 0; i < st->seq_pending_count; i++) {
        void *cmd = st->seq_pending_cmds[i];
        if (out_failed != nullptr && metal_msg_send_id0(st, cmd, "error") != nullptr) {
            *out_failed = true;
        }
        metal_msg_send_void0(st, cmd, "release");
        st->seq_pending_cmds[i] = nullptr;
    }
    st->seq_pending_count = 0;
}

/* Op-start encoder accessor with pipelined rotation. Called ONLY at op
 * boundaries — never mid-op, an op's local `enc` must stay valid across
 * its own dispatches. Returns the current encoder (possibly fresh). */
void *metal_sequence_encoder(struct metal_state *st) {
    if (st == nullptr || !st->sequence_active) {
        return nullptr;
    }
    if (st->seq_rotate_every == 0u || st->sequence_compute_encoder == nullptr ||
        (uint32_t) st->seq_dispatch_count - st->seq_disp_at_rotate < st->seq_rotate_every) {
        return st->sequence_compute_encoder;
    }
    if (st->seq_pending_count >=
        (uint32_t) (sizeof(st->seq_pending_cmds) / sizeof(st->seq_pending_cmds[0]))) {
        /* Backpressure: encode has outrun the GPU by 16 buffers — wait
         * for the oldest before opening another. */
        metal_msg_send_void0(st, st->seq_pending_cmds[0], "waitUntilCompleted");
        metal_msg_send_void0(st, st->seq_pending_cmds[0], "release");
        st->seq_pending_count--;
        memmove(&st->seq_pending_cmds[0],
                &st->seq_pending_cmds[1],
                st->seq_pending_count * sizeof(st->seq_pending_cmds[0]));
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return st->sequence_compute_encoder; /* keep the old buffer */
    }
    metal_msg_send_void0(st, cmd, "retain");
    metal_msg_send_void0(st, enc, "retain");
    metal_msg_send_void0(st, st->sequence_compute_encoder, "endEncoding");
    metal_msg_send_void0(st, st->sequence_command_buffer, "commit");
    metal_msg_send_void0(st, st->sequence_compute_encoder, "release");
    st->seq_pending_cmds[st->seq_pending_count++] =
            st->sequence_command_buffer; /* stays retained until drain */
    st->sequence_command_buffer  = cmd;
    st->sequence_compute_encoder = enc;
    st->seq_disp_at_rotate       = (uint32_t) st->seq_dispatch_count;
    return enc;
}

void metal_release_sequence_objects(struct metal_state *st) {
    if (st == nullptr) {
        return;
    }
    metal_sequence_drain_pending(st, nullptr);
    metal_msg_send_void0(st, st->sequence_compute_encoder, "release");
    metal_msg_send_void0(st, st->sequence_command_buffer, "release");
    st->sequence_compute_encoder = nullptr;
    st->sequence_command_buffer  = nullptr;
    st->sequence_active          = false;
    st->sequence_has_work        = false;
}

void metal_seq_ref_clear(struct metal_state *st) {
    memset(st->seq_ref, 0, sizeof(st->seq_ref));
    st->seq_ref_count    = 0;
    st->seq_ref_overflow = false;
}

void metal_seq_mark_buffer(struct metal_state *st, void *mtl_buf) {
    if (st == nullptr || !st->sequence_active || mtl_buf == nullptr) {
        return;
    }
    const size_t mask = (sizeof(st->seq_ref) / sizeof(st->seq_ref[0])) - 1u;
    size_t       h    = ((uintptr_t) mtl_buf >> 4) & mask;
    for (size_t i = 0; i <= mask; i++) {
        const size_t slot = (h + i) & mask;
        if (st->seq_ref[slot] == mtl_buf) {
            return;
        }
        if (st->seq_ref[slot] == nullptr) {
            st->seq_ref[slot] = mtl_buf;
            if (++st->seq_ref_count > mask - 256u) {
                st->seq_ref_overflow = true;
            }
            return;
        }
    }
    st->seq_ref_overflow = true;
}

bool metal_seq_references(struct metal_state *st, const void *mtl_buf) {
    if (st == nullptr || !st->sequence_active || mtl_buf == nullptr) {
        return false;
    }
    if (st->seq_ref_overflow) {
        return true;
    }
    const size_t mask = (sizeof(st->seq_ref) / sizeof(st->seq_ref[0])) - 1u;
    size_t       h    = ((uintptr_t) mtl_buf >> 4) & mask;
    for (size_t i = 0; i <= mask; i++) {
        const size_t slot = (h + i) & mask;
        if (st->seq_ref[slot] == mtl_buf) {
            return true;
        }
        if (st->seq_ref[slot] == nullptr) {
            return false;
        }
    }
    return true;
}

/* Submit the open batch and start a fresh one of the same kind. Called
 * whenever the host is about to read or overwrite GPU-referenced memory. */
void metal_batch_flush(struct metal_state *st) {
    if (st == nullptr || !st->sequence_active || !st->sequence_has_work) {
        return;
    }
    struct geist_backend                  *be   = st->backend;
    const enum geist_command_sequence_kind kind = st->sequence_kind;
    (void) metal_command_sequence_end(be, st->sequence_token, true);
    metal_seq_ref_clear(st);
    int tok = 0;
    (void) metal_command_sequence_begin(be, kind, &tok);
}

void metal_flush_if_referenced(struct metal_state *st, const void *mtl_buf) {
    if (metal_seq_references(st, mtl_buf)) {
        static _Atomic int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("GEIST_SEQ_COUNT");
            dbg           = (e && e[0]) ? 1 : 0;
        }
        if (dbg) {
            fprintf(stderr, "[flush] buf=%p\n", mtl_buf);
        }
        metal_batch_flush(st);
    }
}

[[nodiscard]] enum geist_status metal_command_sequence_begin(struct geist_backend            *be,
                                                             enum geist_command_sequence_kind kind,
                                                             int *out_token) {

    if (out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_token = 0;
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->sequence_active) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal command sequence: nested begin");
        return GEIST_E_INVALID_ARG;
    }
    switch (kind) {
    case GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY:
    case GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP:
    case GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP:
    case GEIST_COMMAND_SEQUENCE_PREFILL_TEXT:
        break;
    default:
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal command sequence: invalid kind");
        return GEIST_E_INVALID_ARG;
    }

    metal_capture_begin(st, kind);

    static int g_seq_created;
    g_seq_created++;
    if (getenv("GEIST_SEQ_COUNT") && (g_seq_created % 16) == 0)
        fprintf(stderr, "[seqdbg] created=%d\n", g_seq_created);
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal command sequence: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal command sequence: encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_msg_send_void0(st, cmd, "retain");
    metal_msg_send_void0(st, enc, "retain");

    if (st->sequence_token == INT_MAX) {
        st->sequence_token = 0;
    }
    st->sequence_token++;
    st->sequence_kind            = kind;
    st->sequence_command_buffer  = cmd;
    st->sequence_compute_encoder = enc;
    st->sequence_active          = true;
    st->sequence_has_work        = false;
    st->seq_dispatch_count       = 0;
    st->seq_disp_at_rotate       = 0;
    st->seq_begin_ns             = metal_now_ns();
    *out_token                   = st->sequence_token;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
metal_command_sequence_end(struct geist_backend *be, int token, bool submit) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (!st->sequence_active || token == 0 || token != st->sequence_token) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal command sequence: invalid token");
        return GEIST_E_INVALID_ARG;
    }

    void      *cmd               = st->sequence_command_buffer;
    void      *enc               = st->sequence_compute_encoder;
    const bool has_work          = st->sequence_has_work;
    st->sequence_compute_encoder = nullptr;
    st->sequence_command_buffer  = nullptr;
    st->sequence_active          = false;
    st->sequence_has_work        = false;

    metal_msg_send_void0(st, enc, "endEncoding");
    enum geist_status out = GEIST_OK;
    /* Commit even when no work was encoded: an uncommitted command buffer
     * permanently occupies one of the queue's (default 64) slots, and the
     * batched-submit region hooks legitimately close empty sequences. */
    if (submit && !has_work) {
        /* Empty sequence (e.g. the rotation right after a flush): commit
         * to free the queue slot, but skip the ~0.4 ms wait handshake —
         * nothing observes its completion. */
        metal_msg_send_void0(st, cmd, "commit");
    } else if (submit) {
        const enum metal_profile_stage wait_stage =
                metal_profile_wait_stage_for_sequence(st->sequence_kind);
        const uint64_t wait_start_ns = st->profile_enabled ? metal_now_ns() : 0;
        const bool     seq_trace     = metal_env_enabled("GEIST_METAL_SEQ_TRACE");
        const uint64_t commit_ns     = seq_trace ? metal_now_ns() : 0;
        metal_msg_send_void0(st, cmd, "commit");
        metal_msg_send_void0(st, cmd, "waitUntilCompleted");
        /* Pipelined buffers committed before this one are complete now
         * (same queue, commit order); surface their errors and release. */
        bool pending_failed = false;
        metal_sequence_drain_pending(st, &pending_failed);
        if (pending_failed && out == GEIST_OK) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal command sequence: pipelined buffer failed");
            out = GEIST_E_BACKEND;
        }
        metal_profile_add_wait(st, wait_stage, wait_start_ns);
        if (seq_trace) {
            const uint64_t done_ns = metal_now_ns();
            union {
                void *raw;
                double (*fn)(void *, void *);
            } getd                 = {.raw = st->objc_msgSend};
            const double gpu_start = getd.fn(cmd, metal_sel_register_name(st, "GPUStartTime"));
            const double gpu_end   = getd.fn(cmd, metal_sel_register_name(st, "GPUEndTime"));
            fprintf(stderr,
                    "seq_trace kind=%d dispatches=%llu encode=%.1fms "
                    "gpu=%.1fms wall=%.1fms wait=%.1fms\n",
                    (int) st->sequence_kind,
                    (unsigned long long) st->seq_dispatch_count,
                    (double) (commit_ns - st->seq_begin_ns) / 1e6,
                    (gpu_end - gpu_start) * 1e3,
                    (double) (done_ns - st->seq_begin_ns) / 1e6,
                    (double) (done_ns - commit_ns) / 1e6);
        }
        void *err = metal_msg_send_id0(st, cmd, "error");
        if (err != nullptr) {
            geist_backend_set_error(be, GEIST_E_BACKEND, "metal command sequence: command failed");
            out = GEIST_E_BACKEND;
        }
    }

    /* No-op on the normal submit path (already drained after the wait);
     * covers the empty/!submit paths where pendings were committed but
     * never waited on — release only, Metal keeps its internal refs. */
    metal_sequence_drain_pending(st, nullptr);
    metal_capture_end(st);

    metal_msg_send_void0(st, enc, "release");
    metal_msg_send_void0(st, cmd, "release");
    return out;
}
