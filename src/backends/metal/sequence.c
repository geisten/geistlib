/*
 * src/backends/metal/sequence.c — batched-submit command sequencing and flush tracking.
 */
#include "metal_internal.h"

bool metal_ranges_overlap(size_t a_offset, size_t b_offset, size_t n_bytes) {
    if (n_bytes == 0) {
        return false;
    }
    return a_offset < b_offset + n_bytes && b_offset < a_offset + n_bytes;
}

/* ---- Concurrent sequence encoder (#590): see metal_state.seq_concurrent ---- */

static size_t metal_pipeline_access_home(const struct metal_pipeline_h *pipeline) {
    return (size_t) ((((uint64_t) (uintptr_t) pipeline >> 4) * 0x9e3779b97f4a7c15u) >> 40) &
           ((size_t) METAL_PIPELINE_ACCESS_SLOTS - 1u);
}

void metal_pipeline_access_add(struct metal_state            *st,
                               const struct metal_pipeline_h *pipeline,
                               struct metal_function_h       *function) {
    const size_t mask = (size_t) METAL_PIPELINE_ACCESS_SLOTS - 1u;
    const size_t home = metal_pipeline_access_home(pipeline);
    for (size_t i = 0; i <= mask; i++) {
        struct metal_pipeline_access *slot = &st->pipeline_access[(home + i) & mask];
        if (slot->pipeline == nullptr || slot->pipeline == pipeline) {
            *slot = (struct metal_pipeline_access) {.pipeline = pipeline, .function = function};
            return;
        }
    }
    /* Full: the pipeline is not found later and counts as writing everything. */
}

/* Reads which buffer arguments the slot's kernel has and which it may write
 * from a pipeline reflection (a second, discarded pipeline state of the
 * same function: Metal hands the reflection out only at creation). Anything
 * unexpected leaves every argument written, the safe answer. */
static void metal_pipeline_access_resolve(struct metal_state           *st,
                                          struct metal_pipeline_access *slot) {
    slot->resolved = true;
    slot->used = slot->write = UINT32_MAX;
    void *reflection         = nullptr;
    void *err                = nullptr;
    /* 1 = MTLPipelineOptionBindingInfo */
    void *twin = metal_msg_send_id_id_ulong_out_err(
            st,
            st->device,
            "newComputePipelineStateWithFunction:options:reflection:error:",
            slot->function,
            1,
            &reflection,
            &err);
    if (twin != nullptr && metal_responds(st, reflection, "bindings")) {
        void               *bindings = metal_msg_send_id0(st, reflection, "bindings");
        const unsigned long n        = metal_msg_send_ulong0(st, bindings, "count");
        bool                known    = bindings != nullptr;
        uint32_t            used = 0, write = 0;
        for (unsigned long i = 0; i < n && known; i++) {
            void *binding = metal_msg_send_id_ulong(st, bindings, "objectAtIndex:", i);
            if (metal_msg_send_ulong0(st, binding, "type") != 0) { /* 0 = MTLBindingTypeBuffer */
                continue;
            }
            const unsigned long index = metal_msg_send_ulong0(st, binding, "index");
            if (index >= (unsigned long) METAL_BUFFER_ARGS) {
                known = false;
                break;
            }
            used |= 1u << index;
            if (metal_msg_send_ulong0(st, binding, "access") != 0) { /* 0 = ...AccessReadOnly */
                write |= 1u << index;
            }
        }
        if (known) {
            slot->used  = used;
            slot->write = write;
        }
    }
    metal_msg_send_void0(st, twin, "release");
}

/* At a sequence's end, the GPU idle: resolve up to four of the pipelines
 * the sequences bound unresolved, so no token waits for more than a few. */
static void metal_pipeline_access_resolve_some(struct metal_state *st) {
    uint32_t done = 0;
    while (st->pipeline_access_n_wanted != 0 && done < 4u) {
        struct metal_pipeline_access *slot =
                &st->pipeline_access[st->pipeline_access_wanted[--st->pipeline_access_n_wanted]];
        if (!slot->resolved) {
            metal_pipeline_access_resolve(st, slot);
            done++;
        }
    }
}

void metal_seq_note_pipeline(struct metal_state            *st,
                             const struct metal_encoder_h  *enc,
                             const struct metal_pipeline_h *pipeline) {
    if (!st->seq_concurrent || enc != st->sequence_compute_encoder) {
        return;
    }
    st->seq_used = st->seq_write = UINT32_MAX;
    const size_t mask            = (size_t) METAL_PIPELINE_ACCESS_SLOTS - 1u;
    const size_t home            = metal_pipeline_access_home(pipeline);
    for (size_t i = 0; i <= mask; i++) {
        struct metal_pipeline_access *slot = &st->pipeline_access[(home + i) & mask];
        if (slot->pipeline == pipeline) {
            if (slot->resolved) {
                st->seq_used  = slot->used;
                st->seq_write = slot->write;
            } else if (st->pipeline_access_n_wanted < (uint32_t) METAL_PIPELINE_ACCESS_WANTED) {
                /* Listed again at each bind until resolved: cheap, and the
                 * resolver skips a slot it has done. */
                st->pipeline_access_wanted[st->pipeline_access_n_wanted++] =
                        (uint16_t) ((home + i) & mask);
            }
            return;
        }
        if (slot->pipeline == nullptr) {
            return;
        }
    }
}

static bool
metal_seq_has(size_t n, const struct metal_buffer_h *const set[], const struct metal_buffer_h *b) {
    for (size_t i = 0; i < n; i++) {
        if (set[i] == b) {
            return true;
        }
    }
    return false;
}

/* A fresh encoder has no dispatch to order against and an empty argument
 * table. */
static void metal_seq_order_reset(struct metal_state *st) {
    st->seq_nwrote = st->seq_nread = 0;
    st->seq_used = st->seq_write = UINT32_MAX;
    memset(st->seq_bound, 0, sizeof(st->seq_bound));
}

void metal_seq_order(struct metal_state *st, struct metal_encoder_h *enc) {
    if (!st->seq_concurrent || enc != st->sequence_compute_encoder) {
        return;
    }
    /* A dispatch may overlap the ones since the last barrier unless it
     * reads what one of them wrote, or writes what one read or wrote. */
    bool clash =
            (size_t) st->seq_nwrote + (size_t) METAL_BUFFER_ARGS > (size_t) METAL_SEQ_TRACKED ||
            (size_t) st->seq_nread + (size_t) METAL_BUFFER_ARGS > (size_t) METAL_SEQ_TRACKED;
    for (uint32_t i = 0; i < (uint32_t) METAL_BUFFER_ARGS && !clash; i++) {
        if (((st->seq_used >> i) & 1u) == 0 || st->seq_bound[i] == nullptr) {
            continue;
        }
        clash = metal_seq_has(st->seq_nwrote, st->seq_wrote, st->seq_bound[i]) ||
                (((st->seq_write >> i) & 1u) != 0 &&
                 metal_seq_has(st->seq_nread, st->seq_read, st->seq_bound[i]));
    }
    if (clash) {
        metal_msg_send_void_ulong(st, enc, "memoryBarrierWithScope:", 1); /* ...ScopeBuffers */
        st->seq_nwrote = st->seq_nread = 0;
        st->seq_barriers++;
    }
    st->seq_tracked_dispatches++;
    for (uint32_t i = 0; i < (uint32_t) METAL_BUFFER_ARGS; i++) {
        if (((st->seq_used >> i) & 1u) == 0 || st->seq_bound[i] == nullptr) {
            continue;
        }
        const struct metal_buffer_h *b = st->seq_bound[i];
        if (((st->seq_write >> i) & 1u) != 0) {
            if (!metal_seq_has(st->seq_nwrote, st->seq_wrote, b)) {
                st->seq_wrote[st->seq_nwrote++] = b;
            }
        } else if (!metal_seq_has(st->seq_nread, st->seq_read, b)) {
            st->seq_read[st->seq_nread++] = b;
        }
    }
}

/* The compute encoder of a sequence's command buffer: concurrent unless
 * GEIST_METAL_CONCURRENT=0. 1 = MTLDispatchTypeConcurrent. */
static struct metal_encoder_h *metal_seq_new_encoder(struct metal_state    *st,
                                                     struct metal_cmdbuf_h *cmd) {
    return st->seq_concurrent
                   ? metal_msg_send_id_ulong(st, cmd, "computeCommandEncoderWithDispatchType:", 1)
                   : metal_msg_send_id0(st, cmd, "computeCommandEncoder");
}

/* Drain the pipelined (committed, unwaited) command buffers: optionally
 * surface their errors, always release. Buffers execute in commit order,
 * so when the caller has waited on the FINAL buffer these are complete. */
static void metal_sequence_drain_pending(struct metal_state *st, bool *out_failed) {
    for (uint32_t i = 0; i < st->seq_pending_count; i++) {
        struct metal_cmdbuf_h *cmd = st->seq_pending_cmds[i];
        if (out_failed != nullptr && metal_msg_send_id0(st, cmd, "error") != nullptr) {
            *out_failed = true;
        }
        metal_profile_add_cmd_buffer(st, cmd, st->seq_pending_mask[i]);
        metal_msg_send_void0(st, cmd, "release");
        st->seq_pending_cmds[i] = nullptr;
    }
    st->seq_pending_count = 0;
}

/* Op-start encoder accessor with pipelined rotation. Called ONLY at op
 * boundaries — never mid-op, an op's local `enc` must stay valid across
 * its own dispatches. Returns the current encoder (possibly fresh). */
struct metal_encoder_h *metal_sequence_encoder(struct metal_state *st) {
    if (st == nullptr || !metal_seq_mine(st)) {
        return nullptr;
    }
    const uint32_t every = st->sequence_kind == GEIST_COMMAND_SEQUENCE_PREFILL_TEXT
                                   ? st->seq_rotate_prefill
                                   : st->seq_rotate_every;
    if (every == 0u || st->sequence_compute_encoder == nullptr ||
        (uint32_t) st->seq_dispatch_count - st->seq_disp_at_rotate < every) {
        return st->sequence_compute_encoder;
    }
    if (st->seq_pending_count >=
        (uint32_t) (sizeof(st->seq_pending_cmds) / sizeof(st->seq_pending_cmds[0]))) {
        /* Backpressure: encode has outrun the GPU by 16 buffers — wait
         * for the oldest before opening another. */
        metal_msg_send_void0(st, st->seq_pending_cmds[0], "waitUntilCompleted");
        metal_profile_add_cmd_buffer(st, st->seq_pending_cmds[0], st->seq_pending_mask[0]);
        metal_msg_send_void0(st, st->seq_pending_cmds[0], "release");
        st->seq_pending_count--;
        memmove(&st->seq_pending_cmds[0],
                &st->seq_pending_cmds[1],
                st->seq_pending_count * sizeof(st->seq_pending_cmds[0]));
        memmove(&st->seq_pending_mask[0],
                &st->seq_pending_mask[1],
                st->seq_pending_count * sizeof(st->seq_pending_mask[0]));
    }
    struct metal_cmdbuf_h  *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    struct metal_encoder_h *enc = cmd != nullptr ? metal_seq_new_encoder(st, cmd) : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return st->sequence_compute_encoder; /* keep the old buffer */
    }
    metal_msg_send_void0(st, cmd, "retain");
    metal_msg_send_void0(st, enc, "retain");
    metal_msg_send_void0(st, st->sequence_compute_encoder, "endEncoding");
    metal_msg_send_void0(st, st->sequence_command_buffer, "commit");
    metal_msg_send_void0(st, st->sequence_compute_encoder, "release");
    st->seq_pending_mask[st->seq_pending_count] = st->seq_cur_mask;
    st->seq_pending_cmds[st->seq_pending_count++] =
            st->sequence_command_buffer; /* stays retained until drain */
    st->seq_cur_mask             = 0;
    st->sequence_command_buffer  = cmd;
    st->sequence_compute_encoder = enc;
    st->seq_disp_at_rotate       = (uint32_t) st->seq_dispatch_count;
    metal_seq_order_reset(st);
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
    atomic_store_explicit(&st->seq_owner, (uintptr_t) 0, memory_order_relaxed);
}

/* Empties only the occupied slots: a flush-heavy batch (a host fallback per
 * layer) clears the set once per flush. */
void metal_seq_ref_clear(struct metal_state *st) {
    for (size_t i = 0; i < st->seq_ref_count; i++) {
        st->seq_ref[st->seq_ref_used[i]] = (struct metal_seq_ref) {0};
    }
    st->seq_ref_count    = 0;
    st->seq_ref_overflow = false;
}

/* Slot hash of a (MTLBuffer, bind offset) pair. */
static size_t metal_seq_ref_hash(const struct metal_buffer_h *buf, size_t off, size_t mask) {
    const uint64_t x = ((uint64_t) (uintptr_t) buf >> 4) ^ ((uint64_t) off * 0x9e3779b97f4a7c15u);
    return (size_t) (x ^ (x >> 29)) & mask;
}

void metal_seq_mark_buffer(struct metal_state *st, struct metal_buffer_h *mtl_buf, size_t off) {
    if (st == nullptr || !metal_seq_mine(st) || mtl_buf == nullptr) {
        return;
    }
    const size_t mask = (sizeof(st->seq_ref) / sizeof(st->seq_ref[0])) - 1u;
    const size_t h    = metal_seq_ref_hash(mtl_buf, off, mask);
    for (size_t i = 0; i <= mask; i++) {
        const size_t          slot = (h + i) & mask;
        struct metal_seq_ref *e    = &st->seq_ref[slot];
        if (e->buf == mtl_buf && e->off == off) {
            return;
        }
        if (e->buf == nullptr) {
            *e = (struct metal_seq_ref) {.buf = mtl_buf, .off = off};
            st->seq_ref_used[st->seq_ref_count] = (uint16_t) slot;
            if (++st->seq_ref_count > mask - 256u) {
                st->seq_ref_overflow = true;
            }
            return;
        }
    }
    st->seq_ref_overflow = true;
}

/* A scan of the occupied slots, not a probe — the question is about a range
 * of offsets — and it runs only on a host map/upload/download, never per
 * dispatch. Every handle binds at its own base_off (tensor offsets travel
 * in the kernel params), so a bind offset inside the range means a handle
 * there was bound. That is exact while no bound handle starts before a
 * queried view and runs into it: views of one buffer are disjoint slices
 * (#528), and the buffer they are cut from is never bound itself. */
bool metal_seq_references(struct metal_state          *st,
                          const struct metal_buffer_h *mtl_buf,
                          size_t                       off,
                          size_t                       n) {
    if (st == nullptr || !metal_seq_mine(st) || mtl_buf == nullptr) {
        return false;
    }
    if (st->seq_ref_overflow) {
        return true;
    }
    for (size_t i = 0; i < st->seq_ref_count; i++) {
        const struct metal_seq_ref *e = &st->seq_ref[st->seq_ref_used[i]];
        if (e->buf == mtl_buf && e->off >= off && e->off - off < n) {
            return true;
        }
    }
    return false;
}

/* Submit the open batch and start a fresh one of the same kind. Called
 * whenever the host is about to read or overwrite GPU-referenced memory. */
void metal_batch_flush(struct metal_state *st) {
    if (st == nullptr || !metal_seq_mine(st) || !st->sequence_has_work) {
        return;
    }
    struct geist_backend                  *be   = st->backend;
    const enum geist_command_sequence_kind kind = st->sequence_kind;
    (void) metal_command_sequence_end(be, st->sequence_token, true);
    metal_seq_ref_clear(st);
    int tok = 0;
    (void) metal_command_sequence_begin(be, kind, &tok);
}

void metal_flush_if_referenced(struct metal_state          *st,
                               const struct metal_buffer_h *mtl_buf,
                               size_t                       off,
                               size_t                       n) {
    if (metal_seq_references(st, mtl_buf, off, n)) {
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
    bool known = false;
    switch (kind) {
    case GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY:
    case GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP:
    case GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP:
    case GEIST_COMMAND_SEQUENCE_PREFILL_TEXT:
        known = true;
        break;
    }
    if (!known) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal command sequence: invalid kind");
        return GEIST_E_INVALID_ARG;
    }

    /* Everything the sequence autoreleases -- its command buffers (rotation
     * included) and encoders -- drains when _end pops this pool. What must
     * outlive it is retained explicitly: the current buffer and encoder
     * below, the pipelined ones in seq_pending_cmds. */
    struct metal_pool_h *pool = metal_pool_push(st);
    metal_capture_begin(st, kind);

    struct metal_cmdbuf_h *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        metal_pool_pop(st, pool);
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal command sequence: command buffer failed");
        return GEIST_E_BACKEND;
    }
    struct metal_encoder_h *enc = metal_seq_new_encoder(st, cmd);
    if (enc == nullptr) {
        metal_pool_pop(st, pool);
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
    st->sequence_pool            = pool;
    st->sequence_active          = true;
    st->sequence_has_work        = false;
    atomic_store_explicit(&st->seq_owner, (uintptr_t) pthread_self(), memory_order_relaxed);
    metal_seq_order_reset(st);
    st->seq_dispatch_count = 0;
    st->seq_disp_at_rotate = 0;
    st->seq_cur_mask       = 0;
    st->seq_begin_ns       = geist_now_ns();
    *out_token             = st->sequence_token;
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

    struct metal_cmdbuf_h  *cmd      = st->sequence_command_buffer;
    struct metal_encoder_h *enc      = st->sequence_compute_encoder;
    struct metal_pool_h    *pool     = st->sequence_pool;
    const bool              has_work = st->sequence_has_work;
    st->sequence_compute_encoder     = nullptr;
    st->sequence_command_buffer      = nullptr;
    st->sequence_pool                = nullptr;
    st->sequence_active              = false;
    st->sequence_has_work            = false;
    atomic_store_explicit(&st->seq_owner, (uintptr_t) 0, memory_order_relaxed);

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
        const uint64_t wait_start_ns = st->profile_enabled ? geist_now_ns() : 0;
        const bool     seq_trace     = geist_env_flag("GEIST_METAL_SEQ_TRACE", false);
        const uint64_t commit_ns     = seq_trace ? geist_now_ns() : 0;
        metal_msg_send_void0(st, cmd, "commit");
        metal_msg_send_void0(st, cmd, "waitUntilCompleted");
        /* Pipelined buffers committed before this one are complete now
         * (same queue, commit order); surface their errors and release. */
        bool pending_failed = false;
        metal_sequence_drain_pending(st, &pending_failed);
        metal_profile_add_cmd_buffer(st, cmd, st->seq_cur_mask);
        if (pending_failed && out == GEIST_OK) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal command sequence: pipelined buffer failed");
            out = GEIST_E_BACKEND;
        }
        metal_profile_add_wait(st, wait_stage, wait_start_ns);
        if (seq_trace) {
            const uint64_t done_ns = geist_now_ns();
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

    metal_pipeline_access_resolve_some(st);

    /* No-op on the normal submit path (already drained after the wait);
     * covers the empty/!submit paths where pendings were committed but
     * never waited on — release only, Metal keeps its internal refs. */
    metal_sequence_drain_pending(st, nullptr);
    metal_capture_end(st);

    metal_msg_send_void0(st, enc, "release");
    metal_msg_send_void0(st, cmd, "release");
    metal_pool_pop(st, pool);
    return out;
}
