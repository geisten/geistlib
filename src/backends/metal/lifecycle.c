/*
 * src/backends/metal/lifecycle.c — device/queue lifecycle and runtime loading.
 */
#include "metal_internal.h"
#include "parse.h"

static void metal_destroy_state(struct geist_backend *be, struct metal_state *st) {
    if (be == nullptr || st == nullptr) {
        return;
    }
    metal_profile_print_summary(st);
    /* First: the set keeps its members alive, the raw buffers below among
     * them. */
    metal_residency_destroy(st);
    safe_free((void **) &st->buf_reg);
    st->buf_reg_count = 0;
    st->buf_reg_cap   = 0;
    metal_buffer_destroy_internal(be, st->attn_dec_partials_buffer);
    st->attn_dec_partials_buffer = nullptr;
    metal_buffer_destroy_internal(be, st->pq2_xh_buffer);
    st->pq2_xh_buffer = nullptr;
    metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
    metal_buffer_destroy_internal(be, st->attn_vf16_buffer);
    st->attn_kf16_buffer = nullptr;
    st->attn_vf16_buffer = nullptr;
    if (st->objc_msgSend != nullptr && st->sel_registerName != nullptr) {
        if (st->sequence_active) {
            metal_msg_send_void0(st, st->sequence_compute_encoder, "endEncoding");
        }
        metal_release_sequence_objects(st);
        metal_release_pipeline_tables(st);
        metal_msg_send_void0(st, st->argmax_result_buffer, "release");
        metal_msg_send_void0(st, st->argmax_part_buffer, "release");
        metal_msg_send_void0(st, st->dn_scratch, "release");
        metal_msg_send_void0(st, st->command_queue, "release");
        metal_msg_send_void0(st, st->device, "release");
    }
    if (st->metal_handle != nullptr) {
        dlclose(st->metal_handle);
        st->metal_handle = nullptr;
    }
    if (st->objc_handle != nullptr) {
        dlclose(st->objc_handle);
        st->objc_handle = nullptr;
    }
    pthread_mutex_destroy(&st->seq_lock);
    geist_backend_free(be, st);
}

void metal_destroy(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    metal_destroy_state(be, be->state);
    be->state = nullptr;
}

[[nodiscard]] static enum geist_status metal_load_runtime(struct geist_backend *be,
                                                          struct metal_state   *st) {

#if defined(__APPLE__)
    static const char *const metal_paths[] = {
            "/System/Library/Frameworks/Metal.framework/Metal",
            "Metal.framework/Metal",
            nullptr,
    };
    static const char *const objc_paths[] = {
            "/usr/lib/libobjc.A.dylib",
            "libobjc.A.dylib",
            nullptr,
    };

    for (size_t i = 0; metal_paths[i] != nullptr; i++) {
        st->metal_handle = dlopen(metal_paths[i], RTLD_NOW | RTLD_LOCAL);
        if (st->metal_handle != nullptr) {
            break;
        }
    }
    if (st->metal_handle == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "metal: Metal framework is unavailable");
        return GEIST_E_UNSUPPORTED;
    }

    for (size_t i = 0; objc_paths[i] != nullptr; i++) {
        st->objc_handle = dlopen(objc_paths[i], RTLD_NOW | RTLD_LOCAL);
        if (st->objc_handle != nullptr) {
            break;
        }
    }
    if (st->objc_handle == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "metal: Objective-C runtime is unavailable");
        return GEIST_E_UNSUPPORTED;
    }

    st->MTLCreateSystemDefaultDevice =
            metal_dlsym(st->metal_handle, "MTLCreateSystemDefaultDevice");
    st->objc_msgSend             = metal_dlsym(st->objc_handle, "objc_msgSend");
    st->sel_registerName         = metal_dlsym(st->objc_handle, "sel_registerName");
    st->objc_getClass            = metal_dlsym(st->objc_handle, "objc_getClass");
    st->objc_autoreleasePoolPush = metal_dlsym(st->objc_handle, "objc_autoreleasePoolPush");
    st->objc_autoreleasePoolPop  = metal_dlsym(st->objc_handle, "objc_autoreleasePoolPop");
    if (st->MTLCreateSystemDefaultDevice == nullptr || st->objc_msgSend == nullptr ||
        st->sel_registerName == nullptr || st->objc_getClass == nullptr ||
        st->objc_autoreleasePoolPush == nullptr || st->objc_autoreleasePoolPop == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: runtime symbols are incomplete");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
#else
    (void) st;
    geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED, "metal: backend is only available on Apple platforms");
    return GEIST_E_UNSUPPORTED;
#endif
}

[[nodiscard]] enum geist_status metal_create(struct geist_backend            *be,
                                             const struct geist_backend_opts *opts) {

    (void) opts;
    struct metal_state *st = geist_backend_alloc(be, sizeof(*st), alignof(struct metal_state));
    if (st == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "metal: failed to allocate %zu-byte state", sizeof(*st));
        return GEIST_E_OOM;
    }
    *st         = (struct metal_state) {0};
    st->backend = be;
    pthread_mutexattr_t seq_lock_attr;
    pthread_mutexattr_init(&seq_lock_attr);
    pthread_mutexattr_settype(&seq_lock_attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&st->seq_lock, &seq_lock_attr);
    pthread_mutexattr_destroy(&seq_lock_attr);
    metal_tuning_init(be, st);
    /* Command-buffer pipelining: rotate every N dispatches, default 192
     * (~3 buffers per decode token; llama.cpp's n_cb optimum is 2-3).
     * GEIST_METAL_PIPELINE=0 disables, =N sets the rotation period.
     *
     * Prefill rotates every 16 dispatches (GEIST_METAL_PIPELINE_PREFILL=N
     * overrides; it defaults to GEIST_METAL_PIPELINE when that is set). On
     * an M1 Max the same kernels ran ~15 % faster in short buffers: Bonsai
     * 2 27B prefill at 1245 tokens went from ~102 to ~119 tok/s with 8-32
     * dispatches per buffer, 64 was in between, and splitting one buffer
     * into more encoders changed nothing (#77). The sum of the buffers'
     * GPU times fell by the same amount while the GPU stayed busy, so the
     * work itself ran faster: the GPU's performance state follows short
     * buffers better than long ones. */
    const char *pipeline_env = getenv("GEIST_METAL_PIPELINE");
    long        rotate       = 0;
    const bool  rotate_set =
            geist_parse_long(pipeline_env, &rotate) && rotate >= 0 && rotate <= UINT32_MAX;
    st->seq_rotate_every = rotate_set ? (uint32_t) rotate : 192u;
    long rotate_prefill  = 0;
    st->seq_rotate_prefill =
            geist_parse_long(getenv("GEIST_METAL_PIPELINE_PREFILL"), &rotate_prefill) &&
                            rotate_prefill >= 0 && rotate_prefill <= UINT32_MAX
                    ? (uint32_t) rotate_prefill
            : rotate_set ? (uint32_t) rotate
                         : 16u;
    st->profile_enabled = geist_env_flag("GEIST_METAL_PROFILE", false);

    enum geist_status s = metal_load_runtime(be, st);
    if (s != GEIST_OK) {
        metal_destroy_state(be, st);
        return s;
    }

    st->device = metal_create_default_device(st);
    if (st->device == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "metal: no default Metal device");
        metal_destroy_state(be, st);
        return GEIST_E_UNSUPPORTED;
    }
    metal_msg_send_void0(st, st->device, "retain");
    st->ws_budget = metal_msg_send_ulong0(st, st->device, "recommendedMaxWorkingSetSize");
    st->ws_ignore = geist_env_flag("GEIST_METAL_IGNORE_BUDGET", false);
    /* MTLGPUFamilyApple10 = 1010: M5/A19 and later carry matrix units the
     * tensor API reaches (#314). Earlier GPUs run it on the ALUs, no faster
     * than the simdgroup GEMMs (llama.cpp: M2 Ultra -5 %, M4 even). */
    st->tensor_mm =
            geist_env_flag("GEIST_METAL_TENSOR",
                           metal_msg_send_bool_long(st, st->device, "supportsFamily:", 1010));

    st->command_queue = metal_msg_send_id0(st, st->device, "newCommandQueue");
    if (st->command_queue == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: failed to create command queue");
        metal_destroy_state(be, st);
        return GEIST_E_BACKEND;
    }

    /* Keep what the GPU binds wired while the backend is in use (#530).
     * GEIST_METAL_KEEP_ALIVE_S=n: for n s after the last dispatch (default
     * 180, as llama.cpp); 0 leaves the unwiring to macOS. */
    long keep_s;
    if (!geist_parse_long(getenv("GEIST_METAL_KEEP_ALIVE_S"), &keep_s)) {
        keep_s = 180;
    }
    if (keep_s > 0) {
        metal_residency_create(st, (uint64_t) keep_s);
    }

    void       *name = metal_msg_send_id0(st, st->device, "name");
    const char *utf8 = name != nullptr ? metal_msg_send_cstr0(st, name, "UTF8String") : nullptr;
    snprintf(st->device_name,
             sizeof(st->device_name),
             "%s",
             utf8 != nullptr && utf8[0] != '\0' ? utf8 : "Apple Metal GPU");

    be->state = st;
    return GEIST_OK;
}

/* Batched-submit region hooks. The engine brackets each prefill batch and
 * each decode step with these; one command buffer per region carries every
 * op. Host access to GPU-referenced buffers flushes early
 * (see metal_flush_if_referenced). The engine treats the token as opaque;
 * flushes rotate st->sequence_token, so region_end closes the CURRENT
 * sequence, not the original token. */
int metal_parallel_region_begin(struct geist_backend *be, enum geist_parallel_region region) {
    if (be == nullptr || be->state == nullptr) {
        return 0;
    }
    struct metal_state *st = be->state;
    /* Held until region_end: another session's region waits here instead
     * of mistaking this one for an outer region (#544). */
    pthread_mutex_lock(&st->seq_lock);
    if (metal_seq_mine(st)) {
        pthread_mutex_unlock(&st->seq_lock);
        return 0; /* nested region: leave the outer batch in charge */
    }
    const enum geist_command_sequence_kind kind =
            region == GEIST_REGION_PREFILL_BATCH ? GEIST_COMMAND_SEQUENCE_PREFILL_TEXT
                                                 : GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP;
    int tok = 0;
    if (metal_command_sequence_begin(be, kind, &tok) != GEIST_OK) {
        pthread_mutex_unlock(&st->seq_lock);
        return 0;
    }
    metal_seq_ref_clear(st);
    return tok;
}

void metal_parallel_region_end(struct geist_backend *be, int token) {
    if (be == nullptr || be->state == nullptr || token == 0) {
        return;
    }
    struct metal_state *st = be->state;
    if (!metal_seq_mine(st)) {
        return;
    }
    (void) metal_command_sequence_end(be, st->sequence_token, true);
    metal_seq_ref_clear(st);
    pthread_mutex_unlock(&st->seq_lock);
}
