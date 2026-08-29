/*
 * src/backends/metal/profiling.c — env gates, dispatch/wait profiling, capture, debug stats.
 *
 * Layer: BACKEND (metal). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "metal_internal.h"

const char *const metal_profile_stage_names[METAL_PROFILE_STAGE_COUNT] = {
        [METAL_PROFILE_WAIT_DECODE_LAYER_LOOP]        = "wait.decode_layer_loop",
        [METAL_PROFILE_WAIT_DECODE_GREEDY_STEP]       = "wait.decode_greedy_step",
        [METAL_PROFILE_WAIT_VERIFY_GREEDY]            = "wait.verify_greedy",
        [METAL_PROFILE_WAIT_PREFILL_TEXT]             = "wait.prefill_text",
        [METAL_PROFILE_WAIT_FFN_STANDALONE]           = "wait.ffn_standalone",
        [METAL_PROFILE_DISPATCH_RMSNORM_ROWS]         = "dispatch.rmsnorm_rows",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_BASE]     = "dispatch.q4k_gate_up.base",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4]       = "dispatch.q4k_gate_up.n4",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT4]      = "dispatch.q4k_gate_up.nt4",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT8]      = "dispatch.q4k_gate_up.nt8",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_W4A8]     = "dispatch.q4k_gate_up.w4a8",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE]      = "dispatch.q4k_linear.base",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_MM_FAST]   = "dispatch.q4k_linear.mm_fast",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4]        = "dispatch.q4k_linear.n4",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT4]       = "dispatch.q4k_linear.nt4",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT8]       = "dispatch.q4k_linear.nt8",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_W4A8]      = "dispatch.q4k_linear.w4a8",
        [METAL_PROFILE_DISPATCH_Q4K_PLE_GATE_NT8]     = "dispatch.q4k_ple_gate.nt8",
        [METAL_PROFILE_DISPATCH_F32_PLE_GATE]         = "dispatch.f32_ple_gate",
        [METAL_PROFILE_DISPATCH_Q4K_QUANT_X]          = "dispatch.q4k_quant_x",
        [METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE]      = "dispatch.q6k_linear.base",
        [METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4]        = "dispatch.q6k_linear.n4",
        [METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT4]       = "dispatch.q6k_linear.nt4",
        [METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT8]       = "dispatch.q6k_linear.nt8",
        [METAL_PROFILE_DISPATCH_Q4K_QK_BASE]          = "dispatch.q4k_qk.base",
        [METAL_PROFILE_DISPATCH_Q4K_QK_NT4]           = "dispatch.q4k_qk.nt4",
        [METAL_PROFILE_DISPATCH_F32_PLE_PROJ_NORM]    = "dispatch.f32_ple_proj_norm",
        [METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS]     = "dispatch.rmsnorm_add_rows",
        [METAL_PROFILE_DISPATCH_Q_NORM_ROPE]          = "dispatch.q_norm_rope",
        [METAL_PROFILE_DISPATCH_K_NORM_ROPE_APPEND]   = "dispatch.k_norm_rope_append",
        [METAL_PROFILE_DISPATCH_V_NORM_APPEND]        = "dispatch.v_norm_append",
        [METAL_PROFILE_DISPATCH_KV_NORM_APPEND]       = "dispatch.kv_norm_append",
        [METAL_PROFILE_DISPATCH_ROPE_ROWS]            = "dispatch.rope_rows",
        [METAL_PROFILE_DISPATCH_KV_APPEND_ROWS]       = "dispatch.kv_append_rows",
        [METAL_PROFILE_DISPATCH_ATTENTION_ROWS]       = "dispatch.attention_rows",
        [METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS] = "dispatch.attention_qnorm_rows",
        [METAL_PROFILE_DISPATCH_GELU_MUL_ROWS]        = "dispatch.gelu_mul_rows",
        [METAL_PROFILE_DISPATCH_F32_MATMUL]           = "dispatch.f32_matmul",
        [METAL_PROFILE_DISPATCH_EMBED]                = "dispatch.embed",
        [METAL_PROFILE_DISPATCH_ADD_ROWS]             = "dispatch.add_rows",
        [METAL_PROFILE_DISPATCH_MUL_ROWS]             = "dispatch.mul_rows",
        [METAL_PROFILE_DISPATCH_SCALE_ROWS]           = "dispatch.scale_rows",
        [METAL_PROFILE_DISPATCH_GELU_ROWS]            = "dispatch.gelu_rows",
        [METAL_PROFILE_DISPATCH_COPY_U32]             = "dispatch.copy_u32",
        [METAL_PROFILE_DISPATCH_ARGMAX]               = "dispatch.argmax",
        [METAL_PROFILE_DISPATCH_DELTANET_PREFILL]     = "dispatch.deltanet_prefill",
        [METAL_PROFILE_DISPATCH_DELTANET_DECODE]      = "dispatch.deltanet_decode",
        [METAL_PROFILE_DISPATCH_DN_PREP]              = "dispatch.dn.prep",
        [METAL_PROFILE_DISPATCH_DN_NORM]              = "dispatch.dn.norm",
        [METAL_PROFILE_DISPATCH_DN_STAGE]             = "dispatch.dn.stage",
        [METAL_PROFILE_DISPATCH_DN_SUBST]             = "dispatch.dn.subst",
        [METAL_PROFILE_DISPATCH_DN_WIDE]              = "dispatch.dn.wide",
        [METAL_PROFILE_DISPATCH_QGATE_SPLIT]          = "dispatch.qgate_split",
        [METAL_PROFILE_DISPATCH_SIGMOID_MUL]          = "dispatch.sigmoid_mul",
};

#include "metal_shaders.h"
#include "metal_objc.h"

bool metal_env_enabled(const char *name) {
    const char *value = getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

/* Default-on switches: true only when the env var is explicitly "0". */
bool metal_env_disabled(const char *name) {
    const char *value = getenv(name);
    return value != nullptr && strcmp(value, "0") == 0;
}

uint64_t metal_now_ns(void) {
    struct timespec ts = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static uint64_t metal_saturating_add_u64(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static uint64_t metal_profile_workgroups(struct metal_size groups) {
    if (groups.width == 0 || groups.height == 0 || groups.depth == 0) {
        return 0;
    }
    if (groups.width > UINT64_MAX / groups.height) {
        return UINT64_MAX;
    }
    uint64_t xy = (uint64_t) groups.width * (uint64_t) groups.height;
    if (xy > UINT64_MAX / groups.depth) {
        return UINT64_MAX;
    }
    return xy * (uint64_t) groups.depth;
}

void metal_profile_add_wait(struct metal_state      *st,
                            enum metal_profile_stage stage,
                            uint64_t                 start_ns) {
    if (st == nullptr || !st->profile_enabled || stage >= METAL_PROFILE_STAGE_COUNT ||
        start_ns == 0) {
        return;
    }
    const uint64_t end_ns = metal_now_ns();
    if (end_ns < start_ns) {
        return;
    }
    struct metal_profile_stat *stat = &st->profile[stage];
    stat->ns                        = metal_saturating_add_u64(stat->ns, end_ns - start_ns);
    stat->calls                     = metal_saturating_add_u64(stat->calls, 1);
}

/* Subtractive profiler: env GEIST_SKIP_<CAT> drops a whole op category's
 * dispatches so the wait.prefill_text delta reveals that category's GPU time.
 * GEIST_SKIP_H additionally restricts a category skip to dispatches with a
 * matching threadgroup-grid height (mm_sg height = ceil(n_out/64), so this
 * isolates a single GEMM shape). Output is garbage during a skip run —
 * timing only. */
static bool metal_skip_grid_match(struct metal_size groups) {
    const char *h = getenv("GEIST_SKIP_H");
    return h == nullptr || (uint32_t) atoi(h) == groups.height;
}

static bool metal_skip_stage(enum metal_profile_stage s) {
    switch (s) {
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_BASE:
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4:
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT4:
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT8:
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_W4A8:
        return metal_env_enabled("GEIST_SKIP_GEMM") || metal_env_enabled("GEIST_SKIP_GATE_UP");
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_MM_FAST:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT4:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT8:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_W4A8:
    case METAL_PROFILE_DISPATCH_Q4K_QK_BASE:
    case METAL_PROFILE_DISPATCH_Q4K_QK_NT4:
        return metal_env_enabled("GEIST_SKIP_GEMM") || metal_env_enabled("GEIST_SKIP_Q4K_LINEAR");
    case METAL_PROFILE_DISPATCH_Q4K_PLE_GATE_NT8:
    case METAL_PROFILE_DISPATCH_F32_PLE_GATE:
    case METAL_PROFILE_DISPATCH_F32_PLE_PROJ_NORM:
        return metal_env_enabled("GEIST_SKIP_GEMM") || metal_env_enabled("GEIST_SKIP_PLE");
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT4:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT8:
        return metal_env_enabled("GEIST_SKIP_GEMM") || metal_env_enabled("GEIST_SKIP_Q6K");
    case METAL_PROFILE_DISPATCH_RMSNORM_ROWS:
    case METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS:
        return metal_env_enabled("GEIST_SKIP_NORM");
    case METAL_PROFILE_DISPATCH_ATTENTION_ROWS:
    case METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS:
        return metal_env_enabled("GEIST_SKIP_ATTN");
    case METAL_PROFILE_DISPATCH_Q_NORM_ROPE:
    case METAL_PROFILE_DISPATCH_K_NORM_ROPE_APPEND:
    case METAL_PROFILE_DISPATCH_V_NORM_APPEND:
    case METAL_PROFILE_DISPATCH_KV_NORM_APPEND:
    case METAL_PROFILE_DISPATCH_ROPE_ROWS:
        return metal_env_enabled("GEIST_SKIP_ROPE");
    case METAL_PROFILE_DISPATCH_KV_APPEND_ROWS:
        return metal_env_enabled("GEIST_SKIP_KV");
    case METAL_PROFILE_DISPATCH_GELU_MUL_ROWS:
        return metal_env_enabled("GEIST_SKIP_GELU");
    case METAL_PROFILE_DISPATCH_F32_MATMUL:
        return metal_env_enabled("GEIST_SKIP_F32");
    case METAL_PROFILE_DISPATCH_EMBED:
        return metal_env_enabled("GEIST_SKIP_EMBED");
    case METAL_PROFILE_DISPATCH_ADD_ROWS:
    case METAL_PROFILE_DISPATCH_MUL_ROWS:
    case METAL_PROFILE_DISPATCH_SCALE_ROWS:
    case METAL_PROFILE_DISPATCH_GELU_ROWS:
        return metal_env_enabled("GEIST_SKIP_ELEM");
    case METAL_PROFILE_DISPATCH_COPY_U32:
        return metal_env_enabled("GEIST_SKIP_COPY");
    case METAL_PROFILE_DISPATCH_DELTANET_PREFILL:
    case METAL_PROFILE_DISPATCH_DELTANET_DECODE:
        return metal_env_enabled("GEIST_SKIP_DELTANET");
    case METAL_PROFILE_DISPATCH_DN_PREP:
        return metal_env_enabled("GEIST_SKIP_DELTANET") || metal_env_enabled("GEIST_SKIP_DN_PREP");
    case METAL_PROFILE_DISPATCH_DN_NORM:
        return metal_env_enabled("GEIST_SKIP_DELTANET") || metal_env_enabled("GEIST_SKIP_DN_NORM");
    case METAL_PROFILE_DISPATCH_DN_STAGE:
        return metal_env_enabled("GEIST_SKIP_DELTANET") || metal_env_enabled("GEIST_SKIP_DN_STAGE");
    case METAL_PROFILE_DISPATCH_DN_SUBST:
        return metal_env_enabled("GEIST_SKIP_DELTANET") || metal_env_enabled("GEIST_SKIP_DN_SUBST");
    case METAL_PROFILE_DISPATCH_DN_WIDE:
        return metal_env_enabled("GEIST_SKIP_DELTANET") || metal_env_enabled("GEIST_SKIP_DN_WIDE");
    default:
        return false;
    }
}

void metal_profile_add_dispatch(struct metal_state      *st,
                                enum metal_profile_stage stage,
                                struct metal_size        groups) {
    if (st == nullptr || !st->profile_enabled || stage >= METAL_PROFILE_STAGE_COUNT) {
        return;
    }
    if (metal_skip_stage(stage) && metal_skip_grid_match(groups)) {
        st->skip_next_dispatch = true;
    }
    struct metal_profile_stat *stat = &st->profile[stage];
    stat->calls                     = metal_saturating_add_u64(stat->calls, 1);
    stat->workgroups = metal_saturating_add_u64(stat->workgroups, metal_profile_workgroups(groups));
}

enum metal_profile_stage
metal_profile_wait_stage_for_sequence(enum geist_command_sequence_kind kind) {
    switch (kind) {
    case GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP:
        return METAL_PROFILE_WAIT_DECODE_LAYER_LOOP;
    case GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP:
        return METAL_PROFILE_WAIT_DECODE_GREEDY_STEP;
    case GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY:
        return METAL_PROFILE_WAIT_VERIFY_GREEDY;
    case GEIST_COMMAND_SEQUENCE_PREFILL_TEXT:
        return METAL_PROFILE_WAIT_PREFILL_TEXT;
    default:
        return METAL_PROFILE_STAGE_COUNT;
    }
}

void metal_profile_print_summary(const struct metal_state *st) {
    if (st == nullptr || !st->profile_enabled) {
        return;
    }
    bool any = false;
    for (size_t i = 0; i < METAL_PROFILE_STAGE_COUNT; i++) {
        if (st->profile[i].calls != 0) {
            any = true;
            break;
        }
    }
    if (!any) {
        return;
    }
    fprintf(stderr, "metal backend profile:\n");
    fprintf(stderr,
            "  note: wait.* is CPU wall time spent committing/waiting for "
            "submitted Metal command buffers; dispatch.* counts encoded "
            "kernel dispatches and threadgroups.\n");
    for (size_t i = 0; i < METAL_PROFILE_STAGE_COUNT; i++) {
        const struct metal_profile_stat *stat = &st->profile[i];
        if (stat->calls == 0) {
            continue;
        }
        if (stat->ns != 0) {
            fprintf(stderr,
                    "  %-32s %9.2f ms  (%llu calls)\n",
                    metal_profile_stage_names[i],
                    (double) stat->ns / 1000000.0,
                    (unsigned long long) stat->calls);
        } else {
            fprintf(stderr,
                    "  %-32s %9llu dispatches  %llu threadgroups\n",
                    metal_profile_stage_names[i],
                    (unsigned long long) stat->calls,
                    (unsigned long long) stat->workgroups);
        }
    }
}

/* One-shot Xcode GPU capture of the first prefill command sequence, for
 * per-dispatch kernel timing (works on M1; open the .gputrace in Xcode).
 * Usage: METAL_CAPTURE_ENABLED=1 GEIST_METAL_CAPTURE=/path/out.gputrace
 * METAL_CAPTURE_ENABLED must be set at process launch or startCapture fails.
 * GEIST_METAL_CAPTURE_SKIP=N skips the first N prefill sequences (e.g. a
 * bench warmup) before capturing. */
void metal_capture_begin(struct metal_state *st, enum geist_command_sequence_kind kind) {
    if (st->capture_done || kind != GEIST_COMMAND_SEQUENCE_PREFILL_TEXT) {
        return;
    }
    const char *skip_env = getenv("GEIST_METAL_CAPTURE_SKIP");
    if (skip_env != nullptr && st->capture_skipped < atoi(skip_env)) {
        st->capture_skipped++;
        return;
    }
    const char *path = getenv("GEIST_METAL_CAPTURE");
    if (path == nullptr || path[0] == '\0') {
        st->capture_done = true;
        return;
    }
    st->capture_done = true;
    void *mgr_class  = metal_objc_get_class(st, "MTLCaptureManager");
    void *desc_class = metal_objc_get_class(st, "MTLCaptureDescriptor");
    void *str_class  = metal_objc_get_class(st, "NSString");
    void *url_class  = metal_objc_get_class(st, "NSURL");
    if (mgr_class == nullptr || desc_class == nullptr || str_class == nullptr ||
        url_class == nullptr) {
        return;
    }
    void *mgr     = metal_msg_send_id0(st, mgr_class, "sharedCaptureManager");
    void *desc    = metal_msg_send_id0(st, desc_class, "new");
    void *ns_path = metal_msg_send_id_cstr(st, str_class, "stringWithUTF8String:", path);
    if (mgr == nullptr || desc == nullptr || ns_path == nullptr) {
        return;
    }
    void *url = metal_msg_send_id_id(st, url_class, "fileURLWithPath:", ns_path);
    metal_msg_send_id_id(st, desc, "setCaptureObject:", st->device);
    /* 2 = MTLCaptureDestinationGPUTraceDocument */
    metal_msg_send_void_ulong(st, desc, "setDestination:", 2);
    metal_msg_send_id_id(st, desc, "setOutputURL:", url);
    void *err = nullptr;
    if (!metal_msg_send_bool_id_err(st, mgr, "startCaptureWithDescriptor:error:", desc, &err)) {
        const char *msg = metal_nserror_message(st, err);
        fprintf(stderr,
                "geist metal: GPU capture start failed: %s "
                "(launch with METAL_CAPTURE_ENABLED=1)\n",
                msg != nullptr ? msg : "unknown error");
        return;
    }
    st->capture_manager = mgr;
    fprintf(stderr, "geist metal: GPU capture started -> %s\n", path);
}

void metal_capture_end(struct metal_state *st) {
    if (st->capture_manager == nullptr) {
        return;
    }
    metal_msg_send_void0(st, st->capture_manager, "stopCapture");
    st->capture_manager = nullptr;
    fprintf(stderr, "geist metal: GPU capture written\n");
}

/* GEIST_METAL_DEBUG_LINEAR=1: per-linear x/y absmax trace. Every layer
 * stage flows through linear, so the first all-zero input pinpoints which
 * in-between op (rmsnorm/rope/attention/add/mul) lost the data. */
void metal_linear_debug_stats(const float               *x,
                              size_t                     nx,
                              const float               *y,
                              size_t                     ny,
                              const struct geist_weight *w,
                              size_t                     m) {
    static _Atomic int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
        enabled       = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
    }
    if (!enabled) {
        return;
    }
    float  ax   = 0.0f;
    float  ay   = 0.0f;
    size_t nanx = 0;
    size_t nany = 0;
    for (size_t i = 0; i < nx; i++) {
        if (isnan(x[i])) {
            nanx++;
            continue;
        }
        const float a = fabsf(x[i]);
        if (a > ax) {
            ax = a;
        }
    }
    for (size_t i = 0; i < ny; i++) {
        if (isnan(y[i])) {
            nany++;
            continue;
        }
        const float a = fabsf(y[i]);
        if (a > ay) {
            ay = a;
        }
    }
    fprintf(stderr,
            "linear dtype=%u m=%zu %dx%d |x|=%g |y|=%g nanx=%zu nany=%zu\n",
            (unsigned) w->dtype,
            m,
            w->n_out,
            w->n_in,
            (double) ax,
            (double) ay,
            nanx,
            nany);
}
