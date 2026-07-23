/* metal_objc.h — the dlopen/dlsym Objective-C runtime shim for the Metal
 * backend: typed objc_msgSend wrappers over the handles in metal_state.
 *
 * backend.c-private — included exactly once, after struct metal_state is
 * defined (the wrappers read its objc handles); everything is static. */
#ifndef GEIST_METAL_OBJC_H
#define GEIST_METAL_OBJC_H

static void *metal_dlsym(void *handle, const char *name) {
    return handle != nullptr ? dlsym(handle, name) : nullptr;
}

static void *metal_objc_get_class(struct metal_state *st, const char *name) {
    union {
        void *raw;
        void *(*fn)(const char *name);
    } get_class = {.raw = st->objc_getClass};
    return get_class.fn(name);
}

static void *metal_create_default_device(struct metal_state *st) {
    union {
        void *raw;
        void *(*fn)(void);
    } create_device = {.raw = st->MTLCreateSystemDefaultDevice};
    return create_device.fn();
}

static void *metal_sel_register_name(struct metal_state *st, const char *selector) {
    union {
        void *raw;
        void *(*fn)(const char *name);
    } sel_register = {.raw = st->sel_registerName};
    return sel_register.fn(selector);
}

static void *metal_msg_send_id0(struct metal_state *st, void *receiver, const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static void *metal_msg_send_id_size_uint(
        struct metal_state *st, void *receiver, const char *selector, size_t a, unsigned long b) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, size_t, unsigned long);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, b);
}

/* newBufferWithBytes:length:options: — copies host bytes into a new MTLBuffer. */
static void *metal_msg_send_id_ptr_size_uint(struct metal_state *st,
                                             void               *receiver,
                                             const char         *selector,
                                             const void         *ptr,
                                             size_t              len,
                                             unsigned long       opts) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, const void *, size_t, unsigned long);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, ptr, len, opts);
}

static void *metal_msg_send_id_cstr(struct metal_state *st,
                                    void               *receiver,
                                    const char         *selector,
                                    const char         *arg) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, const char *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, arg);
}

static void *
metal_msg_send_id_id(struct metal_state *st, void *receiver, const char *selector, void *arg) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, arg);
}

static void *metal_msg_send_id_id_id_err(struct metal_state *st,
                                         void               *receiver,
                                         const char         *selector,
                                         void               *a,
                                         void               *b,
                                         void              **err) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *, void *, void **);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, b, err);
}

static void *metal_msg_send_id_id_err(
        struct metal_state *st, void *receiver, const char *selector, void *a, void **err) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *, void **);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, err);
}

static void metal_msg_send_void_ulong(struct metal_state *st,
                                      void               *receiver,
                                      const char         *selector,
                                      unsigned long       a) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void (*fn)(void *, void *, unsigned long);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, a);
}

static bool metal_msg_send_bool_id_err(
        struct metal_state *st, void *receiver, const char *selector, void *a, void **err) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        unsigned char (*fn)(void *, void *, void *, void **);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, err) != 0;
}

static const char *
metal_msg_send_cstr0(struct metal_state *st, void *receiver, const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        const char *(*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static const char *metal_nserror_message(struct metal_state *st, void *err) {
    if (st == nullptr || err == nullptr) {
        return nullptr;
    }
    void *desc = metal_msg_send_id0(st, err, "localizedDescription");
    if (desc == nullptr) {
        return nullptr;
    }
    return metal_msg_send_cstr0(st, desc, "UTF8String");
}

static void metal_msg_send_void0(struct metal_state *st, void *receiver, const char *selector) {
    if (receiver == nullptr) {
        return;
    }
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void (*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel);
}

static void metal_msg_send_copy_buffer(struct metal_state *st,
                                       void               *receiver,
                                       const char         *selector,
                                       void               *src,
                                       size_t              src_offset,
                                       void               *dst,
                                       size_t              dst_offset,
                                       size_t              bytes) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void (*fn)(void *, void *, void *, size_t, void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, src, src_offset, dst, dst_offset, bytes);
}

static void metal_seq_mark_buffer(struct metal_state *st, void *mtl_buf);

static void metal_msg_send_set_buffer(
        struct metal_state *st, void *receiver, void *buffer, size_t offset, size_t index) {
    void *sel = metal_sel_register_name(st, "setBuffer:offset:atIndex:");
    metal_seq_mark_buffer(st, buffer);
    union {
        void *raw;
        void (*fn)(void *, void *, void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, buffer, offset, index);
}

static void metal_msg_send_set_bytes(
        struct metal_state *st, void *receiver, const void *bytes, size_t length, size_t index) {
    void *sel = metal_sel_register_name(st, "setBytes:length:atIndex:");
    union {
        void *raw;
        void (*fn)(void *, void *, const void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, bytes, length, index);
}

/* Pipeline bind. Dedup of identical re-binds was tried (2026-07-04) and
 * measured a wash — consecutive redundancy is too low; the per-op GPU
 * command-stream cost scales with op COUNT, so fusion is the lever. */
static void metal_msg_send_set_pipeline(struct metal_state *st, void *receiver, void *pipeline) {
    (void) metal_msg_send_id_id(st, receiver, "setComputePipelineState:", pipeline);
}

static void metal_msg_send_set_threadgroup_memory(struct metal_state *st,
                                                  void               *receiver,
                                                  size_t              length,
                                                  size_t              index) {
    void *sel = metal_sel_register_name(st, "setThreadgroupMemoryLength:atIndex:");
    union {
        void *raw;
        void (*fn)(void *, void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, length, index);
}

static void metal_msg_send_dispatch(struct metal_state *st,
                                    void               *receiver,
                                    struct metal_size   groups,
                                    struct metal_size   threads) {
    void *sel = metal_sel_register_name(st, "dispatchThreadgroups:threadsPerThreadgroup:");
    if (st->skip_next_dispatch) {
        st->skip_next_dispatch = false;
        return;
    }
    union {
        void *raw;
        void (*fn)(void *, void *, struct metal_size, struct metal_size);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, groups, threads);
    st->seq_dispatch_count++;
}

#endif /* GEIST_METAL_OBJC_H */
