/* metal_objc.h — the dlopen/dlsym Objective-C runtime shim for the Metal
 * backend: typed objc_msgSend wrappers over the handles in metal_state.
 *
 * Included from metal_internal.h after struct metal_state is defined (the
 * wrappers read its objc handles); everything is static. */
#ifndef GEIST_METAL_OBJC_H
#define GEIST_METAL_OBJC_H

static inline void *metal_dlsym(void *handle, const char *name) {
    return handle != nullptr ? dlsym(handle, name) : nullptr;
}

static inline void *metal_objc_get_class(struct metal_state *st, const char *name) {
    union {
        void *raw;
        void *(*fn)(const char *name);
    } get_class = {.raw = st->objc_getClass};
    return get_class.fn(name);
}

static inline void *metal_create_default_device(struct metal_state *st) {
    union {
        void *raw;
        void *(*fn)(void);
    } create_device = {.raw = st->MTLCreateSystemDefaultDevice};
    return create_device.fn();
}

static inline void *metal_sel_register_name(struct metal_state *st, const char *selector) {
    union {
        void *raw;
        void *(*fn)(const char *name);
    } sel_register = {.raw = st->sel_registerName};
    return sel_register.fn(selector);
}

/* Autorelease pools. Plain C has no @autoreleasepool; without these the
 * autoreleased command buffers and encoders are never drained (#527).
 * Pools are per thread and LIFO: popping one also pops every pool pushed
 * after it. */
static inline void *metal_pool_push(struct metal_state *st) {
    union {
        void *raw;
        void *(*fn)(void);
    } push = {.raw = st->objc_autoreleasePoolPush};
    return push.fn();
}

static inline void metal_pool_pop(struct metal_state *st, void *pool) {
    union {
        void *raw;
        void (*fn)(void *);
    } pop = {.raw = st->objc_autoreleasePoolPop};
    pop.fn(pool);
}

/* Pool for a standalone submission (commandBuffer ... waitUntilCompleted),
 * popped on every return path:
 *
 *     [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
 *
 * Inside a command sequence it pushes nothing: the sequence's pool covers
 * the op, and a nested one would be popped from under it by a flush. */
struct metal_pool {
    struct metal_state *st;
    void               *token;
};

/* Whether the calling thread has the backend's command sequence open
 * (#544): another session's open sequence is not one to encode into. */
static inline bool metal_seq_mine(const struct metal_state *st) {
    return atomic_load_explicit(&st->seq_owner, memory_order_relaxed) == (uintptr_t) pthread_self();
}

/* Also takes seq_lock for the submission, so it waits out a sequence
 * another thread has open (recursive: free inside the caller's own). */
static inline struct metal_pool metal_standalone_pool(struct metal_state *st) {
    pthread_mutex_lock(&st->seq_lock);
    return (struct metal_pool) {st, metal_seq_mine(st) ? nullptr : metal_pool_push(st)};
}

static inline void metal_pool_end(struct metal_pool *pool) {
    if (pool->token != nullptr) {
        metal_pool_pop(pool->st, pool->token);
    }
    pthread_mutex_unlock(&pool->st->seq_lock);
}

static inline void *
metal_msg_send_id0(struct metal_state *st, void *receiver, const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static inline void *metal_msg_send_id_size_uint(
        struct metal_state *st, void *receiver, const char *selector, size_t a, unsigned long b) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, size_t, unsigned long);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, b);
}

/* newBufferWithBytes:length:options: — copies host bytes into a new MTLBuffer. */
static inline void *metal_msg_send_id_ptr_size_uint(struct metal_state *st,
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

/* newBufferWithBytesNoCopy:length:options:deallocator: — wraps page-aligned,
 * page-multiple host memory as an MTLBuffer without copying it. A nil
 * deallocator leaves ownership of the pages with the caller. */
static inline void *metal_msg_send_id_ptr_size_uint_ptr(struct metal_state *st,
                                                        void               *receiver,
                                                        const char         *selector,
                                                        void               *ptr,
                                                        size_t              len,
                                                        unsigned long       opts,
                                                        void               *deallocator) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *, size_t, unsigned long, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, ptr, len, opts, deallocator);
}

static inline void *metal_msg_send_id_cstr(struct metal_state *st,
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

static inline void *
metal_msg_send_id_id(struct metal_state *st, void *receiver, const char *selector, void *arg) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, arg);
}

static inline void *metal_msg_send_id_id_id_err(struct metal_state *st,
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

static inline void *metal_msg_send_id_id_err(
        struct metal_state *st, void *receiver, const char *selector, void *a, void **err) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *, void **);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, err);
}

/* NSUInteger-returning nullary getter (currentAllocatedSize, ...). */
static inline unsigned long
metal_msg_send_ulong0(struct metal_state *st, void *receiver, const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        unsigned long (*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static inline bool
metal_msg_send_bool0(struct metal_state *st, void *receiver, const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        bool (*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static inline void metal_msg_send_void_ulong(struct metal_state *st,
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

static inline bool metal_msg_send_bool_id_err(
        struct metal_state *st, void *receiver, const char *selector, void *a, void **err) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        unsigned char (*fn)(void *, void *, void *, void **);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, err) != 0;
}

static inline const char *
metal_msg_send_cstr0(struct metal_state *st, void *receiver, const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        const char *(*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static inline const char *metal_nserror_message(struct metal_state *st, void *err) {
    if (st == nullptr || err == nullptr) {
        return nullptr;
    }
    void *desc = metal_msg_send_id0(st, err, "localizedDescription");
    if (desc == nullptr) {
        return nullptr;
    }
    return metal_msg_send_cstr0(st, desc, "UTF8String");
}

static inline void
metal_msg_send_void0(struct metal_state *st, void *receiver, const char *selector) {
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

static inline void metal_msg_send_copy_buffer(struct metal_state *st,
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

/* Defined in sequence.c. */
void metal_seq_mark_buffer(struct metal_state *st, void *mtl_buf, size_t off);

/* Defined in resources.c: the first bind of an MTLBuffer adds it to the
 * residency set (#530). */
void metal_residency_note(struct metal_state *st, void *mtl_buf);

static inline void metal_msg_send_set_buffer(
        struct metal_state *st, void *receiver, void *buffer, size_t offset, size_t index) {
    void *sel = metal_sel_register_name(st, "setBuffer:offset:atIndex:");
    metal_seq_mark_buffer(st, buffer, offset);
    metal_residency_note(st, buffer);
    union {
        void *raw;
        void (*fn)(void *, void *, void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, buffer, offset, index);
}

static inline void metal_msg_send_set_bytes(
        struct metal_state *st, void *receiver, const void *bytes, size_t length, size_t index) {
    void *sel = metal_sel_register_name(st, "setBytes:length:atIndex:");
    union {
        void *raw;
        void (*fn)(void *, void *, const void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, bytes, length, index);
}

/* No dedup of identical re-binds: consecutive repeats are too rare to pay;
 * the per-op cost scales with op count, so fusion is the lever. */
static inline void
metal_msg_send_set_pipeline(struct metal_state *st, void *receiver, void *pipeline) {
    (void) metal_msg_send_id_id(st, receiver, "setComputePipelineState:", pipeline);
}

static inline void metal_msg_send_set_threadgroup_memory(struct metal_state *st,
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

static inline void metal_msg_send_dispatch(struct metal_state *st,
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
