/*
 * tools/probes/metal_trace.m — one record per Metal command buffer of the
 * process it is loaded into (#590): commit time, GPUStartTime/GPUEndTime,
 * kernelStartTime/kernelEndTime, completion time, the time a thread spent
 * in waitUntilCompleted on it, and the compute dispatches it carries per
 * kernel (the MTLFunction name of the bound pipeline).
 *
 * It replaces the implementations of a handful of methods on the concrete
 * Metal classes of the default device (commit, waitUntilCompleted, the
 * encoder constructors, setComputePipelineState:, dispatchThreadgroups:...,
 * dispatchThreads:..., newComputePipelineStateWith...), so it works on any
 * engine without its sources: geist via DYLD_INSERT_LIBRARIES, mlx via
 * ctypes.CDLL before `import mlx`.
 *
 *   cc -dynamiclib -fobjc-arc -O2 -o libmetal_trace.dylib metal_trace.m \
 *      -framework Metal -framework Foundation
 *
 *   METAL_TRACE_OUT=file   write the trace at process exit
 *   METAL_TRACE_FORCE_CONCURRENT=1  experiment: every computeCommandEncoder
 *                          becomes a concurrent one with a buffer-scope
 *                          memory barrier before each dispatch (the serial
 *                          order, expressed with barriers); =2 no barriers
 *                          (wrong results; the bound on what eliding buys);
 *                          =3 a barrier only before a dispatch that reads
 *                          what a dispatch since the last barrier wrote, or
 *                          writes what one read or wrote, by (MTLBuffer,
 *                          bind offset) and the kernels' reflected buffer
 *                          access; =4 the same by MTLBuffer alone. For an
 *                          engine that binds with setBuffer:offset:atIndex:
 *                          and encodes on one thread (geist)
 *   metal_trace_mark(s)    a named time stamp (with the process's CPU time)
 *   metal_trace_dump(path) write the trace now
 *
 * Output: JSON lines. {"k":id,"name":..} per kernel, {"mark":..,"t":..,
 * "cpu":..} per mark, {"c":commit,"gs":..,"ge":..,"ks":..,"ke":..,"d":done,
 * "w0":..,"w1":..,"e":encoders,"cc":of them concurrent,"mb":memory
 * barriers,"b":blit encoders,"n":dispatches,
 * "tg":threadgroups,"k":[[id,dispatches],..]} per command buffer, all times
 * in seconds of the mach_absolute_time clock (the clock of GPUStartTime).
 * Cost per hooked call: one unfair lock and a table probe (tens of ns).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <objc/runtime.h>
#import <os/lock.h>
#import <sys/resource.h>

enum { MAX_K = 48, CHUNK = 4096, MAX_CHUNKS = 4096, TABLE = 1 << 14, MAX_KERNELS = 1024 };

struct rec {
    double   commit, gpu_start, gpu_end, k_start, k_end, done, wait0, wait1;
    uint32_t encoders, concurrent, blits, dispatches, barriers, nk;
    uint64_t threadgroups;
    struct {
        uint16_t id;
        uint32_t n;
    } k[MAX_K];
};

struct slot {
    const void *key;
    uint32_t    val; /* record index (buffers, encoders) or kernel id (pipelines) */
    uint16_t    cur; /* encoders: kernel id bound now */
};

struct mark {
    char   name[48];
    double t, cpu;
};

static os_unfair_lock g_lock = OS_UNFAIR_LOCK_INIT;
static struct rec    *g_chunks[MAX_CHUNKS];
static uint32_t       g_nrec;
static struct slot    g_bufs[TABLE], g_encs[TABLE], g_pipes[TABLE];
static char          *g_kernels[MAX_KERNELS];
static uint32_t       g_nkernels;
static uint32_t       g_kwrite[MAX_KERNELS], g_kused[MAX_KERNELS]; /* buffer-index masks */
static struct mark    g_marks[256];
static uint32_t       g_nmarks;
static double         g_tick;

static double now_s(void) {
    return (double) mach_absolute_time() * g_tick;
}

static struct rec *rec_at(uint32_t i) {
    return &g_chunks[i / CHUNK][i % CHUNK];
}

/* Open addressing with tombstones (key 1). All under g_lock. */
static struct slot *table_find(struct slot *t, const void *key, bool insert) {
    size_t       h    = ((uintptr_t) key >> 4) * 0x9e3779b97f4a7c15u >> 50;
    struct slot *free = nullptr;
    for (size_t i = 0; i < TABLE; i++) {
        struct slot *s = &t[(h + i) & (TABLE - 1)];
        if (s->key == key) {
            return s;
        }
        if (s->key == (const void *) 1 && free == nullptr) {
            free = s;
        }
        if (s->key == nullptr) {
            if (!insert) {
                return nullptr;
            }
            s      = free != nullptr ? free : s;
            s->key = key;
            s->val = UINT32_MAX;
            s->cur = UINT16_MAX;
            return s;
        }
    }
    return nullptr;
}

static uint32_t rec_for_buffer(const void *cmd) {
    struct slot *s = table_find(g_bufs, cmd, true);
    if (s == nullptr) {
        return UINT32_MAX;
    }
    if (s->val == UINT32_MAX) {
        if (g_nrec / CHUNK >= MAX_CHUNKS) {
            return UINT32_MAX;
        }
        if (g_chunks[g_nrec / CHUNK] == nullptr) {
            g_chunks[g_nrec / CHUNK] = calloc(CHUNK, sizeof(struct rec));
        }
        s->val = g_nrec++;
    }
    return s->val;
}

static uint16_t kernel_id(NSString *name) {
    const char *c = name != nil ? name.UTF8String : "?";
    for (uint32_t i = 0; i < g_nkernels; i++) {
        if (strcmp(g_kernels[i], c) == 0) {
            return (uint16_t) i;
        }
    }
    if (g_nkernels == MAX_KERNELS) {
        return 0;
    }
    g_kernels[g_nkernels] = strdup(c);
    return (uint16_t) g_nkernels++;
}

static void pipeline_named(id pipeline, NSString *name) {
    if (pipeline == nil) {
        return;
    }
    os_unfair_lock_lock(&g_lock);
    struct slot *s = table_find(g_pipes, (__bridge const void *) pipeline, true);
    if (s != nullptr) {
        s->val = kernel_id(name);
    }
    os_unfair_lock_unlock(&g_lock);
}

/* ---- replaced methods: each calls the implementation it replaced ---- */

static IMP o_pipe_fn, o_pipe_fn_opt, o_pipe_desc;
static IMP o_commit, o_wait, o_enc, o_enc_desc, o_enc_type, o_blit;
static IMP o_set_pipe, o_disp_groups, o_disp_threads, o_end, o_barrier_scope, o_barrier_res;
static IMP o_set_buffer;
static int g_force; /* METAL_TRACE_FORCE_CONCURRENT, see the head of the file */

/* Modes 3 and 4: what the dispatches since the last barrier of the encoder
 * read and wrote. One encoder is open at a time in the engines traced. */
enum { MAX_BIND = 31, MAX_TRACK = 512 };
struct bind {
    const void *buf;
    size_t      off;
};
static const void *g_track_enc;
static struct bind g_bound[MAX_BIND];
static struct bind g_wrote[MAX_TRACK], g_read[MAX_TRACK];
static uint32_t    g_nwrote, g_nread;

static bool track_has(const struct bind *set, uint32_t n, struct bind b) {
    for (uint32_t i = 0; i < n; i++) {
        if (set[i].buf == b.buf && (g_force == 4 || set[i].off == b.off)) {
            return true;
        }
    }
    return false;
}

/* True when the dispatch of kernel `kid` with the current bindings must not
 * overlap one since the last barrier: it reads what one wrote, or writes
 * what one read or wrote. Records its own reads and writes either way. */
static bool track_dispatch(const void *enc, uint16_t kid) {
    if (enc != g_track_enc) {
        g_track_enc = enc;
        g_nwrote = g_nread = 0;
    }
    const uint32_t used  = kid < MAX_KERNELS ? g_kused[kid] : UINT32_MAX;
    const uint32_t write = kid < MAX_KERNELS ? g_kwrite[kid] : UINT32_MAX;
    bool           clash = kid >= MAX_KERNELS || g_nwrote + MAX_BIND > MAX_TRACK ||
                           g_nread + MAX_BIND > MAX_TRACK;
    for (uint32_t i = 0; i < MAX_BIND && !clash; i++) {
        if (((used >> i) & 1u) == 0 || g_bound[i].buf == nullptr) {
            continue;
        }
        clash = track_has(g_wrote, g_nwrote, g_bound[i]) ||
                (((write >> i) & 1u) != 0 && track_has(g_read, g_nread, g_bound[i]));
    }
    if (clash) {
        g_nwrote = g_nread = 0;
    }
    for (uint32_t i = 0; i < MAX_BIND; i++) {
        if (((used >> i) & 1u) == 0 || g_bound[i].buf == nullptr) {
            continue;
        }
        if (((write >> i) & 1u) != 0) {
            g_wrote[g_nwrote++] = g_bound[i];
        } else if (!track_has(g_read, g_nread, g_bound[i])) {
            g_read[g_nread++] = g_bound[i];
        }
    }
    return clash;
}

static void h_set_buffer(id self, SEL sel, id buffer, NSUInteger offset, NSUInteger index) {
    if (index < MAX_BIND) {
        g_bound[index] = (struct bind) {(__bridge const void *) buffer, offset};
    }
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger)) o_set_buffer)(
            self, sel, buffer, offset, index);
}

/* Which buffer indices the kernel declares and which it may write (not
 * `const device`), from the pipeline's reflection. */
static void kernel_access(id device, id<MTLFunction> fn) {
    MTLComputePipelineReflection *refl = nil;
    (void) ((id (*)(id, SEL, id, NSUInteger, MTLComputePipelineReflection *__autoreleasing *,
                    NSError **)) o_pipe_fn_opt)(
            device,
            sel_registerName("newComputePipelineStateWithFunction:options:reflection:error:"),
            fn,
            MTLPipelineOptionBindingInfo,
            &refl,
            nullptr);
    uint32_t used = 0, write = 0;
    for (id<MTLBinding> b in refl.bindings) {
        if (b.type == MTLBindingTypeBuffer && b.index < MAX_BIND) {
            used |= 1u << b.index;
            if (b.access != MTLBindingAccessReadOnly) {
                write |= 1u << b.index;
            }
        }
    }
    os_unfair_lock_lock(&g_lock);
    const uint16_t kid = kernel_id(fn.name);
    g_kused[kid]       = refl != nil ? used : UINT32_MAX;
    g_kwrite[kid]      = refl != nil ? write : UINT32_MAX;
    os_unfair_lock_unlock(&g_lock);
}

static id h_pipe_fn(id self, SEL sel, id<MTLFunction> fn, NSError **err) {
    id p = ((id (*)(id, SEL, id, NSError **)) o_pipe_fn)(self, sel, fn, err);
    pipeline_named(p, fn.name);
    if (g_force >= 3 && p != nil && o_pipe_fn_opt != nullptr) {
        kernel_access(self, fn);
    }
    return p;
}

static id h_pipe_fn_opt(id            self,
                        SEL           sel,
                        id<MTLFunction> fn,
                        NSUInteger    opt,
                        void         *refl,
                        NSError     **err) {
    id p = ((id (*)(id, SEL, id, NSUInteger, void *, NSError **)) o_pipe_fn_opt)(
            self, sel, fn, opt, refl, err);
    pipeline_named(p, fn.name);
    return p;
}

static id h_pipe_desc(id                            self,
                      SEL                           sel,
                      MTLComputePipelineDescriptor *desc,
                      NSUInteger                    opt,
                      void                         *refl,
                      NSError                     **err) {
    id p = ((id (*)(id, SEL, id, NSUInteger, void *, NSError **)) o_pipe_desc)(
            self, sel, desc, opt, refl, err);
    pipeline_named(p, desc.computeFunction.name);
    return p;
}

static void encoder_made(id cmd, id enc, bool blit, bool concurrent) {
    if (enc == nil) {
        return;
    }
    os_unfair_lock_lock(&g_lock);
    const uint32_t r = rec_for_buffer((__bridge const void *) cmd);
    if (r != UINT32_MAX) {
        if (blit) {
            rec_at(r)->blits++;
        } else {
            rec_at(r)->encoders++;
            rec_at(r)->concurrent += concurrent;
            struct slot *s = table_find(g_encs, (__bridge const void *) enc, true);
            if (s != nullptr) {
                s->val = r;
                s->cur = UINT16_MAX;
            }
        }
    }
    os_unfair_lock_unlock(&g_lock);
}

static id h_enc(id self, SEL sel) {
    if (g_force != 0 && o_enc_type != nullptr) {
        id forced = ((id (*)(id, SEL, NSUInteger)) o_enc_type)(
                self, sel_registerName("computeCommandEncoderWithDispatchType:"), 1);
        encoder_made(self, forced, false, true);
        return forced;
    }
    id e = ((id (*)(id, SEL)) o_enc)(self, sel);
    encoder_made(self, e, false, false);
    return e;
}

static id h_enc_desc(id self, SEL sel, id desc) {
    id e = ((id (*)(id, SEL, id)) o_enc_desc)(self, sel, desc);
    encoder_made(self, e, false, [desc dispatchType] == MTLDispatchTypeConcurrent);
    return e;
}

static id h_enc_type(id self, SEL sel, NSUInteger type) {
    id e = ((id (*)(id, SEL, NSUInteger)) o_enc_type)(self, sel, type);
    encoder_made(self, e, false, type == MTLDispatchTypeConcurrent);
    return e;
}

static id h_blit(id self, SEL sel) {
    id e = ((id (*)(id, SEL)) o_blit)(self, sel);
    encoder_made(self, e, true, false);
    return e;
}

static void h_set_pipe(id self, SEL sel, id pipeline) {
    os_unfair_lock_lock(&g_lock);
    struct slot *e = table_find(g_encs, (__bridge const void *) self, false);
    struct slot *p = table_find(g_pipes, (__bridge const void *) pipeline, false);
    if (e != nullptr) {
        e->cur = p != nullptr ? (uint16_t) p->val : kernel_id(@"(unnamed)");
    }
    os_unfair_lock_unlock(&g_lock);
    ((void (*)(id, SEL, id)) o_set_pipe)(self, sel, pipeline);
}

static void dispatched(id enc, uint64_t groups) {
    os_unfair_lock_lock(&g_lock);
    struct slot *e = table_find(g_encs, (__bridge const void *) enc, false);
    if (e != nullptr && e->val != UINT32_MAX) {
        struct rec *r = rec_at(e->val);
        r->dispatches++;
        r->threadgroups += groups;
        uint32_t i = 0;
        while (i < r->nk && r->k[i].id != e->cur) {
            i++;
        }
        if (i == r->nk && r->nk < MAX_K) {
            r->k[r->nk++].id = e->cur;
        }
        if (i < r->nk) {
            r->k[i].n++;
        }
    }
    os_unfair_lock_unlock(&g_lock);
}

static void barrier(id enc) {
    os_unfair_lock_lock(&g_lock);
    struct slot *e = table_find(g_encs, (__bridge const void *) enc, false);
    if (e != nullptr && e->val != UINT32_MAX) {
        rec_at(e->val)->barriers++;
    }
    os_unfair_lock_unlock(&g_lock);
}

static void h_barrier_scope(id self, SEL sel, NSUInteger scope) {
    barrier(self);
    ((void (*)(id, SEL, NSUInteger)) o_barrier_scope)(self, sel, scope);
}

static void h_barrier_res(id self, SEL sel, const id *resources, NSUInteger count) {
    barrier(self);
    ((void (*)(id, SEL, const id *, NSUInteger)) o_barrier_res)(self, sel, resources, count);
}

static void h_disp_groups(id self, SEL sel, MTLSize groups, MTLSize per) {
    bool need = g_force == 1;
    if (g_force >= 3) {
        os_unfair_lock_lock(&g_lock);
        struct slot *e = table_find(g_encs, (__bridge const void *) self, false);
        need = track_dispatch((__bridge const void *) self, e != nullptr ? e->cur : UINT16_MAX);
        os_unfair_lock_unlock(&g_lock);
    }
    if (need) {
        [(id<MTLComputeCommandEncoder>) self memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    dispatched(self, (uint64_t) groups.width * groups.height * groups.depth);
    ((void (*)(id, SEL, MTLSize, MTLSize)) o_disp_groups)(self, sel, groups, per);
}

static void h_disp_threads(id self, SEL sel, MTLSize threads, MTLSize per) {
    const uint64_t t = (uint64_t) threads.width * threads.height * threads.depth;
    const uint64_t g = (uint64_t) per.width * per.height * per.depth;
    dispatched(self, g != 0 ? (t + g - 1) / g : 0);
    ((void (*)(id, SEL, MTLSize, MTLSize)) o_disp_threads)(self, sel, threads, per);
}

static void h_end(id self, SEL sel) {
    os_unfair_lock_lock(&g_lock);
    struct slot *e = table_find(g_encs, (__bridge const void *) self, false);
    if (e != nullptr) {
        e->key = (const void *) 1;
    }
    os_unfair_lock_unlock(&g_lock);
    ((void (*)(id, SEL)) o_end)(self, sel);
}

static void h_commit(id<MTLCommandBuffer> self, SEL sel) {
    os_unfair_lock_lock(&g_lock);
    const uint32_t r = rec_for_buffer((__bridge const void *) self);
    os_unfair_lock_unlock(&g_lock);
    if (r != UINT32_MAX) {
        [self addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            const double t = now_s();
            os_unfair_lock_lock(&g_lock);
            struct rec *x = rec_at(r);
            x->gpu_start  = cb.GPUStartTime;
            x->gpu_end    = cb.GPUEndTime;
            x->k_start    = cb.kernelStartTime;
            x->k_end      = cb.kernelEndTime;
            x->done       = t;
            struct slot *s = table_find(g_bufs, (__bridge const void *) cb, false);
            if (s != nullptr) {
                s->key = (const void *) 1;
            }
            os_unfair_lock_unlock(&g_lock);
        }];
        os_unfair_lock_lock(&g_lock);
        rec_at(r)->commit = now_s();
        os_unfair_lock_unlock(&g_lock);
    }
    ((void (*)(id, SEL)) o_commit)(self, sel);
}

static void h_wait(id self, SEL sel) {
    os_unfair_lock_lock(&g_lock);
    struct slot   *s = table_find(g_bufs, (__bridge const void *) self, false);
    const uint32_t r = s != nullptr ? s->val : UINT32_MAX;
    os_unfair_lock_unlock(&g_lock);
    const double t0 = now_s();
    ((void (*)(id, SEL)) o_wait)(self, sel);
    const double t1 = now_s();
    if (r != UINT32_MAX) {
        os_unfair_lock_lock(&g_lock);
        rec_at(r)->wait0 = t0;
        rec_at(r)->wait1 = t1;
        os_unfair_lock_unlock(&g_lock);
    }
}

static IMP hook(Class cls, const char *name, IMP replacement) {
    Method m = class_getInstanceMethod(cls, sel_registerName(name));
    if (m == nullptr) {
        return nullptr;
    }
    /* class_replaceMethod adds to cls itself when the method is inherited,
     * so a superclass shared with other classes is left alone. */
    IMP old = method_getImplementation(m);
    class_replaceMethod(cls, sel_registerName(name), replacement, method_getTypeEncoding(m));
    return old;
}

void metal_trace_mark(const char *name);
void metal_trace_dump(const char *path);

void metal_trace_mark(const char *name) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    os_unfair_lock_lock(&g_lock);
    if (g_nmarks < sizeof(g_marks) / sizeof(g_marks[0])) {
        struct mark *m = &g_marks[g_nmarks++];
        strlcpy(m->name, name, sizeof(m->name));
        m->t   = now_s();
        m->cpu = (double) ru.ru_utime.tv_sec + (double) ru.ru_utime.tv_usec * 1e-6 +
                 (double) ru.ru_stime.tv_sec + (double) ru.ru_stime.tv_usec * 1e-6;
    }
    os_unfair_lock_unlock(&g_lock);
}

void metal_trace_dump(const char *path) {
    FILE *f = fopen(path, "w");
    if (f == nullptr) {
        return;
    }
    os_unfair_lock_lock(&g_lock);
    for (uint32_t i = 0; i < g_nkernels; i++) {
        fprintf(f, "{\"kernel\":%u,\"name\":\"%s\"}\n", i, g_kernels[i]);
    }
    for (uint32_t i = 0; i < g_nmarks; i++) {
        fprintf(f,
                "{\"mark\":\"%s\",\"t\":%.9f,\"cpu\":%.6f}\n",
                g_marks[i].name,
                g_marks[i].t,
                g_marks[i].cpu);
    }
    for (uint32_t i = 0; i < g_nrec; i++) {
        const struct rec *r = rec_at(i);
        fprintf(f,
                "{\"c\":%.9f,\"gs\":%.9f,\"ge\":%.9f,\"ks\":%.9f,\"ke\":%.9f,\"d\":%.9f,"
                "\"w0\":%.9f,\"w1\":%.9f,\"e\":%u,\"cc\":%u,\"mb\":%u,\"b\":%u,\"n\":%u,"
                "\"tg\":%llu,\"k\":[",
                r->commit,
                r->gpu_start,
                r->gpu_end,
                r->k_start,
                r->k_end,
                r->done,
                r->wait0,
                r->wait1,
                r->encoders,
                r->concurrent,
                r->barriers,
                r->blits,
                r->dispatches,
                (unsigned long long) r->threadgroups);
        for (uint32_t j = 0; j < r->nk; j++) {
            fprintf(f, "%s[%u,%u]", j != 0 ? "," : "", r->k[j].id, r->k[j].n);
        }
        fprintf(f, "]}\n");
    }
    os_unfair_lock_unlock(&g_lock);
    fclose(f);
}

static void dump_at_exit(void) {
    const char *path = getenv("METAL_TRACE_OUT");
    if (path != nullptr && path[0] != '\0') {
        metal_trace_dump(path);
    }
}

__attribute__((constructor)) static void metal_trace_init(void) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_tick            = (double) tb.numer / (double) tb.denom * 1e-9;
    const char *force = getenv("METAL_TRACE_FORCE_CONCURRENT");
    g_force           = force != nullptr ? atoi(force) : 0;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            return;
        }
        id<MTLCommandQueue>          queue = [dev newCommandQueue];
        id<MTLCommandBuffer>         cmd   = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc   = [cmd computeCommandEncoder];
        Class                        dc = object_getClass(dev), cc = object_getClass(cmd),
                                     ec = object_getClass(enc);
        [enc endEncoding];
        [cmd commit];
        o_pipe_fn     = hook(dc, "newComputePipelineStateWithFunction:error:", (IMP) h_pipe_fn);
        o_pipe_fn_opt = hook(dc,
                             "newComputePipelineStateWithFunction:options:reflection:error:",
                             (IMP) h_pipe_fn_opt);
        o_pipe_desc   = hook(dc,
                             "newComputePipelineStateWithDescriptor:options:reflection:error:",
                             (IMP) h_pipe_desc);
        o_enc         = hook(cc, "computeCommandEncoder", (IMP) h_enc);
        o_enc_desc    = hook(cc, "computeCommandEncoderWithDescriptor:", (IMP) h_enc_desc);
        o_enc_type    = hook(cc, "computeCommandEncoderWithDispatchType:", (IMP) h_enc_type);
        o_blit        = hook(cc, "blitCommandEncoder", (IMP) h_blit);
        o_commit      = hook(cc, "commit", (IMP) h_commit);
        o_wait        = hook(cc, "waitUntilCompleted", (IMP) h_wait);
        o_set_pipe    = hook(ec, "setComputePipelineState:", (IMP) h_set_pipe);
        o_disp_groups =
                hook(ec, "dispatchThreadgroups:threadsPerThreadgroup:", (IMP) h_disp_groups);
        o_disp_threads = hook(ec, "dispatchThreads:threadsPerThreadgroup:", (IMP) h_disp_threads);
        o_end          = hook(ec, "endEncoding", (IMP) h_end);
        o_barrier_scope = hook(ec, "memoryBarrierWithScope:", (IMP) h_barrier_scope);
        o_barrier_res   = hook(ec, "memoryBarrierWithResources:count:", (IMP) h_barrier_res);
        o_set_buffer    = hook(ec, "setBuffer:offset:atIndex:", (IMP) h_set_buffer);
    }
    atexit(dump_at_exit);
}
