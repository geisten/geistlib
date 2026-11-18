#include "qwen-m1-rsqrt-source.h"
#include <objc/message.h>
#include <objc/runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern id MTLCreateSystemDefaultDevice(void);
struct extent {
    size_t x, y, z;
};

static id send(id obj, const char *name) {
    return ((id (*)(id, SEL)) objc_msgSend)(obj, sel_registerName(name));
}
static id send_id(id obj, const char *name, id arg) {
    return ((id (*)(id, SEL, id)) objc_msgSend)(obj, sel_registerName(name), arg);
}
static id str(const char *s) {
    return ((id (*)(id, SEL, const char *)) objc_msgSend)(
            (id) objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"), s);
}
static id make_buffer(size_t n_bytes, id device) {
    return ((id (*)(id, SEL, size_t, size_t)) objc_msgSend)(
            device, sel_registerName("newBufferWithLength:options:"), n_bytes, 0);
}
static void *contents(id object) {
    return ((void *(*)(id, SEL)) objc_msgSend)(object, sel_registerName("contents"));
}
static void set_buffer(id enc, id buffer, size_t index) {
    ((void (*)(id, SEL, id, size_t, size_t)) objc_msgSend)(
            enc, sel_registerName("setBuffer:offset:atIndex:"), buffer, 0, index);
}
static const char *shader =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void exp_bits(constant uint &first [[buffer(0)]],device uint *out "
        "[[buffer(1)]],uint i [[thread_position_in_grid]]){"
        "float x=as_type<float>(i+first);out[i]=as_type<uint>(exp(-fabs(x)));}\n"
        "kernel void rcp_bits(constant uint &first [[buffer(0)]],device uint *out "
        "[[buffer(1)]],uint i [[thread_position_in_grid]]){"
        "float x=as_type<float>(i+first);out[i]=as_type<uint>(1.0f/x);}\n"
        "\n" Q1_RSQRT_SHADER
        "kernel void beta_neg_bits(constant uint &first [[buffer(0)]],device uint *out "
        "[[buffer(1)]],uint i [[thread_position_in_grid]]){"
        "float x=as_type<float>(i+first);out[i]=as_type<uint>(1.0f/(1.0f+exp(x)));}\n"
        "kernel void rsqrt_bits(constant uint &first [[buffer(0)]],device uint *out "
        "[[buffer(1)]],uint i [[thread_position_in_grid]]){"
        "out[i]=as_type<uint>(rsqrt(as_type<float>(i+first)));}\n"
        "kernel void rsqrt_scale(constant uint &first [[buffer(0)]],device uint *out "
        "[[buffer(1)]],uint i [[thread_position_in_grid]]){uint x=i+first;"
        "uint y=as_type<uint>(rsqrt(as_type<float>(q1_rsqrt_key(x))));"
        "out[i]=as_type<uint>(rsqrt(as_type<float>(x)))^q1_rsqrt_scale(x,y);}\n";

static bool arg_bits(const char *s, uint32_t *value) {
    char               *end;
    const unsigned long n = strtoul(s, &end, 16);
    if (end == s || *end || n > UINT32_MAX)
        return false;
    *value = (uint32_t) n;
    return true;
}

int main(int argc, char **argv) {
    uint32_t first, last;
    if (argc != 5 || !arg_bits(argv[2], &first) || !arg_bits(argv[3], &last) || last < first)
        return 2;
    const bool check_exp      = !strcmp(argv[1], "check-exp");
    const bool check_beta     = !strcmp(argv[1], "check-beta-neg");
    const bool check_rsqrt    = !strcmp(argv[1], "check-rsqrt-scale");
    const bool check_constant = check_exp || check_beta || check_rsqrt;
    uint32_t   expected       = 0;
    if (check_constant && !arg_bits(argv[4], &expected))
        return 2;
    const char *kernel;
    if (check_exp || !strcmp(argv[1], "exp"))
        kernel = "exp_bits";
    else if (!strcmp(argv[1], "rcp"))
        kernel = "rcp_bits";
    else if (check_beta || !strcmp(argv[1], "beta-neg"))
        kernel = "beta_neg_bits";
    else if (!strcmp(argv[1], "rsqrt"))
        kernel = "rsqrt_bits";
    else if (check_rsqrt)
        kernel = "rsqrt_scale";
    else
        return 2;
    FILE *target = (check_constant || !strcmp(argv[4], "-")) ? stdout : fopen(argv[4], "wb");
    if (!target)
        return 2;
    id pool = send((id) objc_getClass("NSAutoreleasePool"), "new");
    id dev = MTLCreateSystemDefaultDevice(), error = nullptr;
    id lib = ((id (*)(id, SEL, id, id, id *)) objc_msgSend)(
            dev,
            sel_registerName("newLibraryWithSource:options:error:"),
            str(shader),
            nullptr,
            &error);
    id fn   = send_id(lib, "newFunctionWithName:", str(kernel));
    id pipe = ((id (*)(id, SEL, id, id *)) objc_msgSend)(
            dev, sel_registerName("newComputePipelineStateWithFunction:error:"), fn, &error);
    const size_t capacity = 1u << 22;
    id buf = make_buffer(capacity * sizeof(uint32_t), dev), queue = send(dev, "newCommandQueue");
    if (!pipe || !buf || !queue) {
        const char *message = error == nullptr ? "missing Metal device, buffer or command queue"
                                               : ((const char *(*)(id, SEL)) objc_msgSend)(
                                                         send(error, "localizedDescription"),
                                                         sel_registerName("UTF8String"));
        fprintf(stderr, "Cannot run Metal calibration: %s\n", message);
        return 3;
    }
    for (uint64_t start = first; start <= last;) {
        const uint64_t remaining = (uint64_t) last - start + 1;
        const size_t   n         = remaining < capacity ? (size_t) remaining : capacity;
        id             iteration = send((id) objc_getClass("NSAutoreleasePool"), "new");
        id             cmd = send(queue, "commandBuffer"), enc = send(cmd, "computeCommandEncoder");
        if (!enc)
            return 3;
        ((void (*)(id, SEL, id)) objc_msgSend)(
                enc, sel_registerName("setComputePipelineState:"), pipe);
        const uint32_t base = (uint32_t) start;
        ((void (*)(id, SEL, const void *, size_t, size_t)) objc_msgSend)(
                enc, sel_registerName("setBytes:length:atIndex:"), &base, sizeof(base), 0);
        set_buffer(enc, buf, 1);
        ((void (*)(id, SEL, struct extent, struct extent)) objc_msgSend)(
                enc,
                sel_registerName("dispatchThreads:threadsPerThreadgroup:"),
                (struct extent) {n, 1, 1},
                (struct extent) {128, 1, 1});
        send(enc, "endEncoding");
        send(cmd, "commit");
        send(cmd, "waitUntilCompleted");
        if (((size_t (*)(id, SEL)) objc_msgSend)(cmd, sel_registerName("status")) != 4)
            return 3;
        const uint32_t *values = contents(buf);
        if (check_constant) {
            for (size_t i = 0; i < n; i++) {
                if (values[i] != expected) {
                    fprintf(stderr,
                            "%08x gives %08x, expected %08x\n",
                            (uint32_t) (start + i),
                            values[i],
                            expected);
                    return 1;
                }
            }
        } else if (fwrite(values, sizeof(*values), n, target) != n) {
            return 3;
        }
        send(iteration, "drain");
        start += n;
    }
    send(pool, "drain");
    return fclose(target) != 0;
}
