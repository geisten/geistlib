#include "gguf_ptqtp.h"
#include "heap.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
#include <arm_neon.h>
#endif

/* Bulk fp16 → fp32 conversion. Uses hardware vcvt on NEON, scalar fallback
 * everywhere else. The scalar path mirrors gguf_quant.c's fp16_to_fp32
 * but is inlined here so this file stays standalone. */
static void ptqtp_convert_alpha_fp16_to_fp32(float *dst, const uint16_t *src, size_t n) {
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t v = vld1q_f16((const __fp16 *) (src + i));
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(v)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(v)));
    }
    for (; i < n; i++) {
        __fp16 h;
        memcpy(&h, &src[i], 2);
        dst[i] = (float) h;
    }
#else
    for (size_t i = 0; i < n; i++) {
        const uint16_t h    = src[i];
        const uint32_t sign = (uint32_t) (h >> 15) & 0x1;
        const uint32_t exp  = (uint32_t) (h >> 10) & 0x1F;
        uint32_t       frac = (uint32_t) (h) & 0x3FF;
        uint32_t       out;
        if (exp == 0) {
            if (frac == 0) {
                out = sign << 31;
            } else {
                int e = -1;
                while ((frac & 0x400) == 0) {
                    frac <<= 1;
                    e--;
                }
                frac &= 0x3FF;
                const uint32_t exp32 = (uint32_t) (127 - 13 + e);
                out                  = (sign << 31) | (exp32 << 23) | (frac << 13);
            }
        } else if (exp == 0x1F) {
            out = (sign << 31) | (0xFF << 23) | (frac << 13);
        } else {
            out = (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13);
        }
        memcpy(&dst[i], &out, 4);
    }
#endif
}

struct ptqtp_ctx {
    int                    fd;
    const uint8_t         *map;
    size_t                 map_size;
    uint32_t               version;
    uint32_t               n_tensors;
    uint32_t               group_size;
    uint32_t               n_planes;
    char                  *name_arena; /* concatenated NUL-terminated names */
    size_t                 name_arena_used;
    struct ptqtp_tensor_t *tensors;          /* n_tensors entries */
    float                 *alpha_fp32_arena; /* big FP32 alpha buffer; tensors point in */
};

static const char ERR_OPEN[]    = "ptqtp: open() failed";
static const char ERR_FSTAT[]   = "ptqtp: fstat() failed";
static const char ERR_MMAP[]    = "ptqtp: mmap() failed";
static const char ERR_TRUNC[]   = "ptqtp: file truncated / TOC parse error";
static const char ERR_MAGIC[]   = "ptqtp: bad magic, not a PTQTP file";
static const char ERR_VERSION[] = "ptqtp: unsupported file version";
static const char ERR_NPLANES[] = "ptqtp: unsupported n_planes (only 2 supported)";
static const char ERR_NOMEM[]   = "ptqtp: out of memory";
static const char ERR_DIMS[]    = "ptqtp: tensor dim/group inconsistency";

void ptqtp_close(struct ptqtp_ctx *ctx) {
    if (!ctx)
        return;
    if (ctx->map && ctx->map != MAP_FAILED)
        munmap((void *) ctx->map, ctx->map_size);
    if (ctx->fd >= 0)
        close(ctx->fd);
    safe_free((void **) &ctx->name_arena);
    safe_free((void **) &ctx->tensors);
    safe_free((void **) &ctx->alpha_fp32_arena);
    safe_free((void **) &ctx);
}

struct ptqtp_ctx *ptqtp_open(const char *path, const char **err) {
    if (err)
        *err = nullptr;
    struct ptqtp_ctx *ctx = heap_calloc_array_aligned(struct ptqtp_ctx, 1);
    if (!ctx) {
        if (err)
            *err = ERR_NOMEM;
        return nullptr;
    }
    ctx->fd = -1;

    ctx->fd = open(path, O_RDONLY);
    if (ctx->fd < 0) {
        if (err)
            *err = ERR_OPEN;
        ptqtp_close(ctx);
        return nullptr;
    }

    struct stat st;
    if (fstat(ctx->fd, &st) < 0) {
        if (err)
            *err = ERR_FSTAT;
        ptqtp_close(ctx);
        return nullptr;
    }
    ctx->map_size = (size_t) st.st_size;

    void *m = mmap(nullptr, ctx->map_size, PROT_READ, MAP_PRIVATE, ctx->fd, 0);
    if (m == MAP_FAILED) {
        if (err)
            *err = ERR_MMAP;
        ptqtp_close(ctx);
        return nullptr;
    }
    ctx->map = (const uint8_t *) m;

    /* Parse header: 4 magic + 4*4 fields = 20 bytes */
    if (ctx->map_size < 20) {
        if (err)
            *err = ERR_TRUNC;
        ptqtp_close(ctx);
        return nullptr;
    }
    if (memcmp(ctx->map, "PTQT", 4) != 0) {
        if (err)
            *err = ERR_MAGIC;
        ptqtp_close(ctx);
        return nullptr;
    }
    const uint8_t *p = ctx->map + 4;
    memcpy(&ctx->version, p, 4);
    memcpy(&ctx->n_tensors, p + 4, 4);
    memcpy(&ctx->group_size, p + 8, 4);
    memcpy(&ctx->n_planes, p + 12, 4);
    p += 16;

    if (ctx->version != 1 && ctx->version != 2) {
        if (err) {
            *err = ERR_VERSION;
        }
        ptqtp_close(ctx);
        return nullptr;
    }
    if (ctx->n_planes != 2 && ctx->n_planes != 3) {
        if (err) {
            *err = ERR_NPLANES;
        }
        ptqtp_close(ctx);
        return nullptr;
    }

    /* TOC entry layout:
     *   v1: name_len(4) + name + n_in(4) + n_out(4) + n_groups(4) +
     *       4×u64 offsets/sizes + cos_sim(4)        = 52 + name
     *   v2: same as v1, plus n_planes(1) + reserved(3) trailing
     *                                                = 56 + name
     */
    const uint8_t *toc         = p;
    const size_t   entry_extra = (ctx->version >= 2) ? 4 : 0;
    size_t         arena_bytes = 0;
    {
        const uint8_t *q = toc;
        for (uint32_t i = 0; i < ctx->n_tensors; i++) {
            if ((size_t) (q - ctx->map) + 4 > ctx->map_size) {
                if (err)
                    *err = ERR_TRUNC;
                ptqtp_close(ctx);
                return nullptr;
            }
            uint32_t nlen;
            memcpy(&nlen, q, 4);
            q += 4;
            arena_bytes += (size_t) nlen + 1;
            const size_t entry_tail = (size_t) nlen + 12 + 32 + 4 + entry_extra;
            if ((size_t) (q - ctx->map) + entry_tail > ctx->map_size) {
                if (err)
                    *err = ERR_TRUNC;
                ptqtp_close(ctx);
                return nullptr;
            }
            q += entry_tail;
        }
    }

    ctx->name_arena = heap_alloc_array_aligned(char, arena_bytes);
    ctx->tensors    = heap_calloc_array_aligned(struct ptqtp_tensor_t, ctx->n_tensors);
    if (!ctx->name_arena || !ctx->tensors) {
        if (err)
            *err = ERR_NOMEM;
        ptqtp_close(ctx);
        return nullptr;
    }

    /* Second pass: populate tensor table. */
    {
        const uint8_t *q = toc;
        for (uint32_t i = 0; i < ctx->n_tensors; i++) {
            uint32_t nlen;
            memcpy(&nlen, q, 4);
            q += 4;
            char *dst = ctx->name_arena + ctx->name_arena_used;
            memcpy(dst, q, nlen);
            dst[nlen] = '\0';
            ctx->name_arena_used += (size_t) nlen + 1;
            ctx->tensors[i].name = dst;
            q += nlen;

            memcpy(&ctx->tensors[i].n_in, q, 4);
            memcpy(&ctx->tensors[i].n_out, q + 4, 4);
            memcpy(&ctx->tensors[i].n_groups, q + 8, 4);
            q += 12;

            uint64_t trit_off, trit_sz, alpha_off, alpha_sz;
            memcpy(&trit_off, q, 8);
            memcpy(&trit_sz, q + 8, 8);
            memcpy(&alpha_off, q + 16, 8);
            memcpy(&alpha_sz, q + 24, 8);
            q += 32;
            memcpy(&ctx->tensors[i].cos_sim, q, 4);
            q += 4;

            /* Per-tensor n_planes + storage variant (v2) or copy from header (v1).
             * v2 TOC tail: n_planes(1) + storage(1) + reserved(2). */
            uint8_t per_tensor_planes;
            uint8_t per_tensor_storage = PTQTP_STORE_STANDARD;
            if (ctx->version >= 2) {
                per_tensor_planes  = q[0];
                per_tensor_storage = q[1];
                q += 4;
            } else {
                per_tensor_planes = (uint8_t) ctx->n_planes;
            }
            if (per_tensor_planes != 2 && per_tensor_planes != 3) {
                if (err)
                    *err = ERR_NPLANES;
                ptqtp_close(ctx);
                return nullptr;
            }
            if (per_tensor_storage != PTQTP_STORE_STANDARD &&
                per_tensor_storage != PTQTP_STORE_PACKED5) {
                if (err)
                    *err = ERR_NPLANES; /* reuse error code */
                ptqtp_close(ctx);
                return nullptr;
            }
            if (per_tensor_storage == PTQTP_STORE_PACKED5 && per_tensor_planes != 3) {
                if (err)
                    *err = ERR_NPLANES;
                ptqtp_close(ctx);
                return nullptr;
            }
            ctx->tensors[i].n_planes = per_tensor_planes;
            ctx->tensors[i].storage  = per_tensor_storage;

            const uint32_t n_in     = ctx->tensors[i].n_in;
            const uint32_t n_out    = ctx->tensors[i].n_out;
            const uint32_t n_groups = ctx->tensors[i].n_groups;
            uint64_t       expect_trit_sz;
            if (per_tensor_storage == PTQTP_STORE_PACKED5) {
                if (n_in % 8u != 0u) {
                    if (err)
                        *err = ERR_DIMS;
                    ptqtp_close(ctx);
                    return nullptr;
                }
                expect_trit_sz = (uint64_t) n_out * (n_in / 2u + n_in / 8u); /* 5n_in/8 */
            } else if (per_tensor_planes == 2) {
                expect_trit_sz = (uint64_t) n_out * n_in / 2;
            } else {
                expect_trit_sz = (uint64_t) n_out * n_in;
            }
            if (expect_trit_sz != trit_sz ||
                (uint64_t) n_out * n_groups * per_tensor_planes * sizeof(uint16_t) != alpha_sz ||
                n_groups == 0 || n_in % n_groups != 0) {
                if (err)
                    *err = ERR_DIMS;
                ptqtp_close(ctx);
                return nullptr;
            }
            /* Overflow-safe bounds: the offsets are raw file uint64s, so
             * `off + sz > map_size` could wrap small and slip past. */
            if (trit_off > ctx->map_size || trit_sz > ctx->map_size - trit_off ||
                alpha_off > ctx->map_size || alpha_sz > ctx->map_size - alpha_off) {
                if (err)
                    *err = ERR_TRUNC;
                ptqtp_close(ctx);
                return nullptr;
            }
            ctx->tensors[i].trits = ctx->map + trit_off;
            ctx->tensors[i].alpha = (const uint16_t *) (ctx->map + alpha_off);
        }
    }

    /* Pre-convert all alpha values from fp16 to fp32 once. The kernel hot
     * path otherwise calls a software fp16_to_fp32 ~15M times per token. */
    /* total_alpha_elems is summed from tensor dims read out of the (possibly
     * corrupt/hostile) GGUF file. Every step — the per-tensor product, the
     * running sum, and the final byte-size multiply — is checked for size_t
     * overflow. A wrapped size would under-allocate and let the convert loop
     * below overflow the heap (AGENT.md: correctness first, no silent
     * truncation). */
    size_t total_alpha_elems = 0;
    bool   size_overflow     = false;
    for (uint32_t i = 0; i < ctx->n_tensors; i++) {
        const size_t b     = (size_t) ctx->tensors[i].n_groups;
        const size_t c     = (size_t) ctx->tensors[i].n_planes;
        size_t       elems = (size_t) ctx->tensors[i].n_out;
        if ((b != 0 && elems > SIZE_MAX / b)) {
            size_overflow = true;
            break;
        }
        elems *= b;
        if ((c != 0 && elems > SIZE_MAX / c)) {
            size_overflow = true;
            break;
        }
        elems *= c;
        if (elems > SIZE_MAX - total_alpha_elems) {
            size_overflow = true;
            break;
        }
        total_alpha_elems += elems;
    }
    if (size_overflow || total_alpha_elems > SIZE_MAX / sizeof(float)) {
        if (err)
            *err = ERR_NOMEM;
        ptqtp_close(ctx);
        return nullptr;
    }
    /* Route through heap.h (per AGENT.md) instead of a raw aligned_alloc; it
     * applies its own overflow-checked rounding and >=64-byte alignment. */
    ctx->alpha_fp32_arena = heap_alloc_array_aligned(float, total_alpha_elems);
    if (!ctx->alpha_fp32_arena) {
        if (err)
            *err = ERR_NOMEM;
        ptqtp_close(ctx);
        return nullptr;
    }
    {
        size_t off = 0;
        for (uint32_t i = 0; i < ctx->n_tensors; i++) {
            const size_t    n   = (size_t) ctx->tensors[i].n_out * ctx->tensors[i].n_groups *
                                  ctx->tensors[i].n_planes;
            float          *dst = ctx->alpha_fp32_arena + off;
            const uint16_t *src = ctx->tensors[i].alpha;
            ptqtp_convert_alpha_fp16_to_fp32(dst, src, n);
            ctx->tensors[i].alpha_fp32 = dst;
            off += n;
        }
    }

    return ctx;
}

const struct ptqtp_tensor_t *ptqtp_get_tensor(const struct ptqtp_ctx *ctx, const char *name) {
    if (!ctx || !name)
        return nullptr;
    for (uint32_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return &ctx->tensors[i];
    }
    return nullptr;
}

size_t ptqtp_n_tensors(const struct ptqtp_ctx *ctx) {
    return ctx ? ctx->n_tensors : 0;
}
uint32_t ptqtp_group_size(const struct ptqtp_ctx *ctx) {
    return ctx ? ctx->group_size : 0;
}
uint32_t ptqtp_n_planes(const struct ptqtp_ctx *ctx) {
    return ctx ? ctx->n_planes : 0;
}

bool ptqtp_is_mixed(const struct ptqtp_ctx *ctx) {
    if (!ctx)
        return false;
    const uint8_t header_planes = (uint8_t) ctx->n_planes;
    for (uint32_t i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i].n_planes != header_planes)
            return true;
    }
    return false;
}
