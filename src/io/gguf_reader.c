#include "gguf_reader.h"
#include "heap.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* GGUF metadata value types */
enum {
    GGUF_VT_U8     = 0,
    GGUF_VT_I8     = 1,
    GGUF_VT_U16    = 2,
    GGUF_VT_I16    = 3,
    GGUF_VT_U32    = 4,
    GGUF_VT_I32    = 5,
    GGUF_VT_F32    = 6,
    GGUF_VT_BOOL   = 7,
    GGUF_VT_STRING = 8,
    GGUF_VT_ARRAY  = 9,
    GGUF_VT_U64    = 10,
    GGUF_VT_I64    = 11,
    GGUF_VT_F64    = 12,
};

#define MAGIC_LE 0x46554747u /* "GGUF" little-endian */

/* P1.4.d: metadata KV record. Each entry points into the mmap for both
 * key and value bytes; ctx lifetime keeps the mmap mapped. */
struct gguf_meta_kv_t {
    const char    *key; /* nul-terminated, owned by ctx->meta_keys arena */
    size_t         key_len;
    uint32_t       vt;   /* GGUF_VT_* */
    const uint8_t *vp;   /* start of value payload (after type tag) */
    size_t         vlen; /* total payload bytes */
};

struct gguf_ctx {
    int    fd; /* -1 for the from-memory path */
    void  *map;
    size_t map_size;
    bool   owns_map; /* true: we mmap'd it (munmap on close); false: caller's buffer */

    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    uint64_t alignment;   /* default 32 */
    size_t   data_offset; /* start of tensor data section */

    /* Tensor table */
    struct gguf_tensor_t *tensors; /* tensor_count entries */
    char                 *name_arena;
    size_t                name_arena_used;

    /* Metadata KV table (P1.4.d) */
    struct gguf_meta_kv_t *meta_kvs;
    char                  *meta_key_arena;
    size_t                 meta_key_arena_used;
};

/* ---- dtype info ----------------------------------------------------------- */

struct dtype_row_t {
    gguf_dtype_t dt;
    size_t       block_bytes;
    size_t       block_elems;
    const char  *name;
};

static const struct dtype_row_t DTYPE_ROWS[] = {
        {GGUF_TYPE_F32, 4, 1, "F32"},
        {GGUF_TYPE_F16, 2, 1, "F16"},
        {GGUF_TYPE_BF16, 2, 1, "BF16"},
        {GGUF_TYPE_Q4_0, 18, 32, "Q4_0"}, /* fp16 d + 32 nibbles */
        {GGUF_TYPE_Q4_1, 20, 32, "Q4_1"}, /* fp16 d, fp16 m + 32 nibbles */
        {GGUF_TYPE_Q5_0, 22, 32, "Q5_0"},
        {GGUF_TYPE_Q5_1, 24, 32, "Q5_1"},
        {GGUF_TYPE_Q8_0, 34, 32, "Q8_0"}, /* fp16 d + 32 int8 */
        {GGUF_TYPE_Q8_1, 36, 32, "Q8_1"},
        {GGUF_TYPE_Q2_K, 82, 256, "Q2_K"},
        {GGUF_TYPE_Q3_K, 110, 256, "Q3_K"},
        {GGUF_TYPE_Q4_K, 144, 256, "Q4_K"}, /* k-quant 4-bit super-block */
        {GGUF_TYPE_Q5_K, 176, 256, "Q5_K"},
        {GGUF_TYPE_Q6_K, 210, 256, "Q6_K"},
        {GGUF_TYPE_Q8_K, 292, 256, "Q8_K"},
        {GGUF_TYPE_IQ3_S, 110, 256, "IQ3_S"}, /* fp16 d + 64 qs + 8 qh + 32 signs + 4 scales */
        {GGUF_TYPE_IQ2_S, 82, 256, "IQ2_S"},  /* fp16 d + 64 qs (incl signs) + 8 qh + 8 scales */
        {GGUF_TYPE_TQ1_0, 54, 256, "TQ1_0"},  /* 5-trit packed; 52 + 2-byte fp16 scale */
        {GGUF_TYPE_TQ2_0, 66, 256, "TQ2_0"},  /* 4-trit packed; 64 + 2-byte fp16 scale */
        /* I2_S (BitNet b1.58 official): 256-elem blocks of 64 packed bytes (4
         * trits/byte, 2.0 bpw), NO per-block scale. A single f32 per-TENSOR scale
         * is stored right after the packed bytes (offset n_elems/4); it sits just
         * past the block-computed nbytes but inside the mmap, so the kernel reads
         * it from raw + n_in*n_out/4. (Confirmed against bitnet.cpp quantize_i2_s:
         * one scale_ptr[0] per tensor, not per row.) */
        {GGUF_TYPE_I2_S, 64, 256, "I2_S"},
};
static const size_t DTYPE_ROWS_N = sizeof(DTYPE_ROWS) / sizeof(DTYPE_ROWS[0]);

static const struct dtype_row_t *dtype_lookup(gguf_dtype_t dt) {
    for (size_t i = 0; i < DTYPE_ROWS_N; i++)
        if (DTYPE_ROWS[i].dt == dt)
            return &DTYPE_ROWS[i];
    return nullptr;
}

size_t gguf_dtype_block_size(gguf_dtype_t dt) {
    const struct dtype_row_t *r = dtype_lookup(dt);
    return r ? r->block_bytes : 0;
}
size_t gguf_dtype_block_count(gguf_dtype_t dt) {
    const struct dtype_row_t *r = dtype_lookup(dt);
    return r ? r->block_elems : 0;
}
const char *gguf_dtype_name(gguf_dtype_t dt) {
    const struct dtype_row_t *r = dtype_lookup(dt);
    return r ? r->name : "?";
}

size_t gguf_tensor_elem_count(const struct gguf_tensor_t *t) {
    if (!t)
        return 0;
    size_t n = 1;
    for (int i = 0; i < t->n_dims; i++) {
        const size_t d = (size_t) t->dims[i];
        /* Reject file-provided dims whose product overflows size_t —
         * otherwise a wrapped-small count yields a tiny nbytes that passes
         * the EOF check while the true tensor is huge (OOB downstream). */
        if (d != 0 && n > SIZE_MAX / d)
            return SIZE_MAX;
        n *= d;
    }
    return n;
}

/* ---- low-level stream reader (cursor + bounds-checked) -------------------- */

struct cur_t {
    const uint8_t *p;
    const uint8_t *end;
    bool           ok;
};

[[maybe_unused]] static uint8_t read_u8(struct cur_t *c) {
    if (c->p + 1 > c->end) {
        c->ok = false;
        return 0;
    }
    return *c->p++;
}
static uint32_t read_u32(struct cur_t *c) {
    if (c->p + 4 > c->end) {
        c->ok = false;
        return 0;
    }
    uint32_t v;
    memcpy(&v, c->p, 4);
    c->p += 4;
    return v;
}
static uint64_t read_u64(struct cur_t *c) {
    if (c->p + 8 > c->end) {
        c->ok = false;
        return 0;
    }
    uint64_t v;
    memcpy(&v, c->p, 8);
    c->p += 8;
    return v;
}

/* Read a GGUF string (uint64 len + bytes). Returns pointer (into mmap) and length. */
static const char *read_str(struct cur_t *c, size_t *out_len) {
    uint64_t len = read_u64(c);
    if (!c->ok || c->p + len > c->end) {
        c->ok    = false;
        *out_len = 0;
        return nullptr;
    }
    const char *s = (const char *) c->p;
    c->p += len;
    *out_len = (size_t) len;
    return s;
}

/* Walk a value of given type, advancing cursor. Returns false on bad data. */
static bool skip_value(struct cur_t *c, uint32_t vt);

static bool skip_value(struct cur_t *c, uint32_t vt) {
    switch (vt) {
    case GGUF_VT_U8:
    case GGUF_VT_I8:
    case GGUF_VT_BOOL:
        c->p += 1;
        break;
    case GGUF_VT_U16:
    case GGUF_VT_I16:
        c->p += 2;
        break;
    case GGUF_VT_U32:
    case GGUF_VT_I32:
    case GGUF_VT_F32:
        c->p += 4;
        break;
    case GGUF_VT_U64:
    case GGUF_VT_I64:
    case GGUF_VT_F64:
        c->p += 8;
        break;
    case GGUF_VT_STRING: {
        size_t len;
        (void) read_str(c, &len);
        break;
    }
    case GGUF_VT_ARRAY: {
        uint32_t elem_vt = read_u32(c);
        uint64_t n       = read_u64(c);
        if (!c->ok)
            return false;
        for (uint64_t i = 0; i < n; i++) {
            if (!skip_value(c, elem_vt))
                return false;
            if (!c->ok)
                return false;
        }
        break;
    }
    default:
        c->ok = false;
        return false;
    }
    if (c->p > c->end) {
        c->ok = false;
        return false;
    }
    return c->ok;
}

/* P1.4.d: record a metadata KV entry by spanning the value's bytes and
 * advancing the cursor. Also captures general.alignment when seen.
 * `slot_idx` is the write position in ctx->meta_kvs[]. */
static bool record_meta_kv(struct gguf_ctx *ctx,
                           size_t           slot_idx,
                           const char      *key,
                           size_t           key_len,
                           struct cur_t    *c,
                           uint32_t         vt) {
    const uint8_t *vp_start = c->p;
    if (!skip_value(c, vt) || !c->ok) {
        return false;
    }
    const size_t vlen = (size_t) (c->p - vp_start);

    /* Copy the key into the meta_key_arena and NUL-terminate so callers
     * can use strcmp(); the original is just length-prefixed. */
    char *kdst = ctx->meta_key_arena + ctx->meta_key_arena_used;
    memcpy(kdst, key, key_len);
    kdst[key_len] = '\0';
    ctx->meta_key_arena_used += key_len + 1;

    ctx->meta_kvs[slot_idx] = (struct gguf_meta_kv_t) {
            .key     = kdst,
            .key_len = key_len,
            .vt      = vt,
            .vp      = vp_start,
            .vlen    = vlen,
    };

    /* Inline-capture general.alignment so the data-offset math below
     * stays cheap (no second pass over the meta table). */
    if (key_len == 17 && memcmp(key, "general.alignment", 17) == 0 && vt == GGUF_VT_U32 &&
        vlen == 4) {
        uint32_t a;
        memcpy(&a, vp_start, 4);
        /* Ignore a zero alignment — it would make the data-offset padding
         * math (`% ctx->alignment`) a divide-by-zero (SIGFPE) at open. */
        if (a != 0)
            ctx->alignment = (uint64_t) a;
    }
    return true;
}

/* ---- open / close --------------------------------------------------------- */

static const char *set_err(const char **out, const char *msg) {
    if (out)
        *out = msg;
    return msg;
}

/* Shared parser: `map`/`fsize` already point at the GGUF bytes — an mmap we
 * own (file path) or a caller-provided buffer (from-memory path, owns_map=0).
 * On failure the bytes are released only if we own them. */
static struct gguf_ctx *
gguf_parse(void *map, size_t fsize, int fd, bool owns_map, const char **errmsg) {
    struct cur_t c = {.p = (const uint8_t *) map, .end = (const uint8_t *) map + fsize, .ok = true};

#define GGUF_FAIL(msg)          \
    do {                        \
        if (owns_map) {         \
            munmap(map, fsize); \
            if (fd >= 0)        \
                close(fd);      \
        }                       \
        set_err(errmsg, (msg)); \
        return nullptr;         \
    } while (0)

    uint32_t magic = read_u32(&c);
    if (!c.ok || magic != MAGIC_LE) {
        GGUF_FAIL("bad magic (not GGUF)");
    }
    uint32_t version = read_u32(&c);
    if (!c.ok || version != 3) {
        GGUF_FAIL("unsupported version (need v3)");
    }
    uint64_t tcount = read_u64(&c);
    uint64_t mcount = read_u64(&c);
    if (!c.ok) {
        GGUF_FAIL("header truncated");
    }

    struct gguf_ctx *ctx = heap_calloc_array_aligned(struct gguf_ctx, 1);
    if (!ctx) {
        GGUF_FAIL("calloc ctx");
    }
#undef GGUF_FAIL
    ctx->fd                = fd;
    ctx->map               = map;
    ctx->map_size          = fsize;
    ctx->owns_map          = owns_map;
    ctx->version           = version;
    ctx->tensor_count      = tcount;
    ctx->metadata_kv_count = mcount;
    ctx->alignment         = 32; /* default per spec */

    /* P1.4.d: allocate metadata KV table + key arena, then walk and
     * record every entry. Upper-bound the key arena by the remaining
     * mmap bytes — names + lengths fit within it for any well-formed
     * GGUF (the structure is just length-prefixed strings interleaved
     * with values, and values are at most as large as a couple of
     * float arrays). */
    if (mcount > 0) {
        ctx->meta_kvs = heap_calloc_array_aligned(struct gguf_meta_kv_t, mcount);
        if (!ctx->meta_kvs) {
            gguf_close(ctx);
            set_err(errmsg, "calloc meta_kvs");
            return nullptr;
        }
        ctx->meta_key_arena = heap_alloc_array_aligned(char, (size_t) (c.end - c.p) + 1);
        if (!ctx->meta_key_arena) {
            gguf_close(ctx);
            set_err(errmsg, "malloc meta_key_arena");
            return nullptr;
        }
    }
    for (uint64_t i = 0; i < mcount; i++) {
        size_t      klen;
        const char *key = read_str(&c, &klen);
        uint32_t    vt  = read_u32(&c);
        if (!c.ok) {
            gguf_close(ctx);
            set_err(errmsg, "metadata KV truncated");
            return nullptr;
        }
        if (!record_meta_kv(ctx, (size_t) i, key, klen, &c, vt) || !c.ok) {
            gguf_close(ctx);
            set_err(errmsg, "metadata value walk failed");
            return nullptr;
        }
    }

    /* Tensor info section */
    ctx->tensors = heap_calloc_array_aligned(struct gguf_tensor_t, tcount);
    if (!ctx->tensors) {
        gguf_close(ctx);
        set_err(errmsg, "calloc tensors");
        return nullptr;
    }
    /* Upper-bound name arena by remaining bytes in mmap. */
    ctx->name_arena = heap_alloc_array_aligned(char, (size_t) (c.end - c.p) + 1);
    if (!ctx->name_arena) {
        gguf_close(ctx);
        set_err(errmsg, "malloc name_arena");
        return nullptr;
    }

    for (uint64_t i = 0; i < tcount; i++) {
        size_t      nlen;
        const char *nptr = read_str(&c, &nlen);
        if (!c.ok) {
            gguf_close(ctx);
            set_err(errmsg, "tensor name truncated");
            return nullptr;
        }
        char *name_dst = ctx->name_arena + ctx->name_arena_used;
        memcpy(name_dst, nptr, nlen);
        name_dst[nlen] = '\0';
        ctx->name_arena_used += nlen + 1;
        ctx->tensors[i].name = name_dst;

        uint32_t n_dims = read_u32(&c);
        if (n_dims > GGUF_MAX_DIMS) {
            gguf_close(ctx);
            set_err(errmsg, "tensor n_dims > 4");
            return nullptr;
        }
        ctx->tensors[i].n_dims = (int) n_dims;
        for (uint32_t d = 0; d < n_dims; d++)
            ctx->tensors[i].dims[d] = read_u64(&c);

        uint32_t dt           = read_u32(&c);
        ctx->tensors[i].dtype = (gguf_dtype_t) dt;

        uint64_t off           = read_u64(&c);
        ctx->tensors[i].offset = off;
        if (!c.ok) {
            gguf_close(ctx);
            set_err(errmsg, "tensor info truncated");
            return nullptr;
        }
    }

    /* Compute data section start: align (c.p - map) up to alignment. */
    size_t info_end  = (size_t) (c.p - (const uint8_t *) map);
    size_t pad       = (ctx->alignment - (info_end % ctx->alignment)) % ctx->alignment;
    ctx->data_offset = info_end + pad;
    if (ctx->data_offset > fsize) {
        gguf_close(ctx);
        set_err(errmsg, "data section past EOF");
        return nullptr;
    }

    /* Fix up tensor data pointers + nbytes */
    for (uint64_t i = 0; i < tcount; i++) {
        struct gguf_tensor_t     *t  = &ctx->tensors[i];
        const struct dtype_row_t *dt = dtype_lookup(t->dtype);
        if (!dt) {
            /* Allow unknown dtypes through but mark nbytes=0; reader keeps loading. */
            t->nbytes = 0;
            t->data   = nullptr;
            continue;
        }
        size_t elems = gguf_tensor_elem_count(t);
        if (elems == SIZE_MAX) {
            gguf_close(ctx);
            set_err(errmsg, "tensor dims overflow");
            return nullptr;
        }
        /* All EOF checks below are phrased as `x > avail - y` (never
         * `off + nbytes > fsize`) so a file-controlled offset/size near
         * UINT64_MAX cannot wrap the sum small and slip past the guard.
         * data_offset <= fsize was established above, so avail is valid. */
        const size_t avail = fsize - ctx->data_offset;
        /* Some quants (Microsoft's I2_S) don't fit the "fixed block size"
         * model — their storage is per-row not per-256-element-block.
         * For those entries we register block_bytes=0, block_elems=0 so
         * the reader can recognize the dtype id and still surface the
         * tensor, while the dispatcher in the loader fails on it
         * explicitly. The data pointer is set so callers that DO
         * understand the layout (P1.6 native I2_S resolver) can read
         * the raw bytes — but nbytes is left at 0 to flag the gap. */
        if (dt->block_elems == 0) {
            t->nbytes = 0;
            if (t->offset > avail) {
                gguf_close(ctx);
                set_err(errmsg, "tensor data past EOF");
                return nullptr;
            }
            t->data = (const uint8_t *) map + ctx->data_offset + t->offset;
            continue;
        }
        if (elems % dt->block_elems != 0) {
            gguf_close(ctx);
            set_err(errmsg, "tensor elem count not divisible by block size");
            return nullptr;
        }
        t->nbytes = (elems / dt->block_elems) * dt->block_bytes;
        /* I2_S stores one extra per-tensor f32 scale just past the packed
         * blocks (at offset nbytes); it must stay inside the map too. */
        const size_t tail = (t->dtype == GGUF_TYPE_I2_S) ? sizeof(float) : 0;
        if (t->offset > avail || t->nbytes > avail - t->offset ||
            tail > avail - t->offset - t->nbytes) {
            gguf_close(ctx);
            set_err(errmsg, "tensor data past EOF");
            return nullptr;
        }
        t->data = (const uint8_t *) map + ctx->data_offset + t->offset;
    }

    return ctx;
}

/* Best-effort per-platform mmap hints for the weight mapping — helps larger
 * models. All hints are advisory and silently ignored where unsupported.
 *
 *  - Linux MADV_HUGEPAGE: request transparent huge pages → fewer TLB misses on
 *    the big weight tables. A real win on 4 KB-page Linux (Graviton, generic
 *    ARM/x86 servers); a no-op where THP is off or base pages are already large
 *    (Raspberry Pi 5 = 16 KB pages; macOS Apple Silicon = 16 KB). Disable with
 *    GEIST_NO_HUGEPAGE=1.
 *  - MADV_WILLNEED (opt-in via GEIST_MMAP_PREFETCH=1): prefault the whole
 *    mapping so first-token latency doesn't pay page faults — trades a bigger
 *    upfront read. Off by default (the page cache is usually warm on repeat
 *    runs, and a cold multi-GB prefault just front-loads the same I/O). */
static void apply_mmap_advice(void *map, size_t size) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (getenv("GEIST_NO_HUGEPAGE") == nullptr) {
        (void) madvise(map, size, MADV_HUGEPAGE);
    }
#endif
#if defined(MADV_WILLNEED)
    if (getenv("GEIST_MMAP_PREFETCH") != nullptr) {
        (void) madvise(map, size, MADV_WILLNEED);
    }
#endif
    (void) map;
    (void) size;
}

struct gguf_ctx *gguf_open(const char *path, const char **errmsg) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_err(errmsg, "open() failed");
        return nullptr;
    }

    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        close(fd);
        set_err(errmsg, "fstat() failed");
        return nullptr;
    }
    size_t fsize = (size_t) sb.st_size;

    void *map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        set_err(errmsg, "mmap() failed");
        return nullptr;
    }
    apply_mmap_advice(map, fsize);

    return gguf_parse(map, fsize, fd, /*owns_map=*/true, errmsg);
}

struct gguf_ctx *gguf_open_memory(const void *data, size_t size, const char **errmsg) {
    if (data == nullptr || size < 8) {
        set_err(errmsg, "gguf_open_memory: null or too-small buffer");
        return nullptr;
    }
    /* The caller owns `data`; we alias it read-only and never munmap/free it.
     * The buffer must outlive the model (it is the tensor backing store). */
    return gguf_parse((void *) data, size, /*fd=*/-1, /*owns_map=*/false, errmsg);
}

void gguf_close(struct gguf_ctx *ctx) {
    if (!ctx)
        return;
    if (ctx->owns_map && ctx->map && ctx->map != MAP_FAILED)
        munmap(ctx->map, ctx->map_size);
    if (ctx->fd >= 0)
        close(ctx->fd);
    safe_free((void **) &ctx->name_arena);
    safe_free((void **) &ctx->tensors);
    safe_free((void **) &ctx->meta_kvs);
    safe_free((void **) &ctx->meta_key_arena);
    safe_free((void **) &ctx);
}

/* ---- Metadata KV accessors (P1.4.d) ---------------------------------- */

static const struct gguf_meta_kv_t *meta_find(const struct gguf_ctx *ctx, const char *key) {
    if (!ctx || !ctx->meta_kvs || !key)
        return nullptr;
    for (uint64_t i = 0; i < ctx->metadata_kv_count; i++) {
        if (strcmp(ctx->meta_kvs[i].key, key) == 0) {
            return &ctx->meta_kvs[i];
        }
    }
    return nullptr;
}

const char *gguf_get_meta_string(const struct gguf_ctx *ctx, const char *key, size_t *out_len) {
    const struct gguf_meta_kv_t *kv = meta_find(ctx, key);
    if (!kv || kv->vt != GGUF_VT_STRING || kv->vlen < 8) {
        if (out_len)
            *out_len = 0;
        return nullptr;
    }
    /* String layout: u64 length + bytes. */
    uint64_t slen;
    memcpy(&slen, kv->vp, 8);
    if (8 + slen > kv->vlen) {
        if (out_len)
            *out_len = 0;
        return nullptr;
    }
    if (out_len)
        *out_len = (size_t) slen;
    return (const char *) (kv->vp + 8);
}

bool gguf_get_meta_u32(const struct gguf_ctx *ctx, const char *key, uint32_t *out) {
    const struct gguf_meta_kv_t *kv = meta_find(ctx, key);
    if (!kv || !out)
        return false;
    if (kv->vt == GGUF_VT_U32 && kv->vlen == 4) {
        memcpy(out, kv->vp, 4);
        return true;
    }
    /* Accept narrower unsigned widths too; common for vocab_size which
     * sometimes ships as u64 in newer GGUFs. */
    if (kv->vt == GGUF_VT_U64 && kv->vlen == 8) {
        uint64_t v;
        memcpy(&v, kv->vp, 8);
        if (v > (uint64_t) UINT32_MAX)
            return false;
        *out = (uint32_t) v;
        return true;
    }
    return false;
}

bool gguf_get_meta_u64(const struct gguf_ctx *ctx, const char *key, uint64_t *out) {
    const struct gguf_meta_kv_t *kv = meta_find(ctx, key);
    if (!kv || !out)
        return false;
    if (kv->vt == GGUF_VT_U64 && kv->vlen == 8) {
        memcpy(out, kv->vp, 8);
        return true;
    }
    if (kv->vt == GGUF_VT_U32 && kv->vlen == 4) {
        uint32_t v;
        memcpy(&v, kv->vp, 4);
        *out = (uint64_t) v;
        return true;
    }
    return false;
}

bool gguf_get_meta_f32(const struct gguf_ctx *ctx, const char *key, float *out) {
    const struct gguf_meta_kv_t *kv = meta_find(ctx, key);
    if (!kv || !out || kv->vt != GGUF_VT_F32 || kv->vlen != 4)
        return false;
    memcpy(out, kv->vp, 4);
    return true;
}

bool gguf_get_meta_bool(const struct gguf_ctx *ctx, const char *key, bool *out) {
    const struct gguf_meta_kv_t *kv = meta_find(ctx, key);
    if (!kv || !out || kv->vt != GGUF_VT_BOOL || kv->vlen != 1)
        return false;
    *out = (*kv->vp != 0);
    return true;
}

bool gguf_get_meta_array_info(const struct gguf_ctx *ctx,
                              const char            *key,
                              uint32_t              *out_elem_vt,
                              uint64_t              *out_count,
                              const uint8_t        **out_payload) {
    const struct gguf_meta_kv_t *kv = meta_find(ctx, key);
    if (!kv || kv->vt != GGUF_VT_ARRAY || kv->vlen < 12)
        return false;
    uint32_t elem_vt;
    memcpy(&elem_vt, kv->vp, 4);
    uint64_t count;
    memcpy(&count, kv->vp + 4, 8);
    if (out_elem_vt)
        *out_elem_vt = elem_vt;
    if (out_count)
        *out_count = count;
    if (out_payload)
        *out_payload = kv->vp + 12;
    return true;
}

size_t gguf_tensor_count(const struct gguf_ctx *ctx) {
    return ctx ? (size_t) ctx->tensor_count : 0;
}
const struct gguf_tensor_t *gguf_tensor_at(const struct gguf_ctx *ctx, size_t idx) {
    if (!ctx || idx >= ctx->tensor_count)
        return nullptr;
    return &ctx->tensors[idx];
}
const struct gguf_tensor_t *gguf_get_tensor(const struct gguf_ctx *ctx, const char *name) {
    if (!ctx || !name)
        return nullptr;
    for (size_t i = 0; i < ctx->tensor_count; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return &ctx->tensors[i];
    }
    return nullptr;
}
