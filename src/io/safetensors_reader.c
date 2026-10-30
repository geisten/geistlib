#include "safetensors_reader.h"
#include "checked.h"
#include "file_map.h"
#include "heap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

struct st_ctx {
    int    fd;
    void  *map;
    size_t map_size;
    size_t header_size;
    size_t data_region; // = 8 + header_size

    char  *name_arena;
    size_t name_arena_used;

    struct st_tensor_t *tensors;
    size_t              count;
    size_t              cap;
};

// ---- dtype mapping ----------------------------------------------------------

static const struct {
    const char *str;
    st_dtype_t  dtype;
    size_t      bytes;
} DTYPE_TABLE[] = {
        {"F64", ST_DTYPE_F64, 8},
        {"F32", ST_DTYPE_F32, 4},
        {"F16", ST_DTYPE_F16, 2},
        {"BF16", ST_DTYPE_BF16, 2},
        {"I64", ST_DTYPE_I64, 8},
        {"I32", ST_DTYPE_I32, 4},
        {"I16", ST_DTYPE_I16, 2},
        {"I8", ST_DTYPE_I8, 1},
        {"U64", ST_DTYPE_U64, 8},
        {"U32", ST_DTYPE_U32, 4},
        {"U16", ST_DTYPE_U16, 2},
        {"U8", ST_DTYPE_U8, 1},
        {"BOOL", ST_DTYPE_BOOL, 1},
};
static const size_t DTYPE_TABLE_SIZE = sizeof(DTYPE_TABLE) / sizeof(DTYPE_TABLE[0]);

static st_dtype_t parse_dtype(const char *s, size_t len) {
    for (size_t i = 0; i < DTYPE_TABLE_SIZE; i++) {
        if (strlen(DTYPE_TABLE[i].str) == len && strncmp(DTYPE_TABLE[i].str, s, len) == 0) {
            return DTYPE_TABLE[i].dtype;
        }
    }
    return ST_DTYPE_UNKNOWN;
}

size_t st_dtype_bytes(st_dtype_t dtype) {
    for (size_t i = 0; i < DTYPE_TABLE_SIZE; i++) {
        if (DTYPE_TABLE[i].dtype == dtype)
            return DTYPE_TABLE[i].bytes;
    }
    return 0;
}

const char *st_dtype_name(st_dtype_t dtype) {
    for (size_t i = 0; i < DTYPE_TABLE_SIZE; i++) {
        if (DTYPE_TABLE[i].dtype == dtype)
            return DTYPE_TABLE[i].str;
    }
    return "UNKNOWN";
}

// ---- JSON parsing -----------------------------------------------------------
//
// Cursor-based recursive-descent parser, narrow in scope: only handles the
// JSON shapes that safetensors emits (object of tensor_name -> object with
// "dtype" string, "shape" array of ints, "data_offsets" array of two ints).
// The "__metadata__" key is recognised and skipped.

struct cursor_t {
    const char *p;
    const char *end;
    const char *err;
};

static int at_end(const struct cursor_t *c) {
    return c->p >= c->end;
}

static void skip_ws(struct cursor_t *c) {
    while (!at_end(c)) {
        char ch = *c->p;
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
            c->p++;
        else
            break;
    }
}

static int expect(struct cursor_t *c, char ch) {
    skip_ws(c);
    if (at_end(c) || *c->p != ch) {
        c->err = "unexpected character";
        return 0;
    }
    c->p++;
    return 1;
}

// Parse a JSON string. Sets *out to point inside the buffer, *len to length.
// Does NOT handle escape sequences — safetensors keys/values are plain ASCII
// (tensor names use _ . [] digits letters; dtype strings are like "F32").
static int parse_string(struct cursor_t *c, const char **out, size_t *len) {
    if (!expect(c, '"'))
        return 0;
    const char *start = c->p;
    while (!at_end(c) && *c->p != '"') {
        if (*c->p == '\\') {
            c->err = "JSON escape sequences not supported";
            return 0;
        }
        c->p++;
    }
    if (at_end(c)) {
        c->err = "unterminated string";
        return 0;
    }
    *out = start;
    *len = (size_t) (c->p - start);
    c->p++; // consume closing "
    return 1;
}

static int parse_uint(struct cursor_t *c, uint64_t *out) {
    skip_ws(c);
    if (at_end(c) || *c->p < '0' || *c->p > '9') {
        c->err = "expected unsigned integer";
        return 0;
    }
    uint64_t v = 0;
    while (!at_end(c) && *c->p >= '0' && *c->p <= '9') {
        v = v * 10 + (uint64_t) (*c->p - '0');
        c->p++;
    }
    *out = v;
    return 1;
}

// Skip a JSON value (any type). Handles objects, arrays, strings, numbers,
// true/false/null. Used for "__metadata__" which we don't care about.
static int skip_value(struct cursor_t *c) {
    skip_ws(c);
    if (at_end(c)) {
        c->err = "EOF in skip_value";
        return 0;
    }
    char ch = *c->p;
    if (ch == '{' || ch == '[') {
        char open = ch, close = (ch == '{') ? '}' : ']';
        c->p++;
        int depth = 1;
        while (!at_end(c) && depth > 0) {
            char k = *c->p;
            if (k == '"') {
                const char *dummy_str;
                size_t      dummy_len;
                if (!parse_string(c, &dummy_str, &dummy_len))
                    return 0;
            } else {
                if (k == open)
                    depth++;
                else if (k == close)
                    depth--;
                c->p++;
            }
        }
        return depth == 0;
    }
    if (ch == '"') {
        const char *s;
        size_t      l;
        return parse_string(c, &s, &l);
    }
    // number / true / false / null — consume until ,]} or whitespace
    while (!at_end(c) && *c->p != ',' && *c->p != ']' && *c->p != '}' && *c->p != ' ' &&
           *c->p != '\t' && *c->p != '\n' && *c->p != '\r') {
        c->p++;
    }
    return 1;
}

// ---- Tensor entry parsing ---------------------------------------------------

static int parse_tensor_entry(struct cursor_t *c, struct st_tensor_t *t) {
    if (!expect(c, '{'))
        return 0;
    int      saw_dtype = 0, saw_shape = 0, saw_offsets = 0;
    uint64_t off_start = 0, off_end = 0;

    while (1) {
        skip_ws(c);
        if (at_end(c)) {
            c->err = "EOF in tensor object";
            return 0;
        }
        if (*c->p == '}') {
            c->p++;
            break;
        }

        const char *key;
        size_t      key_len;
        if (!parse_string(c, &key, &key_len))
            return 0;
        if (!expect(c, ':'))
            return 0;
        skip_ws(c);

        if (key_len == 5 && strncmp(key, "dtype", 5) == 0) {
            const char *s;
            size_t      l;
            if (!parse_string(c, &s, &l))
                return 0;
            t->dtype  = parse_dtype(s, l);
            saw_dtype = 1;
        } else if (key_len == 5 && strncmp(key, "shape", 5) == 0) {
            if (!expect(c, '['))
                return 0;
            t->rank = 0;
            skip_ws(c);
            if (!at_end(c) && *c->p == ']') {
                c->p++;
            } else {
                while (1) {
                    if (t->rank >= ST_MAX_RANK) {
                        c->err = "rank exceeds ST_MAX_RANK";
                        return 0;
                    }
                    uint64_t v;
                    if (!parse_uint(c, &v))
                        return 0;
                    t->shape[t->rank++] = (size_t) v;
                    skip_ws(c);
                    if (at_end(c)) {
                        c->err = "EOF in shape array";
                        return 0;
                    }
                    if (*c->p == ']') {
                        c->p++;
                        break;
                    }
                    if (*c->p != ',') {
                        c->err = "expected , or ] in shape";
                        return 0;
                    }
                    c->p++;
                }
            }
            saw_shape = 1;
        } else if (key_len == 12 && strncmp(key, "data_offsets", 12) == 0) {
            if (!expect(c, '['))
                return 0;
            if (!parse_uint(c, &off_start))
                return 0;
            if (!expect(c, ','))
                return 0;
            if (!parse_uint(c, &off_end))
                return 0;
            if (!expect(c, ']'))
                return 0;
            saw_offsets = 1;
        } else {
            // Unknown key — skip its value gracefully.
            if (!skip_value(c))
                return 0;
        }

        skip_ws(c);
        if (at_end(c)) {
            c->err = "EOF after tensor field";
            return 0;
        }
        if (*c->p == ',') {
            c->p++;
            continue;
        }
        if (*c->p == '}') {
            c->p++;
            break;
        }
        c->err = "expected , or } in tensor object";
        return 0;
    }

    if (!saw_dtype || !saw_shape || !saw_offsets) {
        c->err = "tensor entry missing dtype/shape/data_offsets";
        return 0;
    }
    if (off_end < off_start) {
        c->err = "tensor data_offsets reversed";
        return 0;
    }
    t->nbytes = (size_t) (off_end - off_start);
    t->data   = (const void *) (uintptr_t) off_start; // placeholder, fixed up after open
    return 1;
}

// ---- Open / close -----------------------------------------------------------

struct st_ctx *st_open(const char *path, const char **errmsg) {
    struct io_map m;
    if (!io_map_ro(path, MAP_PRIVATE, &m, errmsg)) {
        return nullptr;
    }
    const int    fd    = m.fd;
    void        *map   = m.map;
    const size_t fsize = m.size;
    if (fsize < 8) {
        munmap(map, fsize);
        close(fd);
        io_set_err(errmsg, "file too small for header");
        return nullptr;
    }

    uint64_t header_size;
    memcpy(&header_size, map, 8); // little-endian per spec
    if (header_size > fsize - 8) {
        munmap(map, fsize);
        close(fd);
        io_set_err(errmsg, "header_size exceeds file");
        return nullptr;
    }

    struct st_ctx *ctx = heap_calloc_array_aligned(struct st_ctx, 1);
    if (!ctx) {
        munmap(map, fsize);
        close(fd);
        io_set_err(errmsg, "calloc ctx");
        return nullptr;
    }
    ctx->fd          = fd;
    ctx->map         = map;
    ctx->map_size    = fsize;
    ctx->header_size = (size_t) header_size;
    ctx->data_region = 8 + (size_t) header_size;

    // Upper-bound name_arena size by header bytes.
    ctx->name_arena = heap_alloc_array_aligned(char, (size_t) header_size + 1);
    if (!ctx->name_arena) {
        safe_free((void **) &ctx);
        munmap(map, fsize);
        close(fd);
        io_set_err(errmsg, "malloc name_arena");
        return nullptr;
    }

    struct cursor_t c = {
            .p   = (const char *) map + 8,
            .end = (const char *) map + 8 + header_size,
            .err = nullptr,
    };

    if (!expect(&c, '{')) {
        st_close(ctx);
        io_set_err(errmsg, "header is not a JSON object");
        return nullptr;
    }

    while (1) {
        skip_ws(&c);
        if (at_end(&c)) {
            st_close(ctx);
            io_set_err(errmsg, "EOF in header");
            return nullptr;
        }
        if (*c.p == '}') {
            c.p++;
            break;
        }

        const char *key;
        size_t      key_len;
        if (!parse_string(&c, &key, &key_len)) {
            st_close(ctx);
            io_set_err(errmsg, c.err ? c.err : "parse_string failed");
            return nullptr;
        }
        if (!expect(&c, ':')) {
            st_close(ctx);
            io_set_err(errmsg, "expected ':' after key");
            return nullptr;
        }

        // "__metadata__" is parsed but discarded.
        if (key_len == 12 && strncmp(key, "__metadata__", 12) == 0) {
            if (!skip_value(&c)) {
                st_close(ctx);
                io_set_err(errmsg, c.err ? c.err : "skip __metadata__");
                return nullptr;
            }
        } else {
            if (ctx->count == ctx->cap &&
                !heap_grow_array(ctx->tensors, ctx->cap, ctx->cap ? ctx->cap * 2 : 64)) {
                st_close(ctx);
                io_set_err(errmsg, "alloc tensors");
                return nullptr;
            }
            // Copy name into arena.
            char *name_dst = ctx->name_arena + ctx->name_arena_used;
            memcpy(name_dst, key, key_len);
            name_dst[key_len] = '\0';
            ctx->name_arena_used += key_len + 1;

            struct st_tensor_t *t = &ctx->tensors[ctx->count];
            memset(t, 0, sizeof(*t));
            t->name = name_dst;
            if (!parse_tensor_entry(&c, t)) {
                st_close(ctx);
                io_set_err(errmsg, c.err ? c.err : "parse_tensor_entry");
                return nullptr;
            }
            // Fix up data pointer: parse_tensor_entry stored offset_start.
            uintptr_t offset = (uintptr_t) t->data;
            if (offset + t->nbytes > fsize - ctx->data_region) {
                st_close(ctx);
                io_set_err(errmsg, "tensor offset out of file");
                return nullptr;
            }
            t->data = (const char *) map + ctx->data_region + offset;
            ctx->count++;
        }

        skip_ws(&c);
        if (at_end(&c)) {
            st_close(ctx);
            io_set_err(errmsg, "EOF in header");
            return nullptr;
        }
        if (*c.p == ',') {
            c.p++;
            continue;
        }
        if (*c.p == '}') {
            c.p++;
            break;
        }
        st_close(ctx);
        io_set_err(errmsg, "expected , or } at header level");
        return nullptr;
    }

    return ctx;
}

void st_close(struct st_ctx *ctx) {
    if (!ctx)
        return;
    if (ctx->map && ctx->map != MAP_FAILED)
        munmap(ctx->map, ctx->map_size);
    if (ctx->fd >= 0)
        close(ctx->fd);
    safe_free((void **) &ctx->name_arena);
    safe_free((void **) &ctx->tensors);
    safe_free((void **) &ctx);
}

size_t st_count(const struct st_ctx *ctx) {
    return ctx ? ctx->count : 0;
}

const struct st_tensor_t *st_get(const struct st_ctx *ctx, const char *name) {
    if (!ctx || !name)
        return nullptr;
    for (size_t i = 0; i < ctx->count; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return &ctx->tensors[i];
    }
    return nullptr;
}

// ---- BF16 weight loaders ----------------------------------------------------

static float bf16_bits_to_f32(const unsigned char *p) {
    // memcpy: tensor data sits at any byte offset in the map
    uint16_t raw;
    memcpy(&raw, p, sizeof raw);
    const uint32_t bits = (uint32_t) raw << 16;
    float          f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static const struct st_tensor_t *
bf16_tensor(const struct st_ctx *ctx, const char *name, size_t elems, const char *who) {
    const struct st_tensor_t *t = st_get(ctx, name);
    if (t == nullptr) {
        fprintf(stderr, "%s: missing tensor '%s'\n", who, name);
        return nullptr;
    }
    if (t->dtype != ST_DTYPE_BF16) {
        fprintf(stderr, "%s: '%s' dtype=%s, want BF16\n", who, name, st_dtype_name(t->dtype));
        return nullptr;
    }
    size_t want = 0;
    if (ckd_mul(&want, elems, (size_t) 2) || t->nbytes != want) {
        fprintf(stderr, "%s: '%s' has %zu bytes, want %zu elements\n", who, name, t->nbytes, elems);
        return nullptr;
    }
    return t;
}

float *st_load_bf16_f32(const struct st_ctx *ctx,
                        size_t               n_parts,
                        size_t               part_elems,
                        const char *const    names[static n_parts],
                        const char          *who) {
    size_t total = 0, bytes = 0;
    if (ckd_mul(&total, n_parts, part_elems) || ckd_mul(&bytes, total, sizeof(float))) {
        return nullptr;
    }
    const struct st_tensor_t *parts[4];
    if (n_parts == 0 || n_parts > sizeof parts / sizeof parts[0]) {
        return nullptr;
    }
    for (size_t i = 0; i < n_parts; i++) {
        parts[i] = bf16_tensor(ctx, names[i], part_elems, who);
        if (parts[i] == nullptr) {
            return nullptr;
        }
    }
    float *dst = heap_alloc_large(bytes, alignof(float));
    if (dst == nullptr) {
        fprintf(stderr, "%s: no memory for '%s'\n", who, names[0]);
        return nullptr;
    }
    for (size_t i = 0; i < n_parts; i++) {
        const unsigned char *src = parts[i]->data;
        float               *out = dst + i * part_elems;
        for (size_t j = 0; j < part_elems; j++) {
            out[j] = bf16_bits_to_f32(src + 2 * j);
        }
    }
    return dst;
}

bool st_load_bf16_scalar(const struct st_ctx *ctx, const char *name, const char *who, float *out) {
    const struct st_tensor_t *t = st_get(ctx, name);
    if (t == nullptr || t->dtype != ST_DTYPE_BF16 || t->nbytes != 2) {
        fprintf(stderr, "%s: missing/bad scalar '%s'\n", who, name);
        return false;
    }
    *out = bf16_bits_to_f32(t->data);
    return true;
}

bool st_load_clip4(const struct st_ctx *ctx,
                   const char          *prefix,
                   const char          *who,
                   float               *in_min,
                   float               *in_max,
                   float               *out_min,
                   float               *out_max) {
    static const char *const suffix[4] = {"input_min", "input_max", "output_min", "output_max"};
    float *const             out[4]    = {in_min, in_max, out_min, out_max};
    for (size_t i = 0; i < 4; i++) {
        char name[384];
        snprintf(name, sizeof name, "%s%s", prefix, suffix[i]);
        if (!st_load_bf16_scalar(ctx, name, who, out[i])) {
            return false;
        }
    }
    return true;
}
