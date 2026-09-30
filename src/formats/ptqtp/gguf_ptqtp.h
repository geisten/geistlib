/*
 * gguf_ptqtp — loader for PTQTP-quantized weight files (.ptqtp.bin).
 *
 * The writer, ptqtp_quantize_full.py, is not in this repository
 * (tools/ptqtp_single_tensor.py is the single-tensor validator). The file
 * format stores per-tensor 2-trit-plane representations with per-row-group
 * FP16 scales:
 *
 *   header: magic "PTQT", version, n_tensors, group_size, n_planes
 *   TOC (one entry per tensor): name, n_in, n_out, n_groups, offsets+sizes,
 *                                cos_sim (diagnostic)
 *   data section (16-byte aligned): per tensor [trits, alpha]
 *
 * Per tensor:
 *   trits  uint8[n_out * n_in / 2]   row-major, 4 bits/weight,
 *                                    byte = (idx_col_2j+1 << 4) | idx_col_2j
 *                                    idx = (T1+1)*3 + (T2+1) ∈ {0..8}
 *   alpha  float16[n_out, n_groups, 2]
 *
 * Memory model: opens via mmap; returned tensor pointers are valid for the
 * lifetime of the struct ptqtp_ctx (they alias the mapped file region).
 *
 * Errors are returned via a const char* output parameter on the open call;
 * accessor functions return nullptr on miss (no exceptions, no asserts).
 */
#ifndef GGUF_PTQTP_H
#define GGUF_PTQTP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>

struct ptqtp_ctx;

/* Per-tensor storage layout for trits.
 *   PTQTP_STORE_STANDARD: layout determined by n_planes alone
 *     n_planes=2 → joint nibble (n_in/2 bytes/row, 4 bpw)
 *     n_planes=3 → 1 byte per weight idx 0..26 (n_in bytes/row, 8 bpw)
 *   PTQTP_STORE_PACKED5: only valid with n_planes=3. Two sub-streams per
 *     row — low nibble (n_in/2 bytes) followed by high bit (n_in/8 bytes,
 *     8 weights per byte). Total: 5n_in/8 bytes/row = 5 bpw.
 */
enum ptqtp_storage {
    PTQTP_STORE_STANDARD = 0,
    PTQTP_STORE_PACKED5  = 1,
};

struct ptqtp_tensor_t {
    const char     *name; /* NUL-terminated, owned by ctx */
    uint32_t        n_in;
    uint32_t        n_out;
    uint32_t        n_groups;   /* = n_in / group_size, validated at load */
    uint8_t         n_planes;   /* PER-TENSOR: 2 or 3. v1 files: copied from header. */
    uint8_t         storage;    /* enum ptqtp_storage; v1 files default to 0. */
    const uint8_t  *trits;      /* aliases mmap; layout per (n_planes, storage). */
    const uint16_t *alpha;      /* aliases mmap; n_out * n_groups * n_planes fp16 */
    const float    *alpha_fp32; /* owned by ctx; converted from alpha at load. */
    float           cos_sim;    /* diagnostic, not used at runtime */
};

/* Open + parse a .ptqtp.bin file. Returns nullptr on error with *err set
 * (err string is static, no need to free). On success caller must close. */
[[nodiscard]] struct ptqtp_ctx *ptqtp_open(const char *path, const char **err);
void                            ptqtp_close(struct ptqtp_ctx *ctx);

/* Returns nullptr if name not found. Pointer is valid until close. */
[[nodiscard]] const struct ptqtp_tensor_t *ptqtp_get_tensor(const struct ptqtp_ctx *ctx,
                                                            const char             *name);

/* Header introspection. */
size_t   ptqtp_n_tensors(const struct ptqtp_ctx *ctx);
uint32_t ptqtp_group_size(const struct ptqtp_ctx *ctx);
/* Header-level n_planes. Note: in v2 (mixed) files this is the MAJORITY
 * value; per-tensor n_planes lives in struct ptqtp_tensor_t::n_planes. Use the
 * per-tensor field for kernel dispatch. */
uint32_t ptqtp_n_planes(const struct ptqtp_ctx *ctx);
/* True iff this file mixes per-tensor n_planes (any tensor's n_planes
 * differs from the header value). v1 files always return false. */
bool ptqtp_is_mixed(const struct ptqtp_ctx *ctx);

#endif /* GGUF_PTQTP_H */
