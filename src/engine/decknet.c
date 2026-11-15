/*
 * src/engine/decknet.c — the public entry points of <geist_decknet.h> (#532).
 *
 * Layer: ENGINE, and the composition point for src/archs/decknet/
 * (tools/layers.json). Compiled only with DECKNET=1 (mk/common.mk).
 *
 * decknet is not in arch_registry.c. That registry answers "which decoder,
 * with which towers, does geist_model_load build for this
 * general.architecture", and every reader of a descriptor goes on to a
 * model, a tokenizer and a session; decknet has none of them and its own
 * entry point, so the one name it accepts is checked where it is loaded
 * (decknet_create), fail-closed like the registry lookup. geist_model_load
 * keeps refusing a decknet file with GEIST_E_UNSUPPORTED.
 */
#define GEIST_INTERNAL_ENGINE_LAYER
#include "error.h"
#include "gguf_reader.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../archs/decknet/decknet.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <geist_decknet.h>

#include <math.h>
#include <string.h>

/* set_homographies rejects NaN and Inf by comparison. */
static_assert(__FINITE_MATH_ONLY__ == 0,
              "decknet needs -fno-finite-math-only (mk/common.mk): set_homographies rejects NaN "
              "and Inf by comparison");

/* Closes `g` and reports what decknet_create said about it. */
[[nodiscard]] static enum geist_status
load_from_ctx(const char *fn, struct gguf_ctx *g, struct geist_decknet **out) {
    char                    why[256] = "";
    const enum geist_status st       = decknet_create(sizeof why, g, why, out);
    gguf_close(g);
    if (st != GEIST_OK) {
        geist_error_set_create_time(st, fn, "%s", why);
    }
    return st;
}

enum geist_status geist_decknet_load(const char *path, struct geist_decknet **out) {
    static const char fn[] = "geist_decknet_load";
    if (out == nullptr) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG, fn, "out is null");
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (path == nullptr || path[0] == '\0') {
        geist_error_set_create_time(GEIST_E_INVALID_ARG, fn, "path is null or empty");
        return GEIST_E_INVALID_ARG;
    }
    const char      *why = nullptr;
    struct gguf_ctx *g   = gguf_open(path, &why);
    if (g == nullptr) {
        geist_error_set_create_time(
                GEIST_E_IO, fn, "gguf_open(%s): %s", path, why != nullptr ? why : "unknown error");
        return GEIST_E_IO;
    }
    return load_from_ctx(fn, g, out);
}

enum geist_status
geist_decknet_load_from_memory(size_t size, const uint8_t *data, struct geist_decknet **out) {
    static const char fn[] = "geist_decknet_load_from_memory";
    if (out == nullptr) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG, fn, "out is null");
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (data == nullptr || size == 0) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG, fn, "data is null or empty");
        return GEIST_E_INVALID_ARG;
    }
    const char      *why = nullptr;
    struct gguf_ctx *g   = gguf_open_memory(data, size, &why);
    if (g == nullptr) {
        geist_error_set_create_time(
                GEIST_E_FORMAT, fn, "gguf_open_memory: %s", why != nullptr ? why : "unknown error");
        return GEIST_E_FORMAT;
    }
    return load_from_ctx(fn, g, out);
}

void geist_decknet_destroy(struct geist_decknet *d) {
    decknet_destroy(d);
}

enum geist_status geist_decknet_info(const struct geist_decknet *d,
                                     struct geist_decknet_info  *out) {
    if (d == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = (struct geist_decknet_info) {
            .slots          = d->slots,
            .n_slots        = d->n_slots,
            .image_width    = d->img_w,
            .image_height   = d->img_h,
            .n_planes       = DECKNET_PLANES,
            .planes_mm      = d->planes_mm,
            .n_homographies = d->n_h,
            .n_tips         = d->n_tips,
            .n_tubes        = d->n_tubes,
            .occ            = d->occ,
            .smooth         = d->smooth,
    };
    return GEIST_OK;
}

enum geist_status geist_decknet_set_homographies(struct geist_decknet    *d,
                                                 size_t                   n,
                                                 enum geist_decknet_image image,
                                                 const float             *h) {
    if (d == nullptr || h == nullptr || n != d->n_h ||
        (image != GEIST_DECKNET_IMAGE_CURRENT && image != GEIST_DECKNET_IMAGE_EMPTY)) {
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < n * 9; i++) {
        if (!isfinite(h[i])) {
            return GEIST_E_INVALID_ARG;
        }
    }
    memcpy(image == GEIST_DECKNET_IMAGE_CURRENT ? d->h_cur : d->h_ref, h, n * 9 * sizeof h[0]);
    return GEIST_OK;
}

enum geist_status geist_decknet_run(struct geist_decknet        *d,
                                    size_t                       n_bytes,
                                    const uint8_t               *cur,
                                    const uint8_t               *ref,
                                    struct geist_decknet_result *out) {
    if (out != nullptr) {
        *out = (struct geist_decknet_result) {0};
    }
    if (d == nullptr || cur == nullptr || ref == nullptr || out == nullptr ||
        n_bytes != d->img_bytes) {
        return GEIST_E_INVALID_ARG;
    }
    decknet_run(d, n_bytes, cur, ref);
    *out = d->result;
    return GEIST_OK;
}

enum geist_status
geist_decknet_slot(const struct geist_decknet *d, size_t slot, struct geist_decknet_slot *out) {
    if (out != nullptr) {
        *out = (struct geist_decknet_slot) {0};
    }
    if (d == nullptr || out == nullptr || slot >= d->n_slots) {
        return GEIST_E_INVALID_ARG;
    }
    if (!d->have_run || !d->result.checkable) {
        return GEIST_E_INVALID_STATE;
    }
    *out = d->slot_out[slot];
    return GEIST_OK;
}
