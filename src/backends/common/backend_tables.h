/*
 * src/backends/common/backend_tables.h — defining a backend's tables so a
 * missing required slot is a compile error (#345).
 *
 * Layer: BACKEND (every backend under src/backends/).
 *
 * <geist_backend.h> says which slots of geist_backend_vtbl,
 * geist_backend_primitives and geist_backend_descriptor may be null. These
 * initializers take every other slot as a positional argument, so a table
 * without one does not compile: a missing argument is a preprocessor error,
 * and each is taken by address, which a function (or a table object) has and
 * nullptr has not. The optional slots follow by designator; setting a
 * required one there again is an initializer override, which
 * -Woverride-init / -Winitializer-overrides reject under -Werror.
 *
 *   static const struct geist_backend_vtbl my_vtbl = GEIST_BACKEND_VTBL(
 *           my_destroy, my_buffer_create, ...,
 *           .create = my_create, .buffer_copy = my_buffer_copy);
 *
 * scripts/check_wiring.py refuses a vtbl, primitives or descriptor under
 * src/backends/ defined any other way, and tests/wiring/backend_*.c are the
 * mistakes it must keep failing on.
 *
 * Required, by table (the optional slots are the rest; each says so in
 * <geist_backend.h>):
 *
 *   vtbl        destroy, buffer_create, buffer_destroy,
 *               buffer_create_aliased, buffer_upload, buffer_download,
 *               buffer_map, buffer_unmap
 *   primitives  rmsnorm, add, mul, gelu_tanh, silu, rope_apply, attention
 *   descriptor  name, vtbl, prims
 */
#pragma once

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "backend_tables.h is internal to the backend layer."
#endif

#include <geist_backend.h>

#define GEIST_BACKEND_VTBL(destroy_,                     \
                           buffer_create_,               \
                           buffer_destroy_,              \
                           buffer_create_aliased_,       \
                           buffer_upload_,               \
                           buffer_download_,             \
                           buffer_map_,                  \
                           buffer_unmap_,                \
                           ...)                          \
    {.destroy               = &(destroy_),               \
     .buffer_create         = &(buffer_create_),         \
     .buffer_destroy        = &(buffer_destroy_),        \
     .buffer_create_aliased = &(buffer_create_aliased_), \
     .buffer_upload         = &(buffer_upload_),         \
     .buffer_download       = &(buffer_download_),       \
     .buffer_map            = &(buffer_map_),            \
     .buffer_unmap          = &(buffer_unmap_),          \
     __VA_ARGS__}

#define GEIST_BACKEND_PRIMS(rmsnorm_, add_, mul_, gelu_tanh_, silu_, rope_apply_, attention_, ...) \
    {.rmsnorm    = &(rmsnorm_),                                                                    \
     .add        = &(add_),                                                                        \
     .mul        = &(mul_),                                                                        \
     .gelu_tanh  = &(gelu_tanh_),                                                                  \
     .silu       = &(silu_),                                                                       \
     .rope_apply = &(rope_apply_),                                                                 \
     .attention  = &(attention_),                                                                  \
     __VA_ARGS__}

/* name_ must be a string literal; vtbl_ and prims_ the table objects. */
#define GEIST_BACKEND_DESCRIPTOR(name_, vtbl_, prims_, ...) \
    {.name = "" name_ "", .vtbl = &(vtbl_), .prims = &(prims_), __VA_ARGS__}
