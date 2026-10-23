/*
 * src/archs/transformer/moe_profile.h - a mixture of experts' expert
 * profile, counted at run time and kept with the model (#165).
 *
 * The router's selections per (layer, expert) are counted while sessions
 * run (blk_moe_ffn routes on the host anyway). When the model is
 * destroyed, the counts are added to the ones read at load and written to
 * the profile cache, one file per model:
 *
 *   GEIST_MOE_PROFILE_CACHE=<dir>  the directory; "0" or "": no cache
 *   else $XDG_CACHE_HOME/geist/moe-profile or ~/.cache/geist/moe-profile
 *   file <dir>/<key>.txt, key = 16 hex digits (moe_profile_key)
 *
 * The next load places the GPU's expert copies by them (moe_gpu.c), unless
 * GEIST_MOE_PROFILE names a file, which wins. The format is the USAGE
 * lines of tools/moe_usage, so analyze.py reads it too:
 *
 *   # geist moe profile v1 key <key> layers <L> experts <E>
 *   USAGE <key> profile run-time <tokens> <layer> <c0>,<c1>,...,<c(E-1)>
 *
 * one USAGE line per layer, in layer order. A file whose header names
 * another key, layer or expert count, or whose lines do not parse, is
 * ignored with one line on stderr. Two processes on the same model: the
 * last one to finish writes its counts (the file is replaced whole).
 */
#ifndef GEIST_TRANSFORMER_MOE_PROFILE_H
#define GEIST_TRANSFORMER_MOE_PROFILE_H

#include <stddef.h>
#include <stdint.h>

struct gguf_ctx;
struct transformer_moe_profile;

/* A key for the model in `gguf`: FNV-1a over its tensor directory (name,
 * dtype, dims, offset) and layer 0's router weights, so another file of
 * the same shape (another quantization, a fine-tune) gets another key. */
[[nodiscard]] uint64_t moe_profile_key(const struct gguf_ctx *gguf);

/* Counts for n_layers x n_exp routed experts of the model `key`; `dir` is
 * the cache directory (nullptr: no file is read or written). Reads the
 * file if there is one. nullptr when out of memory. */
[[nodiscard]] struct transformer_moe_profile *
moe_profile_open(size_t n_layers, size_t n_exp, uint64_t key, const char *dir);

/* The cache directory from the environment (see above) into dir[0, n);
 * false: no cache. */
[[nodiscard]] bool moe_profile_dir(size_t n, char dir[static n]);

/* Counts the routed experts of `seq` tokens of layer `layer`: ids holds
 * stride entries per token, the first n_routed of them routed (expert ids
 * < n_exp). A layer past the table (an MTP block) is not counted. Safe from
 * several threads. */
void moe_profile_count(struct transformer_moe_profile *p,
                       size_t                          layer,
                       size_t                          seq,
                       size_t                          stride,
                       size_t                          n_routed,
                       const uint32_t                  ids[static seq * stride]);

/* The counts read at load plus the ones since, out[layer * n_exp + e]; the
 * file's tokens plus this process's in *tokens. */
void moe_profile_counts(const struct transformer_moe_profile *p,
                        size_t                                n,
                        uint64_t                              out[static n],
                        uint64_t                             *tokens);

/* True when the cache file gave counts at load. */
[[nodiscard]] bool moe_profile_loaded(const struct transformer_moe_profile *p);

/* The cache file's path (empty without a cache). */
[[nodiscard]] const char *moe_profile_path(const struct transformer_moe_profile *p);

/* Writes the counts when anything was counted since the load, then frees
 * `p`. nullptr is a no-op. */
void moe_profile_close(struct transformer_moe_profile *p);

#endif
