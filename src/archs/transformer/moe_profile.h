/*
 * src/archs/transformer/moe_profile.h - a mixture of experts' expert
 * profile, counted at run time and kept with the model (#165).
 *
 * The router's selections per (layer, expert) are counted while sessions
 * run (blk_moe_ffn routes on the host anyway; a decode step routed on the
 * device, #201, is not counted -- every expert is on that GPU already). With the cache on, the
 * counts are added to the ones read at load when the model is destroyed
 * and written to the profile cache, one file per model:
 *
 *   GEIST_MOE_PROFILE_CACHE unset, "" or "0"  no cache (the default)
 *   GEIST_MOE_PROFILE_CACHE=1     $XDG_CACHE_HOME/geist/moe-profile or
 *                                 ~/.cache/geist/moe-profile
 *   GEIST_MOE_PROFILE_CACHE=<dir> that directory
 *   file <dir>/<key>.txt, key = 16 hex digits (moe_profile_key)
 *
 * The next load places the GPU's expert copies by them (moe_gpu.c), unless
 * GEIST_MOE_PROFILE names a file, which wins. Opt-in because a profile
 * trades prefill for decode with the model on a discrete GPU: Qwen3.6-35B-A3B
 * on an RTX 2080 Ti decoded ~16 % faster and prefilled ~10 % slower with
 * one (runs 1826/1830), the GPU then holding 79 % of the rows a prefill
 * chunk routes. With the 2,000 hottest experts instead of all 3,862 that
 * fit, prefill was at 568-579 tok/s (run 1845; 507 with all of them, 561
 * without a profile, runs 1840/1841).
 *
 * Prefill rows (#239 follow-up): a prefill chunk's experts are counted a
 * second time, by rows: expert e of a layer with m rows in the chunk adds
 * m to its bucket floor(log2 m) (the last, MOE_PROFILE_BUCKETS - 1, takes
 * every m from 512 on). A staging threshold m_min (a power of two) then
 * reads off the rows a prefill would have staged: the buckets from
 * log2(m_min) on. The GPU backend's host copies of cold experts are
 * chosen by them (moe_gpu.c, moe_stage_order); the resident set still
 * follows the USAGE counts. The last measured threshold is kept too.
 *
 * The format is the USAGE lines of tools/moe_usage, so analyze.py and
 * GEIST_MOE_PROFILE read it too, then the prefill section:
 *
 *   # geist moe profile v2 key <key> layers <L> experts <E>
 *   USAGE <key> profile run-time <tokens> <layer> <c0>,<c1>,...,<c(E-1)>
 *   STAGE <key> m-min <rows>
 *   PREFILL <key> rows-from <2^b> <layer> <r0>,<r1>,...,<r(E-1)>
 *
 * one USAGE line per layer, in layer order; STAGE (0: no threshold
 * measured yet); one PREFILL line per layer and bucket, layer-major. A v1
 * file (the USAGE lines alone) still loads, with no prefill rows. A file
 * whose header names another key, layer or expert count, or whose lines
 * do not parse, is ignored with one line on stderr. Two processes on the
 * same model: the last one to finish writes its counts (the file is
 * replaced whole).
 */
#ifndef GEIST_TRANSFORMER_MOE_PROFILE_H
#define GEIST_TRANSFORMER_MOE_PROFILE_H

#include <stddef.h>
#include <stdint.h>

struct gguf_ctx;
struct transformer_moe_profile;

/* Prefill row buckets per expert: [2^b, 2^(b+1)) rows, the last from 512. */
constexpr size_t MOE_PROFILE_BUCKETS = 10;

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

/* Counts a prefill chunk's rows of layer `layer` by expert: expert e ran
 * row_start[e + 1] - row_start[e] rows (moe_bucket's offsets; the entries
 * past n_exp, a shared expert, are not read). Same rules as
 * moe_profile_count. */
void moe_profile_count_chunk(struct transformer_moe_profile *p,
                             size_t                          layer,
                             size_t                          n_exp,
                             const size_t                    row_start[static n_exp + 1]);

/* The prefill rows a staging threshold of m_min rows would have taken,
 * out[layer * n_exp + e]: the buckets from floor(log2 m_min) on, the
 * file's plus this process's. False (out zeroed) when no prefill was
 * counted at all: a v1 file, a fresh cache, no cache. */
[[nodiscard]] bool moe_profile_prefill_rows(const struct transformer_moe_profile *p,
                                            size_t                                m_min,
                                            size_t                                n,
                                            uint64_t                              out[static n]);

/* The staging threshold kept with the profile: this process's if it set
 * one, else the file's (0: none). */
[[nodiscard]] size_t moe_profile_stage_m_min(const struct transformer_moe_profile *p);
void                 moe_profile_set_stage_m_min(struct transformer_moe_profile *p, size_t m_min);

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
